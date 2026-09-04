"""FeatureStep: evaluates every registered feature expression for a sample,
streaming batches through its reader, and writes the results to the
results store as a (n_scans, n_features) array.

For every feature whose expression is a ratio (outermost operation is
division, see `features.expression.split_ratio`), also evaluates its
numerator/denominator sub-expressions from the same batches and stores them
as `<sample_id>/ratio_components/<feature_id>_numerator`/`..._denominator`
(see `ResultsStore.create_ratio_components`) -- the single place this ever
gets computed from raw data. `Distribution2DStep` and bootstrap read these
back rather than re-streaming the reader and re-evaluating themselves.

`is_done()` compares each registered feature's `feature_identity_hash`
(expression + its referenced peak table entries) against the hashes stamped
on the last-computed array, not just whether the array exists -- so editing
the peak table's definition of a peak id (same id, different wavelength/
continuum/tolerance) is detected and reprocessed, even though the feature_id
and expression string look unchanged. On any mismatch (or a new/missing
feature_id), the whole array is recomputed in one streaming pass -- cheaper
to reason about than splicing individual columns, and the raw file has to be
re-streamed once regardless of how many features actually changed.
"""

import numpy as np

from pylibs.core.features.expression import evaluate_expression, peaks_in_range, split_ratio
from pylibs.core.features.identity import feature_identity_hash
from pylibs.core.features.peak_table import PeakTable
from pylibs.core.io.reader import LibsReader
from pylibs.core.io.store import ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.project.registry import Registry

logger = get_logger(__name__)


class FeatureStep:
    name = "features"
    produces = "features"
    requires: str | None = None
    needs_reader = True

    def __init__(self, peak_table: PeakTable):
        self.peak_table = peak_table

    def is_done(self, store: ResultsStore, sample_id: str, registry: Registry) -> bool:
        array = store.get_features_array(sample_id)
        if array is None:
            return False
        current_hashes = {
            fid: feature_identity_hash(config.expression, self.peak_table)
            for fid, config in registry.features.items()
        }
        return array.attrs.get("feature_hashes", {}) == current_hashes

    def run(
        self, reader: LibsReader | None, sample_id: str, store: ResultsStore, registry: Registry
    ) -> None:
        assert reader is not None, "FeatureStep.needs_reader is True"
        feature_ids = sorted(registry.features)
        if not feature_ids:
            logger.warning("No features registered -- nothing to compute for sample %r", sample_id)
            return

        feature_hashes = {
            fid: feature_identity_hash(registry.features[fid].expression, self.peak_table)
            for fid in feature_ids
        }
        wl_min, wl_max = float(reader.wavelengths.min()), float(reader.wavelengths.max())
        out_of_range_ids: set[str] = set()
        for fid in feature_ids:
            expression = registry.features[fid].expression
            bad_peaks = peaks_in_range(expression, self.peak_table, reader.wavelengths)
            if bad_peaks:
                # INFO, not WARNING: this reaches the project log but not the
                # console. Registering an element's whole line list and letting
                # the ones this instrument can't see fall away is ordinary use
                # (see api.add_element_features), and the ids are stored on the
                # features array anyway, so reports already exclude them.
                logger.info(
                    "Feature %r (%r) references peak id(s) %s outside sample %r's wavelength "
                    "range (%.1f-%.1f nm) -- storing NaN for it.",
                    fid,
                    expression,
                    sorted(set(bad_peaks)),
                    sample_id,
                    wl_min,
                    wl_max,
                )
                out_of_range_ids.add(fid)
        if out_of_range_ids:
            logger.info(
                "Sample %r: %d of %d feature(s) reference peaks outside its %.1f-%.1f nm range "
                "and are stored as NaN: %s",
                sample_id,
                len(out_of_range_ids),
                len(feature_ids),
                wl_min,
                wl_max,
                ", ".join(sorted(out_of_range_ids)),
            )

        array = store.create_features_array(
            sample_id, reader.n_scans, feature_ids, feature_hashes, sorted(out_of_range_ids)
        )

        ratios = {
            fid: split
            for fid in feature_ids
            if fid not in out_of_range_ids
            and (split := split_ratio(registry.features[fid].expression)) is not None
        }
        numerator_chunks: dict[str, list[np.ndarray]] = {fid: [] for fid in ratios}
        denominator_chunks: dict[str, list[np.ndarray]] = {fid: [] for fid in ratios}

        start = 0
        for batch in reader.iter_window():
            values = np.column_stack(
                [
                    np.full(batch.shape[0], np.nan)
                    if fid in out_of_range_ids
                    else evaluate_expression(
                        registry.features[fid].expression,
                        self.peak_table,
                        batch,
                        reader.wavelengths,
                    )
                    for fid in feature_ids
                ]
            )
            store.write_chunk(array, start, values)
            start += batch.shape[0]

            for fid, (num_expr, den_expr) in ratios.items():
                numerator_chunks[fid].append(
                    evaluate_expression(num_expr, self.peak_table, batch, reader.wavelengths)
                )
                denominator_chunks[fid].append(
                    evaluate_expression(den_expr, self.peak_table, batch, reader.wavelengths)
                )

        for fid in ratios:
            # +1 here too, matching expression.py's own Div handling exactly
            # (num_expr/den_expr are typically plain sums of peak ids, with
            # no division of their own to already apply it)
            numerator = np.concatenate(numerator_chunks[fid]) + 1
            denominator = np.concatenate(denominator_chunks[fid]) + 1
            store.create_ratio_components(sample_id, fid, numerator, denominator)
