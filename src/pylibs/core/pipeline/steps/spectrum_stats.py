"""SpectrumStatsStep: computes the min, max, and mean spectrum across every
scan in a sample, streaming batches through its reader, and writes the
result to the results store as a (3, n_pix) array. This backs the
diagnostic-spectra panel in the PDF report.

Also caches the raw file's `wavelengths` array, its per-detector channel
masks (if present, as a proper `detectors` array -- see `ResultsStore.
create_detectors_array`), and its scalar `params` (see `set_raw_params`) --
all already-loaded attributes on the same `reader` this step has open
regardless, so this costs nothing extra, and it's what lets
`get_spectrum_stats`/`generate_feature_report`/
`generate_applied_regression_report`/bootstrap's "local" resampling method
read this header metadata back from the store instead of ever reopening the
raw file themselves.

Not every raw file's `params` necessarily has a `Detectors` key, or spells
its wavelength-calibration key `slambda` (that key isn't itself used for
anything -- the actual wavelength axis always comes from `reader.
wavelengths` independently) -- both are logged as a warning, not an error,
since the pipeline doesn't depend on either.
"""

import numpy as np

from pylibs.core.io.reader import LibsReader
from pylibs.core.io.store import ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.project.registry import Registry

logger = get_logger(__name__)


class SpectrumStatsStep:
    name = "spectrum_stats"
    produces = "spectrum_stats"
    requires: str | None = None
    needs_reader = True

    def is_done(self, store: ResultsStore, sample_id: str, registry: Registry) -> bool:
        return store.has_spectrum_stats(sample_id)

    def run(
        self, reader: LibsReader | None, sample_id: str, store: ResultsStore, registry: Registry
    ) -> None:
        assert reader is not None, "SpectrumStatsStep.needs_reader is True"
        n_pix = len(reader.wavelengths)
        running_min = np.full(n_pix, np.inf)
        running_max = np.full(n_pix, -np.inf)
        running_sum = np.zeros(n_pix)
        n_scans = 0

        for batch in reader.iter_window():
            running_min = np.minimum(running_min, batch.min(axis=0))
            running_max = np.maximum(running_max, batch.max(axis=0))
            running_sum += batch.sum(axis=0)
            n_scans += batch.shape[0]

        mean = running_sum / n_scans if n_scans else running_sum
        store.create_spectrum_stats_array(sample_id, running_min, running_max, mean)
        store.create_wavelengths_array(sample_id, reader.wavelengths)
        store.set_raw_params(sample_id, reader.params)

        self._store_mask(reader, sample_id, store, n_scans)

        if "Detectors" in reader.params:
            store.create_detectors_array(sample_id, reader.params["Detectors"])
        else:
            logger.warning(
                "Sample %r's raw params have no 'Detectors' key -- per-detector channel data "
                "won't be available",
                sample_id,
            )
        if "slambda" not in reader.params:
            logger.warning(
                "Sample %r's raw params have no 'slambda' key -- the raw file's own "
                "wavelength-calibration metadata wasn't found under that name (this doesn't "
                "affect the wavelength axis itself, read from the file's own wavelengths data "
                "independently)",
                sample_id,
            )

    @staticmethod
    def _store_mask(reader: LibsReader, sample_id: str, store: ResultsStore, n_scans: int) -> None:
        """Cache the raw file's scanned-cell mask, if it has one.

        A partially-scanned sample (e.g. a round one in a square frame) records
        fewer scans than its frame has cells, and `PixelAssignmentMatrix` is
        what says which cells those were -- without it `raster.to_raster` can
        only reshape, which such a sample can't satisfy. Absent for a full
        rectangular raster, which is the norm for `.libs`: nothing stored, no
        complaint.

        Only stored when it can actually place this sample's scans -- right
        shape, and exactly one set cell per scan. A mask failing either would
        silently mis-place values, so it's logged and dropped instead.
        """
        raw_mask = reader.params.get("PixelAssignmentMatrix")
        if raw_mask is None:
            return
        mask = np.asarray(raw_mask).astype(bool)

        params = reader.params
        if "HeightPixels" in params and "WidthPixels" in params:
            expected = (int(params["HeightPixels"]), int(params["WidthPixels"]))
            if mask.shape != expected:
                logger.warning(
                    "Sample %r's PixelAssignmentMatrix is %s but its frame is %s -- ignoring it; "
                    "maps will fall back to a plain reshape",
                    sample_id,
                    mask.shape,
                    expected,
                )
                return

        n_set = int(mask.sum())
        if n_set != n_scans:
            logger.warning(
                "Sample %r's PixelAssignmentMatrix marks %d scanned cell(s) but the file has %d "
                "scan(s) -- ignoring it, since it can't say where each scan landed",
                sample_id,
                n_set,
                n_scans,
            )
            return

        store.create_mask_array(sample_id, mask)
        logger.debug(
            "Sample %r: cached a %dx%d scanned-cell mask (%d of %d cells scanned)",
            sample_id,
            mask.shape[0],
            mask.shape[1],
            n_set,
            mask.size,
        )
