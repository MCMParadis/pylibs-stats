"""Distribution2DStep: for every registered feature whose expression is a
ratio (numerator / denominator, split by outermost division -- see
`features.expression.split_ratio`), computes a joint 2D histogram between
the numerator's and denominator's raw per-scan values plus every 2D metric
(see `pipeline.metrics.METRIC_NAMES_2D`), and writes them to the results
store. A feature that isn't a ratio (a bare peak id, a sum, etc.) has
nothing to compute here.

Reads numerator/denominator back from `<sample_id>/ratio_components/` (see
`ResultsStore.get_ratio_components`) rather than streaming the reader and
re-evaluating them -- `FeatureStep` already computed and stored them, as the
one place a ratio's numerator/denominator sub-expressions ever get
evaluated from the raw spectrum. Metric values are persisted generically
(`distribution.metrics`, keyed by the same display names as
`METRIC_NAMES_2D`), so a metric added to `pipeline.metrics` needs no change
here to be stored.

Each ratio feature's joint histogram+metrics is an independent computation
off its own already-stored numerator/denominator, so `n_processes` (opt-in,
default 1 -- sequential) fans them out the same way `Distribution1DStep`
does -- see that module's docstring for the "workers only compute" rationale
and why this stays in-process-only (no `worker_index`/`n_workers`).

`is_done()`/`run()` are per-feature-aware the same way `Distribution1DStep`'s
are: each written distribution stamps its own `feature_hash`, and only a
ratio feature whose hash is missing or mismatched gets recomputed. No ratio
features registered at all is itself a done state (nothing to compute),
rather than the whole-sample flag this used to be, which never got set.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np

from pylibs.core.exceptions import ResultsError
from pylibs.core.features.expression import split_ratio
from pylibs.core.io.reader import LibsReader
from pylibs.core.io.store import ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.pipeline.distributions import Distribution2D, compute_distribution_2d
from pylibs.core.project.registry import Registry
from pylibs.core.runs.executor import WorkerPool, execute

logger = get_logger(__name__)

# feature_id -> (numerator_expression, denominator_expression, numerator, denominator)
RatioRawData = dict[str, tuple[str, str, np.ndarray, np.ndarray]]


@dataclass
class _WorkerState:
    feature_ids: list[str]
    ratio_raw: RatioRawData


_STATE: _WorkerState | None = None


def init_worker(feature_ids: list[str], ratio_raw: RatioRawData) -> None:
    """`ProcessPoolExecutor` initializer (also called directly, once, for
    the sequential `n_processes<=1` path): stash this run's read-only
    inputs in process-global state once per worker process."""
    global _STATE
    _STATE = _WorkerState(feature_ids=feature_ids, ratio_raw=ratio_raw)


def compute_one(index: int) -> Distribution2D:
    state = _STATE
    assert state is not None, "init_worker must run before compute_one"
    num_expr, den_expr, numerator, denominator = state.ratio_raw[state.feature_ids[index]]
    return compute_distribution_2d(numerator, denominator, num_expr, den_expr)


def _ratio_features(registry: Registry) -> dict[str, tuple[str, str]]:
    return {
        feature_id: split
        for feature_id, config in registry.features.items()
        if (split := split_ratio(config.expression)) is not None
    }


def _stale_ratio_feature_ids(
    store: ResultsStore, sample_id: str, feature_ids: list[str]
) -> list[str]:
    """Feature ids (already filtered to ratio, non-out-of-range features)
    whose stored distribution_2d `feature_hash` attribute is missing or
    doesn't match its current one on the features array."""
    features_array = store.get_features_array(sample_id)
    current_hashes = (
        cast(dict[str, str], features_array.attrs.get("feature_hashes", {}))
        if features_array is not None
        else {}
    )
    stale = []
    for fid in feature_ids:
        arrays = store.get_distribution_2d_arrays(sample_id, fid)
        stored_hash = arrays[0].attrs.get("feature_hash") if arrays is not None else None
        if stored_hash != current_hashes.get(fid):
            stale.append(fid)
    return stale


class Distribution2DStep:
    name = "distribution_2d"
    produces = "distribution_2d"
    requires: str | None = "features"
    needs_reader = False

    def __init__(self, n_processes: int = 1, pool: WorkerPool | None = None):
        self.n_processes = n_processes
        # a pool shared with the other distribution step, when the pipeline
        # runner built one -- worker startup is then paid once per pipeline
        # call rather than once per step
        self.pool = pool

    def is_done(self, store: ResultsStore, sample_id: str, registry: Registry) -> bool:
        out_of_range = set(store.get_out_of_range_feature_ids(sample_id))
        feature_ids = [fid for fid in _ratio_features(registry) if fid not in out_of_range]
        if not feature_ids:
            return True
        return not _stale_ratio_feature_ids(store, sample_id, feature_ids)

    def run(
        self, reader: LibsReader | None, sample_id: str, store: ResultsStore, registry: Registry
    ) -> None:
        ratios = _ratio_features(registry)
        if not ratios:
            logger.warning(
                "No ratio features registered -- nothing to compute distribution_2d for sample %r.",
                sample_id,
            )
            return

        out_of_range = set(store.get_out_of_range_feature_ids(sample_id))
        all_ratio_ids = [fid for fid in ratios if fid not in out_of_range]
        feature_ids = _stale_ratio_feature_ids(store, sample_id, all_ratio_ids)
        if not feature_ids:
            return

        features_array = store.get_features_array(sample_id)
        feature_hashes = cast(
            dict[str, str],
            features_array.attrs.get("feature_hashes", {}) if features_array is not None else {},
        )

        ratio_raw: RatioRawData = {}
        for feature_id in feature_ids:
            num_expr, den_expr = ratios[feature_id]
            components = store.get_ratio_components(sample_id, feature_id)
            if components is None:
                raise ResultsError(
                    f"No ratio_components found in the store for feature {feature_id!r} of "
                    f"sample {sample_id!r} -- run the pipeline (RUNPIPELINE) again, e.g. in a "
                    "new project, to compute it."
                )
            numerator_array, denominator_array = components
            ratio_raw[feature_id] = (
                num_expr,
                den_expr,
                np.asarray(numerator_array[:]),
                np.asarray(denominator_array[:]),
            )

        def on_result(index: int, distribution: Distribution2D) -> None:
            feature_id = feature_ids[index]
            num_expr, den_expr = ratios[feature_id]
            num_bound_min, num_bound_max = distribution.numerator_bounds
            den_bound_min, den_bound_max = distribution.denominator_bounds
            attributes = {
                **distribution.metrics,
                "numerator_expression": num_expr,
                "denominator_expression": den_expr,
                "shannon_entropy": distribution.shannon_entropy,
                "numerator_bound_min": num_bound_min,
                "numerator_bound_max": num_bound_max,
                "denominator_bound_min": den_bound_min,
                "denominator_bound_max": den_bound_max,
                "feature_hash": feature_hashes.get(feature_id),
            }
            store.create_distribution_2d(
                sample_id,
                feature_id,
                distribution.counts,
                distribution.bin_edges_num,
                distribution.bin_edges_den,
                distribution.display_counts,
                distribution.display_bin_edges_num,
                distribution.display_bin_edges_den,
                attributes,
            )

        execute(
            list(range(len(feature_ids))),
            work_fn=compute_one,
            on_result=on_result,
            n_processes=self.n_processes,
            pool=self.pool,
            initializer=init_worker,
            initargs=(feature_ids, ratio_raw),
        )
