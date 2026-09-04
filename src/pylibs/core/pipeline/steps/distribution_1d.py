"""Distribution1DStep: computes a histogram + every 1D metric (see
`pipeline.metrics.METRIC_NAMES_1D`) for every registered feature's already-
computed values, and writes them to the results store. This backs the
distribution panel (C) in the PDF report.

Requires FeatureStep to have already run -- it reads the "features" array
rather than streaming the raw reader itself. Metric values are persisted
generically (`distribution.metrics`, keyed by the same display names as
`METRIC_NAMES_1D`), so a metric added to `pipeline.metrics` needs no change
here to be stored.

Each feature's histogram+metrics is an independent computation off the
already-in-memory `values_matrix`, so `n_processes` (opt-in, default 1 --
sequential) fans them out across a `ProcessPoolExecutor` via `core.runs.
executor.execute`, the same "workers only compute, the calling process
writes" split bootstrap/regressions use -- no `worker_index`/`n_workers`
here, see `pipeline.runner.run_pipeline`'s own docstring for why this stays
in-process-only.

`is_done()`/`run()` are per-feature-aware, not just a whole-sample flag: each
written distribution stamps the features array's current `feature_hashes[
feature_id]` (see `features.identity.feature_identity_hash`) as its own
`feature_hash` attribute, and only a feature whose hash is missing or
mismatched gets recomputed -- unlike `FeatureStep`, this doesn't need to
re-stream the raw file, so recomputing only what actually changed (not
every feature) is real, free savings here.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np

from pylibs.core.io.reader import LibsReader
from pylibs.core.io.store import ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.pipeline.distributions import Distribution1D, compute_distribution_1d
from pylibs.core.project.registry import Registry
from pylibs.core.runs.executor import execute

logger = get_logger(__name__)


@dataclass
class _WorkerState:
    feature_ids: list[str]
    values_matrix: np.ndarray  # (n_scans, n_features)


_STATE: _WorkerState | None = None


def init_worker(feature_ids: list[str], values_matrix: np.ndarray) -> None:
    """`ProcessPoolExecutor` initializer (also called directly, once, for
    the sequential `n_processes<=1` path): stash this run's read-only
    inputs in process-global state once per worker process."""
    global _STATE
    _STATE = _WorkerState(feature_ids=feature_ids, values_matrix=values_matrix)


def compute_one(index: int) -> Distribution1D:
    state = _STATE
    assert state is not None, "init_worker must run before compute_one"
    return compute_distribution_1d(state.values_matrix[:, index])


def _stale_feature_indices(
    store: ResultsStore, sample_id: str, feature_ids: list[str], out_of_range: set[str]
) -> list[int]:
    """Indices (into `feature_ids`) of every non-out-of-range feature whose
    stored distribution_1d `feature_hash` attribute is missing or doesn't
    match its current one on the features array."""
    features_array = store.get_features_array(sample_id)
    assert features_array is not None
    current_hashes = cast(dict[str, str], features_array.attrs.get("feature_hashes", {}))
    indices = []
    for i, fid in enumerate(feature_ids):
        if fid in out_of_range:
            continue
        arrays = store.get_distribution_1d_arrays(sample_id, fid)
        stored_hash = arrays[0].attrs.get("feature_hash") if arrays is not None else None
        if stored_hash != current_hashes.get(fid):
            indices.append(i)
    return indices


class Distribution1DStep:
    name = "distribution_1d"
    produces = "distribution_1d"
    requires: str | None = "features"
    needs_reader = False

    def __init__(self, n_processes: int = 1):
        self.n_processes = n_processes

    def is_done(self, store: ResultsStore, sample_id: str, registry: Registry) -> bool:
        features_array = store.get_features_array(sample_id)
        if features_array is None:
            return False
        feature_ids = cast(list[str], features_array.attrs["feature_ids"])
        out_of_range = set(store.get_out_of_range_feature_ids(sample_id))
        return not _stale_feature_indices(store, sample_id, feature_ids, out_of_range)

    def run(
        self, reader: LibsReader | None, sample_id: str, store: ResultsStore, registry: Registry
    ) -> None:
        features_array = store.get_features_array(sample_id)
        if features_array is None:
            logger.warning(
                "No features found in the store for sample %r -- skipping distribution_1d "
                "(run RUNPIPELINE's FeatureStep first).",
                sample_id,
            )
            return

        feature_ids = cast(list[str], features_array.attrs["feature_ids"])
        feature_hashes = cast(dict[str, str], features_array.attrs.get("feature_hashes", {}))
        values_matrix = cast(np.ndarray, features_array[:])
        out_of_range = set(store.get_out_of_range_feature_ids(sample_id))
        indices = _stale_feature_indices(store, sample_id, feature_ids, out_of_range)

        def on_result(index: int, distribution: Distribution1D) -> None:
            feature_id = feature_ids[index]
            amplitude, mu, sigma = distribution.fitted_params or (None, None, None)
            attributes = {
                **distribution.metrics,
                "zero_count": distribution.zero_count,
                "fitted_amplitude": amplitude,
                "fitted_mu": mu,
                "fitted_sigma": sigma,
                "feature_hash": feature_hashes.get(feature_id),
            }
            store.create_distribution_1d(
                sample_id, feature_id, distribution.counts, distribution.bin_edges, attributes
            )

        execute(
            indices,
            work_fn=compute_one,
            on_result=on_result,
            n_processes=self.n_processes,
            initializer=init_worker,
            initargs=(feature_ids, values_matrix),
        )
