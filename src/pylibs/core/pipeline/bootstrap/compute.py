"""Per-iteration bootstrap metric computation: resample a sample's already-
computed feature values (plus, for ratio features, raw numerator/denominator
arrays) at a deterministic subset of scan positions, then compute the exact
same `METRIC_NAMES` vocabulary as `pipeline.metrics.feature_metric_values`
-- just over the resampled subset instead of the whole sample. This module's
`compute_iteration` is the `work_fn` plugged into `core.runs.executor.execute`;
`init_worker` is its matching `initializer`, stashing this run's read-only
inputs in process-global state once per worker process (or once, directly,
for the sequential n_processes<=1 path) rather than re-pickling them on every
call.
"""

from dataclasses import dataclass

import numpy as np

from pylibs.core.pipeline.bootstrap.resampling import get_resampling_strategy
from pylibs.core.pipeline.distributions import compute_distribution_1d, compute_distribution_2d
from pylibs.core.pipeline.metrics import METRIC_NAMES, feature_metric_values

# feature_id -> (numerator_expression, denominator_expression, numerator, denominator)
RatioRawData = dict[str, tuple[str, str, np.ndarray, np.ndarray]]


@dataclass
class _WorkerState:
    feature_ids: list[str]
    feature_values: np.ndarray  # (n_scans, n_features)
    ratio_raw: RatioRawData
    method: str
    npix: int
    shuffle: bool
    base_seed: int
    n_rows: int | None
    n_cols: int | None
    mask: np.ndarray | None


_STATE: _WorkerState | None = None


def init_worker(
    feature_ids: list[str],
    feature_values: np.ndarray,
    ratio_raw: RatioRawData,
    method: str,
    npix: int,
    shuffle: bool,
    base_seed: int,
    n_rows: int | None,
    n_cols: int | None,
    mask: np.ndarray | None = None,
) -> None:
    """`ProcessPoolExecutor` initializer (also called directly, once, for
    the `n_processes<=1` sequential path in `core.runs.executor.execute`):
    stash this run's read-only inputs in process-global state once per
    worker process, so `compute_iteration` doesn't re-pickle large arrays on
    every call."""
    global _STATE
    _STATE = _WorkerState(
        feature_ids=feature_ids,
        feature_values=feature_values,
        ratio_raw=ratio_raw,
        method=method,
        npix=npix,
        shuffle=shuffle,
        base_seed=base_seed,
        n_rows=n_rows,
        n_cols=n_cols,
        mask=mask,
    )


def compute_iteration(iteration_index: int) -> np.ndarray:
    """One bootstrap draw's `(n_features, len(METRIC_NAMES))` metrics row,
    deterministic given `_STATE.base_seed` and `iteration_index` alone.

    Positions are always drawn first, from an rng state that doesn't depend
    on whether shuffling is enabled -- so a shuffled and unshuffled run at
    the same iteration index select the *same* positions and differ only in
    which values sit at them, matching the "permutation test" framing
    (values permuted, not which spatial/random slots get sampled)."""
    state = _STATE
    assert state is not None, "init_worker must run before compute_iteration"
    rng = np.random.default_rng([state.base_seed, iteration_index])
    n_scans = state.feature_values.shape[0]

    strategy = get_resampling_strategy(state.method)
    positions = strategy(
        rng,
        n_scans=n_scans,
        npix=state.npix,
        n_rows=state.n_rows,
        n_cols=state.n_cols,
        mask=state.mask,
    )
    permutation = rng.permutation(n_scans) if state.shuffle else None

    row = np.full((len(state.feature_ids), len(METRIC_NAMES)), np.nan)
    for i, feature_id in enumerate(state.feature_ids):
        column = state.feature_values[:, i]
        if permutation is not None:
            column = column[permutation]
        distribution_1d = compute_distribution_1d(column[positions], fit_gaussian=False)

        distribution_2d = None
        if feature_id in state.ratio_raw:
            num_expr, den_expr, numerator, denominator = state.ratio_raw[feature_id]
            if permutation is not None:
                numerator = numerator[permutation]
                denominator = denominator[permutation]
            distribution_2d = compute_distribution_2d(
                numerator[positions], denominator[positions], num_expr, den_expr
            )

        values = feature_metric_values(distribution_1d, distribution_2d)
        row[i] = [values[name] for name in METRIC_NAMES]
    return row
