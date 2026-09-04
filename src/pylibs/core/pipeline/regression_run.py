"""Parallel dispatch for the regression fitting sweep: mirrors
`pipeline.bootstrap.compute`/`runner` for one (CSV column, metric, feature)
fit at a time -- the independent, CPU-heavy unit here (an OLS fit plus a
permutation test, see `pipeline.regression.fit_least_squares`), the same way
one bootstrap iteration is there. `pipeline.regression` itself stays
untouched (pure algorithm + `compute_feature_regression`); this module only
adds the multiprocessing plumbing around it.

Each cell writes to its own disjoint zarr *leaf* group
(`regressions/<column>/<metric>/<feature_id>/`), not a shared array, so --
unlike bootstrap -- no `core.runs.checkpoint` arrays are needed, just
`core.runs.partition.plan_shard` (static sharding) and `core.runs.executor.
execute` (the dispatch loop), reused directly. Only the leaf is disjoint,
though: the three groups above it (`regressions`, `<column>`, `<metric>`) are
shared by every shard and are created through `store._require_group_concurrent`
for that reason. The stale-prune loop below is also run independently by every
shard, so one shard's `delete_regression_feature` can overlap another's
creates under the same `<metric>` subtree.

Cells are seeded independently (`np.random.default_rng([random_seed,
cell_index])`, matching bootstrap's own `[base_seed, iteration_index]`
pattern) rather than sharing one sequentially-advanced `np.random.Generator`
across the whole sweep -- this makes every cell's result independent of
iteration order and of `n_processes`/`n_workers`, at the cost of shifting
today's stored p-values slightly on the next run (methodology unchanged,
only which permutations get drawn).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pylibs.core.io.store import ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.pipeline.regression import FeatureRegression, compute_feature_regression
from pylibs.core.runs.executor import execute
from pylibs.core.runs.partition import plan_shard

logger = get_logger(__name__)


@dataclass
class _WorkerState:
    column_names: list[str]
    metric_names: list[str]
    feature_ids: list[str]
    metric_tensor: np.ndarray  # (n_points, n_features, n_metrics)
    x_matrix: np.ndarray  # (n_points, n_columns)
    sample_names_used: list[str]
    algorithm: str
    n_permutations: int
    random_seed: int


_STATE: _WorkerState | None = None


def init_worker(
    column_names: list[str],
    metric_names: list[str],
    feature_ids: list[str],
    metric_tensor: np.ndarray,
    x_matrix: np.ndarray,
    sample_names_used: list[str],
    algorithm: str,
    n_permutations: int,
    random_seed: int,
) -> None:
    """`ProcessPoolExecutor` initializer (also called directly, once, for
    the sequential `n_processes<=1` path): stash this sweep's read-only
    inputs in process-global state once per worker process -- mirrors
    `pipeline.bootstrap.compute.init_worker`."""
    global _STATE
    _STATE = _WorkerState(
        column_names=column_names,
        metric_names=metric_names,
        feature_ids=feature_ids,
        metric_tensor=metric_tensor,
        x_matrix=x_matrix,
        sample_names_used=sample_names_used,
        algorithm=algorithm,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )


@dataclass
class CellResult:
    column_name: str
    metric_name: str
    feature_id: str
    regression: FeatureRegression | None


def compute_cell(cell_index: int) -> CellResult:
    """One (column, metric, feature) cell's fit, deterministic given
    `_STATE.random_seed` and `cell_index` alone -- independent of every
    other cell, and of execution order/parallelism degree."""
    state = _STATE
    assert state is not None, "init_worker must run before compute_cell"
    n_columns = len(state.column_names)
    n_metrics = len(state.metric_names)
    n_features = len(state.feature_ids)
    column_index, metric_index, feature_index = (
        int(i) for i in np.unravel_index(cell_index, (n_columns, n_metrics, n_features))
    )
    column_name = state.column_names[column_index]
    metric_name = state.metric_names[metric_index]
    feature_id = state.feature_ids[feature_index]

    x = state.x_matrix[:, column_index]
    y = state.metric_tensor[:, feature_index, metric_index]
    rng = np.random.default_rng([state.random_seed, cell_index])
    regression = compute_feature_regression(
        column_name,
        feature_id,
        metric_name,
        state.algorithm,
        state.sample_names_used,
        x,
        y,
        state.n_permutations,
        rng,
    )
    return CellResult(column_name, metric_name, feature_id, regression)


@dataclass
class RegressionSweepResult:
    column_names: list[str]  # columns with at least one fit stored
    skipped_fits: list[str]
    # The subset of `skipped_fits` a caller might actually want to look at. A
    # (feature, metric) cell whose metric was never computed for that feature
    # -- every METRIC_NAMES_2D metric against a non-ratio feature, say -- can
    # never be fit and is left out, the same way compute_correlations treats
    # its own missing cells as routine.
    unexpected_skipped_fits: list[str]


def run_regression_sweep(
    store: ResultsStore,
    column_names: list[str],
    metrics_to_fit: list[str],
    feature_ids: list[str],
    metric_tensor: np.ndarray,
    x_matrix: np.ndarray,
    sample_names_used: list[str],
    algorithm: str,
    n_permutations: int,
    random_seed: int,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
) -> RegressionSweepResult:
    """Fits every (column, metric, feature) cell -- `len(column_names) *
    len(metrics_to_fit) * len(feature_ids)` of them -- storing each via
    `store.create_regression_arrays` as it completes. Prunes stale features
    (registered under a (column, metric) but no longer in `feature_ids`)
    per (column, metric) subtree before dispatching, same as before this
    was parallelized.

    `n_processes` > 1 parallelizes this invocation's own share of cells
    across an in-process worker pool; `<= 0` uses every CPU available to
    this process (see `core.runs.executor.resolve_n_processes`).
    `worker_index`/`n_workers` statically partition the *full* cell range
    (never a dynamically-filtered subset) across separately-invoked
    processes, e.g. SLURM array-job tasks, with no runtime coordination
    beyond agreeing on `n_workers`; the two knobs compose."""
    for metric_name in metrics_to_fit:
        for column_name in column_names:
            for stale_feature_id in store.list_regression_features(column_name, metric_name):
                if stale_feature_id not in feature_ids:
                    store.delete_regression_feature(column_name, metric_name, stale_feature_id)

    n_cells = len(column_names) * len(metrics_to_fit) * len(feature_ids)
    shard_indices = plan_shard(list(range(n_cells)), worker_index, n_workers)
    logger.info(
        "Running regression sweep for %d cell(s) (worker %d/%d: %d assigned)",
        n_cells,
        worker_index,
        n_workers,
        len(shard_indices),
    )

    skipped_fits: list[str] = []
    unexpected_skipped_fits: list[str] = []
    skipped_by_reason: Counter[str] = Counter()
    columns_with_fits: set[str] = set()

    # A cell is unfittable by construction when its metric holds no finite value
    # for that feature across *any* sample -- i.e. the metric simply isn't
    # computed for it, rather than the data being too sparse. Precomputed once
    # here, since on_result runs per cell.
    metric_exists = np.isfinite(metric_tensor).any(axis=0)  # (n_features, n_metrics)
    feature_index_of = {feature_id: i for i, feature_id in enumerate(feature_ids)}
    metric_index_of = {metric_name: i for i, metric_name in enumerate(metrics_to_fit)}

    def on_result(_cell_index: int, result: CellResult) -> None:
        if result.regression is None:
            expected = not metric_exists[
                feature_index_of[result.feature_id], metric_index_of[result.metric_name]
            ]
            reason = "metric not computed for this feature" if expected else "insufficient data"
            entry = f"{result.column_name}/{result.metric_name}/{result.feature_id} ({reason})"
            skipped_fits.append(entry)
            skipped_by_reason[reason] += 1
            if not expected:
                unexpected_skipped_fits.append(entry)
            # info, not warning: an unfittable cell is routine here, exactly as a
            # missing cell is in compute_correlations. Kept in pylibs.log for
            # anyone who wants the full list, just not on the console.
            logger.info("Skipped fit %s", entry)
            return
        regression = result.regression
        store.create_regression_arrays(
            result.column_name,
            result.metric_name,
            result.feature_id,
            regression.group_mean_x,
            regression.group_mean_y,
            regression.group_std_y,
            attributes={
                "column_name": regression.column_name,
                "feature_id": regression.feature_id,
                "metric_name": regression.metric_name,
                "algorithm": regression.algorithm,
                "slope": regression.slope,
                "intercept": regression.intercept,
                "pearson_r": regression.pearson_r,
                "r_squared": regression.r_squared,
                "mae": regression.mae,
                "p_value": regression.p_value,
                "n": regression.n,
                "x_mean": regression.x_mean,
                "ssxx": regression.ssxx,
                "residual_std": regression.residual_std,
                "x_min": regression.x_min,
                "x_max": regression.x_max,
                "sample_names": regression.sample_names,
                "n_permutations": n_permutations,
                "random_seed": random_seed,
            },
        )
        columns_with_fits.add(result.column_name)

    execute(
        shard_indices,
        work_fn=compute_cell,
        on_result=on_result,
        n_processes=n_processes,
        initializer=init_worker,
        initargs=(
            column_names,
            metrics_to_fit,
            feature_ids,
            metric_tensor,
            x_matrix,
            sample_names_used,
            algorithm,
            n_permutations,
            random_seed,
        ),
    )

    breakdown = ", ".join(f"{reason}: {count}" for reason, count in skipped_by_reason.most_common())
    logger.info(
        "Regression sweep finished: %d/%d cell(s) fit, %d skipped%s",
        len(shard_indices) - len(skipped_fits),
        len(shard_indices),
        len(skipped_fits),
        f" ({breakdown})" if breakdown else "",
    )
    return RegressionSweepResult(
        column_names=[name for name in column_names if name in columns_with_fits],
        skipped_fits=skipped_fits,
        unexpected_skipped_fits=unexpected_skipped_fits,
    )
