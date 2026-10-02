"""Pipeline runner: streams a sample through a project's Steps, writing
results to `<project_root>/results.zarr` and skipping a step whose output
already matches what's currently registered (see `pipeline.step`'s own
docstring -- a sample's raw data still can't be reprocessed in place, but a
feature/peak *definition* change now is).

`run_pipeline_batch` is the across-sample counterpart to `run_pipeline`,
for projects with many samples (see its own docstring for how it composes,
and deliberately does not compose, with `run_pipeline`'s own `n_processes`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from pylibs.core.features.peak_table import load_project_peak_table
from pylibs.core.io.reader import LibsReader, get_reader
from pylibs.core.io.store import RESULTS_DIRNAME, ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.pipeline.step import Step
from pylibs.core.pipeline.steps.distribution_1d import Distribution1DStep
from pylibs.core.pipeline.steps.distribution_2d import Distribution2DStep
from pylibs.core.pipeline.steps.features import FeatureStep
from pylibs.core.pipeline.steps.spectrum_stats import SpectrumStatsStep
from pylibs.core.project.project import Project
from pylibs.core.runs.executor import WorkerPool, execute, resolve_n_processes
from pylibs.core.runs.partition import plan_shard

logger = get_logger(__name__)


class PipelineResult(BaseModel):
    sample_id: str
    n_scans: int
    feature_ids: list[str]
    results_path: Path
    steps_run: list[str]
    steps_skipped: list[str]


def run_pipeline(
    project_root: Path,
    sample_id: str,
    capdata: int | None = None,
    batchsize: int | None = None,
    n_processes: int = 1,
) -> PipelineResult:
    """Run every pipeline Step (feature computation, spectrum statistics,
    1D distribution metrics, 2D distribution metrics for ratio features) for
    `sample_id`, streaming through the sample's reader once for each step
    that isn't already done and still needs it (only FeatureStep and
    SpectrumStatsStep actually stream -- Distribution1DStep/Distribution2DStep
    read FeatureStep's own stored output instead), and write results to
    `<project_root>/results.zarr`. Steps are listed in an order that already
    satisfies their `requires` (Distribution1DStep/Distribution2DStep both
    need FeatureStep's output) -- the runner doesn't topologically sort.

    `n_processes` (opt-in, default 1) affects only Distribution1DStep/
    Distribution2DStep -- each registered feature's histogram+metrics is an
    independent computation once FeatureStep's output is stored, fanned out
    across a local worker pool (see those steps' own docstrings). FeatureStep/
    SpectrumStatsStep stay sequential: they're one coupled streaming pass over
    the raw reader, not easily split without re-reading the raw file per
    worker. `<= 0` uses every CPU available to this process (see
    `core.runs.executor.resolve_n_processes`). For scaling across *samples*
    instead (e.g. many-sample, cluster-deployed projects), see
    `run_pipeline_batch` -- the two don't compose (see its docstring)."""
    logger.info("Running pipeline for sample %r", sample_id)
    project = Project.load(project_root)
    sample = project.get_sample(sample_id)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    table = load_project_peak_table(project_root)
    # One pool for the whole call, shared by both distribution steps: worker
    # startup is a per-process import, so building a pool per step paid it
    # twice. Only created when there is actually more than one worker; the
    # sequential path never builds one.
    pool = WorkerPool(n_processes) if resolve_n_processes(n_processes) > 1 else None
    steps: list[Step] = [
        FeatureStep(table),
        SpectrumStatsStep(),
        Distribution1DStep(n_processes, pool=pool),
        Distribution2DStep(n_processes, pool=pool),
    ]

    # is_done() is re-checked immediately before each step, not all upfront --
    # an earlier step in this same call (e.g. FeatureStep, on a feature
    # identity-hash mismatch) can change the store in a way that makes a
    # later step's own is_done() go from True to False, and checking it only
    # once at the start would miss that.
    reader: LibsReader | None = None
    ran: list[str] = []
    skipped: list[str] = []
    try:
        for step in steps:
            if step.is_done(store, sample_id, project.registry):
                skipped.append(step.name)
                logger.info("%s already computed for sample %r -- skipping", step.name, sample_id)
                continue
            if step.needs_reader and reader is None:
                reader = get_reader(sample.path, capdata=capdata, batchsize=batchsize)
            logger.debug("Running step %r for sample %r", step.name, sample_id)
            step.run(reader, sample_id, store, project.registry)
            ran.append(step.name)
    finally:
        if pool is not None:
            pool.close()

    results_path = store.root / sample_id / "features"
    if ran:
        project.mark_sample_processed(sample_id, results_path)

    features_array = store.get_features_array(sample_id)
    feature_ids = (
        cast(list[str], features_array.attrs["feature_ids"]) if features_array is not None else []
    )
    n_scans = int(features_array.shape[0]) if features_array is not None else 0

    logger.info(
        "Pipeline finished for sample %r: %d step(s) run, %d skipped",
        sample_id,
        len(ran),
        len(skipped),
    )
    return PipelineResult(
        sample_id=sample_id,
        n_scans=n_scans,
        feature_ids=feature_ids,
        results_path=results_path,
        steps_run=ran,
        steps_skipped=skipped,
    )


@dataclass
class _BatchState:
    sample_ids: list[str]
    project_root: Path
    capdata: int | None
    batchsize: int | None


_STATE: _BatchState | None = None


def init_worker(
    sample_ids: list[str],
    project_root: Path,
    capdata: int | None,
    batchsize: int | None,
) -> None:
    """`ProcessPoolExecutor` initializer (also called directly, once, for
    the sequential `n_processes<=1` path): stash this batch's read-only
    inputs in process-global state once per worker process -- mirrors
    `pipeline.bootstrap.compute.init_worker`, even though the payload here
    is small, to keep the same `initializer`/`initargs` calling convention
    `core.runs.executor.execute` expects."""
    global _STATE
    _STATE = _BatchState(
        sample_ids=sample_ids,
        project_root=project_root,
        capdata=capdata,
        batchsize=batchsize,
    )


def compute_sample(index: int) -> PipelineResult:
    """One sample's full `run_pipeline`, run with `n_processes=1` for its
    own Distribution steps -- see `run_pipeline_batch`'s docstring for why
    the two levels of parallelism don't nest. Deliberately writes to the
    results store itself (unlike bootstrap/regression workers) -- see
    `run_pipeline_batch`'s docstring for why that's safe here."""
    state = _STATE
    assert state is not None, "init_worker must run before compute_sample"
    sample_id = state.sample_ids[index]
    return run_pipeline(
        state.project_root,
        sample_id,
        capdata=state.capdata,
        batchsize=state.batchsize,
        n_processes=1,
    )


class PipelineBatchResult(BaseModel):
    sample_ids: list[str]
    n_processed: int
    n_failed: int
    failed_sample_ids: list[str]
    results_path: Path


def run_pipeline_batch(
    project_root: Path,
    sample_ids: list[str] | None = None,
    capdata: int | None = None,
    batchsize: int | None = None,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
) -> PipelineBatchResult:
    """Run `run_pipeline` for every sample in `sample_ids` (default: every
    registered sample, `sorted(project.registry.samples)` -- deterministic
    across independently launched worker processes reading the same
    `registry.json`). Each sample's own pipeline is already idempotent
    (every Step's `is_done` is checked before running it), so re-running an
    already-fully-processed sample here is a cheap no-op -- no separate
    "pending" tracking is needed the way bootstrap's checkpoint arrays need
    one, since every sample writes to its own `<sample_id>/...` zarr group,
    not a shared array.

    `n_processes` > 1 parallelizes this invocation's own share of samples
    across an in-process worker pool; `<= 0` uses every CPU available to
    this process (see `core.runs.executor.resolve_n_processes`).
    `worker_index`/`n_workers` statically partition the *full* sample list
    (never a dynamically-filtered subset, for the same reason
    `core.runs.partition.plan_shard`'s docstring gives for bootstrap) across
    separately-invoked processes, e.g. SLURM array-job tasks, with no
    runtime coordination beyond agreeing on `n_workers`; the two knobs
    compose (each shard can itself use `n_processes`).

    Each worker calls `run_pipeline` (with its own `n_processes` pinned to 1
    for that sample's Distribution steps -- see `run_pipeline`'s own
    `n_processes`) directly, writing to the results store itself, unlike
    bootstrap/regression workers which only compute and let the calling
    process write. This is safe and deliberate: `run_pipeline` already
    loads its own `Project`/`ResultsStore` and writes every Step's output as
    one self-contained call, and every sample writes only to its own
    `<sample_id>/...` group -- disjoint from every other sample -- so there's
    no shared mutable *array* for concurrent writers to race on, unlike
    bootstrap's single shared checkpoint array (which is exactly why that
    needs chunk-size-1 + disjoint index partitioning).

    Two things here are shared and are NOT covered by that argument. The
    store's root group does not necessarily exist before the workers start
    (nothing in this function builds a `ResultsStore`), so on a fresh project
    the workers race to create it -- handled by `ResultsStore.__init__`'s own
    `ContainsGroupError` guard. And `registry.json` is shared mutable
    metadata: every worker writes it via `Registry.mark_sample_processed`,
    which locks against lost updates, though `Registry.load` itself reads it
    unlocked.
    Nesting a second pool inside each worker for that sample's own
    Distribution steps is deliberately not done -- see `run_pipeline`'s
    `n_processes` docstring."""
    project = Project.load(project_root)
    resolved_sample_ids = sample_ids if sample_ids is not None else sorted(project.registry.samples)
    for sample_id in resolved_sample_ids:
        project.get_sample(sample_id)  # raises ProjectError on an unknown sample_id, fail fast
    # ...and likewise for the peak table, which every worker loads for itself:
    # without this an unregistered table surfaces as N identical worker
    # failures reported as "N samples failed", not one setup error.
    load_project_peak_table(project_root)

    shard_indices = plan_shard(list(range(len(resolved_sample_ids))), worker_index, n_workers)
    logger.info(
        "Running pipeline batch for %d sample(s) (worker %d/%d: %d assigned)",
        len(resolved_sample_ids),
        worker_index,
        n_workers,
        len(shard_indices),
    )

    results: list[PipelineResult] = []
    exec_result = execute(
        shard_indices,
        work_fn=compute_sample,
        on_result=lambda _index, result: results.append(result),
        n_processes=n_processes,
        initializer=init_worker,
        initargs=(resolved_sample_ids, project_root, capdata, batchsize),
    )

    failed_sample_ids = [resolved_sample_ids[i] for i in exec_result.failed_indices]
    logger.info(
        "Pipeline batch finished: %d/%d sample(s) processed, %d failed",
        exec_result.n_completed,
        len(shard_indices),
        exec_result.n_failed,
    )
    return PipelineBatchResult(
        sample_ids=resolved_sample_ids,
        n_processed=exec_result.n_completed,
        n_failed=exec_result.n_failed,
        failed_sample_ids=failed_sample_ids,
        results_path=project_root / RESULTS_DIRNAME,
    )
