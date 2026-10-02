"""Executes a run's pending iterations against a `work_fn`, writing results
back through a caller-supplied `on_result` callback that always runs in
*this* (the calling) process/thread -- so all zarr writes for one shard
happen from a single place, and workers (if any) never touch the store
directly. This is what makes in-process multiprocessing (`n_processes`) safe
without any locking: subprocess workers only ever return picklable values
over a pipe, they never open the results store themselves. (Not every
consumer follows this -- see `pipeline.runner.run_pipeline_batch`'s own
docstring for the one deliberate exception, and why it's safe there.)

`work_fn`/`on_result` are generic over the per-item payload type `T` --
bootstrap's own usage returns a metrics row (`np.ndarray`), but nothing here
requires that; a batch pipeline run or a regression fit returns its own
result type instead.
"""

import multiprocessing
import os
import pickle
import shutil
import tempfile
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pylibs.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RunExecutionResult:
    n_submitted: int
    n_completed: int
    n_failed: int
    failed_indices: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # str(exc), aligned with failed_indices


# Loaded work groups, keyed by the state file that defined them. Lives in the
# worker process; the parent never touches it.
_LOADED: dict[str, Callable[[int], Any]] = {}


def _load_group(path: str) -> Callable[[int], Any]:
    """Read one work group's `(initializer, initargs, work_fn)` out of its
    state file, run the initializer once, and remember the work function.

    Called in a worker, on that worker's first task from this group. Loading
    here rather than through `ProcessPoolExecutor`'s own `initializer`/
    `initargs` is what keeps worker startup parallel: those arguments are
    pickled into each new worker's spawn pipe by the parent, one worker at a
    time, and a payload above the pipe buffer blocks the parent until that
    child drains it -- which the child cannot do until it has imported this
    package. Startup then costs one import per worker, paid serially. Passing
    a path instead keeps the pipe payload to a few bytes, so the parent never
    blocks and every worker imports at the same time.
    """
    work_fn = _LOADED.get(path)
    if work_fn is None:
        with open(path, "rb") as handle:
            initializer, initargs, work_fn = pickle.load(handle)
        if initializer is not None:
            initializer(*initargs)
        _LOADED[path] = work_fn
    return work_fn


def _dispatch(path: str, index: int) -> Any:
    """The only callable submitted to a pool: resolve the group, run its
    work function. Module-level and picklable, as `spawn` requires."""
    return _load_group(path)(index)


class WorkerPool:
    """A `spawn` process pool that can serve several work groups in turn.

    One pool per pipeline call rather than one per step: the pool carries no
    initializer of its own, so each group's read-only inputs travel out of
    band through a state file (see `_load_group`) and a worker picks up
    whichever groups it is given tasks from. Workers are created on the first
    submission and reused afterwards.

    `close()` shuts the pool down and removes the state files. They must
    outlive every task that might still need to load them, which is why they
    are owned here and not by an individual `execute` call.
    """

    def __init__(self, n_processes: int) -> None:
        self.n_processes = resolve_n_processes(n_processes)
        self._pool: ProcessPoolExecutor | None = None
        self._state_dir: Path | None = None

    @property
    def pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.n_processes,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._pool

    def state_file(self, initializer, initargs: tuple, work_fn) -> str:
        if self._state_dir is None:
            self._state_dir = Path(tempfile.mkdtemp(prefix="pylibs_pool_"))
        path = self._state_dir / f"{uuid.uuid4().hex}.pkl"
        with open(path, "wb") as handle:
            pickle.dump((initializer, initargs, work_fn), handle, pickle.HIGHEST_PROTOCOL)
        return str(path)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        if self._state_dir is not None:
            shutil.rmtree(self._state_dir, ignore_errors=True)
            self._state_dir = None

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def resolve_n_processes(n_processes: int) -> int:
    """`n_processes <= 0` means "use every CPU available to this process" --
    `len(os.sched_getaffinity(0))` (respects a cgroup/SLURM CPU allocation,
    unlike `os.cpu_count()`) where available, else `os.cpu_count()`. Every
    `n_processes`-accepting entry point in the codebase resolves through
    this one function, so a user only has to know the sentinel once."""
    if n_processes > 0:
        return n_processes
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def execute(
    indices: list[int],
    work_fn: Callable[[int], T],
    on_result: Callable[[int, T], None],
    n_processes: int = 1,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    pool: WorkerPool | None = None,
) -> RunExecutionResult:
    """Compute `work_fn(i)` for every `i` in `indices`.

    `n_processes <= 0` resolves to every CPU available to this process (see
    `resolve_n_processes`) before anything else below applies.

    `n_processes <= 1` (after that resolution): sequential, in this process.
    `initializer(*initargs)` is called once directly (so a work function's
    module-level "stash my read-only inputs" initializer doesn't need a
    separate no-pool code path), then `work_fn(i)` runs for each index in
    turn, calling `on_result(i, result)` immediately after each -- so a
    crash partway through only loses the one in-flight iteration.

    `n_processes > 1`: a `ProcessPoolExecutor(max_workers=n_processes,
    initializer=initializer, initargs=initargs)`, using the `spawn` start
    method explicitly (not the platform default -- `fork` on Linux is unsafe
    to mix with threads and won't be the default forever; `spawn` also
    matches this module's requirement that `work_fn`/`initializer` be
    module-level and picklable, rather than relying on fork's copy-on-write
    memory sharing). Large read-only inputs must travel via `initargs` to
    `initializer`, which runs once per worker *process*, not once per
    iteration. Results are collected via `as_completed` as they finish (not
    submission order) and immediately handed to `on_result`.

    `pool`: an already-running `WorkerPool` to submit into instead of
    building a private one, so several calls in a row (the two distribution
    steps of one pipeline run, say) share one set of worker processes. The
    pool's own worker count applies and `n_processes` is ignored. The caller
    owns the pool and must close it.

    A single iteration raising an exception is caught, logged, and recorded
    in the returned `RunExecutionResult` (`failed_indices`/`errors`) -- it
    does NOT abort the rest of the run or mark that index complete; it
    remains pending and is retried on the next invocation.
    """
    n_processes = pool.n_processes if pool is not None else resolve_n_processes(n_processes)
    n_completed = 0
    failed_indices: list[int] = []
    errors: list[str] = []

    if n_processes <= 1:
        if initializer is not None:
            initializer(*initargs)
        for index in indices:
            try:
                result = work_fn(index)
            except Exception as exc:
                logger.warning("Iteration %d failed: %s", index, exc)
                failed_indices.append(index)
                errors.append(str(exc))
                continue
            on_result(index, result)
            n_completed += 1
    else:
        owned = pool is None
        active = pool if pool is not None else WorkerPool(n_processes)
        try:
            state = active.state_file(initializer, initargs, work_fn)
            futures = {active.pool.submit(_dispatch, state, index): index for index in indices}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.warning("Iteration %d failed: %s", index, exc)
                    failed_indices.append(index)
                    errors.append(str(exc))
                    continue
                on_result(index, result)
                n_completed += 1
        finally:
            if owned:
                active.close()

    return RunExecutionResult(
        n_submitted=len(indices),
        n_completed=n_completed,
        n_failed=len(failed_indices),
        failed_indices=failed_indices,
        errors=errors,
    )
