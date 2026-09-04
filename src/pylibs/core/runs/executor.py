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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TypeVar

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

    A single iteration raising an exception is caught, logged, and recorded
    in the returned `RunExecutionResult` (`failed_indices`/`errors`) -- it
    does NOT abort the rest of the run or mark that index complete; it
    remains pending and is retried on the next invocation.
    """
    n_processes = resolve_n_processes(n_processes)
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
        with ProcessPoolExecutor(
            max_workers=n_processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initializer,
            initargs=initargs,
        ) as pool:
            futures = {pool.submit(work_fn, index): index for index in indices}
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

    return RunExecutionResult(
        n_submitted=len(indices),
        n_completed=n_completed,
        n_failed=len(failed_indices),
        failed_indices=failed_indices,
        errors=errors,
    )
