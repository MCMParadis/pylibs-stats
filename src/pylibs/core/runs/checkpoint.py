"""Generic zarr-backed checkpoint arrays for a resumable run: a `results`
array (shape `(n_iterations, *item_shape)`) and a parallel `completed` int8
array (shape `(n_iterations,)`) marking which rows are actually valid. A
crash mid-iteration simply leaves that row's `completed` flag at 0, so it's
recomputed and rewritten on the next resume -- never left half-written and
marked done.

Both arrays are chunked with chunk size 1 along the iteration axis. This is
the load-bearing safety property for concurrent writers (see
`core.runs.executor`/`core.runs.partition`): zarr v3 has no locking, and two
processes racing to read-modify-write the *same* chunk can silently drop one
writer's update. With chunk size 1, two iteration indices can only ever
collide on a chunk if they're numerically equal -- and `core.runs.partition`
guarantees any two concurrently-running shards are assigned disjoint index
sets.
"""

import math
from dataclasses import dataclass

import numpy as np
import zarr


@dataclass
class CheckpointArrays:
    results: zarr.Array
    completed: zarr.Array


def get_or_create_checkpoint_arrays(
    group: zarr.Group,
    results_name: str,
    completed_name: str,
    n_iterations: int,
    item_shape: tuple[int, ...],
    dtype: str = "float64",
    fill_value: object = math.nan,
    attributes: dict | None = None,
) -> CheckpointArrays:
    """Idempotent get-or-create: if `results_name`/`completed_name` don't
    exist yet, create them sized to `n_iterations`. If they already exist
    with a *smaller* first dimension than `n_iterations`, extend both (never
    shrinks). If they already exist with a first dimension >= `n_iterations`,
    return them as-is.

    Creation races between concurrently-started processes (e.g. two SLURM
    array tasks launched at once, both finding nothing yet) are resolved by
    attempting `create_array(..., overwrite=False)` and, on
    `zarr.errors.ContainsArrayError`, simply re-reading whatever the winner
    created -- safe as long as every racing caller passes the same
    `n_iterations`/`item_shape` (true for one registered run config). This is
    NOT safe for concurrent *extends* -- see `extend_checkpoint_arrays`.
    """
    existing_results = group.get(results_name)
    existing_completed = group.get(completed_name)
    if isinstance(existing_results, zarr.Array) and isinstance(existing_completed, zarr.Array):
        if existing_results.shape[0] < n_iterations:
            return extend_checkpoint_arrays(group, results_name, completed_name, n_iterations)
        return CheckpointArrays(existing_results, existing_completed)

    results: zarr.Array
    try:
        results = group.create_array(
            results_name,
            shape=(n_iterations, *item_shape),
            dtype=dtype,
            chunks=(1, *item_shape),
            fill_value=fill_value,
            attributes=attributes,
            overwrite=False,
        )
    except zarr.errors.ContainsArrayError:
        existing_array = group[results_name]
        assert isinstance(existing_array, zarr.Array)
        results = existing_array

    completed: zarr.Array
    try:
        completed = group.create_array(
            completed_name,
            shape=(n_iterations,),
            dtype="int8",
            chunks=(1,),
            overwrite=False,
        )
    except zarr.errors.ContainsArrayError:
        existing_array = group[completed_name]
        assert isinstance(existing_array, zarr.Array)
        completed = existing_array

    return CheckpointArrays(results, completed)


def extend_checkpoint_arrays(
    group: zarr.Group, results_name: str, completed_name: str, new_n_iterations: int
) -> CheckpointArrays:
    """Explicit resize to grow an existing run's target. Must only be called
    when no shard/worker is concurrently executing iterations against this
    run -- resizing rewrites the arrays' shape metadata, which is not
    chunk-write-safe against concurrent iteration writes the way normal
    per-iteration writes are. Raises `ValueError` if `new_n_iterations` isn't
    strictly greater than the existing size, or if the arrays don't exist
    yet."""
    results = group.get(results_name)
    completed = group.get(completed_name)
    if not isinstance(results, zarr.Array) or not isinstance(completed, zarr.Array):
        raise ValueError(f"No existing checkpoint arrays {results_name!r}/{completed_name!r}")

    current = results.shape[0]
    if new_n_iterations <= current:
        raise ValueError(
            f"new_n_iterations ({new_n_iterations}) must be greater than the current size "
            f"({current})"
        )
    results.resize((new_n_iterations, *results.shape[1:]))
    completed.resize((new_n_iterations,))
    return CheckpointArrays(results, completed)


def pending_indices(arrays: CheckpointArrays) -> list[int]:
    """Every iteration index whose `completed` flag is still 0, in order."""
    completed = np.asarray(arrays.completed[:])
    return np.flatnonzero(completed == 0).tolist()


def mark_done(arrays: CheckpointArrays, index: int, row: np.ndarray) -> None:
    """Write `row` into `results[index]` and only *then* flip
    `completed[index] = 1` -- ordering matters: a crash between these two
    writes leaves `completed[index] == 0`, so `pending_indices` still lists
    it and it gets recomputed, never silently treated as done with stale
    data."""
    arrays.results[index] = row
    arrays.completed[index] = 1
