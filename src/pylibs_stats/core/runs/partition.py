"""Static partitioning of a run's iterations across several independently-
invoked processes (e.g. distinct SLURM array-job tasks), with no runtime
coordination beyond agreeing on `n_workers`.
"""


def plan_shard(indices: list[int], worker_index: int = 0, n_workers: int = 1) -> list[int]:
    """Shard `worker_index` (0-indexed, < n_workers) claims
    `indices[worker_index::n_workers]` -- a round-robin split. `n_workers=1,
    worker_index=0` (the default) claims everything -- i.e. no sharding.

    `indices` MUST be the full, static index range for the run (e.g.
    `list(range(n_iterations))`), the same for every shard -- never a
    dynamically-computed `pending_indices(...)` result. Separately-invoked
    processes start at different times and race ahead at different speeds;
    if each computed its own shard from *its own* snapshot of
    `pending_indices` (which shrinks as any worker finishes iterations), two
    shards' round-robin positions would be computed against
    different-length, different-content lists and could silently miss
    indices altogether -- not just duplicate work, but drop it. Round-robin
    over the fixed full range is a pure function of
    `(n_iterations, worker_index, n_workers)` alone, so every shard is
    disjoint and their union is everything, regardless of timing. Filter a
    shard down to what's not yet done -- i.e. call `pending_indices` first
    and intersect -- only *after* this static split."""
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if not (0 <= worker_index < n_workers):
        raise ValueError(f"worker_index {worker_index} out of range for n_workers {n_workers}")
    return indices[worker_index::n_workers]
