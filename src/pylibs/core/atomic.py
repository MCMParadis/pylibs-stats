"""Atomic file replacement, for the project files several processes read
while one writes.

`Path.write_text`/`write_bytes` open with `O_TRUNC`: the file is zero bytes
from the moment it opens until the buffer flushes. Any concurrent reader
landing in that window sees a truncated file -- for JSON that is a
`JSONDecodeError`, not a retry-and-recover.

That is not hypothetical here. `registry.json` is written by every
`run_pipeline_batch` shard (via `Registry.mark_sample_processed`) and read
*unlocked* by `Project.load`, which `run_pipeline` calls once per sample. The
`flock` in `core.project.registry` serialises writer against writer, so it
stops lost updates -- it does nothing for a reader, which never takes it.
Observed both ways: the read can land in the batch-level `Project.load`, where
the exception kills the worker and its whole shard, or in a per-sample one,
where `core.runs.executor.execute` catches it, logs a warning and continues --
so the command exits 0 having silently skipped a sample.

Writing to a temp file and `os.replace`-ing it over the destination closes the
window entirely: `os.replace` is atomic on POSIX, so a reader sees either the
whole old file or the whole new one, and readers need no lock at all. The temp
name carries the writer's pid so two processes writing the same destination
can't collide on the temp path itself.
"""

import os
from pathlib import Path


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Replace `path` with `payload`, atomically. The temp file is created in
    the destination's own directory, since `os.replace` is only atomic within
    a single filesystem."""
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """`atomic_write_bytes` for text. Encodes UTF-8 explicitly and does no
    newline translation, so a CRLF payload round-trips byte for byte."""
    atomic_write_bytes(path, text.encode("utf-8"))
