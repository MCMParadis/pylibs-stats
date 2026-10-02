"""Slower integration test: `registry.json` is read by processes that hold no
lock while another process is writing it.

`Registry.save()` rewrites the whole file, and `_locked_registry_dir` only
serialises writer against writer -- it stops lost updates, and never runs on
the read path. `Project.load` is unlocked, and `run_pipeline` calls it once
per sample, so a sharded `run_pipeline_batch` interleaves many unlocked reads
with its siblings' writes.

That is safe only because the write is atomic. With a plain
`Path.write_text` (open truncates, then the buffer flushes) a reader landing
in that window gets `JSONDecodeError: Expecting value: line 1 column 1`, and
the damage depends purely on where the read happened: at the batch level it
kills the worker and loses its whole shard, while per-sample it is caught by
`core.runs.executor.execute`, logged as a warning, and the command exits 0
having silently skipped that sample -- results missing, nothing reported.

Making the reads actually overlap the writes is the hard part, and the reason
for the barrier below. The writer's entire job is ~10ms, while `spawn` startup
jitter between siblings is larger than that -- so just starting the processes
and hoping loses the race most of the time, and a reader that runs entirely
before or entirely after the writes finds nothing wrong and passes green
having tested nothing. The barrier releases every process at one instant, so
the reads straddle the writes by construction; `n_reads > 0` then fails loudly
if they somehow still didn't.
"""

import multiprocessing as mp
import time
from multiprocessing.synchronize import Barrier as BarrierT
from multiprocessing.synchronize import Event as EventT
from pathlib import Path

import numpy as np

from pylibs.core import api
from pylibs.core.io.writers.libs_writer import write_libs_file
from pylibs.core.project.registry import Registry

N_WRITES = 150
N_READERS = 2
TIMEOUT = 60


def _writer(root: Path, start: BarrierT, done: EventT) -> None:
    registry = Registry.load(root)
    start.wait(timeout=TIMEOUT)
    try:
        for index in range(N_WRITES):
            registry.mark_sample_processed("sample001", root / f"results{index}.zarr")
    finally:
        # even on failure, release the readers rather than leaving them to
        # spin out their own deadline
        done.set()


def _reader(root: Path, queue: mp.Queue, start: BarrierT, done: EventT) -> None:
    """Read until the writer signals it has finished, never a fixed count.

    Every wait is bounded, so a writer that dies before the barrier leaves
    this reporting zero reads -- which fails the test's non-vacuity assertion
    -- rather than spinning on a `done` that will never be set."""
    failures: list[str] = []
    n_reads = 0
    try:
        start.wait(timeout=TIMEOUT)
    except mp.BrokenBarrierError:
        queue.put((0, failures))
        return
    deadline = time.monotonic() + TIMEOUT
    while not done.is_set() and time.monotonic() < deadline:
        n_reads += 1
        try:
            Registry.load(root)
        except Exception as exc:  # noqa: BLE001 -- any failure here is the bug
            failures.append(f"{type(exc).__name__}: {exc}")
    queue.put((n_reads, failures))


def test_unlocked_readers_never_observe_a_truncated_registry(tmp_path):
    root = tmp_path / "proj"
    # no peak table needed: this exercises Registry.load against
    # mark_sample_processed, and add_sample resolves no peak ids
    libs_path = tmp_path / "sample.libs"
    write_libs_file(
        libs_path,
        data=np.full((4, 8), 10.0),
        wavelengths=np.linspace(200.0, 210.0, 8),
        params={},
    )
    api.create_project("demo", root=root)
    api.add_sample("demo", libs_path, project_root=root)

    # spawn, not fork: matches core.runs.executor's own context, avoids
    # fork()-in-a-multi-threaded-process, and is what the real sharded
    # workflow does anyway
    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    start = context.Barrier(N_READERS + 1)
    done = context.Event()
    readers = [
        context.Process(target=_reader, args=(root, queue, start, done)) for _ in range(N_READERS)
    ]
    writer = context.Process(target=_writer, args=(root, start, done))
    for process in [*readers, writer]:
        process.start()

    outcomes = [queue.get(timeout=TIMEOUT) for _ in readers]
    for process in [*readers, writer]:
        process.join(timeout=TIMEOUT)

    # every process finished cleanly, and the readers actually read: a writer
    # that died on its first line would otherwise leave a stable file, zero
    # failures, and a test that proved nothing
    assert writer.exitcode == 0, f"writer exited {writer.exitcode}"
    for index, reader in enumerate(readers):
        assert reader.exitcode == 0, f"reader {index} exited {reader.exitcode}"
    n_reads = sum(reads for reads, _ in outcomes)
    assert n_reads > 0, "no read overlapped the writes -- the test proved nothing"

    failures = [failure for _, reader_failures in outcomes for failure in reader_failures]
    assert failures == [], (
        f"{len(failures)} of {n_reads} unlocked read(s) saw a partial registry: {failures[:3]}"
    )


N_REGISTRARS = 4
N_EACH = 8


def _registrar(root: Path, index: int, start: BarrierT) -> None:
    registry = Registry.load(root)
    start.wait(timeout=TIMEOUT)
    for i in range(N_EACH):
        registry.add_sample(Path(f"/data/p{index}_s{i}.libs"))
        registry.add_feature(f"F{index}_{i}")


def test_concurrent_registration_loses_nothing(tmp_path):
    """Several processes registering different samples/features at once must
    all survive, with distinct ids.

    Every mutator rewrites the whole registry file. Doing that from a snapshot
    taken when this `Registry` was loaded means the last writer wins with only
    its own view, so a sibling's registration vanishes -- and `_next_id`
    computed from a stale snapshot hands two callers the same id. Against the
    version before `_locked_update`, this scenario kept 25 of 72 samples and 24
    of 72 features.

    This is not the sharded-worker path (those only call
    `mark_sample_processed`, which always re-read under the lock). It is two
    `pylibs project add-sample` invocations, or two DSL scripts, against one
    project."""
    root = tmp_path / "proj"
    libs_path = tmp_path / "sample.libs"
    write_libs_file(
        libs_path,
        data=np.full((4, 8), 10.0),
        wavelengths=np.linspace(200.0, 210.0, 8),
        params={},
    )
    api.create_project("demo", root=root)

    context = mp.get_context("spawn")
    start = context.Barrier(N_REGISTRARS)
    registrars = [
        context.Process(target=_registrar, args=(root, index, start))
        for index in range(N_REGISTRARS)
    ]
    for process in registrars:
        process.start()
    for process in registrars:
        process.join(timeout=TIMEOUT)

    for index, process in enumerate(registrars):
        assert process.exitcode == 0, f"registrar {index} exited {process.exitcode}"

    expected = N_REGISTRARS * N_EACH
    final = Registry.load(root)
    assert len(final.samples) == expected, f"{expected - len(final.samples)} sample(s) lost"
    assert len(final.features) == expected, f"{expected - len(final.features)} feature(s) lost"
    # ids are handed out from the re-read state, so no two registrars collide
    assert len(set(final.samples)) == expected
    assert len({sample.path for sample in final.samples.values()}) == expected
