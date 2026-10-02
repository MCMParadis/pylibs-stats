"""Scenario tests for the two parallel-execution guarantees that a
single-process test cannot express.

The first is the checkpoint write ordering. `CheckpointWriter` batches
finished iterations, so the "results are on disk before their completion
flag" invariant now has to hold across a batch rather than a single pair of
writes. The test interrupts a flush between the two, which a real crash would
do, and asserts that nothing in that batch came back flagged complete.

The second is pool reuse. A pipeline run fans two separate steps out to
workers, and building a pool for each one pays the per-worker import cost
twice. The test counts how many pools a single run actually constructs.
"""

import numpy as np
import pytest
import zarr

from pylibs.core import api
from pylibs.core.io.writers.libs_writer import write_libs_file
from pylibs.core.runs import executor
from pylibs.core.runs.checkpoint import CheckpointArrays, CheckpointWriter


def _arrays(tmp_path, n_iterations: int) -> CheckpointArrays:
    group = zarr.open_group(str(tmp_path / "checkpoint.zarr"), mode="w")
    results = group.create_array(
        "results", shape=(n_iterations, 3), chunks=(1, 3), dtype="f8", fill_value=np.nan
    )
    completed = group.create_array(
        "completed", shape=(n_iterations,), chunks=(1,), dtype="i1", fill_value=0
    )
    return CheckpointArrays(results, completed)


def test_interrupted_flush_leaves_results_written_but_nothing_flagged(tmp_path):
    """A crash between a batch's result writes and its flag writes must leave
    every iteration in that batch pending, never falsely complete."""
    arrays = _arrays(tmp_path, 10)
    writer = CheckpointWriter(arrays, batch_size=4)

    real_selection = type(arrays.completed).set_orthogonal_selection
    calls = {"n": 0}

    def interrupt(self, selection, value, **kwargs):
        # the results array is written first, so the first call to reach the
        # completed array is the flag write -- fail exactly there
        if self is arrays.completed:
            calls["n"] += 1
            raise KeyboardInterrupt("simulated interruption between results and flags")
        return real_selection(self, selection, value, **kwargs)

    rows = {index: np.full(3, float(index)) for index in range(4)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(arrays.completed), "set_orthogonal_selection", interrupt)
        with pytest.raises(KeyboardInterrupt):
            for index, row in rows.items():
                writer.add(index, row)

    assert calls["n"] == 1, "the flag write should have been reached exactly once"
    # results landed ...
    stored = np.asarray(arrays.results[:])
    for index, row in rows.items():
        assert np.array_equal(stored[index], row)
    # ... and not one of them is flagged, so all four stay pending
    assert int(np.asarray(arrays.completed[:]).sum()) == 0


def test_batched_writes_match_unbatched_exactly(tmp_path):
    """Batching is a write-shape change only: the stored arrays must be
    identical to what one-write-per-iteration produces, including when
    iterations arrive out of order."""
    order = [7, 1, 9, 3, 0, 8, 2, 6, 4, 5]
    rows = {index: np.arange(3, dtype=float) * index for index in order}

    batched = _arrays(tmp_path / "a", 10)
    with CheckpointWriter(batched, batch_size=4) as writer:
        for index in order:
            writer.add(index, rows[index])

    single = _arrays(tmp_path / "b", 10)
    with CheckpointWriter(single, batch_size=1) as writer:
        for index in order:
            writer.add(index, rows[index])

    assert np.array_equal(np.asarray(batched.results[:]), np.asarray(single.results[:]))
    assert np.array_equal(np.asarray(batched.completed[:]), np.asarray(single.completed[:]))
    assert int(np.asarray(batched.completed[:]).sum()) == len(order)


def test_one_pipeline_run_builds_one_pool_for_both_distribution_steps(tmp_path):
    """Both distribution steps must submit into the same pool. Counting pool
    constructions is the only way to see it: the results are identical either
    way, which is exactly why the duplication went unnoticed."""
    root = tmp_path / "proj"
    wavelengths = np.linspace(200.0, 215.0, 200)
    n_scans = 24
    data = np.full((n_scans, len(wavelengths)), 10.0)
    for target in (205.0, 210.0):
        data[:, np.argmin(np.abs(wavelengths - target))] = np.arange(n_scans) + 100.0
    libs_path = tmp_path / "sample.libs"
    write_libs_file(libs_path, data=data, wavelengths=wavelengths, params={})
    peak_table = tmp_path / "peaks.csv"
    peak_table.write_text(
        "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\n"
        "A,A205,205.0,,,,1\nB,B210,210.0,,,,1\n"
    )

    api.create_project("demo", root=root)
    api.use_peak_table("demo", peak_table, project_root=root)
    api.add_sample("demo", libs_path, project_root=root)
    # a bare peak and a ratio, so both distribution steps have work to do
    api.add_feature("demo", "A205", project_root=root)
    api.add_feature("demo", "A205/B210", project_root=root)

    built = []
    real_executor = executor.ProcessPoolExecutor

    def counting(*args, **kwargs):
        built.append(1)
        return real_executor(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(executor, "ProcessPoolExecutor", counting)
        api.run_pipeline("demo", "sample001", project_root=root, n_processes=2)

    assert len(built) == 1, f"expected one shared pool, {len(built)} were built"

    # and the run still produced both steps' output
    assert api.get_distribution_1d("demo", "sample001", "feature001", project_root=root) is not None
    assert api.get_distribution_2d("demo", "sample001", "feature002", project_root=root) is not None


def test_sequential_path_builds_no_pool(tmp_path):
    """One worker must stay genuinely in-process: no pool, no spawn, no
    temporary state files."""
    root = tmp_path / "proj"
    wavelengths = np.linspace(200.0, 210.0, 120)
    data = np.full((16, len(wavelengths)), 10.0)
    data[:, np.argmin(np.abs(wavelengths - 205.0))] = np.arange(16) + 100.0
    libs_path = tmp_path / "sample.libs"
    write_libs_file(libs_path, data=data, wavelengths=wavelengths, params={})
    peak_table = tmp_path / "peaks.csv"
    peak_table.write_text(
        "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\nA,A205,205.0,,,,1\n"
    )

    api.create_project("demo", root=root)
    api.use_peak_table("demo", peak_table, project_root=root)
    api.add_sample("demo", libs_path, project_root=root)
    api.add_feature("demo", "A205", project_root=root)

    built = []
    real_executor = executor.ProcessPoolExecutor

    def counting(*args, **kwargs):
        built.append(1)
        return real_executor(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(executor, "ProcessPoolExecutor", counting)
        api.run_pipeline("demo", "sample001", project_root=root, n_processes=1)

    assert built == []
