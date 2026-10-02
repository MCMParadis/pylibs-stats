"""Concurrency tests for a bootstrap run shared by several workers: the real
SLURM-array-job scenario below, plus a deterministic simulation of the
group-creation race it exposed (which needs no processes -- a real one is
microseconds wide).

The array-job test spawns several genuinely separate OS processes (not just
an in-process worker pool) with distinct --worker-index values against one
project, asserting the whole run completes without corruption. This is the
one scenario `core.runs.partition`'s disjoint-static-sharding design is
actually for -- `ProcessPoolExecutor`-based tests elsewhere in the suite cover
in-process parallelism, but not concurrent writers from separate processes.
"""

import subprocess
import sys

import numpy as np
import pytest
import zarr

from pylibs.core import api
from pylibs.core.io.store import BOOTSTRAP_GROUP, RESULTS_DIRNAME, ResultsStore
from pylibs.core.io.writers.libs_writer import write_libs_file
from pylibs.core.pipeline.metrics import METRIC_NAMES

N_WORKERS = 3
N_ITERATIONS = 12


def test_concurrent_sharded_run_bootstrap_completes_without_corruption(tmp_path):
    root = tmp_path / "proj"
    n_scans = 30
    wavelengths = np.linspace(200.0, 210.0, 200)
    data = np.full((n_scans, len(wavelengths)), 10.0)
    peak_idx = np.argmin(np.abs(wavelengths - 205.0))
    data[:, peak_idx] = np.arange(n_scans) + 100.0
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
    api.run_pipeline("demo", "sample001", project_root=root)
    api.add_bootstrap_config(
        "demo", method="random", npix=8, n_iterations=N_ITERATIONS, project_root=root
    )

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from pylibs.interfaces.cli.app import app; app()",
                "project",
                "run-bootstrap",
                "demo",
                "sample001",
                "bootstrap001",
                "--root",
                str(root),
                "--worker-index",
                str(worker_index),
                "--n-workers",
                str(N_WORKERS),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for worker_index in range(N_WORKERS)
    ]
    outcomes = [process.communicate(timeout=60) for process in processes]

    for process, (stdout, stderr) in zip(processes, outcomes, strict=True):
        assert process.returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"

    store = ResultsStore(root / RESULTS_DIRNAME)
    metrics, completed = store.get_bootstrap_arrays("sample001", "bootstrap001")
    assert completed.shape == (N_ITERATIONS,)
    assert list(completed[:]) == [1] * N_ITERATIONS

    mean_index = METRIC_NAMES.index("Mean")
    assert not np.isnan(metrics[:, 0, mean_index]).any()

    progress = api.get_bootstrap_progress("demo", "sample001", "bootstrap001", project_root=root)
    assert progress.n_done == N_ITERATIONS


BOOTSTRAP_ARRAY_KWARGS = dict(
    n_iterations=4,
    feature_ids=["feature001"],
    feature_hashes={"feature001": "abc"},
    metric_names=["Mean"],
    method="random",
    resolved_npix=8,
    shuffle=False,
    base_seed=0,
)


@pytest.mark.parametrize("racing_group", ["sample001", BOOTSTRAP_GROUP, "bootstrap001"])
def test_group_creation_race_returns_the_winners_group(tmp_path, monkeypatch, racing_group):
    """The loser of a concurrent group-creation race must keep going, and must
    end up with the winner's group rather than a fresh one.

    zarr's `require_group` is check-then-create, so when two sharded workers
    both miss and both create a group they share -- every level of
    `<sample>/bootstrap/<bootstrap_id>` -- the loser's create raises
    `ContainsGroupError` and used to kill that worker outright (3 failures in a
    25x run of this suite). A real race is microseconds wide, so it is
    simulated: the winner's state is written first, then `require_group` is
    patched to fail exactly where the loser's create would.

    Parametrized over all three levels because `store._require_group_concurrent`
    guards all three and the failures observed were at two different ones.
    """
    root = tmp_path / "proj"
    store = ResultsStore(root / RESULTS_DIRNAME)

    # the winner gets there first and records real progress, so "returned the
    # winner's group" is distinguishable from "created a fresh one"
    _, winner_completed = store.get_or_create_bootstrap_arrays(
        "sample001", "bootstrap001", **BOOTSTRAP_ARRAY_KWARGS
    )
    winner_completed[:] = 1

    real_require_group = zarr.Group.require_group
    raised = []

    def losing_require_group(self, name, **kwargs):
        group = real_require_group(self, name, **kwargs)
        if name == racing_group and not raised:
            raised.append(name)
            raise zarr.errors.ContainsGroupError(str(root), name)
        return group

    monkeypatch.setattr(zarr.Group, "require_group", losing_require_group)

    metrics, completed = store.get_or_create_bootstrap_arrays(
        "sample001", "bootstrap001", **BOOTSTRAP_ARRAY_KWARGS
    )

    assert raised == [racing_group], "the race was never simulated"
    assert metrics.shape == (4, 1, 1)
    # the winner's completed flags survived -- a fresh group would be all zeros
    assert list(completed[:]) == [1, 1, 1, 1]
