"""Slower integration test: several genuinely separate OS processes (not
just an in-process worker pool) each claiming a disjoint --worker-index
shard of run-pipeline-batch's sample list, and writing directly into the
same results store *and* the same project registry concurrently -- the one
deliberate exception to "workers never touch the store" (see
`pipeline.runner.run_pipeline_batch`'s docstring). This is the empirical
check for that module's two safety arguments:

- every sample writes to its own disjoint `<sample_id>/...` zarr group, so
  concurrent writers should never race there, unlike bootstrap's shared
  checkpoint array (which is why that needs chunk-size-1 + disjoint index
  partitioning instead);
- `Registry.mark_sample_processed` (called once per sample, from whichever
  worker process handled it) is safe against concurrent callers for
  *different* samples -- it locks (`project.registry._locked_registry_dir`)
  and re-reads the registry fresh immediately before merging its own
  update in, rather than trusting a stale in-memory snapshot. Without that,
  this test used to silently lose most samples' `results_path`/
  `processed_at_*` (whichever worker's `save()` landed last would win with
  only its own view of the registry, clobbering its siblings' updates) even
  though every sample's actual zarr results were written correctly -- a
  real bug this test caught.
"""

import subprocess
import sys

import numpy as np

from pylibs.core import api
from pylibs.core.io.readers.libs_reader import write_libs_file
from pylibs.core.io.store import RESULTS_DIRNAME, ResultsStore
from pylibs.core.project.project import Project

N_WORKERS = 3
N_SAMPLES = 9


def test_concurrent_sharded_run_pipeline_batch_completes_without_corruption(tmp_path):
    root = tmp_path / "proj"
    wavelengths = np.linspace(200.0, 210.0, 50)
    peak_idx = np.argmin(np.abs(wavelengths - 205.0))
    peak_table = tmp_path / "peaks.csv"
    peak_table.write_text(
        "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\nA,A205,205.0,,,,1\n"
    )

    api.create_project("demo", root=root)
    api.use_peak_table("demo", peak_table, project_root=root)
    api.add_feature("demo", "A205", project_root=root)
    for index in range(1, N_SAMPLES + 1):
        data = np.full((10, len(wavelengths)), 10.0)
        data[:, peak_idx] = np.arange(10) + 100.0 + index
        libs_path = tmp_path / f"sample{index}.libs"
        write_libs_file(libs_path, data=data, wavelengths=wavelengths, params={})
        api.add_sample("demo", libs_path, project_root=root)

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from pylibs.interfaces.cli.app import app; app()",
                "project",
                "run-pipeline-batch",
                "demo",
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
    for index in range(1, N_SAMPLES + 1):
        sample_id = f"sample{index:03d}"
        assert store.has_distribution_1d(sample_id), sample_id
        features_array = store.get_features_array(sample_id)
        assert features_array is not None
        assert not np.isnan(np.asarray(features_array[:])).any()

    # every sample's registry entry must show it was actually marked
    # processed -- not just N_SAMPLES/N_WORKERS of them (see module
    # docstring: this is the part that used to silently race).
    project = Project.load(root)
    for index in range(1, N_SAMPLES + 1):
        sample_id = f"sample{index:03d}"
        sample = project.get_sample(sample_id)
        assert sample.results_path is not None, f"{sample_id} was never marked processed"
