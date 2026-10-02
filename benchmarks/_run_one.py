"""One timed benchmark run, in its own interpreter.

Not meant to be called by hand -- `benchmark.py` launches it once per
(scenario, worker count, repeat) and reads the JSON it prints on the last
line of stdout.

Why a subprocess per run rather than a loop inside one process. Every timed
run has to start from an empty project, because every stage of the pipeline
skips work it has already done and a second run in the same project would
measure nothing. A fresh interpreter also keeps module import state, the
project-scoped log handler and any allocator warm-up from leaking between
worker counts, so the only thing that differs between two rows of results.csv
is the number of workers.

Everything here goes through `pylibs.core.api`, the same façade a user calls.
"""

from __future__ import annotations

# Must precede any import that pulls in NumPy: the thread limits are read by
# the BLAS backends at load time, and child worker processes inherit this
# environment. One benchmark worker means one thread.
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from pylibs.core import api  # noqa: E402

PROJECT_NAME = "benchmark_project"
LOG_FILENAME = "pylibs.log"

# "2026-09-23 14:02:11,431 DEBUG pylibs.core.pipeline.runner: Running step 'features' ..."
_STEP_LINE = re.compile(
    r"^(?P<stamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*Running step '(?P<step>[^']+)'"
)
_END_LINE = re.compile(r"^(?P<stamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*Pipeline finished")
# Feature extraction and spectrum statistics each stream the raw file once and
# take no worker count; the two distribution steps are the only ones
# `n_processes` reaches. That split is the sequential/parallel boundary the
# Amdahl bound needs, measured rather than assumed.
EXTRACTION_STEPS = {"features", "spectrum_stats"}
METRICS_STEPS = {"distribution_1d", "distribution_2d"}


def _stamp(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S,%f").timestamp()


def stage_times(log_path: Path) -> dict[str, float]:
    """Seconds spent in extraction vs distribution metrics, read off the
    project's own millisecond-stamped debug log.

    The runner logs each step's name immediately before running it and one
    line after the last one finishes, in a fixed order, so consecutive
    timestamps bound each step. Returns an empty dict if the log does not
    carry a complete set of markers -- the caller then reports only the total
    rather than inventing a split."""
    if not log_path.exists():
        return {}
    marks: list[tuple[float, str]] = []
    end: float | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if (match := _STEP_LINE.match(line)) is not None:
            marks.append((_stamp(match.group("stamp")), match.group("step")))
        elif (match := _END_LINE.match(line)) is not None:
            end = _stamp(match.group("stamp"))
    if not marks or end is None:
        return {}
    totals = {"extraction": 0.0, "metrics": 0.0}
    for index, (start, step) in enumerate(marks):
        stop = marks[index + 1][0] if index + 1 < len(marks) else end
        if step in EXTRACTION_STEPS:
            totals["extraction"] += stop - start
        elif step in METRICS_STEPS:
            totals["metrics"] += stop - start
    return totals


def build_project(root: Path, config: dict, sample_paths: list[Path]) -> list[str]:
    """A complete, empty-of-results project: peak table, samples, features."""
    api.create_project(PROJECT_NAME, root=root, description="benchmark")
    api.use_peak_table(PROJECT_NAME, Path(config["data"]["peak_table"]), project_root=root)
    for path in sample_paths:
        # sample_name groups a physical sample's ablation layers, exactly as
        # the bundled example's filename scheme derives it
        api.add_sample(
            PROJECT_NAME, path, sample_name=path.stem.rsplit("_", 1)[0], project_root=root
        )
    for expression in config["features"]:
        api.add_feature(PROJECT_NAME, expression, project_root=root)
    return [sample.sample_id for sample in api.list_samples(PROJECT_NAME, project_root=root)]


def metrics_digest(root: Path, sample_ids: list[str]) -> np.ndarray:
    """Every stored distribution metric, in a fixed order, as one flat array.

    Read back through the façade so the comparison covers what a user would
    actually get, not an internal buffer. Metric dictionaries are sorted by
    name and features by id, so two runs differing only in worker count
    produce element-wise comparable arrays."""
    values: list[float] = []
    for sample_id in sorted(sample_ids):
        features = api.list_features(PROJECT_NAME, project_root=root)
        for feature in sorted(features, key=lambda f: f.feature_id):
            try:
                one_d = api.get_distribution_1d(
                    PROJECT_NAME, sample_id, feature.feature_id, project_root=root
                )
            except Exception:
                continue
            values.extend(one_d.metrics[name] for name in sorted(one_d.metrics))
            try:
                two_d = api.get_distribution_2d(
                    PROJECT_NAME, sample_id, feature.feature_id, project_root=root
                )
            except Exception:
                continue
            values.extend(two_d.metrics[name] for name in sorted(two_d.metrics))
    return np.asarray(values, dtype=float)


def register_bootstrap(root: Path, config: dict):
    """Register the resampling configuration. `pct` is a share of the
    sample's own scan count, resolved by the library; `npix` is a fixed
    count. Exactly one of them is given."""
    settings = config["bootstrap"]
    return api.add_bootstrap_config(
        PROJECT_NAME,
        settings["method"],
        npix=settings.get("npix"),
        pct=settings.get("pct"),
        n_iterations=settings["n_iterations"],
        base_seed=settings["base_seed"],
        shuffle=settings["shuffle"],
        project_root=root,
    )


def scenario_a(root: Path, config: dict, workers: int) -> dict:
    """One sample, every feature, varying the in-process worker count."""
    sample = Path(config["data"]["single_sample"])
    sample_ids = build_project(root, config, [sample])
    api.set_debug_logging(True)
    start = time.perf_counter()
    result = api.run_pipeline(PROJECT_NAME, sample_ids[0], project_root=root, n_processes=workers)
    total = time.perf_counter() - start
    api.set_debug_logging(False)
    stages = stage_times(root / LOG_FILENAME)
    return {
        "t_total_s": total,
        "t_extraction_s": stages.get("extraction", float("nan")),
        "t_metrics_s": stages.get("metrics", float("nan")),
        "n_samples": 1,
        "n_features": len(config["features"]),
        "n_pixels": result.n_scans,
        "B": 0,
        "digest": metrics_digest(root, sample_ids),
    }


def scenario_b(root: Path, config: dict, workers: int, sample_paths: list[Path]) -> dict:
    """A batch of samples, varying the in-process worker count."""
    sample_ids = build_project(root, config, sample_paths)
    start = time.perf_counter()
    api.run_pipeline_batch(PROJECT_NAME, project_root=root, n_processes=workers)
    total = time.perf_counter() - start
    n_pixels = api.run_pipeline(
        PROJECT_NAME, sample_ids[0], project_root=root
    ).n_scans  # already done, so this only reads back the scan count
    return {
        "t_total_s": total,
        "t_extraction_s": float("nan"),
        "t_metrics_s": float("nan"),
        "n_samples": len(sample_ids),
        "n_features": len(config["features"]),
        "n_pixels": n_pixels,
        "B": 0,
        "digest": metrics_digest(root, sample_ids),
    }


def scenario_c(root: Path, config: dict, workers: int) -> dict:
    """Bootstrap resampling over one sample. The pipeline it needs is built
    first and deliberately not timed."""
    sample = Path(config["data"]["single_sample"])
    sample_ids = build_project(root, config, [sample])
    pipeline = api.run_pipeline(PROJECT_NAME, sample_ids[0], project_root=root)
    bootstrap = register_bootstrap(root, config)
    settings = config["bootstrap"]
    start = time.perf_counter()
    api.run_bootstrap(
        PROJECT_NAME,
        sample_ids[0],
        bootstrap.bootstrap_id,
        project_root=root,
        n_processes=workers,
    )
    total = time.perf_counter() - start
    stored = api.get_bootstrap_results(
        PROJECT_NAME, sample_ids[0], bootstrap.bootstrap_id, project_root=root
    )
    return {
        "t_total_s": total,
        "t_extraction_s": float("nan"),
        "t_metrics_s": float("nan"),
        "n_samples": 1,
        "n_features": len(config["features"]),
        "n_pixels": pipeline.n_scans,
        "B": settings["n_iterations"],
        "digest": np.asarray(stored.values, dtype=float).ravel(),
    }


def setup_for_shards(root: Path, config: dict, scenario: str, sample_paths: list[Path]) -> dict:
    """Scenario D's preparation: build the project the shard processes will
    share, and for the bootstrap case run its pipeline and register its
    resampling configuration. Prints what the launcher needs; times nothing."""
    sample_ids = build_project(root, config, sample_paths)
    out = {"sample_ids": sample_ids, "bootstrap_id": None}
    if scenario == "C":
        api.run_pipeline(PROJECT_NAME, sample_ids[0], project_root=root)
        out["bootstrap_id"] = register_bootstrap(root, config).bootstrap_id
    return out


def collect_after_shards(root: Path, scenario: str, sample_ids: list[str], bootstrap_id) -> dict:
    """Scenario D's read-back, once every shard process has exited."""
    if scenario == "C":
        stored = api.get_bootstrap_results(
            PROJECT_NAME, sample_ids[0], bootstrap_id, project_root=root
        )
        digest = np.asarray(stored.values, dtype=float).ravel()
    else:
        digest = metrics_digest(root, sample_ids)
    return {"digest": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True, choices=["A", "B", "C"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--digest-out", default="")
    parser.add_argument("--scratch", default="")
    parser.add_argument("--keep", action="store_true")
    # Scenario D splits one run across separately launched shard processes, so
    # its project has to outlive this interpreter: `setup` builds it and
    # prints where, the launcher runs the shards, `collect` reads the result
    # back. `run` is the single-process path used by A, B and C.
    parser.add_argument("--mode", default="run", choices=["run", "setup", "collect"])
    parser.add_argument("--root", default="")
    parser.add_argument("--bootstrap-id", default="")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    sample_paths = [Path(p) for p in config["_resolved_batch"]]

    if args.mode == "setup":
        root = Path(tempfile.mkdtemp(prefix="pylibs_bench_", dir=args.scratch or None))
        record = setup_for_shards(root, config, args.scenario, sample_paths)
        record["root"] = str(root)
        print(json.dumps(record))
        return

    if args.mode == "collect":
        root = Path(args.root)
        sample_ids = [s.sample_id for s in api.list_samples(PROJECT_NAME, project_root=root)]
        record = collect_after_shards(
            root, args.scenario, sorted(sample_ids), args.bootstrap_id or None
        )
        digest = record.pop("digest")
        np.save(args.digest_out, digest)
        record["n_digest"] = int(digest.size)
        print(json.dumps(record))
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        return

    root = Path(tempfile.mkdtemp(prefix="pylibs_bench_", dir=args.scratch or None))
    try:
        if args.scenario == "A":
            record = scenario_a(root, config, args.workers)
        elif args.scenario == "B":
            record = scenario_b(root, config, args.workers, sample_paths)
        else:
            record = scenario_c(root, config, args.workers)
        digest = record.pop("digest")
        np.save(args.digest_out, digest)
        record["n_digest"] = int(digest.size)
        record["project_root"] = str(root) if args.keep else ""
        print(json.dumps(record))
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
