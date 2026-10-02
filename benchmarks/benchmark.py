"""Parallel-execution benchmark for pylibs.

    python benchmarks/benchmark.py --config benchmarks/config.yaml

Measures wall-clock time and speedup against the number of parallel workers,
for four scenarios:

    A  one sample, every registered feature   (in-process pool)
    B  a batch of samples                      (in-process pool)
    C  bootstrap resampling over one sample    (in-process pool)
    D  B and C again, but split across separately launched shard processes

Everything is driven through `pylibs.core.api` and the `pylibs` command line,
as a user would drive it. No library code is involved beyond that.

Each timed run happens in a fresh interpreter against a fresh, empty project
(see `_run_one.py` for why), so a repeat measures the same work rather than a
no-op skip. The file cache is warm: one untimed warm-up precedes the timed
repeats of every scenario, and the bundled acquisitions are read repeatedly
throughout, so these numbers describe compute, not first-read disk latency.

Outputs land in the configured directory: results.csv (one row per run),
environment.json, summary.csv, speedup.pdf/.png and table.tex.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = Path(__file__).resolve().parent / "_run_one.py"
SCENARIO_LABELS = {
    "A": "A: one sample, 16 features",
    "Aext": "A+: one sample, extended features",
    "B": "B: batch of samples",
    "C": "C: bootstrap resampling",
    "D": "D: shard processes",
}
# One label per drawn curve. Scenarios A and A+ have no sharded counterpart,
# so only their in-process entries appear.
CURVE_LABELS = {
    ("A", "inprocess"): "A, 16 features",
    ("Aext", "inprocess"): "A, 175 features",
    ("B", "inprocess"): "B, pool",
    ("B", "multiprocess"): "B, processes",
    ("C", "inprocess"): "C, pool",
    ("C", "multiprocess"): "C, processes",
}
FIELDS = [
    "scenario",
    "mode",
    "workers",
    "repeat",
    "t_total_s",
    "t_extraction_s",
    "t_metrics_s",
    "n_samples",
    "n_features",
    "n_pixels",
    "B",
    "max_abs_diff_vs_serial",
    "timestamp",
]


# --------------------------------------------------------------- machine ---


def physical_cores() -> int:
    """Physical cores, not hardware threads -- two threads on one core share
    the execution units, so counting them would make efficiency look like it
    collapses past the real core count for reasons that have nothing to do
    with the software."""
    try:
        text = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=True, timeout=20
        ).stdout
        per_socket = int(re.search(r"Core\(s\) per socket:\s*(\d+)", text).group(1))
        sockets = int(re.search(r"Socket\(s\):\s*(\d+)", text).group(1))
        return per_socket * sockets
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def storage_kind(path: Path) -> str:
    """Rotational, solid-state or network, for the device holding `path`."""
    try:
        source = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE", "--target", str(path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.split()
        device, fstype = source[0], source[1]
        if fstype in {"nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"}:
            return f"network ({fstype})"
        if fstype in {"tmpfs", "ramfs"}:
            return f"RAM-backed ({fstype})"
        name = Path(device).name
        out = subprocess.run(
            ["lsblk", "-no", "ROTA,TRAN", f"/dev/{name}"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.split()
        if out:
            rotational = out[0] == "1"
            transport = out[1] if len(out) > 1 else ""
            kind = "HDD" if rotational else ("NVMe SSD" if transport == "nvme" else "SSD")
            return f"{kind} ({fstype})"
        return f"unknown ({fstype})"
    except Exception:
        return "unknown"


def git_state() -> dict:
    """Which revision is being measured.

    Captured before the sweep writes anything, so the dirty flag describes the
    code that ran rather than the output it produced. The tag matters more
    than the commit: a repository mirrored without history keeps its tags but
    not its hashes, so `v1.0.1` resolves where a hash would not."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()

    try:
        version = __import__("pylibs").__version__
    except Exception:
        version = "unknown"
    try:
        return {
            "pylibs_version": version,
            "pylibs_tag": run("describe", "--tags", "--always") or "unknown",
            "pylibs_commit": run("rev-parse", "HEAD") or "unknown",
            "pylibs_dirty": bool(run("status", "--porcelain")),
        }
    except Exception:
        return {
            "pylibs_version": version,
            "pylibs_tag": "unknown",
            "pylibs_commit": "unknown",
            "pylibs_dirty": False,
        }


def environment_record(config: dict, cores: int, scratch: Path, revision: dict) -> dict:
    def version(module: str) -> str:
        try:
            return __import__(module).__version__
        except Exception:
            return "not installed"

    try:
        model = re.search(
            r"^Model name:\s*(.+)$",
            subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout,
            re.MULTILINE,
        ).group(1)
    except Exception:
        model = platform.processor() or "unknown"
    try:
        kib = int(re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text()).group(1))
        ram = f"{kib / 1024 / 1024:.1f} GiB"
    except Exception:
        ram = "unknown"
    data_path = REPO_ROOT / config["data"]["sample_dir"]
    return {
        "cpu_model": model,
        "physical_cores": cores,
        "logical_cores": os.cpu_count(),
        "ram_total": ram,
        "storage_data": storage_kind(data_path),
        "storage_project": storage_kind(scratch),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "zarr": version("zarr"),
        **revision,
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "recorded_at_utc": datetime.now(UTC).isoformat(),
    }


# ----------------------------------------------------------------- runs ---


def expand_features(spec) -> list[str]:
    """A feature set, either listed outright or built from a rule.

    The rule form takes a list of peak ids and, for every ordered pair of
    distinct ones, emits the plain ratio, the normalised ratio x/(x+y), and
    the log of each -- so a much larger workload comes out of the same seven
    lines the bundled peak table already carries, with no extra data."""
    if isinstance(spec, list):
        return list(spec)
    peaks = list(spec["peaks"])
    include = set(spec["include"])
    ratios = [f"{x}/{y}" for x in peaks for y in peaks if x != y]
    normalised = [f"({x})/({x}+{y})" for x in peaks for y in peaks if x != y]
    out: list[str] = []
    if "bare" in include:
        out += peaks
    if "ratio" in include:
        out += ratios
    if "normalised" in include:
        out += normalised
    if "log_ratio" in include:
        out += [f"log({r})" for r in ratios]
    if "log_normalised" in include:
        out += [f"log({n})" for n in normalised]
    return out


def worker_grid(requested: list[int], cores: int) -> list[int]:
    """Powers of two up to the physical core count, plus that count itself
    when it is not one. Never above it."""
    if requested:
        return sorted({w for w in requested if 1 <= w <= cores})
    grid = []
    value = 1
    while value <= cores:
        grid.append(value)
        value *= 2
    if cores not in grid:
        grid.append(cores)
    return sorted(grid)


def launch(config_path: Path, scenario: str, workers: int, digest: Path, scratch: str) -> dict:
    """One timed run, in its own interpreter."""
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config_path),
        "--scenario",
        "A" if scenario == "Aext" else scenario,
        "--workers",
        str(workers),
        "--digest-out",
        str(digest),
        "--scratch",
        scratch,
    ]
    finished = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT)
    if finished.returncode != 0:
        raise RuntimeError(
            f"scenario {scenario} at {workers} worker(s) failed:\n{finished.stderr[-4000:]}"
        )
    return json.loads(finished.stdout.strip().splitlines()[-1])


def run_shards(
    config_path: Path, scenario: str, shards: int, digest: Path, scratch: str
) -> tuple[dict, list[dict]]:
    """Scenario D: the same work split across `shards` separately launched
    processes, each with one in-process worker and its own shard index, all
    writing into one project store -- the array-job mode, run locally.

    Timed from the first process starting to the last one exiting, with each
    process's own start and end recorded so load imbalance between shards is
    visible rather than averaged away."""
    setup = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--scenario",
            scenario,
            "--mode",
            "setup",
            "--scratch",
            scratch,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    prepared = json.loads(setup.stdout.strip().splitlines()[-1])
    root = prepared["root"]

    if scenario == "B":
        base = ["pylibs", "project", "run-pipeline-batch", "benchmark_project", "--root", root]
    else:
        base = [
            "pylibs",
            "project",
            "run-bootstrap",
            "benchmark_project",
            prepared["sample_ids"][0],
            prepared["bootstrap_id"],
            "--root",
            root,
        ]

    started = time.perf_counter()
    processes = []
    for index in range(shards):
        command = base + [
            "--n-processes",
            "1",
            "--worker-index",
            str(index),
            "--n-workers",
            str(shards),
        ]
        processes.append(
            (
                index,
                time.perf_counter(),
                subprocess.Popen(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=REPO_ROOT
                ),
            )
        )
    per_process = []
    failures = []
    for index, begin, process in processes:
        _, stderr = process.communicate()
        end = time.perf_counter()
        per_process.append({"shard": index, "start_s": begin - started, "end_s": end - started})
        if process.returncode != 0:
            failures.append(f"shard {index}: {stderr.decode(errors='replace')[-2000:]}")
    total = time.perf_counter() - started
    if failures:
        raise RuntimeError("\n".join(failures))

    collected = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--scenario",
            scenario,
            "--mode",
            "collect",
            "--root",
            root,
            "--bootstrap-id",
            prepared["bootstrap_id"] or "",
            "--digest-out",
            str(digest),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    json.loads(collected.stdout.strip().splitlines()[-1])
    return {"t_total_s": total}, per_process


# -------------------------------------------------------------- outputs ---


def summarise(rows: list[dict]) -> list[dict]:
    """Median time, speedup and efficiency per (scenario, mode, workers).

    Median, not mean: with three repeats one scheduling hiccup would drag a
    mean well off the typical run, and the median is the number a reader can
    reproduce."""
    keys = sorted({(r["scenario"], r["mode"]) for r in rows})
    out = []
    for scenario, mode in keys:
        subset = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode]
        counts = sorted({r["workers"] for r in subset})
        serial = np.median([r["t_total_s"] for r in subset if r["workers"] == counts[0]])
        for workers in counts:
            times = [r["t_total_s"] for r in subset if r["workers"] == workers]
            median = float(np.median(times))
            speedup = serial / median if median > 0 else float("nan")
            record = {
                "scenario": scenario,
                "mode": mode,
                "workers": workers,
                "median_t_total_s": median,
                "speedup": speedup,
                "efficiency": speedup / workers,
                "amdahl_bound": "",
                "sequential_fraction": "",
            }
            if scenario in {"A", "Aext"} and mode == "inprocess":
                at_one = [
                    r
                    for r in subset
                    if r["workers"] == 1 and np.isfinite(r.get("t_extraction_s", float("nan")))
                ]
                if at_one:
                    fraction = float(
                        np.median([r["t_extraction_s"] / r["t_total_s"] for r in at_one])
                    )
                    record["sequential_fraction"] = fraction
                    record["amdahl_bound"] = 1.0 / (fraction + (1 - fraction) / workers)
            out.append(record)
    return out


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def plot(summary: list[dict], out_dir: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # single-column width, with extra height for the legend strip below the
    # axes, so the fonts stay legible after the journal scales it down
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    colours = {"A": "#1b1b1b", "Aext": "#6a3d9a", "B": "#c1272d", "C": "#0057b7"}
    markers = {"A": "o", "Aext": "v", "B": "s", "C": "^"}

    grid = sorted({row["workers"] for row in summary})
    ax.plot(grid, grid, color="0.6", linewidth=0.9, linestyle=":", label="ideal", zorder=1)

    for scenario in ["A", "Aext", "B", "C"]:
        for mode in ["inprocess", "multiprocess"]:
            subset = sorted(
                (r for r in summary if r["scenario"] == scenario and r["mode"] == mode),
                key=lambda r: r["workers"],
            )
            if not subset:
                continue
            shards = mode == "multiprocess"
            label = CURVE_LABELS[(scenario, mode)]
            ax.plot(
                [r["workers"] for r in subset],
                [r["speedup"] for r in subset],
                marker=markers[scenario],
                markersize=4,
                # shard runs keep their scenario's colour and marker but are
                # drawn dashed and hollow, so the two modes of one scenario
                # read as a pair rather than as four unrelated curves
                markerfacecolor="none" if shards else colours[scenario],
                linewidth=1.3,
                linestyle="--" if shards else "-",
                color=colours[scenario],
                label=label,
            )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(grid, [str(w) for w in grid])
    # below 1 as well as above it: the in-process pool slows scenario A down,
    # and a reader cannot judge by how much without ticks under 1
    y_ticks = [0.25, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(y_ticks, ["0.25", "0.5", "1", "2", "4", "8", "16"])
    ax.set_xlabel("Workers or processes", fontsize=12)
    ax.set_ylabel("Speedup $T(1)/T(p)$", fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    # under the axes rather than inside them: at 16-fold the curves reach the
    # top-left corner the legend used to sit in
    # To the right of the axes, single column. Not inside: the lower-right
    # region looks empty at a glance, but scenario A's in-process curve
    # descends straight through it.
    ax.legend(
        fontsize=8,
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handlelength=1.8,
        labelspacing=0.5,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "speedup.pdf")
    fig.savefig(out_dir / "speedup.png", dpi=dpi)
    plt.close(fig)


def write_table(summary: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Median wall-clock time, speedup $S(p)=T(1)/T(p)$ and parallel "
        r"efficiency $E(p)=S(p)/p$ for each benchmark scenario.}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Scenario & Mode & $p$ & $T$ (s) & $S(p)$ & $E(p)$ \\",
        r"\midrule",
    ]
    previous = None
    for row in summary:
        head = SCENARIO_LABELS[row["scenario"]] if row["scenario"] != previous else ""
        previous = row["scenario"]
        lines.append(
            f"{head} & {row['mode']} & {row['workers']} & "
            f"{row['median_t_total_s']:.2f} & {row['speedup']:.2f} & {row['efficiency']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


# ----------------------------------------------------------------- main ---


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmarks/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="smallest sample, workers 1 and 2")
    parser.add_argument("--keep", action="store_true", help="keep temporary projects")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="rebuild summary.csv, the plot and table.tex from an existing results.csv",
    )
    args = parser.parse_args()

    config = yaml.safe_load((REPO_ROOT / args.config).read_text())
    cores = physical_cores()
    out_dir = REPO_ROOT / (args.out or config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        # no runs, no temporary projects: read back what a previous sweep
        # measured and redraw from it, so a styling change costs nothing
        rows = []
        with (out_dir / "results.csv").open() as handle:
            for row in csv.DictReader(handle):
                row["workers"] = int(row["workers"])
                for key in ("t_total_s", "t_extraction_s", "t_metrics_s"):
                    row[key] = float(row[key]) if row[key] else float("nan")
                rows.append(row)
        summary = summarise(rows)
        write_csv(
            out_dir / "summary.csv",
            summary,
            [
                "scenario",
                "mode",
                "workers",
                "median_t_total_s",
                "speedup",
                "efficiency",
                "sequential_fraction",
                "amdahl_bound",
            ],
        )
        plot(summary, out_dir, int(config["output"]["dpi"]))
        write_table(summary, out_dir / "table.tex")
        print(f"Redrew {out_dir}/speedup.pdf, speedup.png, summary.csv, table.tex")
        return 0
    # before anything is written, so the dirty flag describes the code, not
    # the results this run is about to produce
    revision = git_state()

    scratch = config["run"].get("scratch_dir") or ""
    if scratch:
        # expanded and created up front, so a mistyped path fails here rather
        # than partway through a sweep
        scratch = str(Path(scratch).expanduser())
        Path(scratch).mkdir(parents=True, exist_ok=True)

    feature_sets = {name: expand_features(spec) for name, spec in config["feature_sets"].items()}
    (out_dir / "features_extended.txt").write_text("\n".join(feature_sets["extended"]) + "\n")
    print(
        "feature sets: "
        + ", ".join(f"{name} = {len(values)}" for name, values in sorted(feature_sets.items()))
    )

    sample_dir = REPO_ROOT / config["data"]["sample_dir"]
    batch = [REPO_ROOT / p for p in config["data"]["batch_samples"]] or sorted(
        sample_dir.glob("*.libs")
    )
    cap = int(config["data"].get("max_batch_samples") or 0)
    scenarios = list(config["run"]["scenarios"])
    grid = worker_grid(config["run"]["workers"], cores)
    repeats = int(config["run"]["repeats"])

    if args.dry_run:
        smallest = min(batch, key=lambda p: p.stat().st_size)
        config["data"]["single_sample"] = str(smallest.relative_to(REPO_ROOT))
        batch = sorted(batch)[:2]
        grid = [w for w in (1, 2) if w <= cores]
        repeats = 1
        config["run"]["warmup"] = False
        config["bootstrap"]["n_iterations"] = min(config["bootstrap"]["n_iterations"], 40)
        scenarios = [s for s in scenarios if s in {"A", "Aext", "B", "C", "D"}]
    elif cap:
        batch = batch[:cap]

    config["_resolved_batch"] = [str(p) for p in batch]
    # one config file per feature set: scenario Aext is scenario A run against
    # the larger set, and everything else uses the base set
    config_paths: dict[str, Path] = {}
    for name, values in feature_sets.items():
        path = Path(tempfile.mkstemp(suffix=".json", prefix=f"pylibs_bench_{name}_")[1])
        path.write_text(json.dumps({**config, "features": values}))
        config_paths[name] = path

    def config_for(scenario: str) -> Path:
        return config_paths["extended" if scenario == "Aext" else "base"]

    digest_dir = Path(tempfile.mkdtemp(prefix="pylibs_bench_digest_"))
    rows: list[dict] = []
    alerts: list[str] = []
    n_pixels_seen = ""
    shard_timeline: list[dict] = []

    try:
        for scenario in [s for s in scenarios if s in {"A", "Aext", "B", "C"}]:
            print(f"\n== scenario {scenario} == workers {grid}, {repeats} repeat(s)")
            baseline: np.ndarray | None = None
            if config["run"]["warmup"]:
                print("  warm-up (untimed)")
                launch(config_for(scenario), scenario, grid[0], digest_dir / "warmup.npy", scratch)
            for workers in grid:
                for repeat in range(1, repeats + 1):
                    digest = digest_dir / f"{scenario}_{workers}_{repeat}.npy"
                    record = launch(config_for(scenario), scenario, workers, digest, scratch)
                    values = np.load(digest)
                    if workers == grid[0] and repeat == 1:
                        baseline = values
                        difference = 0.0
                    elif baseline is None or values.shape != baseline.shape:
                        difference = float("nan")
                        alerts.append(
                            f"scenario {scenario}, {workers} worker(s): output shape "
                            f"{values.shape} differs from the serial run's "
                            f"{getattr(baseline, 'shape', None)}"
                        )
                    else:
                        with np.errstate(invalid="ignore"):
                            difference = float(np.nanmax(np.abs(values - baseline)))
                        if difference != 0.0:
                            alerts.append(
                                f"scenario {scenario}, {workers} worker(s), repeat {repeat}: "
                                f"max abs difference vs serial = {difference:.3e} (expected 0)"
                            )
                    rows.append(
                        {
                            **record,
                            "scenario": scenario,
                            "mode": "inprocess",
                            "workers": workers,
                            "repeat": repeat,
                            "max_abs_diff_vs_serial": difference,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    n_pixels_seen = record.get("n_pixels", n_pixels_seen)
                    print(
                        f"  p={workers:<3} repeat {repeat}  "
                        f"{record['t_total_s']:7.2f}s  max|diff|={difference:g}"
                    )

        if "D" in scenarios and config["run"].get("multiprocess", True):
            for scenario in [s for s in ("B", "C") if s in scenarios]:
                print(f"\n== scenario D ({scenario} via shard processes) == shards {grid}")
                baseline = None
                for shards in grid:
                    for repeat in range(1, repeats + 1):
                        digest = digest_dir / f"D{scenario}_{shards}_{repeat}.npy"
                        record, timeline = run_shards(
                            config_paths["base"], scenario, shards, digest, scratch
                        )
                        values = np.load(digest)
                        if shards == grid[0] and repeat == 1:
                            baseline = values
                            difference = 0.0
                        elif baseline is None or values.shape != baseline.shape:
                            difference = float("nan")
                            alerts.append(
                                f"scenario D/{scenario}, {shards} shard(s): shape differs"
                            )
                        else:
                            with np.errstate(invalid="ignore"):
                                difference = float(np.nanmax(np.abs(values - baseline)))
                            if difference != 0.0:
                                alerts.append(
                                    f"scenario D/{scenario}, {shards} shard(s): "
                                    f"max abs difference vs serial = {difference:.3e}"
                                )
                        for entry in timeline:
                            shard_timeline.append(
                                {
                                    "scenario": f"D{scenario}",
                                    "shards": shards,
                                    "repeat": repeat,
                                    **entry,
                                }
                            )
                        rows.append(
                            {
                                "scenario": scenario,
                                "mode": "multiprocess",
                                "workers": shards,
                                "repeat": repeat,
                                "t_total_s": record["t_total_s"],
                                "t_extraction_s": float("nan"),
                                "t_metrics_s": float("nan"),
                                "n_samples": len(batch) if scenario == "B" else 1,
                                "n_features": len(feature_sets["base"]),
                                "n_pixels": n_pixels_seen,
                                "B": config["bootstrap"]["n_iterations"] if scenario == "C" else 0,
                                "max_abs_diff_vs_serial": difference,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                        )
                        print(
                            f"  k={shards:<3} repeat {repeat}  {record['t_total_s']:7.2f}s  "
                            f"max|diff|={difference:g}"
                        )
    finally:
        for path in config_paths.values():
            path.unlink(missing_ok=True)
        if not args.keep:
            shutil.rmtree(digest_dir, ignore_errors=True)

    write_csv(out_dir / "results.csv", rows, FIELDS)
    summary = summarise(rows)
    write_csv(
        out_dir / "summary.csv",
        summary,
        [
            "scenario",
            "mode",
            "workers",
            "median_t_total_s",
            "speedup",
            "efficiency",
            "sequential_fraction",
            "amdahl_bound",
        ],
    )
    if shard_timeline:
        write_csv(
            out_dir / "shard_timeline.csv",
            shard_timeline,
            ["scenario", "shards", "repeat", "shard", "start_s", "end_s"],
        )
    (out_dir / "environment.json").write_text(
        json.dumps(
            environment_record(config, cores, Path(scratch or tempfile.gettempdir()), revision),
            indent=2,
        )
    )
    plot(summary, out_dir, int(config["output"]["dpi"]))
    write_table(summary, out_dir / "table.tex")

    print(f"\nWrote {out_dir}/results.csv, summary.csv, environment.json, speedup.pdf, table.tex")
    if alerts:
        print("\n!! DETERMINISM CHECK FAILED -- outputs are not identical across worker counts:")
        for alert in alerts:
            print(f"   {alert}")
        return 1
    print("Determinism check passed: every worker count reproduced the serial output exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
