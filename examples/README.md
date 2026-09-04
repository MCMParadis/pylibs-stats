# Examples

Every script here runs against the real data in `data/dataset1` (7 physical samples ×
3 ablation layers), not toy fixtures. Run them **from anywhere** — each resolves paths
relative to the repo root itself.

- `.py` files are pylibs DSL scripts: `pylibs run examples/<name>.py`
- `.sh` files are executable: `./examples/<name>.sh`
- `.sbatch` files are SLURM job scripts: `sbatch examples/<dir>/<name>.sbatch`
  (edit `--account` and the `module load` line first — both are placeholders)

## Start here

The two walkthroughs below exist in matching pairs: the `.py` drives everything through
DSL verbs, the `.sh` through `pylibs project ...` commands. Either one alone builds the
same project — pick whichever matches how you intend to script your own runs.

Each starts by registering **`peak_table.csv`** (below), which is not optional: there is no
bundled peak table, so a project can't resolve a peak id until one is copied into it.

| File | What it does |
|---|---|
| `dataset1_example.py` | The full pipeline in one DSL script: 21 samples, 16 features, all-metrics regression sweep, correlations, calibration curves, applied quantification maps, and every report. |
| `dataset1_example.sh` | The same thing through the CLI. Tunables (`PROJECT_NAME`, `N_PERMUTATIONS`, `N_PROCESSES`) are env-overridable, so a quick smoke run needs no edit. |
| `dataset1_bootstrap_example.py` | Bootstrap resampling in one DSL script: four configs (random/local × plain/shuffled) over every sample. |
| `dataset1_bootstrap_example.sh` | The same thing through the CLI, with `PROJECT_NAME`/`N_ITERATIONS`/`NPIX` env-overridable. `NPIX` must stay a perfect square — `local` draws a square block. |

## Splitting work across processes and cluster jobs

`dataset1_multiprocess/` — the same workflow as `dataset1_example.py`, but with the two
heavy stages run as separate parallelized steps, the way you'd actually run it on a
cluster: compute remotely, transfer `results.zarr`, report locally.

| File | What it does |
|---|---|
| `setup_project.py` | Run once: creates the project, registers the 21 samples and 16 features. Never run as part of an array job. |
| `run_pipeline_parallel.sh` | Stage 1 locally — feature and metrics computation for every sample, split across 3 separate OS processes via `run-pipeline-batch --worker-index/--n-workers`. |
| `compute_regressions_parallel.sh` | Stage 2 locally — the all-metrics regression sweep, sharded the same way. Needs stage 1 finished: every sample must have distribution results before any cell can be fit. |
| `report_locally.py` | Stage 3 — the light, sequential half: reads the already-computed `results.zarr` and renders every report. Transferring a project by hand means three files now — `results.zarr`, `registry.json` **and** `peak_table.csv` — or one snapshot, which carries all three. |
| `run_pipeline_slurm.sbatch` | Stage 1 as a real SLURM array job. Same mechanism, launched by SLURM instead of `&`/`wait`. |
| `compute_regressions_slurm.sbatch` | Stage 2 as a real SLURM array job. |
| `submit_slurm_chain.sh` | Submits both sbatch files as one dependency chain (`--dependency=afterok` on the array's own job id), so the sweep starts only after *every* pipeline task succeeds. Prefer this over submitting either sbatch directly. |

`dataset1_bootstrap_multiprocess/` — the same idea applied to bootstrap iterations
rather than samples or regression cells.

| File | What it does |
|---|---|
| `setup_project.py` | Run once: one sample, three features, and a bootstrap config with enough iterations that splitting it is worth demonstrating. |
| `run_bootstrap_parallel.sh` | One config's iterations split across 2 separate OS processes via `run-bootstrap --worker-index/--n-workers`. |
| `run_bootstrap_slurm.sbatch` | The same, as a real SLURM array job. |

## Utilities

| File | What it does |
|---|---|
| `pack_project_example.sh` | Consolidating `results.zarr` (a 100k-file tree on a real project) into one randomly-readable `.zarr.zip`: `pack` + `verify-pack` for a one-off handoff, then snapshots for an iterative multi-machine workflow. `use-snapshot` replaces your working `results.zarr`, so it is only printed unless you pass `DEMO_USE_SNAPSHOT=1`. |
| `dataset1_libs_file_import_example.py` | Reading a raw `.libs` file directly — scan/pixel counts, wavelength range, and streaming spectra in batches with `capdata`/`batchsize`. No project involved. |
| `peak_table.csv` | The 7 peaks (B, C, Ca) every example uses. Not a script: `USEPEAKTABLE` / `use-peak-table` **copies** it into the project, so the project is self-contained afterwards. There is no bundled table, so this step is required before any feature is registered. |

## Re-running

Everything is safe to re-run. `create`, `use-peak-table`, `add-sample` and `add-feature` reuse
what is already registered (re-registering an identical table is a no-op — only a table whose
contents differ counts as a replacement); `run-pipeline` skips any step whose output exists;
`run-bootstrap` resumes rather than recomputing completed iterations. The regression, correlation
and
report stages do recompute, being pure functions of what is already stored.
