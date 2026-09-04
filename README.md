# PyLIBS

[![CI](https://github.com/MCMParadis/pylibs-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/MCMParadis/pylibs-stats/actions/workflows/ci.yml)

Statistical processing of LIBS (Laser-Induced Breakdown Spectroscopy) imaging data —
a CLI, a small DSL, and a Python API, all built on one shared façade (`pylibs.core.api`).

Given a set of raw `.libs` (or `.ELE`) files, `pylibs` extracts named features (single peaks
or ratio expressions like `Fe438 / Li670`), computes per-feature statistics and spatial maps,
and — across a whole project — correlates or regresses those features against an external
per-sample property (e.g. a known concentration), then applies a fitted calibration curve back
onto any sample's raw intensities to produce a quantification map. Everything renders to
PNG/PDF reports; every step is resumable and safe to re-run.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # editable install + lint/type-check/test tools

pip install -e ".[dev,ele]"      # also pulls in the private .ELE reader (needs SSH access to
                                  # a private repo) -- pylibs works fully without it for .libs files
```

Without access to that private repository, the `ele` extra fails during dependency resolution
with an SSH authentication error rather than a message about pylibs — install `.[dev]` instead,
which supports the open `.libs` format fully.

Requires Python 3.11+.

## Quickstart

```bash
# create a project (default location: ./projects/<name>)
pylibs project create demo

# register a peak table -- copied into the project, so everything below
# resolves peak ids against the project's own copy. Required once, before
# any feature is registered; there is no bundled default.
pylibs project use-peak-table demo examples/peak_table.csv

# register a raw file as a sample
pylibs project add-sample demo path/to/sample.libs --sample-name S1
pylibs project add-samples-from-dir demo path/to/dir   # names taken from the filenames

# register a feature -- a bare peak id or a ratio expression, validated
# against the project's registered peak table
pylibs project add-feature demo "B249 / Ca393"
pylibs project add-element-features demo B C Ca     # every peak the table has for each

# run the pipeline: feature extraction, spectrum stats, 1D/2D distribution stats
pylibs project run-pipeline demo sample001

# render (A) a feature map, (B) diagnostic spectra/joint-density plot, and
# (C) a distribution histogram per feature, plus an optional PDF report
pylibs project generate-feature-report demo sample001 --save-pdf
```

From there, `pylibs project --help` lists the full command surface: cross-sample correlations
and regressions against an external CSV of known properties, applying a fitted regression back
onto a sample to get a quantification map, and bootstrap resampling for uncertainty estimates.

Add `--debug` anywhere (`pylibs --debug project run-pipeline ...`) for verbose console output
and a fuller `<project_root>/pylibs.log`.

## DSL scripts

The same operations are available as a small Python-embedded DSL for scripting a whole
workflow at once — see **[`examples/`](examples/README.md)** for runnable, real-data
walkthroughs, one line describing each. `examples/dataset1_example.py` processes a real
7-sample dataset end to end (features, correlations, regressions, applied-regression
quantification maps, and reports), with `examples/dataset1_example.sh` doing the same thing
through the CLI instead. `examples/dataset1_multiprocess/` and
`examples/dataset1_bootstrap_multiprocess/` show that workflow split across worker
processes/SLURM array jobs, via `--n-processes`/`--worker-index`/`--n-workers` on
`run-pipeline-batch`, `compute-regressions`, and `run-bootstrap`.

```python
CREATEPROJECT("demo")
USEPEAKTABLE("demo", "examples/peak_table.csv")  # copied in; required before ADDFEATURE
ADDSAMPLE("demo", "path/to/sample.libs", sample_name="S1")
ADDSAMPLESFROMDIR("demo", "path/to/dir")  # sample1_2.libs -> name "sample1", layer "2"
ADDFEATURE("demo", "B249 / Ca393")
ADDELEMENTFEATURES("demo", ["B", "Ca"])  # or ("demo", "Ca", only_default=True)
RUNPIPELINE("demo", "sample001")
GENERATEFEATUREREPORT("demo", "sample001", save_pdf=True)
```

```bash
pylibs run my_script.py
```

## Packing & snapshots

`results.zarr` is a directory tree of many small chunk files — fine locally, awkward to hand
off or transfer at scale. `pack` consolidates it into one `.zarr.zip` (still randomly
readable, no full unzip needed) plus a `.manifest.json` recording the git commit, the project's
own `registry.json`, and a checksum, for a one-off handoff:

```bash
pylibs project pack demo
pylibs project verify-pack projects/demo/results.zarr.zip
```

For an iterative workflow — run the pipeline/bootstrap on one machine, report or add features on
another, reprocess back on the first — use timestamped snapshots instead, which never overwrite
each other (the timestamp carries milliseconds, and a name that still collides gets a `-1`/`-2`
suffix rather than replacing the archive already there):

```bash
pylibs project pack-snapshot demo                       # projects/demo/snapshots/<UTC timestamp>.zarr.zip
pylibs project list-snapshots demo
pylibs project use-snapshot demo                        # extracts the most recent snapshot as the
                                                          # working results.zarr (+ registry.json
                                                          # and peak_table.csv);
                                                          # --timestamp to pick a specific one,
                                                          # --force to overwrite an existing working copy
```

Every one of these is also a DSL verb (`PACKPROJECT`, `VERIFYPACK`, `PACKSNAPSHOT`,
`LISTSNAPSHOTS`, `USESNAPSHOT`). A snapshot carries the project's `registry.json` **and** its
`peak_table.csv`, so a machine that received only the archive can still resolve peak ids. Adding
or editing a feature/peak definition (not the raw sample data) is detected and reprocessed in
place automatically the next time `run-pipeline`/`run-bootstrap` runs — for a peak-table
correction, re-register the corrected table with `use-peak-table` first, since the project holds
its own copy.

## Development

See [`CLAUDE.md`](CLAUDE.md) for the architecture, conventions, and per-module notes.

```bash
ruff check . && ruff format --check . && mypy src && pytest -q   # CI-equivalent local check
```

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
