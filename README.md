# PyLIBS

[![CI](https://github.com/MCMParadis/pylibs-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/MCMParadis/pylibs-stats/actions/workflows/ci.yml)

Statistical processing of LIBS (Laser-Induced Breakdown Spectroscopy) imaging data. It offers
a command line interface, a small DSL, and a Python API, all built on one shared façade
(`pylibs.core.api`).

Given a set of raw `.libs` or `.ELE` files, `pylibs` extracts named features, which are either
single peaks or ratio expressions such as `Fe438 / Li670`, and computes per-feature statistics
and spatial maps. Across a whole project it then correlates or regresses those features against
an external per-sample property such as a known concentration, and applies a fitted calibration
curve back onto any sample's raw intensities to produce a quantification map. Everything renders
to PNG and PDF reports, and every step is resumable and safe to re-run.

## Supported inputs

`pylibs` reads the open **`.libs`** format: a plain NumPy `.npz` archive holding a
`(n_scans, n_pixels)` spectra array, a wavelength axis, and a JSON metadata object.
It is uncompressed, it depends on no private code, and it reads in three lines of NumPy:

```python
import numpy as np, json

with np.load("sample.libs", allow_pickle=False) as archive:
    data, wavelengths = archive["data"], archive["wavelengths"]
    params = json.loads(str(archive["params"]))
```

The full specification is in [docs/libs_format.md](docs/libs_format.md).

**Any other format converts to it.** There is no proprietary step. If you can get your data
into a NumPy array, then `write_libs` turns it into a sample that the whole pipeline accepts.

```bash
# from a CSV spectral matrix: one row per pixel, x,y columns, one column per
# wavelength, the header giving each column's wavelength
pylibs write-libs converted.libs --csv matrix.csv

# from arrays, with an explicit raster shape and pixel pitch
pylibs write-libs converted.libs --spectra cube.npy --wavelengths axis.npy \
    --grid 41x21 --step 0.1
```

```python
from pylibs.core import api

api.write_libs("converted.libs", spectra, wavelengths, grid=(41, 21), step=0.1)
```

Spectra may be an array, a memory map, or an iterable of row blocks. Blocks stream straight
into the archive, so converting a scan larger than memory costs one block of RAM.

Worked conversions from **ENVI**, **FITS** and a **CSV matrix** are in
[`examples/conversion/`](examples/conversion/). The ENVI and FITS readers there use `spectral`
and `astropy`, which are optional packages used only by those examples and never by `pylibs`
itself.

The vendor **`.ELE`** format is also read directly, through the optional `ele` extra described
below. The command `pylibs convert-ele-to-libs` turns such a file into a `.libs` one that needs
no private code thereafter. Nothing in the documented workflow requires that extra.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # editable install + lint/type-check/test tools

pip install -e ".[dev,ele]"      # also pulls in the private .ELE reader, which needs SSH
                                  # access to a private repo. pylibs works fully without it
                                  # for .libs files.
```

Without access to that private repository, the `ele` extra fails during dependency resolution
with an SSH authentication error rather than a message about pylibs. Install `.[dev]` instead,
which supports the open `.libs` format fully.

Requires Python 3.11+.

## Quickstart

```bash
# create a project (default location: ./projects/<name>)
pylibs project create demo

# register a peak table. It is copied into the project, so everything below
# resolves peak ids against the project's own copy. Required once, before
# any feature is registered. There is no bundled default.
pylibs project use-peak-table demo examples/peak_table.csv

# register a raw file as a sample
pylibs project add-sample demo path/to/sample.libs --sample-name S1
pylibs project add-samples-from-dir demo path/to/dir   # names taken from the filenames

# register a feature, either a bare peak id or a ratio expression. It is
# validated against the project's registered peak table.
pylibs project add-feature demo "B249 / Ca393"
pylibs project add-element-features demo B C Ca     # every peak the table has for each

# run the pipeline: feature extraction, spectrum stats, 1D/2D distribution stats
pylibs project run-pipeline demo sample001

# render (A) a feature map, (B) diagnostic spectra/joint-density plot, and
# (C) a distribution histogram per feature, plus an optional PDF report
pylibs project generate-feature-report demo sample001 --save-pdf
```

Regressions treat the physical sample as the unit of inference by default. Acquisitions that
share a sample name, such as the three ablation layers of one pellet, are not independent
evidence about that pellet, so counting each as its own point would overstate the precision.
The fitted line, R² and MAE are pooled over every acquisition as usual, while the confidence
band and the permutation test use one point per sample name. Pass
`--no-group-by-sample-name` to disable it, and note that it does nothing when no name repeats.

From there, `pylibs project --help` lists the full command surface. That covers cross-sample
correlations and regressions against an external CSV of known properties, applying a fitted
regression back onto a sample to get a quantification map, and bootstrap resampling for
uncertainty estimates.

Add `--debug` anywhere, as in `pylibs --debug project run-pipeline ...`, for verbose console
output and a fuller `<project_root>/pylibs.log`.

## DSL scripts

The same operations are available as a small Python-embedded DSL, for scripting a whole
workflow at once. See **[`examples/`](examples/README.md)** for runnable walkthroughs against
real data, each described in one line.

`examples/dataset1_example.py` processes a real seven-sample dataset from end to end, covering
features, correlations, regressions, applied-regression quantification maps, and reports.
`examples/dataset1_example.sh` does the same thing through the command line instead.
`examples/dataset1_multiprocess/` and `examples/dataset1_bootstrap_multiprocess/` show that
workflow split across worker processes or SLURM array jobs, using the `--n-processes`,
`--worker-index` and `--n-workers` options on `run-pipeline-batch`, `compute-regressions` and
`run-bootstrap`. `examples/conversion/` converts ENVI, FITS and CSV spectral matrices
into `.libs`.

```python
CREATEPROJECT("demo")
USEPEAKTABLE("demo", "examples/peak_table.csv")  # copied in; required before ADDFEATURE
ADDSAMPLE("demo", "path/to/sample.libs", sample_name="S1")
ADDSAMPLESFROMDIR("demo", "path/to/dir")  # sample1_2.libs -> name "sample1", layer "2"
WRITELIBS("converted.libs", spectra, wavelengths, grid=(41, 21), step=0.1)  # from any source
ADDFEATURE("demo", "B249 / Ca393")
ADDELEMENTFEATURES("demo", ["B", "Ca"])  # or ("demo", "Ca", only_default=True)
RUNPIPELINE("demo", "sample001")
GENERATEFEATUREREPORT("demo", "sample001", save_pdf=True)
```

```bash
pylibs run my_script.py
```

## Packing & snapshots

`results.zarr` is a directory tree of many small chunk files. That is fine locally, but it is
awkward to hand off or transfer at scale. For a one-off handoff, `pack` consolidates it into a
single `.zarr.zip`, which stays randomly readable and needs no full unzip, together with a
`.manifest.json` recording the git commit, the project's own `registry.json`, and a checksum.

```bash
pylibs project pack demo
pylibs project verify-pack projects/demo/results.zarr.zip
```

For an iterative workflow, use timestamped snapshots instead. Such a workflow might run the
pipeline or bootstrap on one machine, report or add features on another, and reprocess back on
the first. Snapshots never overwrite each other: the timestamp carries milliseconds, and a name
that still collides gets a `-1` or `-2` suffix rather than replacing the archive already
there.

```bash
pylibs project pack-snapshot demo                       # projects/demo/snapshots/<UTC timestamp>.zarr.zip
pylibs project list-snapshots demo
pylibs project use-snapshot demo                        # extracts the most recent snapshot as the
                                                          # working results.zarr, along with
                                                          # registry.json and peak_table.csv.
                                                          # Use --timestamp to pick a specific one,
                                                          # or --force to overwrite a working copy.
```

Every one of these is also a DSL verb: `PACKPROJECT`, `VERIFYPACK`, `PACKSNAPSHOT`,
`LISTSNAPSHOTS` and `USESNAPSHOT`. A snapshot carries the project's `registry.json` **and** its
`peak_table.csv`, so a machine that received only the archive can still resolve peak ids.

Adding or editing a feature or a peak definition is detected and reprocessed in place
automatically the next time `run-pipeline` or `run-bootstrap` runs. This does not extend to the
raw sample data, which cannot be reprocessed in place. For a peak-table correction, re-register
the corrected table with `use-peak-table` first, because the project holds its own copy.

## Development

See [`CLAUDE.md`](CLAUDE.md) for the architecture, the conventions and the per-module notes,
[`docs/libs_format.md`](docs/libs_format.md) for the `.libs` specification, and
[`CHANGELOG.md`](CHANGELOG.md) for what changed between releases.
[`benchmarks/`](benchmarks/) holds a reproducible parallel-execution benchmark that runs on the
bundled data.

```bash
ruff check . && ruff format --check . && mypy src && pytest -q   # CI-equivalent local check
```

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
