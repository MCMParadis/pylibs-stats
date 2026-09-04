# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`pylibs`: statistical processing of LIBS (Laser-Induced Breakdown Spectroscopy) imaging data,
exposed as a CLI, a small DSL, and (eventually) other interfaces, all built on one façade module.

## Install / build / lint / test

```bash
pip install -e ".[dev]"          # editable install + dev tools
pip install -e ".[dev,ele]"      # also pulls in the private .ELE reader (needs SSH access)

ruff check .                     # lint
ruff format .                    # format (format --check . to verify without writing)
mypy src                         # type check

pytest -q                        # full test suite
pytest -q tests/unit/test_expression.py                    # one file
pytest -q tests/unit/test_expression.py::test_evaluate_ratio   # one test
```

CI-equivalent local check before considering work done: `ruff check . && ruff format --check . && mypy src && pytest -q`.

## Architecture

**One rule drives the whole layout: every interface (CLI, DSL, future GUI/DB) calls through
`pylibs.core.api` — nothing outside `core/` ever imports `core.pipeline`, `core.reporting`, or
`core.io` internals directly.** When adding a capability, implement it in `core/`, expose it as a
function on `core.api`, then add a thin CLI command and DSL verb that just call that function.

```
src/pylibs/
  core/
    api.py              # the façade -- the only module CLI/DSL/etc. are allowed to import from
    config.py           # pydantic models: ProjectConfig, SampleConfig, FeatureConfig
    atomic.py           # atomic_write_bytes()/atomic_write_text(): temp file + os.replace, for
                        # the project files one process writes while others read. write_text()
                        # truncates first, and registry.json is read UNLOCKED by Project.load --
                        # once per sample inside run_pipeline -- while sharded siblings write
                        # it; a reader in that window got a JSONDecodeError. Used by
                        # project/registry.py, features/peak_table.py and io/snapshot.py
    exceptions.py       # PylibsError and subclasses (ProjectError, ResultsError, ReaderError,
                        # CorrelationError, RegressionError, ...)
    logging.py          # get_logger(name) -- library code never calls basicConfig;
                        # set_debug_logging(enabled) toggles DEBUG level process-wide (set once,
                        # via core.api.set_debug_logging, by the CLI's --debug flag before any
                        # command runs); configure_console_logging(debug)/configure_project_
                        # logging(root) are the only places handlers get attached -- the latter,
                        # called by Project.create/load, additionally points a per-project file
                        # handler (<root>/pylibs.log) at the "pylibs" logger, switching file
                        # target when a different project is loaded in the same process
    project/            # Registry (registry.json on disk) + Project wrapper. Every mutator
                        # goes through Registry._locked_update(): flock, re-read from disk,
                        # mutate that, save, adopt -- because save() rewrites the WHOLE file,
                        # so mutating a snapshot taken at load time drops any concurrent
                        # sibling's update and _next_id off a stale snapshot hands two callers
                        # the same id. _locked_registry_dir is NOT reentrant (flock is keyed to
                        # the open file description), so a locking mutator must never be called
                        # while already holding it -- see get_or_create. save() itself writes
                        # through core.atomic, which is what makes the unlocked Project.load
                        # reads safe; the lock never runs on the read path
      sample_properties.py  # loads an external per-sample scalar-property CSV (stdlib csv,
                             # no pandas), joined on a sample_name column
      filename_scheme.py     # compiles a "{sample_name}_{layer}"-style template into an
                             # anchored regex, so add_samples_from_dir can read a sample's
                             # name off its filename; fields are greedy left-to-right, so
                             # the default scheme splits at the LAST underscore
    runs/                # generic resumable/checkpointed execution -- knows nothing about
                        # bootstrap, features, or LIBS data; bootstrap is its first consumer
      checkpoint.py       # results + completed zarr arrays, both chunk-size-1 along the
                          # iteration axis; mark_done() writes results before flipping
                          # completed -- a crash between the two leaves the slot pending, not
                          # falsely done; get_or_create_checkpoint_arrays()/extend
      partition.py         # plan_shard(): round-robin split of a run's iterations across
                            # separately-invoked processes -- MUST shard the static full
                            # range, never a dynamically-shrinking pending_indices() result
                            # (see the module docstring for the exact bug that causes)
      executor.py           # execute(): sequential or ProcessPoolExecutor (spawn context);
                            # on_result always runs in the calling process, so subprocess
                            # workers never touch the results store themselves;
                            # resolve_n_processes(n) -- n<=0 resolves to every CPU available
                            # to this process, shared by every n_processes-taking entry point
    io/
      reader.py          # LibsReader Protocol + get_reader() dispatch by file extension
      readers/           # libs_reader.py (.libs, built-in), ele_reader.py (.ELE, needs libs_importer)
      convert.py          # convert_ele_to_libs(): reads a whole .ELE file and writes it out as a
                          # self-contained .libs archive (no project/registry involved)
      store.py           # ResultsStore: zarr v3 results.zarr, one group per sample_id; bulky
                          # array-valued raw params (wavelengths/detectors/mask) each get their
                          # own array, since set_raw_params keeps only scalar entries.
                          # _require_group_concurrent() wraps zarr's require_group, which is
                          # check-then-create: sharded processes racing for the SAME group
                          # (bootstrap's <sample>/bootstrap/<id>, the regression sweep's
                          # regressions/<column>/<metric>) can both miss, and the loser's
                          # create raises ContainsGroupError. Single catch-and-reread, matching
                          # ResultsStore.__init__'s own guard for the root group and
                          # core.runs.checkpoint's for arrays -- the group layer between them
                          # was the gap, and cost ~3 spurious CI failures in every 25 runs
      pack.py             # pack_results()/verify_pack(): consolidate a results.zarr directory
                          # tree into one .zarr.zip (plain zipfile, still randomly readable via
                          # zarr's own ZipStore) plus a .manifest.json (git sha/dirty, the
                          # project's own registry.json AND peak_table.csv, sizes, checksum);
                          # list_manifests() globs a directory tree for these. The table is
                          # carried as raw text read/written as bytes -- read_text() would
                          # strip the \r from a CRLF table (the real ones are CRLF)
      snapshot.py          # pack_snapshot_results()/use_snapshot_results(): a timestamped,
                            # never-overwritten sibling to pack.py's single-canonical archive
                            # (ms-resolution timestamp; a colliding name is
                            # suffixed -1/-2 via pack_results(exclusive=True),
                            # never truncated),
                            # for a multi-machine workflow -- pack on one machine, extract
                            # (use_snapshot_results) on another to keep working from exactly
                            # that state -- restoring registry.json and peak_table.csv from
                            # the manifest too, so a machine that received only the archive
                            # can load the project and resolve peak ids. Refuses to extract
                            # over an existing results.zarr, or over either of those two files
                            # if it holds different content, unless force=True (this repo's
                            # first directory-tree deletion)
    features/
      peak_table.py       # element/id/peak/continuum CSV lookup. There is NO bundled table:
                           # a project registers its own via api.use_peak_table, which COPIES it
                           # to <project_root>/peak_table.csv (PEAK_TABLE_FILENAME); presence of
                           # that file is the registration, so nothing records a path in
                           # registry.json that could disagree with it. install_peak_table
                           # validates before copying and writes via os.replace (atomic, so a
                           # sharded run never reads half a CSV) and no-ops when the source *is*
                           # the project's own copy; load_project_peak_table is the single place
                           # "no peak table registered" is raised, for api/runner/workers alike.
                           # Loading is strict -- a malformed row or an empty table raises
                           # PeakTableError, not a bare KeyError -- since a user-supplied file is
                           # on this path. peaks_for_element() selects a whole element's peaks in
                           # the table's own row order, optionally narrowed to its one
                           # `_isdefault`-flagged peak -- which some elements lack entirely
      expression.py        # parses "Fe438 / Li670"-style expressions into per-scan values;
                            # split_ratio() finds the outermost division (respecting parens);
                            # referenced_peak_ids() lists every peak-id-shaped token, known or not
      identity.py           # feature_identity_hash(expression, peak_table): a feature's true
                            # identity is its expression AND every referenced peak id's resolved
                            # element/peak/continuum/tol_peak/tol_continuum -- the peak table can
                            # change (a corrected wavelength) without the expression or feature_id
                            # changing at all. Stamped on the features array and propagated to
                            # distribution_1d/2d/bootstrap, so is_done() can detect "this feature
                            # means something different now" and reprocess it in place
      extraction.py        # single-peak intensity extraction from a raw spectrum batch
    pipeline/
      step.py             # Step Protocol (name/produces/requires/needs_reader + is_done/run);
                           # is_done() takes the registry too, not just the store, so it can
                           # detect a feature/peak-definition change (feature_identity_hash), not
                           # just "does output exist" -- raw sample data still can't be
                           # reprocessed in place, but this narrower case now is
      runner.py           # run_pipeline(): streams a sample through every Step, in a fixed order
                           # that already satisfies `requires` -- no topological sort;
                           # run_pipeline_batch(): many samples in one call, sharded via
                           # core.runs.partition/executor (worker_index/n_workers across
                           # separately-invoked processes, n_processes as an in-process pool)
      metrics.py           # every metric's actual math, 1D and 2D, and only there -- see its own
                            # module docstring for "how to add a metric". METRIC_NAMES_1D/_2D name
                            # vocabulary + feature_metric_values() (merges a Distribution1D's and
                            # Distribution2D's own `metrics` dicts) also live here
      distributions.py      # Distribution1D/Distribution2D (siblings, not parent/child) + their
                            # compute_distribution_1d/2d orchestrators, which build a histogram
                            # and loop metrics.py's registries to fill `metrics: dict[str, float]`
                            # (`@property` wrappers like `.gini`/`.mean` read straight out of it).
                            # Both 1D (histogram/color scale) and 2D (each side's own marginal,
                            # for panel B's joint-density plot) window via skew_adjusted_bounds()/
                            # quantile_skewness() -- a boxplot-whisker fence (Q1/Q3 +/- 1.5*IQR)
                            # with an exponential term that widens/narrows each side based on the
                            # data's own quantile skewness, then narrowed to the real data min/max
                            # within the fence (matplotlib boxplot-whisker convention); empty
                            # input returns (0.0, 0.0) rather than raising. 2D's display_counts
                            # (panel B) keeps only scans within BOTH sides' own fence -- dropped,
                            # not clipped, so outliers can't pile into the edge bins
      correlations.py       # cross-sample only (not per-sample, unlike everything else here), and
                            # pure formatting -- never fits anything itself. Reads each (feature,
                            # metric) cell's already-stored regression (see regression.py) and
                            # reshapes them into a (feature x metric) matrix; a cell with no stored
                            # regression is left nan and logged, never recomputed
      regression.py          # also cross-sample, and the *only* place fit_least_squares (or any
                             # future algorithm) is ever called. Fits either one selected metric
                             # (default "Mean") or, with metric=None, every metric in
                             # metrics.METRIC_NAMES in one "all-metrics sweep" -- the latter is
                             # what gives correlations.py something to read for every cell. Via a
                             # registered algorithm (REGRESSION_ALGORITHMS, currently just
                             # least_squares); stores everything needed to redraw the fit and its
                             # confidence band; REGRESSION_APPLICATORS is the inverse-direction
                             # registry, applying a stored fit back to one sample's own raw
                             # per-pixel feature values
      regression_run.py      # parallel dispatch for the sweep above: one (column, metric,
                             # feature) fit is one independent, CPU-heavy unit, the same way one
                             # bootstrap iteration is for bootstrap/ below. Each cell writes its
                             # own disjoint zarr group, so unlike bootstrap no checkpoint arrays
                             # are needed, just core.runs.partition (worker_index/n_workers) +
                             # core.runs.executor (n_processes), reused directly; regression.py
                             # itself stays untouched, pure algorithm only
      steps/
        features.py         # FeatureStep: evaluates every registered expression -> features array;
                             # for ratio features also computes+stores +1-shifted numerator/
                             # denominator (ratio_components/) -- the one place either is ever
                             # evaluated from the raw spectrum, so Distribution2DStep/bootstrap just
                             # read it back instead of re-streaming/re-evaluating themselves
        spectrum_stats.py   # SpectrumStatsStep: sample-wide min/max/mean spectrum
        distribution_1d.py  # Distribution1DStep: per-feature 1D metrics (needs features);
                             # n_processes parallelizes across a sample's own features
        distribution_2d.py  # Distribution2DStep: per-ratio-feature 2D metrics (reads
                             # ratio_components from the store, needs features -- no reader
                             # stream); n_processes parallelizes across ratio features likewise
      bootstrap/            # NOT a Step -- many resumable iterations per sample, not one-shot
        resampling.py        # RESAMPLING_STRATEGIES: "random" (with replacement) / "local"
                             # (contiguous square raster block, restricted to blocks lying
                             # entirely inside the scanned region when the sample has a mask);
                             # a third method is one function + one registry entry
        compute.py            # init_worker()/compute_iteration(): resample a sample's
                              # features (+ raw numerator/denominator for ratio features) at
                              # deterministic positions, compute the same METRIC_NAMES
                              # vocabulary as metrics.py's feature_metric_values
        runner.py             # run_bootstrap(): resolves resolved_npix/raster dims, gets-or-
                              # creates checkpoint arrays via ResultsStore, shards+executes
                              # pending iterations via core.runs
        csv_export.py          # export_bootstrap_csv(): one flat CSV per (sample_id,
                               # bootstrap_id) -- a plain-file counterpart to reporting/'s
                               # PNGs/PDFs, written to <project_root>/exports/
    reporting/
      figures.py            # matplotlib figure builders (feature map, spectra, distribution,
                             # joint-density panel, combined (A)(B)(C) diagnostics figure)
      report.py             # generate_feature_maps(): one diagnostics PNG per feature
      pdf.py                # build_pdf_report(): cover + params + one page per feature;
                            # also cover_page()/add_page_number(), shared with correlations_report.py
      correlations_report.py  # build_correlations_report(): color-scaled, best/worst-ranked
                               # Pearson r / R² / p-value tables, one PDF per external CSV column
      regression_report.py    # build_regression_report(): one calibration-curve page per
                              # regressed feature (error-barred points, dashed fit line, 95% CI
                              # band), one PDF per external CSV column
      applied_regression_report.py  # build_applied_regression_report(): one page per applied
                                    # (column, feature) pair -- reuses figures._draw_feature_diagnostics
                                    # directly (map + 1D distribution, no diagnostic-spectra/joint-
                                    # density panel), one PDF per sample
      raster.py             # per-scan values -> (n_rows, n_cols) grid, undoing the serpentine
                             # scan order; with a sample's scanned-cell mask (a partially-scanned,
                             # e.g. round sample) it places each scan at its own cell in that same
                             # order and leaves the rest NaN, instead of reshaping;
                             # resolve_physical_extent()
                             # derives (width_img, height_img) in mm from a raw file's params --
                             # direct WidthIMG/HeightIMG, else Resolution (mm/pixel) * pixel count,
                             # else (None, None) -- shared by report.py and applied_regression_report.py
      data/logo.png         # the PyLIBS wordmark drawn on every report's cover page by
                             # pdf.cover_page(); bundled into the wheel by hatchling's
                             # default "everything under src/pylibs" rule, so it needs no
                             # package-data stanza; read via importlib.resources
                             # (pdf._logo_path)
  interfaces/
    cli/app.py            # Typer app; every command is a thin wrapper over core.api; a root
                          # @app.callback() (`main`) handles the global --debug flag, firing
                          # before any subcommand dispatches -- including nested `project`
                          # subcommands -- so it's the one place a process-wide flag can live
    dsl/
      verbs.py             # UPPERCASE verb functions (CREATEPROJECT, ADDFEATURE, ...), thin
                            # wrappers over core.api, collected in VERBS
      runner.py             # run_script(): exec()s a script with VERBS pre-injected as globals
```

### Data model

A **project** (`core/project`) is a directory with a `registry.json` (samples + registered feature
expressions), a `peak_table.csv` (the project's own copy of a peak table, put there by
`use_peak_table`/`USEPEAKTABLE` -- see below), a `results.zarr` (per-sample computed results,
written by the pipeline), and a `pylibs.log` (every log message emitted while this project was
the active one, at INFO level by default or DEBUG with the CLI's `--debug` flag -- see
`configure_project_logging`). A **sample**
is one raw LIBS file (`.libs` or `.ELE`), registered either individually (`add_sample`) or a whole
directory at a time (`add_samples_from_dir`, which derives each file's `sample_name` from its name
via a `filename_scheme` template, keeps the full stem as `sample_stem`, and stores the template's
other fields as `name_parts`; a file the template can't parse is logged and skipped); a **feature**
is a named expression over peak ids (e.g. `Fe438`, `Fe438 / Li670`, or a nested ratio like
`(Fe438/Li670)/(Fe438/Li670+P214/Li670)`). Every peak id in an expression resolves against the
**project's own** peak table, which must be registered first: `use_peak_table`/`USEPEAKTABLE`/
`pylibs project use-peak-table` copies a CSV to `<project_root>/peak_table.csv`, and
`add_feature`/`add_element_features`/`run_pipeline`/`run_pipeline_batch`/
`generate_feature_report` all read that copy -- there is no per-command `--peak-table` flag and no
bundled fallback, so calling it out of order raises rather than silently resolving against
something else. Copying (rather than storing a path) is what makes a project self-contained:
`pack_snapshot`/`use_snapshot` carry the table to another machine, and editing the source
afterwards can't retroactively change what an existing project means -- the flip side being that
correcting a peak means re-registering (or editing the project's copy in place; the file is the
truth either way). `run_bootstrap` needs no table at all: it reads feature identity hashes off
`FeatureStep`'s stored output. Features are registered either one expression at a time
(`add_feature`) or a whole element at a time (`add_element_features`: every peak the peak table
lists for `"Fe"`, or just its flagged default, each becoming its own bare-peak feature -- an
element that contributes nothing is skipped, following `validate_expression`'s log-don't-raise
precedent -- but at INFO, i.e. to `pylibs.log` and not the console, since passing a standing list
of elements and letting the ones the table has no lines for fall away is ordinary use, not a typo
worth interrupting a run over; the summary line names any that were skipped).

`results.zarr` is a directory tree with one small file per chunk -- fine locally, bad for quota/
transfer at scale (a real project can have 100k+ files; see `core/io/pack.py`'s module docstring).
`pack_project`/`PACKPROJECT` (CLI `pylibs project pack` -- the one command whose CLI name
doesn't mirror its api/DSL one, unlike `pack_snapshot`/`PACKSNAPSHOT`/`pack-snapshot`)
consolidates it into one `.zarr.zip`, still randomly readable via
zarr's own `ZipStore`, for a one-off handoff; `pack_snapshot`/`PACKSNAPSHOT` and `use_snapshot`/
`USESNAPSHOT` (`core/io/snapshot.py`) -- whose manifest embeds the project's `peak_table.csv`
alongside its `registry.json`, both restored on extraction (and both covered by the same `force`
conflict check), so a machine that received only the archive can still resolve peak ids -- are the
iterative-workflow counterpart -- process on one
machine, pack a timestamped snapshot, `use_snapshot` it on another to keep reporting/reprocessing
from exactly that state, refusing to overwrite an existing `results.zarr` unless `force=True`.

A sample needn't fill its rectangular frame: a **round** one records fewer scans than the frame has
cells, and its raw file says which cells were scanned via a `PixelAssignmentMatrix` (0/1, one 1 per
scan -- `.ELE` files carry it; a `.libs` file may too, and a converted one keeps it). `SpectrumStatsStep`
caches it as the sample's `mask` array -- only if its shape and set-cell count can actually place
that sample's scans, else it's logged and dropped -- and reports place each scan at its own cell,
leaving unscanned cells NaN (drawn white). Without a mask, the scan is assumed to fill the frame and
is simply reshaped, which is the `.libs` norm and unchanged. A project processed before masks were
stored must be re-run to gain one.

`run_pipeline` streams a sample's reader once per `Step` that isn't already done, and writes
results to `results.zarr`. Every division -- a plain ratio, or one wrapped in `log()`/`exp()` --
adds 1 to both sides first (`features.expression`'s `_eval`), so a genuinely zero-intensity peak
never divides by zero or feeds `log(0)`; a **ratio feature** (outermost operation is division, see
`split_ratio`) additionally gets 2D joint-distribution stats between its (+1-shifted) numerator and
denominator -- computed once by `FeatureStep` alongside the feature's own value and stored in
`ratio_components/`, never re-evaluated from the raw spectrum by `Distribution2DStep` or bootstrap;
a **bare-peak feature** (expression is exactly one peak id) gets a diagnostic-spectra panel instead.
Reprocessing a sample's *raw data* in place isn't supported -- start a new project (new
`results.zarr`) if the `.libs`/`.ELE` file itself changes. A feature/peak *definition* change is
different: `is_done()` at every layer (`FeatureStep`, `Distribution1DStep`/`Distribution2DStep`,
`run_bootstrap`) compares `features.identity.feature_identity_hash(expression, peak_table)` -- not
just the feature_id/expression string, but every referenced peak id's resolved element/peak/
continuum/tol_peak/tol_continuum too -- against what's already stored, and reprocesses just what
changed automatically the next time `RUNPIPELINE`/`RUNBOOTSTRAP` runs (editing the peak table's
entry for a peak id, with the expression and feature_id unchanged, is exactly the case this exists
for). `FeatureStep` recomputes its whole array in one pass on any mismatch (the raw file has to be
re-streamed regardless of how many features changed); `Distribution1DStep`/`Distribution2DStep`
recompute only the specific feature(s) whose hash changed, since that's cheap and already
per-feature. A bootstrap run has no per-feature granularity -- its `completed` checkpoint flag
(`core.runs.checkpoint`) is atomically all-features-done-or-not per iteration -- so `run_bootstrap`
resets (`ResultsStore.reset_bootstrap_run`) the *whole* run for that (sample, bootstrap_id) on any
feature-hash mismatch rather than leave stale values silently marked done. Note the direction of
that last one: bootstrap compares against the hashes on `FeatureStep`'s *stored* output, so a peak
definition change only reaches a bootstrap run once `RUNPIPELINE` has refreshed that output --
re-register, re-run the pipeline, then re-run bootstrap.

`use_peak_table` itself invalidates nothing: it replaces the project's copy and returns. Every
recomputation above stays lazy, detected from stored hashes on the next run, which is why editing
`<project_root>/peak_table.csv` in place works identically to re-registering an edited source.
That matches `compute_regressions`' own precedent of not auto-invalidating downstream consumers.

### Report layout

`generate_feature_report` produces one combined diagnostics image per feature: (A) a feature map, (B)
diagnostic spectra (bare-peak) or a joint-density plot (ratio feature) or nothing (composite,
non-ratio expression), and (C) a distribution histogram -- plus an optional multi-page PDF with a
cover page, a parameters table, and one page + metrics table per feature.

The cover page's logo (`reporting/data/logo.png`) is placed in a fixed `add_axes((0.25, 0.6,
0.5, 0.15))` on an A4 figure -- 4.14 x 1.75in, i.e. 1240px wide at the 300 dpi every page is
saved at. `imshow` letterboxes rather than stretches, so a replacement must keep the artwork's
2.402 aspect ratio to fill that box, and be at least 1240px wide to avoid being upscaled into
it; anything wider is just downsampled back to 1240 and only grows the source file. Nothing in
`pdf.py` reads the image's size, so a swap needs no code change.

Panel (A)'s map axes read `x / mm`/`y / mm` when the raw file's params carry a physical extent --
direct `WidthIMG`/`HeightIMG`, or derived as `Resolution` (mm per pixel) times the pixel count when
only `Resolution` is present (`raster.resolve_physical_extent`, shared with the applied-regression
report below) -- falling back to `x / px`/`y / px` when neither is available. Every report's
cover-page title and per-page caption uses the sample's display **stem** (e.g. `"sample1_1"`, from
`SampleConfig.sample_stem`), never the internal `sample_id` (e.g. `"sample010"`) or the
correlations join key `sample_name` -- consistent across `generate_feature_report` and
`generate_applied_regression_report`.

Every map/distribution panel colors out-of-window pixels/bars via a `clamp_color` param
(`figures._draw_feature_map`/`_draw_distribution`/`_draw_feature_diagnostics`/
`plot_feature_diagnostics`, all threaded through): `None` (the plain feature report's default)
clamps to the colormap's own boundary color; an explicit color (e.g. `clamp_color="black"`, used by
`generate_applied_regression_report`) flags out-of-range pixels distinctly instead -- there,
"out of range" means the model is extrapolating past its own calibration limits (`x_min`/`x_max`),
a stronger signal than the plain report's display-only windowing. The colorbar's `extend` wedge
("neither"/"min"/"max"/"both") is auto-detected from the true full-data min/max vs. the resolved
`vmin`/`vmax`, not passed explicitly. Panel (C)'s legend includes a genuine blue-to-red gradient
swatch for the histogram itself (`figures._HandlerColormapSwatch`, a custom matplotlib legend
handler), not a flat-color patch. Every saved PNG/PDF page is 300 dpi. A standalone feature/
applied-regression PNG uses the exact same `figsize`/`bottom` as its own PDF page counterpart, so
the two are pixel-for-pixel the same shape -- there is no separate "standalone" image size.

Panel (B)'s joint-density plot (`figures._draw_joint_distribution`, ratio features only) forces its
imshow square by hand rather than via matplotlib's own aspect machinery, sized to the largest square
that fits both the row's own height and the width available before panel (C)'s colorbar -- its own
colorbar (`cbar_x0`, passed in by `_draw_feature_diagnostics` via `_colorbar_x0`) is placed by
solving for the imshow's position that lands the colorbar there exactly, so it lines up with panel
(C)'s colorbar regardless of which of those two bounds actually binds. The top/right marginal 1D
histograms (`_marginal_bar_plot`) are forced to the same physical thickness (`MARGINAL_RATIO` of the
imshow's own side) and the same physical gap from the imshow (`marginal_gap_in`) on both sides, not
each independently sized by the (non-square) GridSpec cell they started in; their bars set
`edgecolor` to their own `facecolor` and `antialiased=False` so many thin, densely-packed bars read
as solid color rather than faded/seamed. Both axes' scientific-notation offset text (e.g. `"1e3"`)
is drawn as a plain replacement `Text` pinned to the axis's own end (not matplotlib's default
placement, and not the last *visible* tick label, which can sit well short of the axis's true end).

Panel (B)'s diagnostic-spectra plot (`_draw_diagnostic_spectra`, bare-peak features only) scales its
y-axis off the tallest of the three spectra within `peak_wavelength +/- peak_scale_window` (default
5nm) -- not the tallest point anywhere across the wider *display* window (`window`, default 7nm/21nm
either side) -- so an unrelated, taller nearby line can't dictate the scale (and now may get clipped
at the top of the plot instead); its legend has a solid white background (`framealpha=1.0`) so a
clipped line can't visually pass through it.

### Regression (calibration curves)

`compute_regressions` is the one project-wide (not per-sample) analysis that ever fits anything:
given a CSV of external scalar properties (one row per sample, joined on a `sample_name` column),
it fits an actual `y = slope*x + intercept` model (X = the external CSV column, Y = a registered
feature's chosen metric -- one of 12 always-available 1D metrics (mean, median, gini, 1D Shannon
entropy, ...) plus 5 more for ratio features (2D gini, Pearson-CR, KL divergences, ...), see
`pipeline.metrics.METRIC_NAMES`) against each numeric CSV column. `metric` (default `"Mean"`) picks
one; `metric=None` instead fits **every** metric for every selected feature in one call -- an
"all-metrics sweep", the only thing that gives `compute_correlations` (below) a fully populated
matrix to read back. Sample-gathering: a sample present in the CSV but unprocessed is skipped;
samples sharing a `sample_name` (e.g. ablation layers) each contribute their own data point against
that row's broadcast value. `method` must be `"univariate"` (`"multivariate"` is reserved, raises
`RegressionError`); `algorithm` is a registry (`REGRESSION_ALGORITHMS`, currently only
`"least_squares"`, mirroring bootstrap's `RESAMPLING_STRATEGIES` pattern); `select_features` takes
feature *expressions* (resolved to `feature_id`s, raising on an unresolvable one) and defaults to
every registered feature. A single (feature, metric) fit that turns out degenerate (too few finite
pairs, or constant x/y) is skipped and recorded in `skipped_fits`, not fatal. Most skips in an
all-metrics sweep aren't degenerate at all but unfittable by construction -- every
`METRIC_NAMES_2D` metric of every non-ratio feature -- so each is classified by whether its
metric holds *any* finite value for that feature across *any* sample: none at all means the
metric simply isn't computed for it ("metric not computed for this feature"), otherwise it's a
real "insufficient data". Every skip is logged at INFO with a per-reason breakdown, following
`compute_correlations`' own missing-cell precedent; only the unexplained ones
(`unexpected_skipped_fits`) reach the CLI's console, capped, with the line naming the log file
holding the rest -- dataset1's 16 features give 126 skips, 124 of them structural.

Each (CSV column, metric, feature) triple stores its fitted `slope`/`intercept`/`pearson_r`/
`r_squared`/`mae`/`p_value` plus everything needed to redraw the 95% confidence band later purely
from stored scalars (`n`/`x_mean`/`ssxx`/`residual_std`, via `confidence_band_half_width`) -- no
per-scan data re-read -- and a per-physical-`sample_name` grouped summary (mean_x, mean_y, std_y)
used only for plotting, not for the fit itself, in `results.zarr`'s top-level
`regressions/<column>/<metric>/<feature_id>/` group -- `metric` is an explicit path segment (not
just an attribute) specifically so the *same* feature can hold a regression under every metric
simultaneously. `p_value` is a permutation test on the fit's own residual `mae` (shuffle the
pairing, refit, recompute MAE; fraction of shuffled fits with a lower MAE than the real one, over
`n_permutations`, default 10000, deterministic via `random_seed`, default 0), not
`scipy.stats.linregress`'s analytic p-value. Pruning is scoped to one (column, metric) subtree: a
feature stored under a metric being fit this call that's no longer in a narrower `select_features`
is removed; every other metric's own features are untouched -- so regressing different metrics for
the same column (e.g. one call's `"Mean"`, a later call's `"Median"`, or one `metric=None` sweep)
accumulates instead of overwriting each other. `generate_regression_report` renders one
calibration-curve page per (feature, metric) pair -- every stored metric's, or just one
`metric_name`'s if given (worth narrowing after an all-metrics sweep, which otherwise stores, and
would report, every feature under every metric at once) -- titled `"<feature> -- <metric>"`: a
half-page-width, square (50:50) axes with an error-barred point per physical sample, a dashed
fitted line, and a borderless light-red confidence-interval band, annotated with R², Pearson r,
MAE, permutation p-value, and the fit equation in the line's own legend label.

### Cross-sample correlations

`compute_correlations` is pure formatting over `compute_regressions`'s already-stored output --
**it never fits anything itself**. For each (feature, metric) cell of every numeric column in a
sample-properties CSV, it looks up that cell's already-stored regression and reshapes
`pearson_r`/`r_squared`/`mae`/`p_value` into a (feature x metric) matrix, stored in `results.zarr`'s
top-level `correlations/` group (not under a `sample_id`, since it spans every sample); a cell with
no stored regression is left `nan` and logged (one summary line: how many cells were missing, out
of the total), never recomputed -- run `compute_regressions(..., metric=None)` first to populate
every cell. `generate_correlations_report` renders those matrices into a PDF: nine pages (Pearson r,
R², MAE, MAE p-value, each best/worst, plus a cover page), every table ranking its rows by each
feature's own most-extreme value among its metrics rather than any single fixed metric. MAE has no
fixed, metric-independent range like the other three, so its color scale is instead the
interquartile range `[Q1, Q3]` of every finite MAE value in the whole table -- exactly half the
cells get a real gradient, a quarter clip to "best", a quarter to "worst", robust to one metric
(e.g. "c-KL divergence (vs. normal)") having wildly larger raw units than the rest.

### Applying a fitted regression (quantification maps)

`apply_regressions(project_name, sample_id, ...)` is the actual point of a calibration curve:
inverting a stored regression to turn a sample's raw per-pixel feature intensity into a predicted
per-pixel value of the external property (e.g. turn a boron-intensity map into a predicted-
concentration map), for *any* processed sample -- not just the ones used to fit the regression.
Inversion is algorithm-keyed (`REGRESSION_APPLICATORS`, mirroring `REGRESSION_ALGORITHMS` and
dispatched by each regression's own stored `algorithm`) since a future non-linear algorithm needs
its own inverse, not just `(y - intercept) / slope`. Applies every currently-stored (column, metric,
feature) regression by default -- e.g. every metric from an all-metrics sweep, not just one --
(skipping, not crashing, any feature whose raw values aren't in this particular sample, e.g.
registered after it was processed); `columns`/`select_features` narrow it, mirroring
`compute_regressions`'s own params. Stored the same way, under `<sample_id>/applied_regressions/
<column>/<metric>/<feature_id>/` (`metric` an explicit path segment, same reason as `regressions/`
above).

Stores both the predicted per-scan array and its own freshly-computed 1D distribution stats
(`fit_gaussian=False`, same rationale as bootstrap) in `results.zarr`'s `<sample_id>/
applied_regressions/<column>/<feature_id>/` group -- sample-scoped, unlike top-level `regressions/`.
Re-running `compute_regressions` does **not** auto-invalidate a stale applied prediction;
`apply_regressions` must be explicitly re-run to refresh it, matching `distribution_1d`/
`distribution_2d`'s own precedent of not auto-invalidating downstream consumers. `generate_applied_regression_report`
renders one page per applied (column, feature) pair -- a spatially-resolved map (A) and a 1D
distribution (B) of the predicted property, reusing `figures.plot_feature_diagnostics` (with its
`value_label` param overriding the default "Normalized intensity" axis/colorbar text so it correctly
reads e.g. "Predicted c_B / wt%"), no diagnostic-spectra/joint-density panel since this is a derived
scalar, not spectral data. Passes `show_expected_gaussian=False` -- the mean/std-implied curve isn't
a meaningful reference for a derived, predicted-property distribution (the "Fitted" curve is already
absent too, since the underlying stats use `fit_gaussian=False`, as above). Map axes use the same
`x / mm`/`y / mm` (or `x / px`/`y / px` fallback) convention as `generate_feature_report`.

### Bootstrap resampling

`BootstrapConfig` (method, npix-or-pct, n_iterations, base_seed, shuffle) is registered once per
project like a feature (`add_bootstrap_config`, auto-id `bootstrap001`; re-registering the same
identity with a larger `n_iterations` grows it in place). `run_bootstrap(project, sample_id,
bootstrap_id)` computes every not-yet-done iteration for that sample: each iteration is a pure
function of `(base_seed, iteration_index)` -- draw `npix` scan positions via the registered
resampling strategy, optionally permute values across positions first if `shuffle` (a shuffled and
unshuffled run at the same iteration index draw the *same* positions, differing only in which
values sit there -- the "permutation test" framing), then compute metrics on the resampled subset
via the same `compute_distribution_1d`/`compute_distribution_2d`/`feature_metric_values` the rest
of the pipeline uses (`compute_distribution_1d(..., fit_gaussian=False)` here, since bootstrap
never reads the fitted curve and fitting one per feature per iteration would only waste time and
risk convergence warnings). Results land in `results.zarr`'s `<sample_id>/bootstrap/<bootstrap_id>/`
group (`metrics` + `completed` arrays, chunk-size-1, via `core.runs.checkpoint`).

This is deliberately not a `Step` -- a bootstrap run is many resumable iterations across possibly
many sessions/processes, not a one-shot per-sample computation. Resuming a partial run, and
extending a finished one to more iterations, are both just "call `run_bootstrap` again" --
`extend_bootstrap_config` (or `run_bootstrap`'s own `n_iterations` convenience kwarg) raises the
registered target first if needed. `n_processes` parallelizes one invocation's own share of pending
iterations across an in-process pool; `worker_index`/`n_workers` statically partition a run's
iterations across separately-invoked processes (e.g. SLURM array-job tasks) with no runtime
coordination beyond agreeing on `n_workers` -- see `core.runs.partition.plan_shard`'s docstring for
why that partition must be computed over the *static* full iteration range, never a dynamically-
shrinking pending list, and the concurrent-subprocess test in `tests/integration/` that guards it.

`run_bootstrap`'s `save_csv` flag (mirroring `generate_feature_report`'s `save_pdf`) exports every
iteration's metrics -- including not-yet-computed ones, as blank cells -- to a flat
`<project_root>/exports/<sample_id>_<bootstrap_id>.csv` via `core.pipeline.bootstrap.csv_export`;
`export_bootstrap_csv` is also callable standalone to re-export without rerunning.

## Conventions

- Only commit when explicitly asked; stage explicit file lists, never `git add -A`/`git add .`.
- `.ELE` support depends on the private `libs_importer` package (`pip install pylibs[ele]`, needs
  SSH access to a private repo) -- `pylibs` must still work fully without it for `.libs` files.
- `examples/` scripts run against real data in `data/` (e.g. `data/dataset1/sample1_1.libs`), not
  toy fixtures -- there's no separate `examples/data/`.
- The test suite is deliberately small (`tests/unit/test_expression.py` for the feature-expression
  parser, `tests/unit/test_cli.py` for dense CLI-integration workflow tests, plus `tests/integration/`
  for concurrency) -- CLI-level happy-path/error-boundary coverage is the primary safety net, not a
  `test_<module>.py` per source file. Don't reintroduce that per-module pattern or re-add direct
  numeric-correctness tests for statistics/correlation/regression/bootstrap math; extend the
  existing CLI workflow tests instead, or ask first if a change seems to call for a new kind of
  test.
- `tests/integration/` files are named per *scenario*, not per module -- `test_bootstrap_concurrent`,
  `test_run_pipeline_batch_concurrent`, `test_registry_concurrent` -- and hold what a CLI workflow
  test structurally cannot express: several real OS processes contending for one artifact. Each
  races something specific (a shared zarr group, a shared registry.json). A concurrency test that
  can pass without the contention actually happening is worse than none, so each one asserts that
  it raced: every worker's exit status is checked, every wait is bounded, and a counter proves the
  overlap occurred (`raised == [racing_group]`, `n_reads > 0`).
