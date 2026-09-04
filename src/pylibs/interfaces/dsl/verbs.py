"""DSL verbs: each one is a thin wrapper over `core.api`, named for use as a
bare function call in DSL scripts (e.g. `CREATEPROJECT("demo", root=".")`).
"""

from pathlib import Path

from pylibs.core import api
from pylibs.core.api import (
    DEFAULT_SCHEME,
    AppliedRegression,
    AppliedRegressionsResult,
    BootstrapConfig,
    BootstrapProgress,
    BootstrapResults,
    BootstrapRunResult,
    CorrelationMatrices,
    CorrelationsResult,
    Distribution2D,
    FeatureConfig,
    FeatureMapsResult,
    FeatureRegression,
    FeatureResults,
    LibsReader,
    PackManifest,
    PackResult,
    PeakTableResult,
    PipelineBatchResult,
    PipelineResult,
    ProjectConfig,
    RegressionsResult,
    SampleConfig,
    SnapshotUseResult,
    VerifyResult,
)


def CREATEPROJECT(
    project_name: str, root: str | Path | None = None, description: str | None = None
) -> ProjectConfig:
    """Get the project `project_name`, creating it if it doesn't exist yet."""
    return api.create_project(
        project_name=project_name,
        root=Path(root) if root is not None else None,
        description=description,
    )


def USEPEAKTABLE(
    project_name: str,
    peak_table: str | Path,
    project_root: str | Path | None = None,
) -> PeakTableResult:
    """Register a peak table with `project_name` by copying it into the
    project as `peak_table.csv`. Call once, right after CREATEPROJECT and
    before ADDFEATURE/ADDELEMENTFEATURES/RUNPIPELINE -- they all read the
    project's own copy.

    Copying is what makes a project self-contained: PACKSNAPSHOT/USESNAPSHOT
    carry the table to another machine, and editing the source afterwards
    can't silently change what this project means. To correct a peak, edit
    your table and call this again (or edit the project's copy directly);
    only the features whose definitions actually changed are recomputed on
    the next RUNPIPELINE."""
    return api.use_peak_table(
        project_name=project_name,
        peak_table=Path(peak_table),
        project_root=Path(project_root) if project_root is not None else None,
    )


def ADDSAMPLE(
    project_name: str,
    path: str | Path,
    sample_name: str | None = None,
    sample_stem: str | None = None,
    project_root: str | Path | None = None,
) -> SampleConfig:
    """Get the sample for `path` on project `project_name`, registering it if it
    isn't yet. The sample's id (e.g. 'sample001') is auto-assigned.

    `sample_name` and `sample_stem` each default independently to `path`'s
    stem. `sample_name` is the correlations join key (see
    COMPUTECORRELATIONS) and can be shared across several samples (e.g.
    ablation-layer samples of one physical sample); `sample_stem` stays a
    unique per-sample label used for report titles."""
    return api.add_sample(
        project_name=project_name,
        path=Path(path),
        sample_name=sample_name,
        sample_stem=sample_stem,
        project_root=Path(project_root) if project_root is not None else None,
    )


def ADDSAMPLESFROMDIR(
    project_name: str,
    directory: str | Path,
    scheme: str = DEFAULT_SCHEME,
    recursive: bool = False,
    project_root: str | Path | None = None,
) -> list[SampleConfig]:
    """Register every readable sample file in `directory` on `project_name` --
    ADDSAMPLE for a whole directory at once, taking each sample's name from
    its filename.

    `scheme` is a filename template, default "{sample_name}_{layer}": so
    "sample1_2.libs" registers with sample_name "sample1", sample_stem
    "sample1_2", and name_parts {"layer": "2"}. Pass a different template for
    a different layout, e.g. "{sample_name}_{param}-{info1}_{layer}".

    Only .libs/.ELE files count, so an unrelated CSV alongside them is
    ignored. `recursive` walks subdirectories too. A file whose name the
    scheme can't parse is logged and skipped, not raised on. Re-running
    doesn't duplicate already-registered samples."""
    return api.add_samples_from_dir(
        project_name=project_name,
        directory=Path(directory),
        scheme=scheme,
        recursive=recursive,
        project_root=Path(project_root) if project_root is not None else None,
    )


def ADDFEATURE(
    project_name: str,
    expression: str,
    feature_name: str | None = None,
    project_root: str | Path | None = None,
) -> FeatureConfig:
    """Get the feature for `expression` on project `project_name`, registering it
    if it isn't yet. The feature's id (e.g. 'feature001') is auto-assigned.
    `expression` is validated against the project's registered peak table,
    so USEPEAKTABLE must have run first."""
    return api.add_feature(
        project_name=project_name,
        expression=expression,
        feature_name=feature_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def ADDELEMENTFEATURES(
    project_name: str,
    element: str | list[str],
    project_root: str | Path | None = None,
    only_default: bool = False,
) -> list[FeatureConfig]:
    """Register every peak the peak table lists for `element` as its own
    bare-peak feature on `project_name` -- ADDFEATURE for a whole element at
    once. `element` takes one name or a list of them ("Fe" or ["Al", "Fe"]),
    matched case-insensitively but exactly ("ca" finds "Ca", never "CaF").

    `only_default` narrows each element to its one flagged default peak.
    Peaks come from the project's registered peak table (USEPEAKTABLE), so
    the elements it knows are the only ones available. An element with nothing
    to contribute (unknown, or no flagged peak) is skipped, not raised on,
    and noted in the project log rather than on the console -- passing a
    standing list of elements and letting the ones this table has no lines
    for fall away is a normal way to use this. Re-running doesn't duplicate
    already-registered peaks."""
    return api.add_element_features(
        project_name=project_name,
        element=element,
        project_root=Path(project_root) if project_root is not None else None,
        only_default=only_default,
    )


def LISTSAMPLES(project_name: str, project_root: str | Path | None = None) -> list[SampleConfig]:
    """List every sample registered with `project_name`, ordered by sample_id."""
    return api.list_samples(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def LISTFEATURES(project_name: str, project_root: str | Path | None = None) -> list[FeatureConfig]:
    """List every feature registered with `project_name`, ordered by feature_id."""
    return api.list_features(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def IMPORTLIBSFILE(
    path: str | Path, capdata: int | None = None, batchsize: int | None = None
) -> LibsReader:
    """Open a raw LIBS data file directly (reader chosen by file extension), with
    no project or registry involved. Returns a reader exposing `.wavelengths`,
    `.params`, `.n_scans`, and `.iter_window(batch_size)`.

    `capdata` caps the total number of spectra read/reported. `batchsize` sets
    the default batch size used by `iter_window` when called without one."""
    return api.import_libs_file(Path(path), capdata=capdata, batchsize=batchsize)


def CONVERTELETOLIBS(
    ele_path: str | Path,
    libs_path: str | Path,
    capdata: int | None = None,
    batchsize: int | None = None,
) -> Path:
    """Convert a .ELE file to the open .libs format (needs the private
    libs_importer package to read the source; the output needs it no
    longer). No project or registry involved. Returns `libs_path`."""
    return api.convert_ele_to_libs(
        ele_path=Path(ele_path),
        libs_path=Path(libs_path),
        capdata=capdata,
        batchsize=batchsize,
    )


def RUNPIPELINE(
    project_name: str,
    sample_id: str,
    project_root: str | Path | None = None,
    capdata: int | None = None,
    batchsize: int | None = None,
    n_processes: int = 1,
) -> PipelineResult:
    """Run every pipeline step (feature computation, spectrum statistics) for
    `sample_id`, streaming through its reader in batches, and write results
    to `<project_root>/results.zarr`. Each step is skipped independently if
    its output already exists (see `result.steps_run`/`steps_skipped`).

    `n_processes` (opt-in, default 1) affects only the distribution-metrics
    steps -- see `api.run_pipeline`'s own docstring. For scaling across
    *samples* instead, see RUNPIPELINEBATCH."""
    return api.run_pipeline(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
        capdata=capdata,
        batchsize=batchsize,
        n_processes=n_processes,
    )


def RUNPIPELINEBATCH(
    project_name: str,
    sample_ids: list[str] | None = None,
    project_root: str | Path | None = None,
    capdata: int | None = None,
    batchsize: int | None = None,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
) -> PipelineBatchResult:
    """Run RUNPIPELINE for every sample in `sample_ids` (default: every
    registered sample) -- the across-sample counterpart to RUNPIPELINE, for
    projects with many samples. `n_processes` > 1 parallelizes this
    invocation's own share of samples across an in-process worker pool.
    `worker_index`/`n_workers` statically partition the sample list across
    separately invoked processes (e.g. SLURM array-job tasks); these two
    knobs compose. See `api.run_pipeline_batch`'s own docstring for why this
    doesn't nest with each sample's own `n_processes`."""
    return api.run_pipeline_batch(
        project_name=project_name,
        sample_ids=sample_ids,
        project_root=Path(project_root) if project_root is not None else None,
        capdata=capdata,
        batchsize=batchsize,
        n_processes=n_processes,
        worker_index=worker_index,
        n_workers=n_workers,
    )


def GETRESULTS(
    project_name: str, sample_id: str, project_root: str | Path | None = None
) -> FeatureResults:
    """Read back the computed feature values for `sample_id`. Raises if the
    sample hasn't been processed yet -- call RUNPIPELINE first. Returns a
    `FeatureResults` exposing `.feature_ids`, `.feature_names`, `.values`
    (shape (n_scans, n_features)), and `.get(feature_id)` for one column."""
    return api.get_results(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def GETDISTRIBUTION2D(
    project_name: str,
    sample_id: str,
    feature_id: str,
    project_root: str | Path | None = None,
) -> Distribution2D:
    """Read back precomputed 2D distribution metrics (joint histogram, Gini
    index, Shannon entropy, Pearson correlation, KL divergences) for a ratio
    feature. Raises if the sample hasn't been processed yet, or if
    `feature_id` isn't a ratio feature -- call RUNPIPELINE first."""
    return api.get_distribution_2d(
        project_name=project_name,
        sample_id=sample_id,
        feature_id=feature_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def COMPUTECORRELATIONS(
    project_name: str,
    csv_path: str | Path,
    project_root: str | Path | None = None,
) -> CorrelationsResult:
    """Reshape every registered feature's already-stored regressions (see
    COMPUTEREGRESSIONS) into a Pearson-r/R²/MAE/MAE permutation-p-value
    matrix per numeric column of `csv_path`, stored in
    `<project_root>/results.zarr`. Never fits anything itself -- run
    COMPUTEREGRESSIONS with `all_metrics=True` first to populate every
    (feature, metric) cell; a cell with no stored regression is left NaN."""
    return api.compute_correlations(
        project_name=project_name,
        csv_path=Path(csv_path),
        project_root=Path(project_root) if project_root is not None else None,
    )


def GETCORRELATIONS(
    project_name: str,
    column_name: str,
    project_root: str | Path | None = None,
) -> CorrelationMatrices:
    """Read back precomputed correlation matrices (Pearson r, R², MAE, MAE
    permutation-p-value) for `column_name`. Raises if that column hasn't
    been computed yet -- call COMPUTECORRELATIONS first."""
    return api.get_correlations(
        project_name=project_name,
        column_name=column_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def LISTCORRELATIONS(project_name: str, project_root: str | Path | None = None) -> list[str]:
    """List every external CSV column with computed correlations for
    `project_name`."""
    return api.list_correlations(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def GENERATECORRELATIONSREPORT(
    project_name: str,
    column_name: str,
    project_root: str | Path | None = None,
    top_n: int = 20,
) -> Path:
    """Build a color-scaled, sorted best/worst PDF report (Pearson r, R²,
    MAE, MAE permutation-p-value) for one CSV column's precomputed
    correlations, saved to `<project_root>/reports/<column_name>_correlations.pdf`.
    Raises if that column hasn't been computed yet -- call
    COMPUTECORRELATIONS first."""
    return api.generate_correlations_report(
        project_name=project_name,
        column_name=column_name,
        project_root=Path(project_root) if project_root is not None else None,
        top_n=top_n,
    )


def COMPUTEREGRESSIONS(
    project_name: str,
    csv_path: str | Path,
    project_root: str | Path | None = None,
    method: str = "univariate",
    select_features: list[str] | None = None,
    algorithm: str = "least_squares",
    metric: str = "Mean",
    all_metrics: bool = False,
    n_permutations: int = 10000,
    random_seed: int = 0,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
) -> RegressionsResult:
    """Fit `metric` (default 'Mean') of every selected feature against every
    numeric column of `csv_path` (a sample-properties CSV joined on a
    `sample_name` column matching each registered sample), via `algorithm`
    (default 'least_squares'). `all_metrics=True` fits every metric instead
    of just `metric` -- an "all-metrics sweep", needed to give
    COMPUTECORRELATIONS a fully populated matrix to read back (it never fits
    anything itself). `method` must be 'univariate' (the only implemented
    method). `select_features` is a list of feature expressions (as passed
    to ADDFEATURE); `None` means every registered feature. Storing the fit
    and a per-physical-sample summary (for plotting) in
    `<project_root>/results.zarr`. Each fit's p-value is a permutation-test
    p-value on its own residual MAE, not an analytic test; `n_permutations`
    (default 10000) and `random_seed` (default 0) control it -- each
    (column, metric, feature) cell is seeded independently, so results don't
    depend on `n_processes`/`n_workers`.

    The fitting sweep -- one independent unit of work per (column, metric,
    feature) cell -- is parallelized like RUNBOOTSTRAP: `n_processes` > 1
    parallelizes this invocation's own share of cells across an in-process
    worker pool; `worker_index`/`n_workers` statically partition the full
    cell range across separately invoked processes (e.g. SLURM array-job
    tasks); these two knobs compose."""
    return api.compute_regressions(
        project_name=project_name,
        csv_path=Path(csv_path),
        project_root=Path(project_root) if project_root is not None else None,
        method=method,
        select_features=select_features,
        algorithm=algorithm,
        metric=None if all_metrics else metric,
        n_permutations=n_permutations,
        random_seed=random_seed,
        n_processes=n_processes,
        worker_index=worker_index,
        n_workers=n_workers,
    )


def GETREGRESSIONRESULTS(
    project_name: str,
    column_name: str,
    metric_name: str | None = None,
    select: list[tuple[str, str]] | None = None,
    project_root: str | Path | None = None,
) -> list[FeatureRegression]:
    """Read back every feature's regression for `column_name` -- every
    stored metric's, just `metric_name`'s, or exactly the `(feature_
    expression, metric_name)` pairs in `select` (mutually exclusive with
    `metric_name`), if given. Raises if nothing's been computed (for that
    metric, or every `select` pair, if given) yet -- call COMPUTEREGRESSIONS
    first."""
    return api.get_regression_results(
        project_name=project_name,
        column_name=column_name,
        metric_name=metric_name,
        select=select,
        project_root=Path(project_root) if project_root is not None else None,
    )


def LISTREGRESSIONS(project_name: str, project_root: str | Path | None = None) -> list[str]:
    """List every external CSV column with computed regressions for
    `project_name`."""
    return api.list_regressions(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def GENERATEREGRESSIONREPORT(
    project_name: str,
    column_name: str,
    metric_name: str | None = None,
    select: list[tuple[str, str]] | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Build a calibration-curve PDF report (one page per feature, per
    metric) for one CSV column's precomputed regressions -- every stored
    metric's, just `metric_name`'s, or exactly the `(feature_expression,
    metric_name)` pairs in `select` (mutually exclusive with `metric_name`,
    e.g. to bundle a curated, mixed-metric set of calibration curves into
    one PDF) -- saved to `<project_root>/reports/<column_name>_regressions.pdf`.
    Raises if nothing's been computed (for that metric, or every `select`
    pair, if given) yet -- call COMPUTEREGRESSIONS first."""
    return api.generate_regression_report(
        project_name=project_name,
        column_name=column_name,
        metric_name=metric_name,
        select=select,
        project_root=Path(project_root) if project_root is not None else None,
    )


def APPLYREGRESSIONS(
    project_name: str,
    sample_id: str,
    project_root: str | Path | None = None,
    columns: list[str] | None = None,
    select_features: list[str] | None = None,
    select: list[tuple[str, str]] | None = None,
) -> AppliedRegressionsResult:
    """Applies every stored regression (or a narrower `columns`/
    `select_features`/`select` subset -- `select` is a list of
    `(feature_expression, metric_name)` pairs, mutually exclusive with
    `select_features`, applying only those exact combinations) to
    `sample_id`'s own raw per-scan feature values, inverting each fit to
    predict a per-pixel value of the external property, storing the
    predicted array and its own distribution stats in
    `<project_root>/results.zarr`."""
    return api.apply_regressions(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
        columns=columns,
        select_features=select_features,
        select=select,
    )


def GETAPPLIEDREGRESSIONRESULTS(
    project_name: str,
    sample_id: str,
    project_root: str | Path | None = None,
    column_name: str | None = None,
    select: list[tuple[str, str]] | None = None,
) -> list[AppliedRegression]:
    """Read back every applied regression for `sample_id` (or just
    `column_name`'s, or exactly the `(feature_expression, metric_name)`
    pairs in `select`, if given). Raises if nothing has been applied yet --
    call APPLYREGRESSIONS first."""
    return api.get_applied_regression_results(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
        column_name=column_name,
        select=select,
    )


def LISTAPPLIEDREGRESSIONS(
    project_name: str, sample_id: str, project_root: str | Path | None = None
) -> list[str]:
    """List every external CSV column with applied regressions for
    `sample_id`."""
    return api.list_applied_regressions(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def GENERATEAPPLIEDREGRESSIONREPORT(
    project_name: str,
    sample_id: str,
    project_root: str | Path | None = None,
    select: list[tuple[str, str]] | None = None,
) -> Path:
    """Build a PDF report (one page per applied (column, feature) pair) of
    `sample_id`'s precomputed applied regressions -- or exactly the
    `(feature_expression, metric_name)` pairs in `select`, if given -- saved
    to `<project_root>/reports/<sample_id>_applied_regressions.pdf`. Raises
    if nothing has been applied yet -- call APPLYREGRESSIONS first."""
    return api.generate_applied_regression_report(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
        select=select,
    )


def GENERATEFEATUREREPORT(
    project_name: str,
    sample_id: str,
    project_root: str | Path | None = None,
    save_pdf: bool = False,
) -> FeatureMapsResult:
    """Generate one combined diagnostics PNG per registered feature for
    `sample_id` -- (A) a feature map, (B) diagnostic spectra (bare-peak
    features only, not a ratio), and (C) a distribution histogram -- saved
    to `<project_root>/reports/img/features/<sample_id>/<feature_id>.png`,
    plus a separate metrics-table PNG at
    `<project_root>/reports/img/features/<sample_id>/<feature_id>_table.png`.
    Requires the sample to already be processed -- call RUNPIPELINE first.
    If `save_pdf` is set, also bundles everything into
    `<project_root>/reports/<sample_id>_features.pdf` (no LaTeX needed)."""
    return api.generate_feature_report(
        project_name=project_name,
        sample_id=sample_id,
        project_root=Path(project_root) if project_root is not None else None,
        save_pdf=save_pdf,
    )


def ADDBOOTSTRAPCONFIG(
    project_name: str,
    method: str,
    npix: int | None = None,
    pct: float | None = None,
    n_iterations: int = 100,
    base_seed: int = 0,
    shuffle: bool = False,
    project_root: str | Path | None = None,
) -> BootstrapConfig:
    """Get the bootstrap config for (method, npix-or-pct, base_seed, shuffle)
    on `project_name`, registering it if it isn't yet. The config's id (e.g.
    'bootstrap001') is auto-assigned; re-registering the same identity with a
    larger `n_iterations` extends it in place. Exactly one of `npix`/`pct`
    must be given."""
    return api.add_bootstrap_config(
        project_name=project_name,
        method=method,
        npix=npix,
        pct=pct,
        n_iterations=n_iterations,
        base_seed=base_seed,
        shuffle=shuffle,
        project_root=Path(project_root) if project_root is not None else None,
    )


def EXTENDBOOTSTRAPCONFIG(
    project_name: str,
    bootstrap_id: str,
    n_iterations: int,
    project_root: str | Path | None = None,
) -> BootstrapConfig:
    """Raise `bootstrap_id`'s registered `n_iterations` target -- RUNBOOTSTRAP
    picks this new target up on its next invocation. Must not be called while
    a run against `bootstrap_id` is concurrently executing."""
    return api.extend_bootstrap_config(
        project_name=project_name,
        bootstrap_id=bootstrap_id,
        n_iterations=n_iterations,
        project_root=Path(project_root) if project_root is not None else None,
    )


def LISTBOOTSTRAPCONFIGS(
    project_name: str, project_root: str | Path | None = None
) -> list[BootstrapConfig]:
    """List every bootstrap config registered with `project_name`, ordered by
    bootstrap_id."""
    return api.list_bootstrap_configs(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def RUNBOOTSTRAP(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    project_root: str | Path | None = None,
    n_iterations: int | None = None,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
    batchsize: int | None = None,
    save_csv: bool = False,
) -> BootstrapRunResult:
    """Run every not-yet-computed iteration of `bootstrap_id`'s registered
    target for `sample_id`. Requires the sample to already be processed --
    call RUNPIPELINE first.

    Resuming a partially-done run and extending a fully-done one are both
    just "call this again": resume happens automatically; extend happens
    automatically if `n_iterations` is given and greater than the config's
    current target. `n_processes` > 1 parallelizes this invocation's own
    share of pending iterations across an in-process worker pool.
    `worker_index`/`n_workers` statically partition the run's pending
    iterations across several separately invoked processes (e.g. SLURM
    array-job tasks); these two knobs compose.

    If `save_csv` is set, also exports every iteration's metrics to
    `<project_root>/exports/<sample_id>_<bootstrap_id>.csv` (see
    EXPORTBOOTSTRAPCSV to re-export standalone, without rerunning)."""
    return api.run_bootstrap(
        project_name=project_name,
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        project_root=Path(project_root) if project_root is not None else None,
        n_iterations=n_iterations,
        n_processes=n_processes,
        worker_index=worker_index,
        n_workers=n_workers,
        batchsize=batchsize,
        save_csv=save_csv,
    )


def GETBOOTSTRAPPROGRESS(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    project_root: str | Path | None = None,
) -> BootstrapProgress:
    """Read back (n_done, n_iterations) for `bootstrap_id` on `sample_id`.
    Raises if that run hasn't been started yet -- call RUNBOOTSTRAP first."""
    return api.get_bootstrap_progress(
        project_name=project_name,
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def GETBOOTSTRAPRESULTS(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    project_root: str | Path | None = None,
) -> BootstrapResults:
    """Read back every computed metrics row for `bootstrap_id` on
    `sample_id`, plus which iterations are actually done (`.completed`).
    Raises if that run hasn't been started yet -- call RUNBOOTSTRAP first."""
    return api.get_bootstrap_results(
        project_name=project_name,
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def EXPORTBOOTSTRAPCSV(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    project_root: str | Path | None = None,
) -> Path:
    """Export every iteration's metrics for `bootstrap_id` on `sample_id` to
    `<project_root>/exports/<sample_id>_<bootstrap_id>.csv` -- one row per
    iteration (including not-yet-computed ones), one column per
    feature+metric pair. Raises if that run hasn't been started yet -- call
    RUNBOOTSTRAP first."""
    return api.export_bootstrap_csv(
        project_name=project_name,
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def GETBOOTSTRAPSTATUS(
    project_name: str, bootstrap_id: str, project_root: str | Path | None = None
) -> dict[str, BootstrapProgress]:
    """Per-sample progress for one registered bootstrap config, across every
    sample that has any results for it."""
    return api.get_bootstrap_status(
        project_name=project_name,
        bootstrap_id=bootstrap_id,
        project_root=Path(project_root) if project_root is not None else None,
    )


def PACKPROJECT(
    project_name: str,
    out_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> PackResult:
    """Pack the project's results.zarr into a single .zarr.zip archive
    (default `<project_root>/results.zarr.zip`) plus a .manifest.json
    provenance sidecar. Non-destructive."""
    return api.pack_project(
        project_name=project_name,
        out_path=Path(out_path) if out_path is not None else None,
        project_root=Path(project_root) if project_root is not None else None,
    )


def VERIFYPACK(path: str | Path) -> VerifyResult:
    """Sanity-check a .zarr.zip archive produced by PACKPROJECT."""
    return api.verify_pack(Path(path))


def INDEXRUNS(root: str | Path) -> list[PackManifest]:
    """Every manifest (from PACKPROJECT) found anywhere under `root`,
    sorted by when it was packed."""
    return api.index_packs(Path(root))


def PACKSNAPSHOT(project_name: str, project_root: str | Path | None = None) -> PackResult:
    """Pack the project's results.zarr into a new, timestamped snapshot
    under <project_root>/snapshots/ (a colliding timestamp gets a -1/-2
    suffix) -- unlike PACKPROJECT, never overwrites
    a prior snapshot."""
    return api.pack_snapshot(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def LISTSNAPSHOTS(project_name: str, project_root: str | Path | None = None) -> list[PackManifest]:
    """Every snapshot packed for this project, sorted by when it was packed."""
    return api.list_snapshots(
        project_name=project_name,
        project_root=Path(project_root) if project_root is not None else None,
    )


def USESNAPSHOT(
    project_name: str,
    timestamp: str | None = None,
    project_root: str | Path | None = None,
    force: bool = False,
) -> SnapshotUseResult:
    """Extract a snapshot (the most recent one, if `timestamp` is omitted)
    into the project's results.zarr, so the rest of this script keeps
    working from exactly that state. Raises if results.zarr already exists
    and `force` isn't set."""
    return api.use_snapshot(
        project_name=project_name,
        timestamp=timestamp,
        project_root=Path(project_root) if project_root is not None else None,
        force=force,
    )


VERBS = {
    "CREATEPROJECT": CREATEPROJECT,
    "USEPEAKTABLE": USEPEAKTABLE,
    "ADDSAMPLE": ADDSAMPLE,
    "ADDSAMPLESFROMDIR": ADDSAMPLESFROMDIR,
    "ADDFEATURE": ADDFEATURE,
    "ADDELEMENTFEATURES": ADDELEMENTFEATURES,
    "LISTSAMPLES": LISTSAMPLES,
    "LISTFEATURES": LISTFEATURES,
    "IMPORTLIBSFILE": IMPORTLIBSFILE,
    "CONVERTELETOLIBS": CONVERTELETOLIBS,
    "RUNPIPELINE": RUNPIPELINE,
    "RUNPIPELINEBATCH": RUNPIPELINEBATCH,
    "GETRESULTS": GETRESULTS,
    "GETDISTRIBUTION2D": GETDISTRIBUTION2D,
    "COMPUTECORRELATIONS": COMPUTECORRELATIONS,
    "GETCORRELATIONS": GETCORRELATIONS,
    "LISTCORRELATIONS": LISTCORRELATIONS,
    "GENERATECORRELATIONSREPORT": GENERATECORRELATIONSREPORT,
    "COMPUTEREGRESSIONS": COMPUTEREGRESSIONS,
    "GETREGRESSIONRESULTS": GETREGRESSIONRESULTS,
    "LISTREGRESSIONS": LISTREGRESSIONS,
    "GENERATEREGRESSIONREPORT": GENERATEREGRESSIONREPORT,
    "APPLYREGRESSIONS": APPLYREGRESSIONS,
    "GETAPPLIEDREGRESSIONRESULTS": GETAPPLIEDREGRESSIONRESULTS,
    "LISTAPPLIEDREGRESSIONS": LISTAPPLIEDREGRESSIONS,
    "GENERATEAPPLIEDREGRESSIONREPORT": GENERATEAPPLIEDREGRESSIONREPORT,
    "GENERATEFEATUREREPORT": GENERATEFEATUREREPORT,
    "ADDBOOTSTRAPCONFIG": ADDBOOTSTRAPCONFIG,
    "EXTENDBOOTSTRAPCONFIG": EXTENDBOOTSTRAPCONFIG,
    "LISTBOOTSTRAPCONFIGS": LISTBOOTSTRAPCONFIGS,
    "RUNBOOTSTRAP": RUNBOOTSTRAP,
    "GETBOOTSTRAPPROGRESS": GETBOOTSTRAPPROGRESS,
    "GETBOOTSTRAPRESULTS": GETBOOTSTRAPRESULTS,
    "EXPORTBOOTSTRAPCSV": EXPORTBOOTSTRAPCSV,
    "GETBOOTSTRAPSTATUS": GETBOOTSTRAPSTATUS,
    "PACKPROJECT": PACKPROJECT,
    "VERIFYPACK": VERIFYPACK,
    "INDEXRUNS": INDEXRUNS,
    "PACKSNAPSHOT": PACKSNAPSHOT,
    "LISTSNAPSHOTS": LISTSNAPSHOTS,
    "USESNAPSHOT": USESNAPSHOT,
}
