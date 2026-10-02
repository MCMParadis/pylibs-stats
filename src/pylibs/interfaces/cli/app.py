import math
from pathlib import Path

import typer

from pylibs.core import api
from pylibs.core.api import DEFAULT_SCHEME
from pylibs.core.exceptions import ProjectError, PylibsError
from pylibs.core.logging import LOG_FILENAME
from pylibs.interfaces.dsl.runner import run_script

# How many unexplained skipped fits to name on the console before summarising
# the rest; the complete list is always in the project log.
MAX_SKIPPED_FITS_SHOWN = 10

app = typer.Typer(name="pylibs")
project_app = typer.Typer(name="project", help="Manage pylibs projects.")
app.add_typer(project_app)


def _parse_select(values: list[str] | None) -> list[tuple[str, str]] | None:
    """Parses repeatable `--select "<expression>:<metric_name>"` CLI values
    into `(feature_expression, metric_name)` pairs, splitting each on the
    last `:` (feature expressions never contain one)."""
    if not values:
        return None
    pairs = []
    for value in values:
        if ":" not in value:
            raise typer.BadParameter(
                f"{value!r} must be '<feature expression>:<metric name>'", param_hint="--select"
            )
        expression, _, metric_name = value.rpartition(":")
        pairs.append((expression, metric_name))
    return pairs


@app.callback()
def main(
    debug: bool = typer.Option(
        False, "--debug", help="Log at DEBUG level, to the console and any project's log file."
    ),
) -> None:
    # runs before any subcommand -- including nested `project` subcommands --
    # dispatches, so this is the one place a global flag needs to live
    api.set_debug_logging(debug)


@project_app.command("create")
def project_create(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory to create the project in. Defaults to ./projects/<project_name>.",
    ),
    description: str | None = typer.Option(None, "--description", help="Project description."),
) -> None:
    """Create a pylibs project, or reuse one that already exists at `root`."""
    try:
        config = api.create_project(project_name=project_name, root=root, description=description)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Project {config.name!r} ready at {config.root}")


@project_app.command("use-peak-table")
def project_use_peak_table(
    project_name: str,
    peak_table_path: Path,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Register a peak table with the project by copying it in. Must be run
    once after `create`, before registering features or running the pipeline
    -- every command that resolves a peak id reads the project's own copy."""
    try:
        result = api.use_peak_table(
            project_name=project_name, peak_table=peak_table_path, project_root=root
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    # naming the count and elements makes registering the wrong CSV obvious
    typer.echo(
        f"Peak table ready for project {project_name!r} at {result.path} "
        f"({result.n_peaks} peaks: {', '.join(result.elements)})"
    )
    if result.replaced_existing:
        typer.echo(
            "Replaced the previous peak table -- any feature whose peak definitions changed "
            "will be recomputed on the next run-pipeline."
        )


@project_app.command("add-sample")
def project_add_sample(
    project_name: str,
    path: Path,
    sample_name: str | None = typer.Option(
        None,
        "--sample-name",
        help="Correlations join key for the sample (see compute-correlations). "
        "Can be shared by several samples. Defaults to the file's stem.",
    ),
    sample_stem: str | None = typer.Option(
        None,
        "--sample-stem",
        help="Unique per-sample display label, used for report titles. "
        "Defaults to the file's stem.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Register a sample with an existing project, or reuse it if already registered."""
    try:
        sample = api.add_sample(
            project_name=project_name,
            path=path,
            sample_name=sample_name,
            sample_stem=sample_stem,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Sample {sample.sample_id!r} ready in project {project_name!r} at {sample.path}")


@project_app.command("add-samples-from-dir")
def project_add_samples_from_dir(
    project_name: str,
    directory: Path,
    scheme: str = typer.Option(
        DEFAULT_SCHEME,
        "--scheme",
        help="Filename template naming each sample, e.g. '{sample_name}_{layer}'. "
        "The {sample_name} field becomes the correlations join key; every other "
        "field is stored alongside it.",
    ),
    recursive: bool = typer.Option(
        False, "--recursive", help="Search subdirectories too, not just the given directory."
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Register every .libs/.ELE file in a directory as a sample, taking each
    one's name from its filename. A file the scheme can't parse is logged and
    skipped."""
    try:
        samples = api.add_samples_from_dir(
            project_name=project_name,
            directory=directory,
            scheme=scheme,
            recursive=recursive,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not samples:
        typer.echo("No samples registered.")
        return
    for sample in samples:
        typer.echo(
            f"Sample {sample.sample_id!r} ready in project {project_name!r} "
            f"at {sample.path} (name={sample.sample_name})"
        )


@project_app.command("add-feature")
def project_add_feature(
    project_name: str,
    expression: str,
    feature_name: str | None = typer.Option(
        None, "--feature-name", help="Display name for the feature."
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Register a feature expression with an existing project, or update it if
    the same expression is already registered."""
    try:
        feature = api.add_feature(
            project_name=project_name,
            expression=expression,
            feature_name=feature_name,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Feature {feature.feature_id!r} ready in project {project_name!r}: {expression}")


@project_app.command("add-element-features")
def project_add_element_features(
    project_name: str,
    elements: list[str],
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    only_default: bool = typer.Option(
        False,
        "--only-default",
        help="Register only each element's flagged default peak, not all of its peaks.",
    ),
) -> None:
    """Register every peak the peak table lists for each named element as its
    own feature -- add-feature for a whole element at once. An element with
    nothing to contribute (unknown, or no flagged default peak) is skipped and
    noted in the project log, not on the console."""
    try:
        features = api.add_element_features(
            project_name=project_name,
            element=elements,
            project_root=root,
            only_default=only_default,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not features:
        typer.echo("No features registered.")
        return
    for feature in features:
        typer.echo(
            f"Feature {feature.feature_id!r} ready in project {project_name!r}: "
            f"{feature.expression}"
        )


@project_app.command("list-samples")
def project_list_samples(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every sample registered with the project."""
    try:
        samples = api.list_samples(project_name=project_name, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not samples:
        typer.echo("No samples registered.")
        return
    for sample in samples:
        status = "processed" if sample.results_path is not None else "not processed"
        typer.echo(
            f"  {sample.sample_id} (name={sample.sample_name}, stem={sample.sample_stem}): "
            f"{sample.path} [{status}]"
        )


@project_app.command("list-features")
def project_list_features(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every feature registered with the project."""
    try:
        features = api.list_features(project_name=project_name, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not features:
        typer.echo("No features registered.")
        return
    for feature in features:
        suffix = f" ({feature.feature_name})" if feature.feature_name != feature.expression else ""
        typer.echo(f"  {feature.feature_id}{suffix}: {feature.expression}")


@project_app.command("inspect-sample")
def project_inspect_sample(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Open a registered sample's raw data and report scan/pixel counts and wavelength range."""
    try:
        inspection = api.inspect_sample(
            project_name=project_name, sample_id=sample_id, project_root=root
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Sample {inspection.sample_id!r} at {inspection.path}")
    typer.echo(f"  {inspection.n_scans} scans x {inspection.n_pix} pixels")
    typer.echo(
        f"  wavelength range: {inspection.wavelength_min:.2f}-{inspection.wavelength_max:.2f}"
    )


@project_app.command("run-pipeline")
def project_run_pipeline(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    capdata: int | None = typer.Option(
        None, "--capdata", help="Cap the total number of spectra processed."
    ),
    batchsize: int | None = typer.Option(
        None, "--batchsize", help="Batch size to stream the sample in."
    ),
    n_processes: int = typer.Option(
        1,
        "--n-processes",
        help="Worker processes for this sample's distribution-metrics steps (<=0 for every "
        "CPU available). Feature computation itself stays sequential.",
    ),
) -> None:
    """Run every pipeline step (feature computation, spectrum statistics) for
    a sample and write results to <root>/results.zarr. Each step is skipped
    independently if its output already exists."""
    try:
        result = api.run_pipeline(
            project_name=project_name,
            sample_id=sample_id,
            project_root=root,
            capdata=capdata,
            batchsize=batchsize,
            n_processes=n_processes,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{len(result.feature_ids)} feature(s) for {result.n_scans} scans")
    if result.steps_run:
        typer.echo(f"  ran: {', '.join(result.steps_run)}")
    if result.steps_skipped:
        typer.echo(f"  skipped (already computed): {', '.join(result.steps_skipped)}")
    typer.echo(f"  results: {result.results_path}")


@project_app.command("run-pipeline-batch")
def project_run_pipeline_batch(
    project_name: str,
    sample_id: list[str] | None = typer.Option(
        None,
        "--sample-id",
        help="Sample id(s) to process (repeatable). Defaults to every registered sample.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    capdata: int | None = typer.Option(
        None, "--capdata", help="Cap the total number of spectra processed per sample."
    ),
    batchsize: int | None = typer.Option(
        None, "--batchsize", help="Batch size to stream each sample in."
    ),
    n_processes: int = typer.Option(
        1,
        "--n-processes",
        help="Worker processes for this invocation's own share of samples (<=0 for every CPU "
        "available, e.g. $SLURM_CPUS_PER_TASK).",
    ),
    worker_index: int = typer.Option(
        0,
        "--worker-index",
        help="This invocation's shard index, for splitting the sample list across separately "
        "invoked processes (e.g. $SLURM_ARRAY_TASK_ID).",
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        help="Total number of shards the sample list is split across (e.g. "
        "$SLURM_ARRAY_TASK_COUNT).",
    ),
) -> None:
    """Run run-pipeline for many samples at once (default: every registered
    sample), optionally in parallel -- the across-sample counterpart to
    run-pipeline, for projects with many samples."""
    try:
        result = api.run_pipeline_batch(
            project_name=project_name,
            sample_ids=sample_id,
            project_root=root,
            capdata=capdata,
            batchsize=batchsize,
            n_processes=n_processes,
            worker_index=worker_index,
            n_workers=n_workers,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Pipeline batch: {result.n_processed}/{len(result.sample_ids)} sample(s) processed")
    if result.n_failed:
        typer.echo(f"  {result.n_failed} sample(s) failed: {result.failed_sample_ids}")
    typer.echo(f"  results: {result.results_path}")


@project_app.command("get-results")
def project_get_results(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Print a per-feature summary (mean/min/max) of a sample's computed results."""
    try:
        results = api.get_results(project_name=project_name, sample_id=sample_id, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Results for sample {sample_id!r} ({results.values.shape[0]} scans):")
    for feature_id in results.feature_ids:
        values = results.get(feature_id)
        name = results.feature_names.get(feature_id, feature_id)
        typer.echo(
            f"  {feature_id} ({name}): mean={values.mean():.4g} "
            f"min={values.min():.4g} max={values.max():.4g}"
        )


@project_app.command("get-distribution-2d")
def project_get_distribution_2d(
    project_name: str,
    sample_id: str,
    feature_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Print a ratio feature's 2D distribution metrics (joint histogram Gini
    index, Shannon entropy, Pearson correlation, KL divergences), computed
    by Distribution2DStep. Raises if `feature_id` isn't a ratio feature."""
    try:
        stats = api.get_distribution_2d(
            project_name=project_name,
            sample_id=sample_id,
            feature_id=feature_id,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"2D distribution stats for {feature_id!r} "
        f"({stats.numerator_expression} / {stats.denominator_expression}):"
    )
    typer.echo(f"  Gini index: {stats.gini:.4g}")
    typer.echo(
        f"  Shannon entropy: {stats.shannon_entropy:.4g} "
        f"(boundary-normalized: {stats.shannon_entropy_boundary:.4g})"
    )
    if stats.pearson_correlation is not None:
        typer.echo(f"  Pearson-CR: {stats.pearson_correlation:.4g}")
    if stats.kl_divergence_gaussians is not None:
        typer.echo(f"  c-KL divergence: {stats.kl_divergence_gaussians:.4g}")
    typer.echo(f"  d-KL divergence: {stats.kl_divergence_histograms:.4g}")


@project_app.command("compute-correlations")
def project_compute_correlations(
    project_name: str,
    csv_path: Path,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Reshape every registered feature's already-stored regressions (see
    compute-regressions) into a Pearson-r/R²/MAE/MAE permutation-p-value
    matrix per numeric column of a sample-properties CSV, stored in
    <root>/results.zarr. Never fits anything itself -- run
    compute-regressions --all-metrics first to populate every (feature,
    metric) cell."""
    try:
        result = api.compute_correlations(
            project_name=project_name,
            csv_path=csv_path,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Computed correlations for {len(result.column_names)} column(s):")
    for column_name in result.column_names:
        typer.echo(f"  {column_name}")
    if result.skipped_columns:
        typer.echo(f"Skipped non-numeric column(s): {', '.join(result.skipped_columns)}")
    typer.echo(f"{result.found_cells}/{result.found_cells + result.missing_cells} cell(s) found")
    if result.missing_cells:
        typer.echo(
            f"{result.missing_cells} cell(s) had no stored regression -- run "
            "compute-regressions --all-metrics to fill them"
        )
    typer.echo(f"results: {result.results_path}")


@project_app.command("list-correlations")
def project_list_correlations(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every external CSV column with computed correlations."""
    try:
        columns = api.list_correlations(project_name=project_name, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not columns:
        typer.echo("No correlations computed yet.")
        return
    for column_name in columns:
        typer.echo(f"  {column_name}")


@project_app.command("get-correlations")
def project_get_correlations(
    project_name: str,
    column_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    top: int = typer.Option(10, "--top", help="How many |Pearson r| pairs to print."),
) -> None:
    """Print the strongest feature+metric correlations for one CSV column --
    a sanity glance, not the full matrix (see generate-correlations-report
    for the full color-scaled, sorted best/worst table)."""
    try:
        matrices = api.get_correlations(
            project_name=project_name, column_name=column_name, project_root=root
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    pairs = [
        (
            feature_id,
            metric_name,
            matrices.pearson_r[i, j],
            matrices.r_squared[i, j],
            matrices.mae[i, j],
            matrices.mae_p_value[i, j],
        )
        for i, feature_id in enumerate(matrices.feature_ids)
        for j, metric_name in enumerate(matrices.metric_names)
        if not math.isnan(matrices.pearson_r[i, j])
    ]
    pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)

    total_cells = matrices.pearson_r.size
    undefined = total_cells - len(pairs)
    typer.echo(f"Top {min(top, len(pairs))} |Pearson r| for column {column_name!r}:")
    for feature_id, metric_name, r, r_squared, mae, mae_p in pairs[:top]:
        typer.echo(
            f"  {feature_id} / {metric_name}: r={r:.4g} R²={r_squared:.4g} "
            f"mae={mae:.4g} mae_p={mae_p:.4g}"
        )
    typer.echo(f"{undefined}/{total_cells} cells undefined")


@project_app.command("generate-correlations-report")
def project_generate_correlations_report(
    project_name: str,
    column_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    top_n: int = typer.Option(
        20, "--top-n", help="How many features to show in each best/worst table."
    ),
) -> None:
    """Build a color-scaled, sorted best/worst PDF report (Pearson r, R²,
    p-value) for one CSV column's precomputed correlations, saved to
    <root>/reports/<column_name>_correlations.pdf."""
    try:
        path = api.generate_correlations_report(
            project_name=project_name, column_name=column_name, project_root=root, top_n=top_n
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Correlations report: {path}")


@project_app.command("compute-regressions")
def project_compute_regressions(
    project_name: str,
    csv_path: Path,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    method: str = typer.Option(
        "univariate", "--method", help="Regression method. Only 'univariate' is implemented."
    ),
    select_features: list[str] | None = typer.Option(
        None,
        "--select-features",
        help="Feature expression(s) to regress (repeatable), e.g. --select-features B249 "
        "--select-features B249/C229. Defaults to every registered feature.",
    ),
    algorithm: str = typer.Option(
        "least_squares", "--algorithm", help="Regression algorithm. Only 'least_squares' for now."
    ),
    metric: str = typer.Option(
        "Mean", "--metric", help="Which of a feature's metrics to regress against (default Mean)."
    ),
    all_metrics: bool = typer.Option(
        False,
        "--all-metrics",
        help="Fit every metric instead of just --metric -- an all-metrics sweep, needed to give "
        "compute-correlations a fully populated matrix to read back.",
    ),
    n_permutations: int = typer.Option(
        10000, "--n-permutations", help="Iterations for the MAE permutation test."
    ),
    random_seed: int = typer.Option(
        0, "--random-seed", help="Seed for the MAE permutation test's RNG (reproducibility)."
    ),
    group_by_sample_name: bool = typer.Option(
        True,
        "--group-by-sample-name/--no-group-by-sample-name",
        help="Treat the physical sample, not the acquisition, as the unit of inference. "
        "The fitted line, R2 and MAE are unchanged; the confidence band and the permutation "
        "test use one point per sample name. No-op when no name repeats.",
    ),
    n_processes: int = typer.Option(
        1,
        "--n-processes",
        help="Worker processes for this invocation's own share of (column, metric, feature) "
        "cells (<=0 for every CPU available, e.g. $SLURM_CPUS_PER_TASK).",
    ),
    worker_index: int = typer.Option(
        0,
        "--worker-index",
        help="This invocation's shard index, for splitting the cell sweep across separately "
        "invoked processes (e.g. $SLURM_ARRAY_TASK_ID).",
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        help="Total number of shards the cell sweep is split across (e.g. "
        "$SLURM_ARRAY_TASK_COUNT).",
    ),
) -> None:
    """Fit `metric` (or every metric, with --all-metrics) of every selected
    feature against every numeric column of a sample-properties CSV (joined
    on a `sample_name` column matching each registered sample), storing the
    fit and a per-physical-sample summary (for plotting) in
    <root>/results.zarr."""
    try:
        result = api.compute_regressions(
            project_name=project_name,
            csv_path=csv_path,
            project_root=root,
            method=method,
            select_features=select_features,
            algorithm=algorithm,
            metric=None if all_metrics else metric,
            n_permutations=n_permutations,
            random_seed=random_seed,
            n_processes=n_processes,
            worker_index=worker_index,
            n_workers=n_workers,
            group_by_sample_name=group_by_sample_name,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Computed regressions for {len(result.column_names)} column(s):")
    for column_name in result.column_names:
        typer.echo(f"  {column_name}")
    if result.skipped_columns:
        typer.echo(f"Skipped non-numeric column(s): {', '.join(result.skipped_columns)}")
    typer.echo(f"{len(result.sample_ids_used)} sample(s) used")
    if result.skipped_samples:
        typer.echo(f"Skipped sample(s): {', '.join(result.skipped_samples)}")
    if result.skipped_fits:
        # The full list lives in the project log -- an all-metrics sweep skips
        # every 2D metric of every non-ratio feature, which is routine and can
        # run to thousands of lines. Only the ones that aren't explained that
        # way are worth the console.
        log_path = result.results_path.parent / LOG_FILENAME
        typer.echo(f"Skipped {len(result.skipped_fits)} fit(s) -- full list in {log_path}")
        unexpected = result.unexpected_skipped_fits
        for entry in unexpected[:MAX_SKIPPED_FITS_SHOWN]:
            typer.echo(f"  {entry}")
        if len(unexpected) > MAX_SKIPPED_FITS_SHOWN:
            typer.echo(f"  ... and {len(unexpected) - MAX_SKIPPED_FITS_SHOWN} more")
    typer.echo(f"results: {result.results_path}")


@project_app.command("list-regressions")
def project_list_regressions(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every external CSV column with computed regressions."""
    try:
        columns = api.list_regressions(project_name=project_name, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not columns:
        typer.echo("No regressions computed yet.")
        return
    for column_name in columns:
        typer.echo(f"  {column_name}")


@project_app.command("get-regression-results")
def project_get_regression_results(
    project_name: str,
    column_name: str,
    metric: str | None = typer.Option(
        None, "--metric", help="Only this metric's regressions. Defaults to every stored metric."
    ),
    select: list[str] | None = typer.Option(
        None,
        "--select",
        help="'<feature expression>:<metric name>' (repeatable) -- exactly these pairs, mutually "
        "exclusive with --metric.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Print slope/intercept/R²/MAE/p-value/n for every feature regressed
    against one CSV column."""
    try:
        regressions = api.get_regression_results(
            project_name=project_name,
            column_name=column_name,
            metric_name=metric,
            select=_parse_select(select),
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Regressions against column {column_name!r}:")
    for regression in regressions:
        typer.echo(
            f"  {regression.feature_id} ({regression.metric_name}): "
            f"slope={regression.slope:.4g} intercept={regression.intercept:.4g} "
            f"R²={regression.r_squared:.4g} MAE={regression.mae:.4g} "
            f"{api.format_p_value(regression.p_value, regression.n_permutations)} "
            f"[{api.format_permutation_mode(regression)}] "
            f"{api.format_sample_count(regression)}"
        )


@project_app.command("generate-regression-report")
def project_generate_regression_report(
    project_name: str,
    column_name: str,
    metric: str | None = typer.Option(
        None, "--metric", help="Only this metric's regressions. Defaults to every stored metric."
    ),
    select: list[str] | None = typer.Option(
        None,
        "--select",
        help="'<feature expression>:<metric name>' (repeatable) -- exactly these pairs, mutually "
        "exclusive with --metric.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Build a calibration-curve PDF report (one page per feature, per
    metric) for one CSV column's precomputed regressions, saved to
    <root>/reports/<column_name>_regressions.pdf."""
    try:
        path = api.generate_regression_report(
            project_name=project_name,
            column_name=column_name,
            metric_name=metric,
            select=_parse_select(select),
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Regressions report: {path}")


@project_app.command("apply-regressions")
def project_apply_regressions(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    columns: list[str] | None = typer.Option(
        None,
        "--columns",
        help="CSV column(s) to apply (repeatable). Defaults to every column with a "
        "computed regression.",
    ),
    select_features: list[str] | None = typer.Option(
        None,
        "--select-features",
        help="Feature expression(s) to apply (repeatable), e.g. --select-features B249. "
        "Defaults to every feature with a computed regression. Mutually exclusive with --select.",
    ),
    select: list[str] | None = typer.Option(
        None,
        "--select",
        help="'<feature expression>:<metric name>' (repeatable) -- exactly these pairs, mutually "
        "exclusive with --select-features.",
    ),
) -> None:
    """Apply every stored regression (or a narrower --columns/--select-features/
    --select subset) to sample_id's own raw per-scan feature values, inverting
    each fit to predict a per-pixel value of the external property, storing the
    predicted array and its own distribution stats in <root>/results.zarr."""
    try:
        result = api.apply_regressions(
            project_name=project_name,
            sample_id=sample_id,
            project_root=root,
            columns=columns,
            select_features=select_features,
            select=_parse_select(select),
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Applied regressions for {len(result.column_names)} column(s):")
    for column_name in result.column_names:
        typer.echo(f"  {column_name}")
    if result.skipped:
        typer.echo(f"Skipped: {', '.join(result.skipped)}")
    typer.echo(f"results: {result.results_path}")


@project_app.command("list-applied-regressions")
def project_list_applied_regressions(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every external CSV column with applied regressions for sample_id."""
    try:
        columns = api.list_applied_regressions(
            project_name=project_name, sample_id=sample_id, project_root=root
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not columns:
        typer.echo("No applied regressions computed yet.")
        return
    for column_name in columns:
        typer.echo(f"  {column_name}")


@project_app.command("get-applied-regression-results")
def project_get_applied_regression_results(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    column_name: str | None = typer.Option(
        None, "--column-name", help="Narrow to one CSV column. Defaults to every column."
    ),
    select: list[str] | None = typer.Option(
        None,
        "--select",
        help="'<feature expression>:<metric name>' (repeatable) -- exactly these pairs.",
    ),
) -> None:
    """Print mean/median/std of the predicted values for every applied
    (column, feature) pair of sample_id."""
    try:
        applied = api.get_applied_regression_results(
            project_name=project_name,
            sample_id=sample_id,
            project_root=root,
            column_name=column_name,
            select=_parse_select(select),
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Applied regressions for sample {sample_id!r}:")
    for item in applied:
        typer.echo(
            f"  {item.column_name}/{item.feature_id} ({item.metric_name}): "
            f"mean={item.distribution.mean:.4g} median={item.distribution.median:.4g} "
            f"std={item.distribution.std:.4g}"
        )


@project_app.command("generate-applied-regression-report")
def project_generate_applied_regression_report(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    select: list[str] | None = typer.Option(
        None,
        "--select",
        help="'<feature expression>:<metric name>' (repeatable) -- exactly these pairs.",
    ),
) -> None:
    """Build a PDF report (one page per applied (column, feature) pair) of
    sample_id's precomputed applied regressions, saved to
    <root>/reports/<sample_id>_applied_regressions.pdf."""
    try:
        path = api.generate_applied_regression_report(
            project_name=project_name,
            sample_id=sample_id,
            project_root=root,
            select=_parse_select(select),
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Applied regressions report: {path}")


@project_app.command("generate-feature-report")
def project_generate_feature_report(
    project_name: str,
    sample_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    save_pdf: bool = typer.Option(
        False, "--save-pdf", help="Also bundle everything into a PDF report (no LaTeX needed)."
    ),
) -> None:
    """Generate one combined diagnostics PNG per registered feature -- (A) a
    feature map, (B) diagnostic spectra (bare-peak features only), and (C) a
    distribution histogram -- saved to
    <root>/reports/img/features/<sample_id>/<feature_id>.png, plus a
    separate metrics-table PNG at
    <root>/reports/img/features/<sample_id>/<feature_id>_table.png."""
    try:
        result = api.generate_feature_report(
            project_name=project_name,
            sample_id=sample_id,
            project_root=root,
            save_pdf=save_pdf,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Generated {len(result.image_paths)} feature diagnostics "
        f"({len(result.table_paths)} with a metrics table):"
    )
    for path in result.image_paths:
        typer.echo(f"  {path}")
    if result.pdf_path is not None:
        typer.echo(f"PDF report: {result.pdf_path}")


@project_app.command("add-bootstrap-config")
def project_add_bootstrap_config(
    project_name: str,
    method: str,
    npix: int | None = typer.Option(
        None, "--npix", help="Number of scan positions per draw. Exactly one of npix/pct."
    ),
    pct: float | None = typer.Option(
        None,
        "--pct",
        help="Percentage of a sample's scans per draw, resolved to npix on first run. "
        "Exactly one of npix/pct.",
    ),
    n_iterations: int = typer.Option(100, "--n-iterations", help="Number of draws to compute."),
    base_seed: int = typer.Option(0, "--base-seed", help="Seed this run's draws are derived from."),
    shuffle: bool = typer.Option(
        False, "--shuffle", help="Permute values across positions before resampling."
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Register a bootstrap config (method, npix-or-pct, base_seed, shuffle), or grow its
    n_iterations in place if that exact identity is already registered."""
    try:
        config = api.add_bootstrap_config(
            project_name=project_name,
            method=method,
            npix=npix,
            pct=pct,
            n_iterations=n_iterations,
            base_seed=base_seed,
            shuffle=shuffle,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Bootstrap config {config.bootstrap_id!r} ready in project {project_name!r}: "
        f"method={config.method} npix={config.npix} pct={config.pct} "
        f"n_iterations={config.n_iterations} shuffle={config.shuffle}"
    )


@project_app.command("extend-bootstrap-config")
def project_extend_bootstrap_config(
    project_name: str,
    bootstrap_id: str,
    n_iterations: int,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Raise a bootstrap config's registered n_iterations target. Must not be
    called while a run against it is concurrently executing."""
    try:
        config = api.extend_bootstrap_config(
            project_name=project_name,
            bootstrap_id=bootstrap_id,
            n_iterations=n_iterations,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Bootstrap config {config.bootstrap_id!r} extended to {config.n_iterations}")


@project_app.command("list-bootstrap-configs")
def project_list_bootstrap_configs(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every bootstrap config registered with the project."""
    try:
        configs = api.list_bootstrap_configs(project_name=project_name, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not configs:
        typer.echo("No bootstrap configs registered.")
        return
    for config in configs:
        npix_or_pct = f"npix={config.npix}" if config.npix is not None else f"pct={config.pct}"
        typer.echo(
            f"  {config.bootstrap_id} (method={config.method}, {npix_or_pct}, "
            f"n_iterations={config.n_iterations}, base_seed={config.base_seed}, "
            f"shuffle={config.shuffle})"
        )


@project_app.command("run-bootstrap")
def project_run_bootstrap(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
    n_iterations: int | None = typer.Option(
        None,
        "--n-iterations",
        help="Extend the config's target to this many iterations before running, if greater.",
    ),
    n_processes: int = typer.Option(
        1,
        "--n-processes",
        help="Worker processes for this invocation's own share of pending iterations "
        "(e.g. $SLURM_CPUS_PER_TASK).",
    ),
    worker_index: int = typer.Option(
        0,
        "--worker-index",
        help="This invocation's shard index, for splitting a run across separately "
        "invoked processes (e.g. $SLURM_ARRAY_TASK_ID).",
    ),
    n_workers: int = typer.Option(
        1,
        "--n-workers",
        help="Total number of shards a run is split across (e.g. $SLURM_ARRAY_TASK_COUNT).",
    ),
    batchsize: int | None = typer.Option(
        None, "--batchsize", help="Batch size to stream ratio features' raw data in."
    ),
    save_csv: bool = typer.Option(
        False,
        "--save-csv",
        help="Also export every iteration's metrics to "
        "<root>/exports/<sample_id>_<bootstrap_id>.csv.",
    ),
) -> None:
    """Compute every not-yet-done iteration of a bootstrap config's registered
    target for a sample (resuming automatically). Requires run-pipeline to
    have already run for this sample."""
    try:
        result = api.run_bootstrap(
            project_name=project_name,
            sample_id=sample_id,
            bootstrap_id=bootstrap_id,
            project_root=root,
            n_iterations=n_iterations,
            n_processes=n_processes,
            worker_index=worker_index,
            n_workers=n_workers,
            batchsize=batchsize,
            save_csv=save_csv,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Bootstrap {result.bootstrap_id!r} on sample {result.sample_id!r}: "
        f"{result.n_done}/{result.n_iterations} done "
        f"({result.n_newly_computed} newly computed this run)"
    )
    if result.n_failed:
        typer.echo(f"  {result.n_failed} iteration(s) failed: {result.failed_indices}")
    typer.echo(f"  results: {result.results_path}")
    if result.csv_path is not None:
        typer.echo(f"  csv: {result.csv_path}")


@project_app.command("bootstrap-progress")
def project_bootstrap_progress(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Print (n_done/n_iterations) for a bootstrap run on one sample."""
    try:
        progress = api.get_bootstrap_progress(
            project_name=project_name,
            sample_id=sample_id,
            bootstrap_id=bootstrap_id,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"{progress.n_done}/{progress.n_iterations} done")


@project_app.command("bootstrap-status")
def project_bootstrap_status(
    project_name: str,
    bootstrap_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Print per-sample progress for one bootstrap config, across every
    sample with any results for it."""
    try:
        status = api.get_bootstrap_status(
            project_name=project_name, bootstrap_id=bootstrap_id, project_root=root
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not status:
        typer.echo("No samples have results for this bootstrap config yet.")
        return
    for sample_id, progress in status.items():
        typer.echo(f"  {sample_id}: {progress.n_done}/{progress.n_iterations} done")


@project_app.command("get-bootstrap-results")
def project_get_bootstrap_results(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Print a per-feature mean-across-completed-iterations summary of a
    bootstrap run's computed metrics."""
    try:
        results = api.get_bootstrap_results(
            project_name=project_name,
            sample_id=sample_id,
            bootstrap_id=bootstrap_id,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    n_done = int(results.completed.sum())
    typer.echo(
        f"Bootstrap {bootstrap_id!r} results for sample {sample_id!r} "
        f"({n_done}/{len(results.completed)} iterations done):"
    )
    done_values = results.values[results.completed]
    for i, feature_id in enumerate(results.feature_ids):
        typer.echo(f"  {feature_id}:")
        for j, metric_name in enumerate(results.metric_names):
            mean = done_values[:, i, j].mean() if n_done else float("nan")
            typer.echo(f"    {metric_name}: mean={mean:.4g}")


@project_app.command("export-bootstrap-csv")
def project_export_bootstrap_csv(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Export every iteration's metrics for a bootstrap run to
    <root>/exports/<sample_id>_<bootstrap_id>.csv -- one row per iteration
    (including not-yet-computed ones), one column per feature+metric pair."""
    try:
        path = api.export_bootstrap_csv(
            project_name=project_name,
            sample_id=sample_id,
            bootstrap_id=bootstrap_id,
            project_root=root,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Bootstrap CSV: {path}")


@project_app.command("pack")
def project_pack(
    project_name: str,
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Archive path. Defaults to <project_root>/results.zarr.zip.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Pack a project's results.zarr into a single .zarr.zip archive (still
    randomly readable, no unpack step needed) plus a .manifest.json
    provenance sidecar. Non-destructive -- the original directory is left
    untouched."""
    try:
        result = api.pack_project(project_name=project_name, out_path=out, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Packed {result.n_files_packed} files ({result.source_size_bytes} bytes) into "
        f"{result.out_path} ({result.size_bytes} bytes); manifest: {result.manifest_path}"
    )


@project_app.command("verify-pack")
def project_verify_pack(path: Path) -> None:
    """Sanity-check a .zarr.zip archive produced by `pylibs project pack`:
    validate every member's checksum and confirm it opens as a
    structurally intact zarr store."""
    result = api.verify_pack(path)
    if not result.ok:
        reason = result.bad_member or result.error or "unknown error"
        typer.echo(f"Error: {path} failed verification: {reason}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {result.n_members} members, {result.n_samples} samples")


@project_app.command("pack-snapshot")
def project_pack_snapshot(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Pack results.zarr into a new, timestamped snapshot under
    <project_root>/snapshots/ -- unlike `pack`, never overwrites a prior
    one (a colliding timestamp gets a -1/-2 suffix instead)
    snapshot. Meant for an iterative multi-machine workflow: pack a
    snapshot after each meaningful bit of progress, `use-snapshot` it
    elsewhere to keep working from exactly that state."""
    try:
        result = api.pack_snapshot(project_name=project_name, project_root=root)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Packed snapshot: {result.out_path} ({result.size_bytes} bytes)")


@project_app.command("list-snapshots")
def project_list_snapshots(
    project_name: str,
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """List every snapshot packed for this project, oldest first."""
    manifests = api.list_snapshots(project_name=project_name, project_root=root)
    if not manifests:
        typer.echo("No snapshots found")
        return
    for manifest in manifests:
        sha = manifest.git_sha[:8] if manifest.git_sha else "-"
        # resolved + flagged, same as `runs index` -- see resolved_archive_path
        archive_path = manifest.resolved_archive_path
        missing = "" if archive_path.exists() else "  [MISSING]"
        typer.echo(
            f"{manifest.packed_at_utc.isoformat()}  sha={sha}  "
            f"{manifest.archive_size_bytes} bytes  {archive_path}{missing}"
        )


@project_app.command("use-snapshot")
def project_use_snapshot(
    project_name: str,
    timestamp: str | None = typer.Option(
        None, "--timestamp", help="Snapshot to use. Defaults to the most recently packed one."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Discard an already-existing results.zarr to extract this snapshot over it.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Directory the project lives in. Defaults to ./projects/<project_name>.",
    ),
) -> None:
    """Extract a snapshot into <project_root>/results.zarr so DSL scripts/
    reports/further processing keep working from exactly that state. Fails
    if results.zarr already exists, unless --force is given -- pack your
    current state first if you want to keep it."""
    try:
        result = api.use_snapshot(
            project_name=project_name, timestamp=timestamp, project_root=root, force=force
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Using snapshot {result.snapshot_path} (packed {result.manifest.packed_at_utc})"
        + (" -- replaced existing results.zarr" if result.replaced_existing else "")
    )


runs_app = typer.Typer(name="runs", help="Inspect packed project archives.")
app.add_typer(runs_app)


@runs_app.command("index")
def runs_index(root: Path) -> None:
    """List every packed archive's manifest found anywhere under `root`."""
    manifests = api.index_packs(root)
    if not manifests:
        typer.echo(f"No manifests found under {root}")
        return
    for manifest in manifests:
        sha = manifest.git_sha[:8] if manifest.git_sha else "-"
        # the archive's real location, derived from where this manifest was
        # actually found -- not the path recorded at pack time, which is
        # relative to that run's working directory and goes stale on its own
        archive_path = manifest.resolved_archive_path
        # only genuinely absent archives are flagged; a healthy one listed
        # from another directory used to print a dead path with no hint
        missing = "" if archive_path.exists() else "  [MISSING]"
        typer.echo(
            f"{manifest.packed_at_utc.isoformat()}  {manifest.project_name:<30}  "
            f"sha={sha}  {manifest.archive_size_bytes} bytes  "
            f"{manifest.n_files_packed} files  {archive_path}{missing}"
        )


@app.command("convert-ele-to-libs")
def convert_ele_to_libs(
    ele_path: Path,
    libs_path: Path,
    capdata: int | None = typer.Option(
        None, "--capdata", help="Cap the total number of spectra converted."
    ),
    batchsize: int | None = typer.Option(
        None, "--batchsize", help="Batch size to stream the source .ELE file in."
    ),
) -> None:
    """Convert a .ELE file to the open .libs format (needs the private
    libs_importer package to read the source; the output needs it no
    longer)."""
    try:
        path = api.convert_ele_to_libs(
            ele_path=ele_path, libs_path=libs_path, capdata=capdata, batchsize=batchsize
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {path}")


@app.command("write-libs")
def write_libs(
    out_path: Path,
    csv_path: Path | None = typer.Option(
        None, "--csv", help="CSV spectral matrix: x,y then one column per wavelength."
    ),
    spectra_path: Path | None = typer.Option(
        None, "--spectra", help="(n_scans, n_pixels) array, .npy or .npz."
    ),
    wavelengths_path: Path | None = typer.Option(
        None, "--wavelengths", help="(n_pixels,) array, .npy or .npz."
    ),
    coordinates_path: Path | None = typer.Option(
        None, "--coordinates", help="(n_scans, 2) x/y positions, .npy or .npz."
    ),
    grid: str | None = typer.Option(
        None, "--grid", help="Raster shape as ROWSxCOLS, e.g. 41x21. Alternative to --coordinates."
    ),
    step: float | None = typer.Option(
        None, "--step", help="Pixel spacing, in --units. Gives maps physical axes."
    ),
    units: str = typer.Option("mm", "--units", help="Spatial units for --step."),
    order: str = typer.Option(
        "raster", "--order", help="Row order of the input: 'raster' or 'serpentine'."
    ),
    dtype: str = typer.Option("float64", "--dtype", help="Stored dtype: float64 or float32."),
    npz_key: str | None = typer.Option(
        None, "--npz-key", help="Member to read from a multi-array .npz (spectra only)."
    ),
) -> None:
    """Write a .libs sample file from arrays or a CSV spectral matrix.

    Two input shapes. Either --csv, holding one row per pixel with x and y
    columns and one column per wavelength (the header gives each column's
    wavelength), or --spectra with --wavelengths as .npy/.npz arrays.

    Position comes from --coordinates or --grid. A CSV carries its own
    coordinates, so neither is needed with it.
    """
    try:
        if csv_path is not None:
            if spectra_path is not None or wavelengths_path is not None:
                raise ProjectError("--csv cannot be combined with --spectra/--wavelengths")
            spectra, wavelengths, coordinates = api.load_csv_matrix(csv_path)
        else:
            if spectra_path is None or wavelengths_path is None:
                raise ProjectError("either --csv, or both --spectra and --wavelengths, is required")
            spectra = api.load_array(spectra_path, key=npz_key)
            wavelengths = api.load_array(wavelengths_path)
            coordinates = api.load_array(coordinates_path) if coordinates_path is not None else None

        parsed_grid = None
        if grid is not None:
            try:
                rows, cols = (int(part) for part in grid.lower().split("x"))
            except ValueError:
                raise ProjectError(f"--grid must look like ROWSxCOLS, got {grid!r}") from None
            parsed_grid = (rows, cols)
        if parsed_grid is None and coordinates is None:
            raise ProjectError("one of --grid or --coordinates is required")

        path = api.write_libs(
            out_path,
            spectra,
            wavelengths,
            grid=parsed_grid if coordinates is None else None,
            coordinates=coordinates,
            step=step,
            units=units,
            dtype=dtype,
            order=order,
        )
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote {path}")


@app.command("run")
def run(script: Path) -> None:
    """Run a pylibs DSL script."""
    try:
        run_script(script)
    except PylibsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
