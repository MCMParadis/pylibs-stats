"""Public façade for pylibs.

Every interface (CLI, DSL, future GUI/DB) calls through this module.
Nothing outside `core/` should import from `processing` or `reporting`
directly.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
import zarr
from pydantic import BaseModel

from pylibs.core.config import BootstrapConfig, FeatureConfig, ProjectConfig, SampleConfig
from pylibs.core.exceptions import (
    BootstrapError,
    CorrelationError,
    ProjectError,
    RegressionError,
    ResultsError,
)
from pylibs.core.features.expression import validate_expression
from pylibs.core.features.peak_table import (
    PeakEntry,
    PeakTableResult,
    elements_in_table,
    install_peak_table,
    load_project_peak_table,
    peaks_for_element,
)
from pylibs.core.io.convert import convert_ele_to_libs as _convert_ele_to_libs
from pylibs.core.io.pack import PackManifest, PackResult, VerifyResult
from pylibs.core.io.pack import list_manifests as _list_manifests
from pylibs.core.io.pack import pack_results as _pack_results
from pylibs.core.io.pack import verify_pack as _verify_pack
from pylibs.core.io.reader import SUPPORTED_EXTENSIONS, LibsReader, get_reader
from pylibs.core.io.snapshot import SNAPSHOTS_DIRNAME, SnapshotUseResult
from pylibs.core.io.snapshot import new_timestamp as _new_timestamp
from pylibs.core.io.snapshot import pack_snapshot_results as _pack_snapshot_results
from pylibs.core.io.snapshot import use_snapshot_results as _use_snapshot_results
from pylibs.core.io.store import RESULTS_DIRNAME, ResultsStore
from pylibs.core.logging import configure_console_logging, get_logger
from pylibs.core.logging import set_debug_logging as _set_debug_logging
from pylibs.core.pipeline.bootstrap.csv_export import export_bootstrap_csv as _export_bootstrap_csv
from pylibs.core.pipeline.bootstrap.resampling import get_resampling_strategy
from pylibs.core.pipeline.bootstrap.runner import BootstrapResults, BootstrapRunResult
from pylibs.core.pipeline.bootstrap.runner import run_bootstrap as _run_bootstrap
from pylibs.core.pipeline.correlations import CorrelationMatrices, compute_correlation_matrices
from pylibs.core.pipeline.distributions import Distribution1D, Distribution2D
from pylibs.core.pipeline.metrics import (
    METRIC_NAMES,
    METRIC_NAMES_1D,
    METRIC_NAMES_2D,
    feature_metric_values,
)
from pylibs.core.pipeline.regression import (
    AppliedRegression,
    FeatureRegression,
    compute_applied_regression,
    get_regression_algorithm,
)
from pylibs.core.pipeline.regression_run import run_regression_sweep
from pylibs.core.pipeline.runner import PipelineBatchResult, PipelineResult
from pylibs.core.pipeline.runner import run_pipeline as _run_pipeline
from pylibs.core.pipeline.runner import run_pipeline_batch as _run_pipeline_batch
from pylibs.core.project.filename_scheme import DEFAULT_SCHEME, compile_scheme, parse_stem
from pylibs.core.project.project import Project
from pylibs.core.project.sample_properties import load_sample_properties
from pylibs.core.reporting.applied_regression_report import build_applied_regression_report
from pylibs.core.reporting.correlations_report import TOP_N, build_correlations_report
from pylibs.core.reporting.regression_report import build_regression_report
from pylibs.core.reporting.report import (
    FeatureMapsResult,
    SpectrumStatsData,
    generate_feature_maps,
)

DEFAULT_PROJECTS_DIR = Path("projects")

logger = get_logger(__name__)


class FeatureResults:
    """Computed feature values for one sample, read back from the results
    store. `values` has shape (n_scans, len(feature_ids)); `feature_ids[i]`
    labels column i. `feature_names` maps feature_id to its display name."""

    def __init__(
        self,
        sample_id: str,
        feature_ids: list[str],
        feature_names: dict[str, str],
        values: np.ndarray,
    ):
        self.sample_id = sample_id
        self.feature_ids = feature_ids
        self.feature_names = feature_names
        self.values = values

    def get(self, feature_id: str) -> np.ndarray:
        """Column of values (shape (n_scans,)) for a single feature id."""
        if feature_id not in self.feature_ids:
            raise ResultsError(f"No feature {feature_id!r} in these results")
        return self.values[:, self.feature_ids.index(feature_id)]


class SpectrumStats:
    """Sample-wide min/max/mean spectrum, read back from the results store.
    All four arrays share shape (n_pix,) and line up with `wavelengths`."""

    def __init__(
        self,
        sample_id: str,
        wavelengths: np.ndarray,
        minimum: np.ndarray,
        maximum: np.ndarray,
        mean: np.ndarray,
    ):
        self.sample_id = sample_id
        self.wavelengths = wavelengths
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean


class SampleInspection(BaseModel):
    sample_id: str
    path: Path
    n_scans: int
    n_pix: int
    wavelength_min: float
    wavelength_max: float
    params: dict


class BootstrapProgress(BaseModel):
    sample_id: str
    bootstrap_id: str
    n_iterations: int
    n_done: int


class CorrelationsResult(BaseModel):
    project_name: str
    column_names: list[str]
    skipped_columns: list[str]
    feature_ids: list[str]
    metric_names: list[str]
    found_cells: (
        int  # (feature, metric) cells with an already-stored regression, across all columns
    )
    missing_cells: int  # ditto, with none -- left nan; run compute_regressions(metric=None) to fill
    results_path: Path


class RegressionsResult(BaseModel):
    project_name: str
    column_names: list[str]
    skipped_columns: list[str]
    feature_ids: list[str]
    metric_names: list[
        str
    ]  # every metric fit this call -- [metric], or all of METRIC_NAMES if None
    algorithm: str
    method: str
    sample_ids_used: list[str]
    skipped_samples: list[str]
    skipped_fits: list[str]
    # The subset of `skipped_fits` worth a caller's attention: a cell whose
    # metric was never computed for that feature (every 2D metric against a
    # non-ratio feature, say) can never be fit, so it is excluded here.
    unexpected_skipped_fits: list[str]
    results_path: Path


class AppliedRegressionsResult(BaseModel):
    project_name: str
    sample_id: str
    column_names: list[str]
    feature_ids: list[str]
    skipped: list[str]
    results_path: Path


def _letters_only_filename(text: str) -> str:
    """Collapse `text` to its letter-only "words" joined by underscores, e.g.
    "c_B / wt%" -> "c_B_wt" -- keeps correlation-report filenames short and
    plain regardless of what punctuation an external CSV's column header
    contains."""
    return "_".join(re.findall(r"[A-Za-z]+", text)) or "column"


def _resolve_feature_ids(project: Project, expressions: list[str]) -> list[str]:
    """Resolves feature *expressions* (as passed to `ADDFEATURE`) to their
    registered `feature_id`s by exact match, in order. Raises
    `RegressionError` listing every expression that isn't registered as any
    feature, rather than silently dropping it."""
    feature_ids = []
    unresolved = []
    for expression in expressions:
        match = next(
            (
                fid
                for fid, feature in project.registry.features.items()
                if feature.expression == expression
            ),
            None,
        )
        if match is None:
            unresolved.append(expression)
        else:
            feature_ids.append(match)
    if unresolved:
        raise RegressionError(
            f"expression(s) not registered as any feature: {unresolved!r} -- register them "
            "first (ADDFEATURE)."
        )
    return feature_ids


def _feature_regression_from_arrays(
    arrays: tuple[zarr.Array, zarr.Array, zarr.Array],
) -> FeatureRegression:
    """Builds a `FeatureRegression` from a `store.get_regression_arrays`
    result -- shared by every regression read-back path (`get_regression_results`'s
    default and `select`-filtered branches, `apply_regressions`)."""
    group_mean_x_array, group_mean_y_array, group_std_y_array = arrays
    attrs = group_mean_x_array.attrs
    return FeatureRegression(
        column_name=str(cast(str, attrs["column_name"])),
        feature_id=str(cast(str, attrs["feature_id"])),
        metric_name=str(cast(str, attrs["metric_name"])),
        algorithm=str(cast(str, attrs["algorithm"])),
        slope=float(cast(float, attrs["slope"])),
        intercept=float(cast(float, attrs["intercept"])),
        pearson_r=float(cast(float, attrs["pearson_r"])),
        r_squared=float(cast(float, attrs["r_squared"])),
        mae=float(cast(float, attrs["mae"])),
        p_value=float(cast(float, attrs["p_value"])),
        n=int(cast(int, attrs["n"])),
        x_mean=float(cast(float, attrs["x_mean"])),
        ssxx=float(cast(float, attrs["ssxx"])),
        residual_std=float(cast(float, attrs["residual_std"])),
        x_min=float(cast(float, attrs["x_min"])),
        x_max=float(cast(float, attrs["x_max"])),
        sample_names=cast(list[str], attrs["sample_names"]),
        group_mean_x=cast(np.ndarray, group_mean_x_array[:]),
        group_mean_y=cast(np.ndarray, group_mean_y_array[:]),
        group_std_y=cast(np.ndarray, group_std_y_array[:]),
    )


def _applied_regression_from_arrays(
    sample_id: str, arrays: tuple[zarr.Array, zarr.Array, zarr.Array]
) -> AppliedRegression:
    """Builds an `AppliedRegression` from a `store.get_applied_regression_arrays`
    result -- shared by `get_applied_regression_results`'s default and
    `select`-filtered branches."""
    predicted_array, counts_array, bin_edges_array = arrays
    attrs = predicted_array.attrs
    fitted_params: tuple[float, float, float] | None = None
    if attrs.get("fitted_amplitude") is not None:
        fitted_params = (
            float(cast(float, attrs["fitted_amplitude"])),
            float(cast(float, attrs["fitted_mu"])),
            float(cast(float, attrs["fitted_sigma"])),
        )
    distribution = Distribution1D(
        counts=cast(np.ndarray, counts_array[:]),
        bin_edges=cast(np.ndarray, bin_edges_array[:]),
        metrics={name: float(cast(float, attrs[name])) for name in METRIC_NAMES_1D},
        zero_count=int(cast(int, attrs["zero_count"])),
        fitted_params=fitted_params,
    )
    return AppliedRegression(
        sample_id=sample_id,
        column_name=str(cast(str, attrs["column_name"])),
        feature_id=str(cast(str, attrs["feature_id"])),
        metric_name=str(cast(str, attrs["metric_name"])),
        algorithm=str(cast(str, attrs["algorithm"])),
        predicted=cast(np.ndarray, predicted_array[:]),
        distribution=distribution,
        x_min=float(cast(float, attrs["x_min"])),
        x_max=float(cast(float, attrs["x_max"])),
    )


def set_debug_logging(enabled: bool) -> None:
    """Toggle DEBUG-level logging process-wide -- call once, before any
    other `core.api` function, so it's in effect for everything that
    follows (including commands that never touch a project, like
    `convert_ele_to_libs`). See `core.logging.set_debug_logging`."""
    _set_debug_logging(enabled)
    configure_console_logging(enabled)


def create_project(
    project_name: str, root: Path | None = None, description: str | None = None
) -> ProjectConfig:
    """Get the project at `root` (default `./projects/<project_name>`), creating it
    if it doesn't exist yet. Never overwrites an existing project."""
    if root is None:
        root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.create(project_name=project_name, root=root, description=description)
    return project.config


def use_peak_table(
    project_name: str,
    peak_table: Path,
    project_root: Path | None = None,
) -> PeakTableResult:
    """Register `peak_table` with project `project_name` (default root
    `./projects/<project_name>`) by **copying** it to
    `<project_root>/peak_table.csv`. Every later operation that resolves a
    peak id -- `add_feature`, `add_element_features`, `run_pipeline`,
    `run_pipeline_batch`, `generate_feature_report` -- reads that copy, so it
    must be called once before any of them (after `create_project`).

    Copying rather than referencing a path is what makes a project
    self-contained: `pack_snapshot`/`use_snapshot` carry the table to another
    machine, and editing or moving the source afterwards can't silently
    change what an existing project means. The flip side is that correcting a
    peak means editing your table and calling this again (or editing the
    project's own copy in place -- the file is the truth either way).

    Re-registering a different table simply replaces the copy; nothing is
    invalidated here. Reprocessing stays lazy and is detected on the next run
    from stored feature identity hashes -- `FeatureStep` recomputes its whole
    array, `Distribution1DStep`/`Distribution2DStep` only the features that
    actually changed, and `run_bootstrap` resets a run whose features moved,
    but only once `run_pipeline` has refreshed the features array it reads
    those hashes from."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    # load the project first, so calling this before create_project raises
    # ProjectError rather than dropping a CSV into an empty directory, and so
    # the copy is recorded in that project's own log
    Project.load(project_root)
    result = install_peak_table(project_root, Path(peak_table))
    if result.replaced_existing:
        # info, not warning: the console line is the interface's job (the CLI
        # echoes it, the DSL returns replaced_existing), and logging at WARNING
        # too printed the same fact twice -- once on stdout, once on stderr
        logger.info(
            "Replaced the peak table for project %r with %s (%d peak(s), elements: %s) -- any "
            "feature whose peak definitions changed will be recomputed on the next run_pipeline.",
            project_name,
            result.source,
            result.n_peaks,
            ", ".join(result.elements),
        )
    else:
        logger.info(
            "Registered peak table %s for project %r (%d peak(s), elements: %s)",
            result.source,
            project_name,
            result.n_peaks,
            ", ".join(result.elements),
        )
    return result


def add_sample(
    project_name: str,
    path: Path,
    sample_name: str | None = None,
    sample_stem: str | None = None,
    project_root: Path | None = None,
) -> SampleConfig:
    """Get the sample for `path` on project `project_name` (default root
    `./projects/<project_name>`), registering it if it isn't registered yet.
    The sample's id (e.g. 'sample001') is auto-assigned, so registering the
    same path again updates that same sample instead of adding a duplicate.

    `sample_name` and `sample_stem` each default independently to `path`'s
    stem. `sample_name` is the correlations join key (see
    `compute_correlations`) and can be shared across several samples (e.g.
    ablation-layer samples of one physical sample); `sample_stem` stays a
    unique per-sample label used for report titles."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    return project.add_sample(path, sample_name, sample_stem)


def add_samples_from_dir(
    project_name: str,
    directory: Path,
    scheme: str = DEFAULT_SCHEME,
    recursive: bool = False,
    project_root: Path | None = None,
) -> list[SampleConfig]:
    """Register every readable sample file in `directory` on project
    `project_name` (default root `./projects/<project_name>`), deriving each
    one's `sample_name` from its filename instead of taking it per call --
    the directory-wide counterpart to `add_sample`.

    `scheme` is a filename template (see `project.filename_scheme`), default
    `"{sample_name}_{layer}"`: the `{sample_name}` field becomes the sample's
    correlations join key, every other field is stored as its `name_parts`,
    and `sample_stem` is always the file's full stem. A malformed template
    raises `FilenameSchemeError` before any file is touched; a file whose stem
    the template can't parse is logged as an error and skipped, so one oddly
    named file doesn't abandon the rest.

    Only files whose extension some reader handles are considered
    (`io.reader.SUPPORTED_EXTENSIONS`, matched case-insensitively, so both
    `.libs` and `.ELE` count and e.g. a CSV sitting alongside them doesn't).
    `recursive` walks subdirectories too. Files are taken in sorted path
    order, so the sequential `sample_id`s a run assigns are reproducible.

    Returns every registered sample. Registration goes through the same
    `add_sample` path, so re-running is idempotent."""
    pattern = compile_scheme(scheme)
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    directory = Path(directory)
    if not directory.is_dir():
        raise ProjectError(f"No such directory to add samples from: {directory}")
    # load before scanning, so a missing project fails the same way here as in
    # every other entry point rather than passing quietly on an empty directory
    project = Project.load(project_root)

    candidates = sorted(
        path
        for path in (directory.rglob("*") if recursive else directory.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        logger.warning(
            "No sample files (%s) found in %s%s",
            ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            directory,
            "" if recursive else " -- pass recursive=True to search subdirectories",
        )
        return []

    # an already-registered sample reached by a differently-spelled path (say
    # a relative directory now given absolutely) must update that sample, not
    # add a second entry pointing at the same file -- which would silently
    # count it twice in every cross-sample fit
    registered_by_resolved = {
        sample.path.resolve(): sample.path for sample in project.registry.samples.values()
    }

    samples: list[SampleConfig] = []
    skipped = 0
    for path in candidates:
        fields = parse_stem(path.stem, pattern)
        if fields is None:
            logger.error(
                "Filename %r doesn't match scheme %r -- skipping. Register it with "
                "add_sample, or re-run with a scheme it fits.",
                path.name,
                scheme,
            )
            skipped += 1
            continue
        name_parts = {field: value for field, value in fields.items() if field != "sample_name"}
        samples.append(
            project.add_sample(
                registered_by_resolved.get(path.resolve(), path),
                sample_name=fields["sample_name"],
                sample_stem=path.stem,
                name_parts=name_parts,
            )
        )

    logger.info(
        "Registered %d sample(s) on project %r from %s (%d skipped, %d file(s) found)",
        len(samples),
        project_name,
        directory,
        skipped,
        len(candidates),
    )
    return samples


def add_feature(
    project_name: str,
    expression: str,
    feature_name: str | None = None,
    project_root: Path | None = None,
) -> FeatureConfig:
    """Get the feature for `expression` on project `project_name` (default root
    `./projects/<project_name>`), registering it if it isn't registered yet.
    The feature's id (e.g. 'feature001') is auto-assigned, so registering the
    same expression again updates that same feature instead of adding a
    duplicate.

    `expression` is validated against the project's own registered peak table
    (`use_peak_table`, which must have run first): any peak-id-shaped token
    that isn't a known id is logged as an error, but registration still
    proceeds -- it doesn't raise."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    # load the project *before* the peak table: a missing project is the more
    # fundamental failure, and reporting "no peak table registered" for a root
    # that holds no project at all just sends you after the wrong thing
    project = Project.load(project_root)
    validate_expression(expression, load_project_peak_table(project_root))
    return project.add_feature(expression, feature_name)


def add_element_features(
    project_name: str,
    element: str | list[str],
    project_root: Path | None = None,
    only_default: bool = False,
) -> list[FeatureConfig]:
    """Register every peak the project's registered peak table (`use_peak_table`)
    lists for `element` as its own bare-peak feature on project `project_name` (default
    root `./projects/<project_name>`) -- the element-wide counterpart to
    `add_feature`'s one expression at a time.

    `element` takes one name or a list of them, matched case-insensitively but
    exactly ('ca' finds 'Ca', never 'CaF'). `only_default` narrows each element
    to its one `_isdefault`-flagged peak instead of all of them.

    Returns every registered feature, in element-then-table order. Registration
    goes through the same `add_feature` path, so re-running is idempotent: an
    already-registered peak id yields its existing feature rather than a
    duplicate. An element that contributes nothing -- unknown, or with no
    flagged peak under `only_default` -- is skipped rather than raised on, so
    one such name doesn't abandon the rest. That's logged at INFO, i.e. to the
    project log but not the console: passing a standing list of elements of
    interest and letting the ones this peak table has no lines for fall away
    is a normal way to use this, not a mistake worth interrupting a run over.
    The returned list and the summary log line both say what was actually
    registered."""
    elements = [element] if isinstance(element, str) else list(element)
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    # project before peak table, same reasoning as add_feature
    project = Project.load(project_root)
    table = load_project_peak_table(project_root)

    features: list[FeatureConfig] = []
    skipped: list[str] = []
    for name in elements:
        entries = peaks_for_element(table, name, only_default=only_default)
        if not entries:
            skipped.append(name)
            all_entries = peaks_for_element(table, name) if only_default else []
            if all_entries:
                logger.info(
                    "No peak flagged as the default for element %r -- registering nothing for "
                    "it. Register its peaks individually, or drop only_default to add all %d.",
                    name,
                    len(all_entries),
                )
            else:
                logger.info(
                    "No peaks for element %r in the peak table -- registering nothing for it. "
                    "Known elements: %s",
                    name,
                    ", ".join(elements_in_table(table)),
                )
            continue
        features.extend(project.add_feature(entry.id) for entry in entries)

    logger.info(
        "Registered %d feature(s) on project %r for element(s): %s%s",
        len(features),
        project_name,
        ", ".join(name for name in elements if name not in skipped) or "none",
        f" (no peaks in the table for: {', '.join(skipped)})" if skipped else "",
    )
    return features


def list_samples(project_name: str, project_root: Path | None = None) -> list[SampleConfig]:
    """List every sample registered with `project_name` (default root
    `./projects/<project_name>`), ordered by sample_id."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    return sorted(project.registry.samples.values(), key=lambda sample: sample.sample_id)


def list_features(project_name: str, project_root: Path | None = None) -> list[FeatureConfig]:
    """List every feature registered with `project_name` (default root
    `./projects/<project_name>`), ordered by feature_id."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    return sorted(project.registry.features.values(), key=lambda feature: feature.feature_id)


def import_libs_file(
    path: Path, capdata: int | None = None, batchsize: int | None = None
) -> LibsReader:
    """Open a raw LIBS data file directly (reader chosen by file extension), with
    no project or registry involved. Gives full access to wavelengths, params,
    and spectra -- useful for ad hoc exploration of a file outside a project.

    `capdata` caps the total number of spectra read/reported. `batchsize` sets
    the default batch size used by `iter_window` when called without one."""
    return get_reader(path, capdata=capdata, batchsize=batchsize)


def convert_ele_to_libs(
    ele_path: Path,
    libs_path: Path,
    capdata: int | None = None,
    batchsize: int | None = None,
) -> Path:
    """Convert a `.ELE` file to the open `.libs` format (requires the private
    `libs_importer` package to read the source -- the output needs it no
    longer). No project or registry involved.

    Useful for producing a small, shareable, dependency-free sample from
    proprietary data -- not meant for production-scale conversions, since the
    whole result is held in memory before writing. `capdata`/`batchsize`
    control how the source is streamed while reading it.

    Returns `libs_path`."""
    _convert_ele_to_libs(Path(ele_path), Path(libs_path), capdata=capdata, batchsize=batchsize)
    return Path(libs_path)


def run_pipeline(
    project_name: str,
    sample_id: str,
    project_root: Path | None = None,
    capdata: int | None = None,
    batchsize: int | None = None,
    n_processes: int = 1,
) -> PipelineResult:
    """Run every pipeline step (feature computation, spectrum statistics) for
    `sample_id`, streaming through its reader in batches, and write results
    to `<project_root>/results.zarr` (default root `./projects/<project_name>`).
    Each step is skipped independently if its output already exists --
    reprocessing a sample from scratch means starting a new project.

    `n_processes` (opt-in, default 1) affects only the 1D/2D distribution-
    metrics steps -- each registered feature's histogram+metrics is an
    independent computation once feature values are stored, fanned out
    across a local worker pool; `<= 0` uses every CPU available to this
    process (see `core.runs.executor.resolve_n_processes`). Feature
    computation itself stays sequential (one streaming pass over the raw
    reader) -- for scaling across *samples* instead, see
    `run_pipeline_batch`, which deliberately does not nest with this
    parameter (see its own docstring)."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    return _run_pipeline(
        project_root=project_root,
        sample_id=sample_id,
        capdata=capdata,
        batchsize=batchsize,
        n_processes=n_processes,
    )


def run_pipeline_batch(
    project_name: str,
    sample_ids: list[str] | None = None,
    project_root: Path | None = None,
    capdata: int | None = None,
    batchsize: int | None = None,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
) -> PipelineBatchResult:
    """Run `run_pipeline` for every sample in `sample_ids` (default: every
    sample registered with `project_name`, default root
    `./projects/<project_name>`) -- the across-sample counterpart to
    `run_pipeline`, for projects with many samples (e.g. large,
    cluster-deployed datasets). Each sample's own pipeline is already
    idempotent, so re-running an already-processed sample is a cheap no-op.

    `n_processes` > 1 parallelizes this invocation's own share of samples
    across an in-process worker pool (`<= 0` uses every CPU available to
    this process, see `core.runs.executor.resolve_n_processes`).
    `worker_index`/`n_workers` statically partition the full sample list
    across separately-invoked processes (e.g. SLURM array-job tasks) with no
    runtime coordination beyond agreeing on `n_workers`; the two knobs
    compose. Each sample's own Distribution steps always run with
    `n_processes=1` regardless of this call's own `n_processes` -- see
    `pipeline.runner.run_pipeline_batch`'s docstring for why the two axes of
    parallelism deliberately don't nest."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    return _run_pipeline_batch(
        project_root=project_root,
        sample_ids=sample_ids,
        capdata=capdata,
        batchsize=batchsize,
        n_processes=n_processes,
        worker_index=worker_index,
        n_workers=n_workers,
    )


def get_results(
    project_name: str, sample_id: str, project_root: Path | None = None
) -> FeatureResults:
    """Read back the computed feature values for `sample_id` from
    `<project_root>/results.zarr` (default root `./projects/<project_name>`).
    Raises `ResultsError` if the sample hasn't been processed yet -- run
    `run_pipeline` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    array = store.get_features_array(sample_id)
    if array is None:
        raise ResultsError(
            f"No results for sample {sample_id!r} -- run the pipeline first (RUNPIPELINE)."
        )
    feature_ids = cast(list[str], array.attrs["feature_ids"])
    feature_names = {
        fid: project.registry.features[fid].feature_name or fid
        for fid in feature_ids
        if fid in project.registry.features
    }
    return FeatureResults(
        sample_id=sample_id,
        feature_ids=feature_ids,
        feature_names=feature_names,
        values=cast(np.ndarray, array[:]),
    )


def get_spectrum_stats(
    project_name: str, sample_id: str, project_root: Path | None = None
) -> SpectrumStats:
    """Read back the sample-wide min/max/mean spectrum for `sample_id` from
    `<project_root>/results.zarr` (default root `./projects/<project_name>`).
    Raises `ResultsError` if the sample hasn't been processed yet -- run
    `run_pipeline` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    array = store.get_spectrum_stats_array(sample_id)
    if array is None:
        raise ResultsError(
            f"No spectrum stats for sample {sample_id!r} -- run the pipeline first (RUNPIPELINE)."
        )
    wavelengths_array = store.get_wavelengths_array(sample_id)
    if wavelengths_array is None:
        raise ResultsError(
            f"No wavelengths for sample {sample_id!r} -- run the pipeline first (RUNPIPELINE)."
        )
    values = cast(np.ndarray, array[:])
    return SpectrumStats(
        sample_id=sample_id,
        wavelengths=cast(np.ndarray, wavelengths_array[:]),
        minimum=values[0],
        maximum=values[1],
        mean=values[2],
    )


def get_distribution_1d(
    project_name: str, sample_id: str, feature_id: str, project_root: Path | None = None
) -> Distribution1D:
    """Read back precomputed 1D distribution metrics (see
    `pipeline.metrics.METRIC_NAMES_1D`) for one feature of `sample_id` from
    `<project_root>/results.zarr` (default root `./projects/<project_name>`).
    Raises `ResultsError` if the sample hasn't been processed yet -- run
    `run_pipeline` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    arrays = store.get_distribution_1d_arrays(sample_id, feature_id)
    if arrays is None:
        raise ResultsError(
            f"No distribution_1d found in the store for feature {feature_id!r} of sample "
            f"{sample_id!r} -- run the pipeline first (RUNPIPELINE)."
        )
    counts_array, bin_edges_array = arrays
    attrs = counts_array.attrs
    fitted_params: tuple[float, float, float] | None = None
    if attrs.get("fitted_amplitude") is not None:
        fitted_params = (
            float(cast(float, attrs["fitted_amplitude"])),
            float(cast(float, attrs["fitted_mu"])),
            float(cast(float, attrs["fitted_sigma"])),
        )
    return Distribution1D(
        counts=cast(np.ndarray, counts_array[:]),
        bin_edges=cast(np.ndarray, bin_edges_array[:]),
        metrics={name: float(cast(float, attrs[name])) for name in METRIC_NAMES_1D},
        zero_count=int(cast(int, attrs["zero_count"])),
        fitted_params=fitted_params,
    )


def get_distribution_2d(
    project_name: str, sample_id: str, feature_id: str, project_root: Path | None = None
) -> Distribution2D:
    """Read back precomputed 2D distribution metrics (see
    `pipeline.metrics.METRIC_NAMES_2D`) for a ratio feature of `sample_id`
    from `<project_root>/results.zarr` (default root
    `./projects/<project_name>`). Raises `ResultsError` if the sample hasn't
    been processed yet, or if `feature_id` isn't a ratio feature -- run
    `run_pipeline` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    arrays = store.get_distribution_2d_arrays(sample_id, feature_id)
    if arrays is None:
        raise ResultsError(
            f"No distribution_2d found in the store for feature {feature_id!r} of sample "
            f"{sample_id!r} -- run the pipeline first (RUNPIPELINE), and check that it's a "
            "ratio feature."
        )
    (
        counts_array,
        bin_edges_num_array,
        bin_edges_den_array,
        display_counts_array,
        display_bin_edges_num_array,
        display_bin_edges_den_array,
    ) = arrays
    attrs = counts_array.attrs
    return Distribution2D(
        numerator_expression=str(cast(str, attrs["numerator_expression"])),
        denominator_expression=str(cast(str, attrs["denominator_expression"])),
        counts=cast(np.ndarray, counts_array[:]),
        bin_edges_num=cast(np.ndarray, bin_edges_num_array[:]),
        bin_edges_den=cast(np.ndarray, bin_edges_den_array[:]),
        metrics={name: float(cast(float, attrs[name])) for name in METRIC_NAMES_2D},
        shannon_entropy=float(cast(float, attrs["shannon_entropy"])),
        display_counts=cast(np.ndarray, display_counts_array[:]),
        display_bin_edges_num=cast(np.ndarray, display_bin_edges_num_array[:]),
        display_bin_edges_den=cast(np.ndarray, display_bin_edges_den_array[:]),
        numerator_bounds=(
            float(cast(float, attrs["numerator_bound_min"])),
            float(cast(float, attrs["numerator_bound_max"])),
        ),
        denominator_bounds=(
            float(cast(float, attrs["denominator_bound_min"])),
            float(cast(float, attrs["denominator_bound_max"])),
        ),
    )


def compute_correlations(
    project_name: str, csv_path: Path, project_root: Path | None = None
) -> CorrelationsResult:
    """Reshape every registered feature's already-stored regressions (see
    `compute_regressions`) into a (feature x metric) matrix per numeric
    column of `csv_path` -- **never fits anything itself**. For each
    (feature, metric) cell, reads that cell's already-stored regression
    (`regressions/<column>/<metric>/<feature_id>/`, see `pipeline.
    regression.fit_least_squares`, called only by `compute_regressions`); a
    cell with no stored regression is left `nan`, not recomputed -- run
    `compute_regressions(..., metric=None)` (an all-metrics sweep) first to
    populate every cell for a column. `csv_path` is only consulted for its
    column names (same file used for `compute_regressions`), never
    re-fit against.

    Stores a (feature x metric) Pearson-r, R², MAE (residual of a
    univariate OLS fit of the metric against the external property), and
    MAE's permutation-test p-value matrix per CSV column in
    `<project_root>/results.zarr`'s `correlations` group (default root
    `./projects/<project_name>`), fully overwriting any previously computed
    columns.

    Raises `CorrelationError` if `csv_path` has no numeric columns at all."""
    logger.info("Computing correlations for project %r against %s", project_name, csv_path)
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)

    properties = load_sample_properties(csv_path)
    if not properties.column_names:
        raise CorrelationError(
            f"No numeric column(s) found in {csv_path} -- nothing to correlate against."
        )
    feature_ids = sorted(project.registry.features)

    column_names: list[str] = []
    found_cells = 0
    missing_cells = 0
    missing_by_metric: Counter[str] = Counter()
    for column_name in properties.column_names:
        pearson_r = np.full((len(feature_ids), len(METRIC_NAMES)), np.nan)
        r_squared = np.full((len(feature_ids), len(METRIC_NAMES)), np.nan)
        mae = np.full((len(feature_ids), len(METRIC_NAMES)), np.nan)
        mae_p_value = np.full((len(feature_ids), len(METRIC_NAMES)), np.nan)

        for i, feature_id in enumerate(feature_ids):
            for j, metric_name in enumerate(METRIC_NAMES):
                arrays = store.get_regression_arrays(column_name, metric_name, feature_id)
                if arrays is None:
                    missing_cells += 1
                    missing_by_metric[metric_name] += 1
                    logger.debug(
                        "No regression found in the store for (column=%r, metric=%r, "
                        "feature=%r) -- leaving this correlation cell nan.",
                        column_name,
                        metric_name,
                        feature_id,
                    )
                    continue
                found_cells += 1
                group_mean_x_array, _, _ = arrays
                attrs = group_mean_x_array.attrs
                pearson_r[i, j] = float(cast(float, attrs["pearson_r"]))
                r_squared[i, j] = float(cast(float, attrs["r_squared"]))
                mae[i, j] = float(cast(float, attrs["mae"]))
                mae_p_value[i, j] = float(cast(float, attrs["p_value"]))

        matrices = compute_correlation_matrices(
            column_name, feature_ids, METRIC_NAMES, pearson_r, r_squared, mae, mae_p_value
        )
        store.create_correlation_arrays(
            column_name,
            matrices.feature_ids,
            matrices.metric_names,
            matrices.pearson_r,
            matrices.r_squared,
            matrices.mae,
            matrices.mae_p_value,
        )
        column_names.append(column_name)

    if missing_cells:
        breakdown = ", ".join(f"{name}: {count}" for name, count in missing_by_metric.most_common())
        # info, not warning -- a missing cell is routine (e.g. the 5 METRIC_NAMES_2D metrics
        # never apply to a non-ratio feature), not a problem; still worth having in
        # pylibs.log for anyone who wants to check the pattern, just not on the console
        logger.info(
            "%d/%d (feature, metric) cell(s) had no stored regression for project %r -- run "
            "COMPUTEREGRESSIONS with all_metrics=True to fill them. Missing by metric: %s",
            missing_cells,
            found_cells + missing_cells,
            project_name,
            breakdown,
        )
    logger.info(
        "Correlations finished for project %r: %d column(s), %d/%d cell(s) found in the store",
        project_name,
        len(column_names),
        found_cells,
        found_cells + missing_cells,
    )
    return CorrelationsResult(
        project_name=project_name,
        column_names=column_names,
        skipped_columns=properties.skipped_columns,
        feature_ids=feature_ids,
        metric_names=METRIC_NAMES,
        found_cells=found_cells,
        missing_cells=missing_cells,
        results_path=project_root / RESULTS_DIRNAME,
    )


def get_correlations(
    project_name: str, column_name: str, project_root: Path | None = None
) -> CorrelationMatrices:
    """Read back precomputed correlation matrices (Pearson r, R², MAE, MAE
    permutation-test p-value) for `column_name` from
    `<project_root>/results.zarr` (default root `./projects/<project_name>`).
    Raises `CorrelationError` if that column hasn't been computed yet -- run
    `compute_correlations` first (also raised for a column computed before
    this codebase stored `mae`/`mae_p_value`, requiring a rerun)."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    Project.load(project_root)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    arrays = store.get_correlation_arrays(column_name)
    if arrays is None:
        raise CorrelationError(
            f"No correlations computed for column {column_name!r} in project "
            f"{project_name!r} -- run compute_correlations (COMPUTECORRELATIONS) first."
        )
    pearson_array, r_squared_array, mae_array, mae_p_value_array = arrays
    attrs = pearson_array.attrs
    return CorrelationMatrices(
        column_name=str(cast(str, attrs["column_name"])),
        feature_ids=cast(list[str], attrs["feature_ids"]),
        metric_names=cast(list[str], attrs["metric_names"]),
        pearson_r=cast(np.ndarray, pearson_array[:]),
        r_squared=cast(np.ndarray, r_squared_array[:]),
        mae=cast(np.ndarray, mae_array[:]),
        mae_p_value=cast(np.ndarray, mae_p_value_array[:]),
    )


def list_correlations(project_name: str, project_root: Path | None = None) -> list[str]:
    """List every external CSV column with computed correlations for
    `project_name` (default root `./projects/<project_name>`), ordered
    alphabetically."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    return sorted(store.list_correlation_columns())


def generate_correlations_report(
    project_name: str,
    column_name: str,
    project_root: Path | None = None,
    top_n: int = TOP_N,
) -> Path:
    """Build a color-scaled, sorted best/worst PDF report over
    `column_name`'s precomputed correlation matrices (see
    `compute_correlations`/`get_correlations`), saved to
    `<project_root>/reports/<letters-only column_name>_correlations.pdf`
    (default root `./projects/<project_name>`; the filename keeps only
    `column_name`'s letter characters, e.g. "c_B / wt%" -> "c_B_wt", so it
    stays plain regardless of what punctuation the CSV header contains).
    Also saves each of the 8 tables as its own uncaptioned PNG (the PDF page
    still carries its caption), under
    `<project_root>/reports/img/correlations/<letters-only column_name>/`.
    Raises `CorrelationError` if that column hasn't been computed yet -- run
    `compute_correlations` first.

    Each of the report's eight tables (Pearson r, R², MAE, MAE p-value --
    best/worst) ranks its `top_n` rows by every feature's own most-extreme
    value among its metrics."""
    logger.info(
        "Generating correlations report for project %r, column %r", project_name, column_name
    )
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    matrices = get_correlations(project_name, column_name, project_root)

    feature_labels = {
        feature_id: project.registry.features[feature_id].feature_name or feature_id
        for feature_id in matrices.feature_ids
        if feature_id in project.registry.features
    }

    safe_column_name = _letters_only_filename(column_name)
    path = project_root / "reports" / f"{safe_column_name}_correlations.pdf"
    build_correlations_report(path, project_name, matrices, feature_labels, top_n)
    logger.info("Correlations report finished for project %r: %s", project_name, path)
    return path


def compute_regressions(
    project_name: str,
    csv_path: Path,
    project_root: Path | None = None,
    method: str = "univariate",
    select_features: list[str] | None = None,
    algorithm: str = "least_squares",
    metric: str | None = "Mean",
    n_permutations: int = 10000,
    random_seed: int = 0,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
) -> RegressionsResult:
    """Fit `metric` (default `'Mean'`, one of `pipeline.metrics.METRIC_NAMES`)
    of every selected feature against every numeric column of `csv_path` (a
    table of external per-sample scalar properties, joined on `sample_name`,
    default root `./projects/<project_name>`) via `algorithm` (default
    `'least_squares'`). `metric=None` fits **every** name in `METRIC_NAMES`
    for every selected feature instead of just one -- an "all-metrics
    sweep", the only thing that gives `compute_correlations` a fully
    populated (feature x metric) matrix to read back (it never fits
    anything itself -- see `pipeline.correlations`). A sample present in the
    CSV but never processed is skipped, not an error, and multiple samples
    sharing one `sample_name` (e.g. several ablation-layer files) each
    contribute their own data point against that shared CSV row's value
    (pseudo-replication caveat).

    `method` must be `'univariate'` (the only implemented method;
    `'multivariate'` is reserved and raises `RegressionError`).
    `select_features` is a list of feature *expressions* (as passed to
    `ADDFEATURE`), resolved to registered `feature_id`s by exact match;
    `None` means every registered feature. Raises `RegressionError` for an
    unknown `method`/`algorithm`/`metric`, an unresolvable `select_features`
    expression, or if no sample ends up usable at all.

    A feature whose fit turns out degenerate for a given (column, metric)
    (too few finite pairs, or constant x/y) is skipped, not fatal --
    recorded in the returned result's `skipped_fits`.

    Stores one regression per (CSV column, metric, feature) triple in
    `<project_root>/results.zarr`'s `regressions/<column>/<metric>/`
    groups. Pruning is scoped to one (column, metric) subtree: a feature
    stored under a metric being fit this call that's no longer in this
    call's `select_features` is removed; every other metric's own features
    (fit by an earlier call) are untouched -- so regressing different
    metrics for the same column (e.g. a `metric='Mean'` call, later a
    separate `metric='Median'` call, or one `metric=None` sweep) accumulates
    instead of overwriting each other.

    Each fit's `p_value` is a permutation-test p-value on its own residual
    MAE (shuffle the pairing, refit, recompute MAE; fraction of shuffled
    fits with a lower MAE than the real one), not an analytic test -- see
    `pipeline.regression.fit_least_squares`. `n_permutations` (default
    10000) controls its iteration count; `random_seed` (default 0) makes it
    reproducible -- each (column, metric, feature) cell gets its own
    independent seed derived from `random_seed` and the cell's own index
    (see `pipeline.regression_run`), so results are identical regardless of
    `n_processes`/`n_workers` or how many cells there are to fit.

    The actual fitting -- one independent unit of work per (column, metric,
    feature) cell -- is parallelized like `run_bootstrap`: `n_processes` > 1
    parallelizes this invocation's own share of cells across an in-process
    worker pool (`<= 0` uses every CPU available to this process, see
    `core.runs.executor.resolve_n_processes`); `worker_index`/`n_workers`
    statically partition the full cell range across separately-invoked
    processes (e.g. SLURM array-job tasks) with no runtime coordination
    beyond agreeing on `n_workers`; the two knobs compose."""
    if method != "univariate":
        raise RegressionError(
            f"method={method!r} isn't implemented yet -- only 'univariate' is supported."
        )
    get_regression_algorithm(algorithm)  # raises RegressionError if unknown, fail fast
    if metric is not None and metric not in METRIC_NAMES:
        raise RegressionError(f"Unknown metric {metric!r}; must be one of {METRIC_NAMES}")
    metrics_to_fit = METRIC_NAMES if metric is None else [metric]

    logger.info(
        "Computing regressions for project %r against %s (metric=%s)",
        project_name,
        csv_path,
        "all" if metric is None else metric,
    )
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)

    if select_features is None:
        feature_ids = sorted(project.registry.features)
    else:
        feature_ids = _resolve_feature_ids(project, select_features)

    sample_ids_by_name: dict[str, list[str]] = {}
    for sample in project.registry.samples.values():
        name = sample.sample_name or sample.sample_id
        sample_ids_by_name.setdefault(name, []).append(sample.sample_id)

    properties = load_sample_properties(csv_path)

    sample_ids_used: list[str] = []
    sample_names_used: list[str] = []
    skipped_samples: list[str] = []
    metric_rows: list[np.ndarray] = []
    used_row_indices: list[int] = []

    for row_index, name in enumerate(properties.sample_names):
        sample_ids = sample_ids_by_name.get(name)
        if not sample_ids:
            skipped_samples.append(f"{name} (no matching sample)")
            continue

        for sample_id in sample_ids:
            if not store.has_distribution_1d(sample_id):
                skipped_samples.append(f"{sample_id} (not processed)")
                logger.debug(
                    "No distribution_1d found in the store for sample %r -- skipping it for "
                    "this regression run.",
                    sample_id,
                )
                continue

            row = np.full((len(feature_ids), len(metrics_to_fit)), np.nan)
            for feature_index, feature_id in enumerate(feature_ids):
                try:
                    distribution_1d = get_distribution_1d(
                        project_name, sample_id, feature_id, project_root
                    )
                except ResultsError:
                    logger.debug(
                        "No distribution_1d found in the store for (sample=%r, feature=%r) -- "
                        "skipping it for this regression run.",
                        sample_id,
                        feature_id,
                    )
                    continue
                try:
                    distribution_2d = get_distribution_2d(
                        project_name, sample_id, feature_id, project_root
                    )
                except ResultsError:
                    distribution_2d = None
                values = feature_metric_values(distribution_1d, distribution_2d)
                row[feature_index] = [values[m] for m in metrics_to_fit]

            metric_rows.append(row)
            sample_ids_used.append(sample_id)
            sample_names_used.append(name)
            used_row_indices.append(row_index)

    if not metric_rows:
        raise RegressionError(
            f"No usable samples found for {csv_path} -- every referenced sample is either "
            "unregistered or not yet processed (run the pipeline first)."
        )

    metric_tensor = np.stack(metric_rows, axis=0)  # (n_points, n_features, n_metrics_to_fit)
    x_matrix = properties.values[used_row_indices, :]  # (n_points, n_csv_columns)

    sweep_result = run_regression_sweep(
        store,
        properties.column_names,
        metrics_to_fit,
        feature_ids,
        metric_tensor,
        x_matrix,
        sample_names_used,
        algorithm,
        n_permutations,
        random_seed,
        n_processes=n_processes,
        worker_index=worker_index,
        n_workers=n_workers,
    )

    logger.info(
        "Regressions finished for project %r: %d column(s), %d metric(s), %d sample(s) used, "
        "%d skipped, %d fit(s) skipped",
        project_name,
        len(sweep_result.column_names),
        len(metrics_to_fit),
        len(sample_ids_used),
        len(skipped_samples),
        len(sweep_result.skipped_fits),
    )
    return RegressionsResult(
        project_name=project_name,
        column_names=sweep_result.column_names,
        skipped_columns=properties.skipped_columns,
        feature_ids=feature_ids,
        metric_names=metrics_to_fit,
        algorithm=algorithm,
        method=method,
        sample_ids_used=sample_ids_used,
        skipped_samples=skipped_samples,
        skipped_fits=sweep_result.skipped_fits,
        unexpected_skipped_fits=sweep_result.unexpected_skipped_fits,
        results_path=project_root / RESULTS_DIRNAME,
    )


def get_regression_results(
    project_name: str,
    column_name: str,
    metric_name: str | None = None,
    select: list[tuple[str, str]] | None = None,
    project_root: Path | None = None,
) -> list[FeatureRegression]:
    """Read back regressions for `column_name` (default root
    `./projects/<project_name>`). Three modes, mutually exclusive
    (`RegressionError` if both `metric_name` and `select` are given):

    - Default (`metric_name=None`, `select=None`): every stored metric's (a
      feature regressed under several metrics appears once per metric).
    - `metric_name`: only that metric's, for every feature that has it.
    - `select`: a list of `(feature_expression, metric_name)` pairs (feature
      expressions as passed to `ADDFEATURE`), returned in the given order --
      lets one call pick an arbitrary, mixed-metric set of calibration
      curves (e.g. one feature's `"Mean"` alongside another's `"Median"`)
      out of a broad sweep, e.g. for a single curated report. Since this
      function is inherently single-column, a requested pair with no stored
      regression is a real error, not a cross-column ambiguity (contrast
      `apply_regressions`'s own, lenient `select`) -- raises `RegressionError`
      listing every missing pair.

    Also raises `RegressionError` if nothing has been computed for that
    column (and metric, if given) at all -- run `compute_regressions`
    first."""
    if metric_name is not None and select is not None:
        raise RegressionError("metric_name and select are mutually exclusive.")
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)

    if select is not None:
        feature_ids = _resolve_feature_ids(project, [expression for expression, _ in select])
        pairs = list(zip(feature_ids, (name for _, name in select), strict=True))
        results = []
        missing = []
        for feature_id, sel_metric_name in pairs:
            arrays = store.get_regression_arrays(column_name, sel_metric_name, feature_id)
            if arrays is None:
                missing.append(f"{feature_id}/{sel_metric_name}")
                continue
            results.append(_feature_regression_from_arrays(arrays))
        if missing:
            raise RegressionError(
                f"select has pair(s) with no stored regression for column {column_name!r} in "
                f"project {project_name!r}: {missing!r} -- run compute_regressions first."
            )
        return results

    metric_names = (
        [metric_name] if metric_name is not None else store.list_regression_metrics(column_name)
    )
    results = []
    for name in metric_names:
        for feature_id in store.list_regression_features(column_name, name):
            arrays = store.get_regression_arrays(column_name, name, feature_id)
            assert arrays is not None  # feature_id came from list_regression_features itself
            results.append(_feature_regression_from_arrays(arrays))
    if not results:
        raise RegressionError(
            f"No regressions computed for column {column_name!r}"
            f"{f' and metric {metric_name!r}' if metric_name is not None else ''} in project "
            f"{project_name!r} -- run compute_regressions (COMPUTEREGRESSIONS) first."
        )
    return results


def list_regressions(project_name: str, project_root: Path | None = None) -> list[str]:
    """List every external CSV column with computed regressions for
    `project_name` (default root `./projects/<project_name>`), ordered
    alphabetically."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    return sorted(store.list_regression_columns())


def generate_regression_report(
    project_name: str,
    column_name: str,
    metric_name: str | None = None,
    select: list[tuple[str, str]] | None = None,
    project_root: Path | None = None,
) -> Path:
    """Build a calibration-curve PDF (one page per feature, per metric) over
    `column_name`'s precomputed regressions (see
    `compute_regressions`/`get_regression_results`) -- every stored metric's
    by default, just `metric_name`'s if given, or exactly the `(feature_
    expression, metric_name)` pairs in `select` if given (mutually exclusive
    with `metric_name`) -- e.g. picking a curated, mixed-metric set of
    calibration curves (one feature's `"Mean"`, another's `"Median"`, ...)
    out of a broad all-metrics sweep into one report, which otherwise stores
    -- and would report -- every feature under every metric at once. Saved
    to `<project_root>/reports/<letters-only column_name>_regressions.pdf`
    (default root `./projects/<project_name>`; the filename keeps only
    `column_name`'s letter characters, matching
    `generate_correlations_report`'s own filename convention) -- or
    `..._<letters-only metric_name>_regressions.pdf` when `metric_name` is
    given, so calling this once per metric for the same column never
    overwrites a previous call's PDF (`select`, spanning several metrics by
    design, keeps the plain filename). Also saves each calibration-curve
    page as its own uncaptioned PNG (the PDF page still carries its
    caption), under `<project_root>/reports/img/regressions/<letters-only
    column_name>/`. Raises `RegressionError` if that column (and metric, or
    every `select` pair, if given) hasn't been computed yet -- run
    `compute_regressions` first."""
    logger.info("Generating regression report for project %r, column %r", project_name, column_name)
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    regressions = get_regression_results(
        project_name,
        column_name,
        metric_name=metric_name,
        select=select,
        project_root=project_root,
    )

    feature_labels = {
        regression.feature_id: (
            project.registry.features[regression.feature_id].feature_name or regression.feature_id
        )
        for regression in regressions
        if regression.feature_id in project.registry.features
    }

    safe_column_name = _letters_only_filename(column_name)
    filename_stem = safe_column_name
    if metric_name is not None:
        filename_stem = f"{safe_column_name}_{_letters_only_filename(metric_name)}"
    path = project_root / "reports" / f"{filename_stem}_regressions.pdf"
    build_regression_report(path, project_name, column_name, regressions, feature_labels)
    logger.info("Regression report finished for project %r: %s", project_name, path)
    return path


def apply_regressions(
    project_name: str,
    sample_id: str,
    project_root: Path | None = None,
    columns: list[str] | None = None,
    select_features: list[str] | None = None,
    select: list[tuple[str, str]] | None = None,
) -> AppliedRegressionsResult:
    """Applies every currently-stored regression (or a narrower `columns`/
    `select_features`/`select` subset) to `sample_id`'s own raw per-scan
    feature values, inverting each fit to predict a per-pixel value of the
    external property (e.g. turn a Mean-intensity feature into a predicted
    concentration map) -- the same sample doesn't need to have been one of
    the ones used to fit the regression. A regressed feature whose raw
    values aren't available in this particular sample (e.g. registered
    after this sample was processed) is skipped, not fatal -- recorded in
    the returned result's `skipped`.

    `columns` defaults to every column with any stored regression;
    `select_features` (feature *expressions*, as passed to `ADDFEATURE`)
    defaults to every feature with a stored regression for each column --
    every metric it was stored under (e.g. from an all-metrics sweep) gets
    applied. `select` is a more precise alternative, mutually exclusive with
    `select_features`: a list of `(feature_expression, metric_name)` pairs,
    applying only those exact combinations. A pair with no stored regression
    for a given considered column is skipped for that column, not an error
    (unlike `get_regression_results`'s own `select`, which is single-column
    and therefore has no such ambiguity) -- one column's own `skipped`
    entry, since the same pair may validly apply to a different column.
    Raises `RegressionError` for an unknown `columns` entry, an unresolvable
    `select_features`/`select` expression, both `select_features` and
    `select` given, or if nothing at all has been computed yet.

    Stores each (column, metric, feature) triple's predicted array and its
    own fresh 1D distribution stats in `<project_root>/results.zarr`'s
    `<sample_id>/applied_regressions/<column>/<metric>/` group.
    Re-running `compute_regressions` later does NOT automatically refresh a
    previously-applied prediction -- call `apply_regressions` again after
    any refit to update it."""
    if select is not None and select_features is not None:
        raise RegressionError("select and select_features are mutually exclusive.")
    logger.info("Applying regressions for sample %r in project %r", sample_id, project_name)
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    results = get_results(project_name, sample_id, project_root)

    if columns is None:
        column_names = store.list_regression_columns()
    else:
        unknown_columns = [c for c in columns if c not in store.list_regression_columns()]
        if unknown_columns:
            raise RegressionError(
                f"No regressions computed for column(s) {unknown_columns!r} in project "
                f"{project_name!r} -- run compute_regressions (COMPUTEREGRESSIONS) first."
            )
        column_names = list(columns)

    if not column_names:
        raise RegressionError(
            f"No regressions computed at all for project {project_name!r} -- run "
            "compute_regressions (COMPUTEREGRESSIONS) first."
        )

    resolved_feature_ids: list[str] | None = None
    if select_features is not None:
        resolved_feature_ids = _resolve_feature_ids(project, select_features)

    resolved_select: list[tuple[str, str]] | None = None  # (feature_id, metric_name) pairs
    if select is not None:
        select_feature_ids = _resolve_feature_ids(project, [expression for expression, _ in select])
        resolved_select = list(
            zip(select_feature_ids, (metric_name for _, metric_name in select), strict=True)
        )

    out_of_range = set(store.get_out_of_range_feature_ids(sample_id))

    applied_column_names: list[str] = []
    applied_feature_ids: set[str] = set()
    skipped: list[str] = []
    for column_name in column_names:
        if not store.list_regression_metrics(column_name):
            continue  # nothing currently registered for this column
        if resolved_select is not None:
            regressions = []
            for feature_id, metric_name in resolved_select:
                arrays = store.get_regression_arrays(column_name, metric_name, feature_id)
                if arrays is None:
                    skipped.append(
                        f"{column_name}/{feature_id}/{metric_name} "
                        "(no stored regression for this (feature, metric) pair)"
                    )
                    continue
                regressions.append(_feature_regression_from_arrays(arrays))
        else:
            regressions = get_regression_results(
                project_name, column_name, project_root=project_root
            )
            if resolved_feature_ids is not None:
                regressions = [r for r in regressions if r.feature_id in resolved_feature_ids]

        any_applied = False
        for regression in regressions:
            if regression.feature_id in out_of_range:
                skipped.append(
                    f"{column_name}/{regression.feature_id} "
                    "(feature out of this sample's wavelength range)"
                )
                continue
            try:
                raw_values = results.get(regression.feature_id)
            except ResultsError:
                skipped.append(
                    f"{column_name}/{regression.feature_id} (feature not in this sample's results)"
                )
                continue

            applied = compute_applied_regression(sample_id, regression, raw_values)
            distribution = applied.distribution
            fitted_params = distribution.fitted_params
            attributes = {
                **distribution.metrics,
                "column_name": applied.column_name,
                "feature_id": applied.feature_id,
                "metric_name": applied.metric_name,
                "algorithm": applied.algorithm,
                "zero_count": distribution.zero_count,
                "fitted_amplitude": fitted_params[0] if fitted_params is not None else None,
                "fitted_mu": fitted_params[1] if fitted_params is not None else None,
                "fitted_sigma": fitted_params[2] if fitted_params is not None else None,
                "x_min": applied.x_min,
                "x_max": applied.x_max,
            }
            store.create_applied_regression_arrays(
                sample_id,
                column_name,
                regression.metric_name,
                regression.feature_id,
                applied.predicted,
                distribution.counts,
                distribution.bin_edges,
                attributes,
            )
            applied_feature_ids.add(regression.feature_id)
            any_applied = True

        if any_applied:
            applied_column_names.append(column_name)

    logger.info(
        "Applied regressions finished for sample %r: %d column(s), %d feature(s), %d skipped",
        sample_id,
        len(applied_column_names),
        len(applied_feature_ids),
        len(skipped),
    )
    return AppliedRegressionsResult(
        project_name=project_name,
        sample_id=sample_id,
        column_names=applied_column_names,
        feature_ids=sorted(applied_feature_ids),
        skipped=skipped,
        results_path=project_root / RESULTS_DIRNAME,
    )


def get_applied_regression_results(
    project_name: str,
    sample_id: str,
    project_root: Path | None = None,
    column_name: str | None = None,
    select: list[tuple[str, str]] | None = None,
) -> list[AppliedRegression]:
    """Read back every applied regression for `sample_id` (or just
    `column_name`'s, if given) from `<project_root>/results.zarr` (default
    root `./projects/<project_name>`) -- or exactly the `(feature_expression,
    metric_name)` pairs in `select`, if given, applied leniently per
    considered column: a pair simply not applied for a given column is
    absent from that column's results, not an error (mirrors
    `apply_regressions`'s own `select`, and for the same reason -- this
    function can span several columns at once). Raises `ResultsError` if
    nothing has been applied yet -- run `apply_regressions` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)
    store = ResultsStore(project_root / RESULTS_DIRNAME)

    if column_name is not None:
        column_names = [column_name]
    else:
        column_names = store.list_applied_regression_columns(sample_id)

    resolved_select: list[tuple[str, str]] | None = None
    if select is not None:
        select_feature_ids = _resolve_feature_ids(project, [expression for expression, _ in select])
        resolved_select = list(
            zip(select_feature_ids, (metric_name for _, metric_name in select), strict=True)
        )

    results: list[AppliedRegression] = []
    for name in column_names:
        if resolved_select is not None:
            for feature_id, metric_name in resolved_select:
                arrays = store.get_applied_regression_arrays(
                    sample_id, name, metric_name, feature_id
                )
                if arrays is None:
                    continue
                results.append(_applied_regression_from_arrays(sample_id, arrays))
        else:
            for applied_metric_name in store.list_applied_regression_metrics(sample_id, name):
                for feature_id in store.list_applied_regression_features(
                    sample_id, name, applied_metric_name
                ):
                    arrays = store.get_applied_regression_arrays(
                        sample_id, name, applied_metric_name, feature_id
                    )
                    assert arrays is not None  # came from list_applied_regression_features itself
                    results.append(_applied_regression_from_arrays(sample_id, arrays))

    if not results:
        raise ResultsError(
            f"No applied regressions computed for sample {sample_id!r} in project "
            f"{project_name!r} -- run apply_regressions (APPLYREGRESSIONS) first."
        )
    return results


def list_applied_regressions(
    project_name: str, sample_id: str, project_root: Path | None = None
) -> list[str]:
    """List every external CSV column with applied regressions for
    `sample_id` (default root `./projects/<project_name>`), ordered
    alphabetically."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    project.get_sample(sample_id)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    return sorted(store.list_applied_regression_columns(sample_id))


def generate_applied_regression_report(
    project_name: str,
    sample_id: str,
    project_root: Path | None = None,
    select: list[tuple[str, str]] | None = None,
) -> Path:
    """Build a PDF (one page per (column, feature) pair) of `sample_id`'s
    precomputed applied regressions (see `apply_regressions`/
    `get_applied_regression_results`): a spatially-resolved map and a 1D
    distribution of each predicted external property, saved to
    `<project_root>/reports/<sample_id>_applied_regressions.pdf` (default
    root `./projects/<project_name>`). `select` (a list of `(feature_
    expression, metric_name)` pairs) narrows this to exactly those, the same
    as `get_applied_regression_results`'s own `select` -- useful after
    applying a broad all-metrics sweep, which otherwise reports every
    applied (feature, metric) pair at once. Also saves, per (column,
    feature) pair, a map+distribution PNG and a separate metrics-table PNG
    (neither captioned -- the PDF page still carries both), under
    `<project_root>/reports/img/applied_regressions/<sample_id>/
    <letters-only column_name>/` -- the column gets its own directory so two
    differently-punctuated column names can never collide. Raises
    `ResultsError` if nothing has been applied yet -- run `apply_regressions`
    first."""
    logger.info(
        "Generating applied regression report for project %r, sample %r", project_name, sample_id
    )
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    sample = project.get_sample(sample_id)
    applied = get_applied_regression_results(project_name, sample_id, project_root, select=select)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    params = store.get_raw_params(sample_id)
    if params is None:
        raise ResultsError(
            f"No raw params for sample {sample_id!r} -- run the pipeline first (RUNPIPELINE)."
        )

    feature_labels = {
        item.feature_id: (
            project.registry.features[item.feature_id].feature_name or item.feature_id
        )
        for item in applied
        if item.feature_id in project.registry.features
    }

    path = project_root / "reports" / f"{sample_id}_applied_regressions.pdf"
    sample_stem = sample.sample_stem or sample_id
    build_applied_regression_report(
        path,
        project_name,
        sample_id,
        sample_stem,
        applied,
        params,
        feature_labels,
        mask=store.get_mask_array(sample_id),
    )
    logger.info(
        "Applied regression report finished for project %r, sample %r: %s",
        project_name,
        sample_id,
        path,
    )
    return path


def generate_feature_report(
    project_name: str,
    sample_id: str,
    project_root: Path | None = None,
    save_pdf: bool = False,
) -> FeatureMapsResult:
    """Generate one feature-map PNG per registered feature for `sample_id`,
    saved to `<project_root>/reports/img/features/<sample_id>/<feature_id>.png`
    (default root `./projects/<project_name>`). Requires the sample to
    already be processed -- run `run_pipeline` first.

    Also saves a distribution panel per feature (if distribution stats have
    been computed) and, for features whose expression is exactly one bare
    peak id (e.g. "Li610", not a ratio like "Fe438 / Fe266"), a
    diagnostic-spectra panel (if spectrum stats have been computed) -- or,
    for a ratio feature, a joint-density plot (if 2D distribution stats have
    been computed) -- plus a separate metrics-table PNG (if distribution
    stats have been computed) at
    `<project_root>/reports/img/features/<sample_id>/<feature_id>_table.png`.

    If `save_pdf` is set, also bundles a cover page, a parameters table, and
    every feature map into `<project_root>/reports/<sample_id>_features.pdf`
    -- pure matplotlib, no LaTeX needed."""
    logger.info("Generating feature report for project %r, sample %r", project_name, sample_id)
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    sample = project.get_sample(sample_id)
    results = get_results(project_name, sample_id, project_root)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    params = store.get_raw_params(sample_id)
    wavelengths_array = store.get_wavelengths_array(sample_id)
    if params is None or wavelengths_array is None:
        raise ResultsError(
            f"No raw params/wavelengths for sample {sample_id!r} -- run the pipeline first "
            "(RUNPIPELINE)."
        )
    wavelengths = cast(np.ndarray, wavelengths_array[:])

    out_of_range = set(store.get_out_of_range_feature_ids(sample_id))
    feature_ids = [fid for fid in results.feature_ids if fid not in out_of_range]

    table = load_project_peak_table(project_root)
    peak_entries: dict[str, PeakEntry | None] = {
        fid: table.get(project.registry.features[fid].expression.strip())
        for fid in feature_ids
        if fid in project.registry.features
    }

    spectrum_stats_data = None
    try:
        stats = get_spectrum_stats(project_name, sample_id, project_root)
        spectrum_stats_data = SpectrumStatsData(
            wavelengths=stats.wavelengths,
            minimum=stats.minimum,
            maximum=stats.maximum,
            mean=stats.mean,
        )
    except ResultsError:
        pass

    def _get_distribution_1d(feature_id: str) -> Distribution1D | None:
        try:
            return get_distribution_1d(project_name, sample_id, feature_id, project_root)
        except ResultsError:
            return None

    def _get_distribution_2d(feature_id: str) -> Distribution2D | None:
        try:
            return get_distribution_2d(project_name, sample_id, feature_id, project_root)
        except ResultsError:
            return None

    result = generate_feature_maps(
        sample_id=sample_id,
        sample_name=sample.sample_stem or sample_id,
        project_name=project_name,
        params=params,
        wavelengths=wavelengths,
        n_scans=results.values.shape[0],
        feature_ids=feature_ids,
        feature_names=results.feature_names,
        get_values=results.get,
        reports_dir=project_root / "reports",
        peak_entries=peak_entries,
        spectrum_stats=spectrum_stats_data,
        get_distribution_1d=_get_distribution_1d,
        get_distribution_2d=_get_distribution_2d,
        save_pdf=save_pdf,
        mask=store.get_mask_array(sample_id),
    )
    logger.info(
        "Feature report finished for project %r, sample %r: %d feature(s)",
        project_name,
        sample_id,
        len(result.image_paths),
    )
    return result


def inspect_sample(
    project_name: str, sample_id: str, project_root: Path | None = None
) -> SampleInspection:
    """Open sample `sample_id`'s raw data file (reader chosen by file extension)
    and report basic info: scan/pixel counts and wavelength range."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    sample = project.get_sample(sample_id)
    reader = get_reader(sample.path)
    wavelengths = reader.wavelengths
    return SampleInspection(
        sample_id=sample.sample_id,
        path=sample.path,
        n_scans=reader.n_scans,
        n_pix=len(wavelengths),
        wavelength_min=float(wavelengths.min()),
        wavelength_max=float(wavelengths.max()),
        params=reader.params,
    )


def add_bootstrap_config(
    project_name: str,
    method: str,
    npix: int | None = None,
    pct: float | None = None,
    n_iterations: int = 100,
    base_seed: int = 0,
    shuffle: bool = False,
    project_root: Path | None = None,
) -> BootstrapConfig:
    """Get the bootstrap config for (method, npix-or-pct, base_seed, shuffle)
    on project `project_name` (default root `./projects/<project_name>`),
    registering it if it isn't registered yet. The config's id (e.g.
    'bootstrap001') is auto-assigned; re-registering the same identity with a
    larger `n_iterations` extends it in place.

    Exactly one of `npix`/`pct` must be given. `method` must be a registered
    resampling strategy (see
    `pipeline.bootstrap.resampling.RESAMPLING_STRATEGIES`). A 'local,
    no-shuffle' sweep and a 'local, with-shuffle' sweep are different
    identities and so get different ids, tracked as fully independent,
    independently resumable runs (see `run_bootstrap`)."""
    if (npix is None) == (pct is None):
        raise BootstrapError("Exactly one of npix or pct must be given")
    get_resampling_strategy(method)  # raises BootstrapError if unknown
    if method == "local" and npix is not None:
        side = math.isqrt(npix)
        if side * side != npix:
            raise BootstrapError(f"method='local' requires npix to be a perfect square, got {npix}")
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    return project.add_bootstrap_config(method, npix, pct, n_iterations, base_seed, shuffle)


def extend_bootstrap_config(
    project_name: str, bootstrap_id: str, n_iterations: int, project_root: Path | None = None
) -> BootstrapConfig:
    """Raise `bootstrap_id`'s registered `n_iterations` target (default root
    `./projects/<project_name>`) -- `run_bootstrap` picks this new target up
    on its next invocation and fills the newly added slots. Must not be
    called while a run against `bootstrap_id` is concurrently executing.
    Raises `BootstrapError` if `n_iterations` isn't strictly greater than the
    current value."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    return project.extend_bootstrap_config(bootstrap_id, n_iterations)


def list_bootstrap_configs(
    project_name: str, project_root: Path | None = None
) -> list[BootstrapConfig]:
    """List every bootstrap config registered with `project_name` (default
    root `./projects/<project_name>`), ordered by bootstrap_id."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    return sorted(project.registry.bootstrap_configs.values(), key=lambda c: c.bootstrap_id)


def run_bootstrap(
    project_name: str,
    sample_id: str,
    bootstrap_id: str,
    project_root: Path | None = None,
    n_iterations: int | None = None,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
    batchsize: int | None = None,
    save_csv: bool = False,
) -> BootstrapRunResult:
    """Run every not-yet-computed iteration of `bootstrap_id`'s registered
    target for `sample_id` (default root `./projects/<project_name>`).
    Requires the sample to already be processed -- run `run_pipeline` first.

    Resuming a partially-done run and extending a fully-done one are both
    just "call this again": resume happens automatically; extend happens
    automatically if `n_iterations` is given and greater than the config's
    current target (this bumps the registered target via
    `extend_bootstrap_config` first, then proceeds).

    `n_processes` > 1 parallelizes this invocation's own share of pending
    iterations across an in-process worker pool (`<= 0` uses every CPU
    available to this process, see `core.runs.executor.resolve_n_processes`).
    `worker_index`/`n_workers` statically partition the run's pending
    iterations across several *separately invoked* processes (e.g. SLURM
    array-job tasks) -- each invocation claims a disjoint slice and writes to
    the same results store with no coordination beyond agreeing on
    `n_workers`; these two knobs compose (each shard can itself use
    `n_processes`).

    If `save_csv` is set, also exports every iteration's metrics to
    `<project_root>/exports/<sample_id>_<bootstrap_id>.csv` (see
    `export_bootstrap_csv` to re-export standalone, without rerunning)."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    if n_iterations is not None:
        project = Project.load(project_root)
        config = project.get_bootstrap_config(bootstrap_id)
        if n_iterations > config.n_iterations:
            project.extend_bootstrap_config(bootstrap_id, n_iterations)
    return _run_bootstrap(
        project_root=project_root,
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        n_processes=n_processes,
        worker_index=worker_index,
        n_workers=n_workers,
        batchsize=batchsize,
        save_csv=save_csv,
    )


def get_bootstrap_progress(
    project_name: str, sample_id: str, bootstrap_id: str, project_root: Path | None = None
) -> BootstrapProgress:
    """Read back (n_done, n_iterations) for `bootstrap_id` on `sample_id`
    (default root `./projects/<project_name>`). Raises `BootstrapError` if
    that run hasn't been started yet -- run `run_bootstrap` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    progress = store.get_bootstrap_progress(sample_id, bootstrap_id)
    if progress is None:
        raise BootstrapError(
            f"No bootstrap run {bootstrap_id!r} for sample {sample_id!r} -- "
            "run run_bootstrap (RUNBOOTSTRAP) first."
        )
    n_done, n_iterations = progress
    return BootstrapProgress(
        sample_id=sample_id, bootstrap_id=bootstrap_id, n_iterations=n_iterations, n_done=n_done
    )


def get_bootstrap_results(
    project_name: str, sample_id: str, bootstrap_id: str, project_root: Path | None = None
) -> BootstrapResults:
    """Read back every computed metrics row for `bootstrap_id` on
    `sample_id` (default root `./projects/<project_name>`), plus which
    iterations are actually done (`completed`). Raises `BootstrapError` if
    that run hasn't been started yet -- run `run_bootstrap` first."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    arrays = store.get_bootstrap_arrays(sample_id, bootstrap_id)
    if arrays is None:
        raise BootstrapError(
            f"No bootstrap run {bootstrap_id!r} for sample {sample_id!r} -- "
            "run run_bootstrap (RUNBOOTSTRAP) first."
        )
    metrics, completed = arrays
    return BootstrapResults(
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        feature_ids=cast(list[str], metrics.attrs["feature_ids"]),
        metric_names=cast(list[str], metrics.attrs["metric_names"]),
        values=cast(np.ndarray, metrics[:]),
        completed=cast(np.ndarray, completed[:]).astype(bool),
    )


def export_bootstrap_csv(
    project_name: str, sample_id: str, bootstrap_id: str, project_root: Path | None = None
) -> Path:
    """Export every iteration's metrics for `bootstrap_id` on `sample_id`
    (default root `./projects/<project_name>`) to
    `<project_root>/exports/<sample_id>_<bootstrap_id>.csv` -- one row per
    iteration (including not-yet-computed ones, as blank cells), one column
    per feature+metric pair. Callable standalone, any time after
    `run_bootstrap` has been called at least once -- does not itself run
    anything (see `run_bootstrap`'s own `save_csv` to export automatically
    right after running). Raises `BootstrapError` if that run hasn't been
    started yet."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    arrays = store.get_bootstrap_arrays(sample_id, bootstrap_id)
    if arrays is None:
        raise BootstrapError(
            f"No bootstrap run {bootstrap_id!r} for sample {sample_id!r} -- "
            "run run_bootstrap (RUNBOOTSTRAP) first."
        )
    metrics, completed = arrays
    feature_ids = cast(list[str], metrics.attrs["feature_ids"])
    feature_labels = {
        feature_id: project.registry.features[feature_id].feature_name or feature_id
        for feature_id in feature_ids
        if feature_id in project.registry.features
    }
    return _export_bootstrap_csv(
        project_root,
        sample_id,
        bootstrap_id,
        feature_ids,
        feature_labels,
        cast(list[str], metrics.attrs["metric_names"]),
        cast(np.ndarray, metrics[:]),
        cast(np.ndarray, completed[:]),
    )


def get_bootstrap_status(
    project_name: str, bootstrap_id: str, project_root: Path | None = None
) -> dict[str, BootstrapProgress]:
    """Per-sample progress for one registered bootstrap config, across every
    sample that has any results for it (default root
    `./projects/<project_name>`)."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    project = Project.load(project_root)
    store = ResultsStore(project_root / RESULTS_DIRNAME)
    status: dict[str, BootstrapProgress] = {}
    for sample_id in project.registry.samples:
        progress = store.get_bootstrap_progress(sample_id, bootstrap_id)
        if progress is None:
            continue
        n_done, n_iterations = progress
        status[sample_id] = BootstrapProgress(
            sample_id=sample_id,
            bootstrap_id=bootstrap_id,
            n_iterations=n_iterations,
            n_done=n_done,
        )
    return status


def pack_project(
    project_name: str, out_path: Path | None = None, project_root: Path | None = None
) -> PackResult:
    """Pack `<project_root>/results.zarr` (default root
    `./projects/<project_name>`) into a single `.zarr.zip` archive (default
    `<project_root>/results.zarr.zip`) plus a `.manifest.json` provenance
    sidecar -- still randomly readable via zarr's own `ZipStore`, no unpack
    step needed. Non-destructive: the original directory is left untouched;
    delete it yourself once `verify_pack` confirms the archive is good."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    if out_path is None:
        out_path = project_root / f"{RESULTS_DIRNAME}.zip"
    return _pack_results(project_root / RESULTS_DIRNAME, out_path, project_name)


def verify_pack(path: Path) -> VerifyResult:
    """Sanity-check a `.zarr.zip` archive produced by `pack_project`:
    validate every member's checksum and confirm it opens as a
    structurally intact zarr store."""
    return _verify_pack(path)


def index_packs(root: Path) -> list[PackManifest]:
    """Every `*.manifest.json` (written by `pack_project`/`pack_snapshot`)
    found anywhere under `root`, parsed and sorted by `packed_at_utc`."""
    return _list_manifests(Path(root))


def pack_snapshot(project_name: str, project_root: Path | None = None) -> PackResult:
    """Pack `<project_root>/results.zarr` (default root
    `./projects/<project_name>`) into a new, timestamped
    `<project_root>/snapshots/<UTC timestamp>.zarr.zip` archive plus its own
    `.manifest.json` sidecar -- unlike `pack_project`, never overwrites a
    prior snapshot: the timestamp carries milliseconds, and a name that
    somehow still collides is disambiguated with a `-1`/`-2` suffix rather
    than truncating the archive already there. Meant for an iterative
    multi-machine workflow: pack a snapshot after each meaningful bit of
    progress, `use_snapshot` it elsewhere to keep working from exactly that
    state."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    return _pack_snapshot_results(project_root, project_name, _new_timestamp())


def list_snapshots(project_name: str, project_root: Path | None = None) -> list[PackManifest]:
    """Every snapshot packed for this project (default root
    `./projects/<project_name>`), sorted by `packed_at_utc`."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    return _list_manifests(project_root / SNAPSHOTS_DIRNAME)


def use_snapshot(
    project_name: str,
    timestamp: str | None = None,
    project_root: Path | None = None,
    force: bool = False,
) -> SnapshotUseResult:
    """Extract a snapshot (the most recent one, if `timestamp` is omitted)
    into `<project_root>/results.zarr` (default root
    `./projects/<project_name>`), so DSL scripts/reports/further processing
    keep working from exactly that state. Raises `SnapshotError` if
    `results.zarr` already exists and `force` isn't set -- pack your current
    state first (`pack_snapshot`) if you want to keep it, or pass
    `force=True` to discard it."""
    if project_root is None:
        project_root = DEFAULT_PROJECTS_DIR / project_name
    return _use_snapshot_results(project_root, timestamp, force)
