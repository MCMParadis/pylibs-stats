"""Zarr v3 results store: one store per project (`<project_root>/results.zarr`),
holding a `<sample_id>/features` array per processed sample -- shape
(n_scans, n_features), with the feature-id column order kept as an
array attribute -- a `<sample_id>/spectrum_stats` array -- shape
(3, n_pix), rows ordered min/max/mean -- a `<sample_id>/wavelengths` array
(shape (n_pix,)), a `<sample_id>/detectors` array (shape (n_detectors,
n_pix), bool, with a `detector_ids` attribute labeling each row -- which raw
per-pixel detector-channel mask it is, when the raw file has one), and a
`raw_params` attribute on the sample's own group (the raw file's own
`params` dict, scalar entries only -- e.g. HeightPixels/WidthPixels/
Resolution/AnalysisDate -- its own bulky array-valued entries, like a
wavelength-calibration array or detector masks, are what `wavelengths`/
`detectors` store properly instead) -- all cached once by `SpectrumStatsStep`
(which already has the raw file open) so report generation never needs to
reopen it just for this header metadata -- a `<sample_id>/ratio_components/
<feature_id>_numerator`/`..._denominator` array pair (each shape (n_scans,),
+1-shifted -- see `features.expression`'s own `_eval` `Div` handling) for
every ratio feature (`features.expression.split_ratio(expression) is not
None`), computed once by `FeatureStep` alongside the "features" array itself
and read back by `Distribution2DStep`/bootstrap rather than ever
re-evaluated from the raw spectrum -- a `<sample_id>/distribution_1d/
<feature_id>_counts` histogram array (every `pipeline.metrics.METRIC_NAMES_1D`
value stored directly as an attribute, keyed by its own display name, plus
`zero_count`/`fitted_amplitude`/`fitted_mu`/`fitted_sigma`) plus its matching
`..._bin_edges` array, and, for ratio features, a `<sample_id>/
distribution_2d/<feature_id>_counts` joint-histogram array (every
`METRIC_NAMES_2D` value likewise stored as an attribute, plus
`numerator_expression`/`denominator_expression`/`shannon_entropy` (raw,
unnormalized)/`numerator_bound_min`/`numerator_bound_max`/
`denominator_bound_min`/`denominator_bound_max`) plus its matching
`..._bin_edges_num`/`..._bin_edges_den` arrays and a `..._display_counts`
array (a second joint histogram windowed to each side's own skew-adjusted
boxplot-whisker fence, see `pipeline.distributions.skew_adjusted_bounds`,
for report rendering -- not used for the stored metrics) plus its matching
`..._display_bin_edges_num`/
`..._display_bin_edges_den` arrays. See `pipeline.metrics`/`pipeline.
distributions` for what each metric means and how to add one.

Also holds a `regressions/<quoted column name>/<quoted metric name>/
<feature_id>/` group per (external CSV column, metric, feature) triple with
a computed univariate regression (see `pipeline.regression`) -- top level,
not under a sample_id, since it's a cross-sample analysis over the whole
project's already-computed per-sample metrics. `metric_name` is an explicit
path segment (not just an attribute) specifically so the *same* feature can
hold a regression under every metric simultaneously -- e.g. `compute_
regressions(..., metric=None)` (an "all-metrics sweep") fits and stores one
regression per (feature, metric) pair for a column, which is what lets
`correlations/` (below) be pure formatting rather than its own computation.
Each feature group holds `group_mean_x`/`group_mean_y`/`group_std_y` arrays
(one entry per distinct `sample_name`, for plotting one error-barred point
per physical sample), with every other `FeatureRegression` field (`slope`,
`intercept`, `pearson_r`, `r_squared`, `mae`, `p_value`, `n`, `x_mean`,
`ssxx`, `residual_std`, `x_min`, `x_max`, `metric_name`, `algorithm`,
`sample_names`) as attributes on `group_mean_x` -- `p_value` is a
permutation-test p-value on `mae`, not an analytic one (see
`pipeline.regression.fit_least_squares`). `x_min`/`x_max` are the fit's own
quantification limits (the finite-masked x range the OLS fit was actually
computed over), not derived from `group_mean_x`'s own (coarser, grouped)
range. Pruning is scoped to one (column, metric) subtree: re-running
`compute_regressions` for a given metric only prunes a feature that's gone
stale under that *same* metric (no longer in a narrower `select_features`),
leaving every other metric's own features untouched (see
`delete_regression_feature`).

Also holds a `correlations/<quoted column name>/` group per external
sample-properties CSV column -- also top-level, for the same cross-sample
reason as `regressions/`. Each holds `pearson`/`r_squared`/`mae`/
`mae_p_value` arrays, shape (n_features, n_metrics), with `feature_ids`/
`metric_names`/`column_name` as attributes on `pearson`. Unlike
`regressions/`, nothing here is ever computed by `core.pipeline` itself --
`compute_correlations` purely reads each (feature, metric) cell's already-
stored regression (see above) and reshapes it into these four matrices; a
cell with no stored regression is `nan`, logged, never recomputed (see
`pipeline.correlations`).

Also holds a `<sample_id>/applied_regressions/<quoted column name>/
<feature_id>/` group per (sample, external CSV column, feature) combination
that a stored regression has been *applied* to (see
`pipeline.regression.compute_applied_regression`) -- unlike `regressions/`,
this lives under the sample_id, since a fitted regression is applied to one
sample's own raw per-scan feature values at a time, inverting the fit to
predict a per-pixel value of the external property. Holds `predicted` (the
per-scan predicted array; carries `column_name`/`feature_id`/`metric_name`/
`algorithm` plus every `pipeline.metrics.METRIC_NAMES_1D` value (keyed by
display name, matching `distribution_1d`'s own storage exactly) and
`zero_count`/`fitted_amplitude`/`fitted_mu`/`fitted_sigma` as attributes,
plus the underlying regression's own `x_min`/`x_max` quantification limits,
copied over redundantly here since
this array is reconstructed straight from its own attributes at report time,
never re-joined with `regressions/`), plus `counts`/`bin_edges` (the
histogram of `predicted`, no attributes). Re-running `compute_regressions`
does not automatically invalidate a stale `applied_regressions` entry fit
from an old slope/intercept -- `apply_regressions` must be explicitly
re-run to refresh it, the same as `distribution_1d`/`distribution_2d` don't
auto-invalidate their own downstream consumers either.

Finally holds a `<sample_id>/bootstrap/<bootstrap_id>/` group per registered
`BootstrapConfig` run against that sample (see `pipeline.bootstrap`) -- a
`metrics` array, shape `(n_iterations, n_features, n_metrics)` (with
`feature_ids`/`metric_names`/`method`/`resolved_npix`/`shuffle`/`base_seed`
as attributes), and a parallel `completed` int8 array, shape
`(n_iterations,)`, marking which iterations have actually been computed --
this is the checkpoint that makes a bootstrap run resumable/extendable
across many sessions (see `core.runs.checkpoint`). Both are chunked with
chunk size 1 along the iteration axis, which is what makes concurrent writes
from separate shard/worker processes safe (zarr v3 has no locking; disjoint
chunks can never collide).
"""

from pathlib import Path
from typing import cast
from urllib.parse import quote, unquote

import numpy as np
import zarr

from pylibs.core.runs.checkpoint import get_or_create_checkpoint_arrays

RESULTS_DIRNAME = "results.zarr"
CORRELATIONS_GROUP = "correlations"
REGRESSIONS_GROUP = "regressions"
APPLIED_REGRESSIONS_GROUP = "applied_regressions"
BOOTSTRAP_GROUP = "bootstrap"
BOOTSTRAP_METRICS_NAME = "metrics"
BOOTSTRAP_COMPLETED_NAME = "completed"


def _require_group_concurrent(parent: zarr.Group, name: str) -> zarr.Group:
    """`parent.require_group(name)`, tolerating another process creating the
    same group at the same moment.

    zarr's `require_group` is check-then-create ("get it; on KeyError, create
    it"), so two separately-invoked processes racing for the *same* group name
    -- every `run_bootstrap` shard shares `<sample>/bootstrap/<bootstrap_id>`,
    every `compute_regressions` shard shares `regressions/<column>/<metric>`
    -- can both miss, and the loser's create raises `ContainsGroupError`
    instead of returning the winner's group. Nothing is corrupted (zarr's
    LocalStore writes each node atomically); the run just dies.

    This is the same non-atomicity `ResultsStore.__init__` already handles for
    the root group, and `core.runs.checkpoint` handles one layer below for
    arrays -- the group layer in between was the gap between them.

    A single re-read, not a retry loop, matching both of those. A loop would
    also be unsafe here: `reset_bootstrap_run` deletes a run group, and every
    sharded worker calls it on a feature-hash mismatch, so retrying could
    re-create a group a sibling is mid-delete and turn that interleaving into
    a silently half-populated run. Better it surfaces. For the same reason a
    re-read that doesn't find a group re-raises rather than papering over
    `ContainsArrayError`/`ContainsArrayAndGroupError`, which mean something
    else entirely."""
    try:
        return parent.require_group(name)
    except zarr.errors.ContainsGroupError:
        existing = parent.get(name)
        if not isinstance(existing, zarr.Group):
            raise
        return existing


class ResultsStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        try:
            self._group = zarr.open_group(str(self.root), mode="a")
        except zarr.errors.ContainsGroupError:
            # `mode="a"` isn't atomic: it first checks whether a group
            # already exists at this path, and only creates one if not --
            # two separate processes constructing a ResultsStore at the
            # same not-yet-existing root concurrently (e.g. run_pipeline_
            # batch's worker_index/n_workers workers, each racing to be the
            # first to process any sample at all in a brand new project)
            # can both pass that check before either has finished creating
            # it, and the loser hits this instead of a clean open. Safe to
            # just open what the winner created -- every ResultsStore
            # instance for the same root is equivalent.
            self._group = zarr.open_group(str(self.root), mode="a")

    def has_features(self, sample_id: str) -> bool:
        return self.get_features_array(sample_id) is not None

    def get_features_array(self, sample_id: str) -> zarr.Array | None:
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        array = sample_group.get("features")
        return array if isinstance(array, zarr.Array) else None

    def create_features_array(
        self,
        sample_id: str,
        n_scans: int,
        feature_ids: list[str],
        feature_hashes: dict[str, str],
        out_of_range_feature_ids: list[str] | None = None,
    ) -> zarr.Array:
        sample_group = self._group.require_group(sample_id)
        return sample_group.create_array(
            "features",
            shape=(n_scans, len(feature_ids)),
            dtype="float64",
            attributes={
                "feature_ids": feature_ids,
                "feature_hashes": feature_hashes,
                "out_of_range_feature_ids": out_of_range_feature_ids or [],
            },
            overwrite=True,
        )

    def get_out_of_range_feature_ids(self, sample_id: str) -> list[str]:
        """Feature ids whose expression referenced a peak outside this
        sample's wavelength range (see `FeatureStep.run`) -- computed as NaN
        with no downstream stats stored, rather than a real (if degenerate)
        result. Empty for a sample processed before this attribute existed."""
        array = self.get_features_array(sample_id)
        if array is None:
            return []
        return cast(list[str], array.attrs.get("out_of_range_feature_ids", []))

    def write_chunk(self, array: zarr.Array, start: int, values: np.ndarray) -> None:
        array[start : start + values.shape[0]] = values

    def has_spectrum_stats(self, sample_id: str) -> bool:
        return self.get_spectrum_stats_array(sample_id) is not None

    def get_spectrum_stats_array(self, sample_id: str) -> zarr.Array | None:
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        array = sample_group.get("spectrum_stats")
        return array if isinstance(array, zarr.Array) else None

    def create_spectrum_stats_array(
        self, sample_id: str, minimum: np.ndarray, maximum: np.ndarray, mean: np.ndarray
    ) -> zarr.Array:
        sample_group = self._group.require_group(sample_id)
        array = sample_group.create_array(
            "spectrum_stats",
            shape=(3, minimum.shape[0]),
            dtype="float64",
            attributes={"stats": ["min", "max", "mean"]},
            overwrite=True,
        )
        array[0] = minimum
        array[1] = maximum
        array[2] = mean
        return array

    def get_wavelengths_array(self, sample_id: str) -> zarr.Array | None:
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        array = sample_group.get("wavelengths")
        return array if isinstance(array, zarr.Array) else None

    def create_wavelengths_array(self, sample_id: str, wavelengths: np.ndarray) -> zarr.Array:
        sample_group = self._group.require_group(sample_id)
        array = sample_group.create_array(
            "wavelengths", shape=wavelengths.shape, dtype="float64", overwrite=True
        )
        array[:] = wavelengths
        return array

    def create_detectors_array(
        self, sample_id: str, detectors: dict[str, list[bool]]
    ) -> zarr.Array:
        try:
            detector_ids = sorted(detectors, key=int)
        except ValueError:
            detector_ids = sorted(detectors)
        sample_group = self._group.require_group(sample_id)
        array = sample_group.create_array(
            "detectors",
            shape=(len(detector_ids), len(detectors[detector_ids[0]])),
            dtype="bool",
            attributes={"detector_ids": detector_ids},
            overwrite=True,
        )
        array[:] = np.array([detectors[detector_id] for detector_id in detector_ids])
        return array

    def create_mask_array(self, sample_id: str, mask: np.ndarray) -> zarr.Array:
        """Store the raw file's scanned-cell mask (shape (n_rows, n_cols), True
        where a scan was actually taken) as its own array, the same way
        `wavelengths`/`detectors` get theirs -- it's a bulky array-valued param
        that `set_raw_params` drops on purpose."""
        sample_group = self._group.require_group(sample_id)
        array = sample_group.create_array("mask", shape=mask.shape, dtype="bool", overwrite=True)
        array[:] = mask.astype(bool)
        return array

    def get_mask_array(self, sample_id: str) -> np.ndarray | None:
        """The sample's scanned-cell mask, or None if its raw file had none
        (the norm for a full rectangular raster -- see
        `reporting.raster.to_raster`, which then just reshapes)."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        array = sample_group.get("mask")
        if not isinstance(array, zarr.Array):
            return None
        return np.asarray(array[:], dtype=bool)

    def get_raw_params(self, sample_id: str) -> dict | None:
        """The raw file's own `params` dict (see `io.reader.LibsReader.params`),
        cached at `SpectrumStatsStep` time -- e.g. HeightPixels/WidthPixels/
        WidthIMG/HeightIMG/Resolution/AnalysisDate/AnalysisTime -- so report
        generation never needs to reopen the raw file just for this. Only
        scalar entries (see `set_raw_params`) -- bulky array-valued ones like
        `wavelengths`/`detectors` live in their own dedicated arrays instead."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        params = sample_group.attrs.get("raw_params")
        return dict(cast(dict, params)) if params is not None else None

    def set_raw_params(self, sample_id: str, params: dict) -> None:
        """Stores only `params`'s scalar entries (str/int/float/bool/None) --
        a list/dict-valued entry (e.g. a wavelength-calibration array or a
        per-detector channel mask) would bloat this group's JSON attributes
        with data that belongs in its own proper array instead (see
        `create_wavelengths_array`/`create_detectors_array`), so it's
        dropped here rather than duplicated."""
        sample_group = self._group.require_group(sample_id)
        scalar_params = {k: v for k, v in params.items() if not isinstance(v, (list, dict))}
        sample_group.attrs["raw_params"] = scalar_params

    def get_ratio_components(
        self, sample_id: str, feature_id: str
    ) -> tuple[zarr.Array, zarr.Array] | None:
        """Returns (numerator, denominator) -- each shape (n_scans,), already
        +1-shifted -- for `feature_id`, or None if not computed (only ratio
        features, per `features.expression.split_ratio`, have these)."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        ratio_group = sample_group.get("ratio_components")
        if not isinstance(ratio_group, zarr.Group):
            return None
        numerator = ratio_group.get(f"{feature_id}_numerator")
        denominator = ratio_group.get(f"{feature_id}_denominator")
        if not isinstance(numerator, zarr.Array) or not isinstance(denominator, zarr.Array):
            return None
        return numerator, denominator

    def create_ratio_components(
        self, sample_id: str, feature_id: str, numerator: np.ndarray, denominator: np.ndarray
    ) -> None:
        sample_group = self._group.require_group(sample_id)
        ratio_group = sample_group.require_group("ratio_components")
        numerator_array = ratio_group.create_array(
            f"{feature_id}_numerator", shape=numerator.shape, dtype="float64", overwrite=True
        )
        numerator_array[:] = numerator
        denominator_array = ratio_group.create_array(
            f"{feature_id}_denominator", shape=denominator.shape, dtype="float64", overwrite=True
        )
        denominator_array[:] = denominator

    def has_distribution_1d(self, sample_id: str) -> bool:
        sample_group = self._group.get(sample_id)
        return isinstance(sample_group, zarr.Group) and "distribution_1d" in sample_group

    def get_distribution_1d_arrays(
        self, sample_id: str, feature_id: str
    ) -> tuple[zarr.Array, zarr.Array] | None:
        """Returns (counts, bin_edges) for `feature_id`, or None if not computed."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        dist_group = sample_group.get("distribution_1d")
        if not isinstance(dist_group, zarr.Group):
            return None
        counts = dist_group.get(f"{feature_id}_counts")
        bin_edges = dist_group.get(f"{feature_id}_bin_edges")
        if not isinstance(counts, zarr.Array) or not isinstance(bin_edges, zarr.Array):
            return None
        return counts, bin_edges

    def create_distribution_1d(
        self,
        sample_id: str,
        feature_id: str,
        counts: np.ndarray,
        bin_edges: np.ndarray,
        attributes: dict,
    ) -> None:
        sample_group = self._group.require_group(sample_id)
        dist_group = sample_group.require_group("distribution_1d")
        counts_array = dist_group.create_array(
            f"{feature_id}_counts",
            shape=counts.shape,
            dtype="float64",
            attributes=attributes,
            overwrite=True,
        )
        counts_array[:] = counts
        bin_edges_array = dist_group.create_array(
            f"{feature_id}_bin_edges",
            shape=bin_edges.shape,
            dtype="float64",
            overwrite=True,
        )
        bin_edges_array[:] = bin_edges

    def has_distribution_2d(self, sample_id: str) -> bool:
        sample_group = self._group.get(sample_id)
        return isinstance(sample_group, zarr.Group) and "distribution_2d" in sample_group

    def get_distribution_2d_arrays(
        self, sample_id: str, feature_id: str
    ) -> tuple[zarr.Array, zarr.Array, zarr.Array, zarr.Array, zarr.Array, zarr.Array] | None:
        """Returns (counts, bin_edges_num, bin_edges_den, display_counts,
        display_bin_edges_num, display_bin_edges_den) for `feature_id`, or
        None if not computed."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        dist_group = sample_group.get("distribution_2d")
        if not isinstance(dist_group, zarr.Group):
            return None
        counts = dist_group.get(f"{feature_id}_counts")
        bin_edges_num = dist_group.get(f"{feature_id}_bin_edges_num")
        bin_edges_den = dist_group.get(f"{feature_id}_bin_edges_den")
        display_counts = dist_group.get(f"{feature_id}_display_counts")
        display_bin_edges_num = dist_group.get(f"{feature_id}_display_bin_edges_num")
        display_bin_edges_den = dist_group.get(f"{feature_id}_display_bin_edges_den")
        if (
            not isinstance(counts, zarr.Array)
            or not isinstance(bin_edges_num, zarr.Array)
            or not isinstance(bin_edges_den, zarr.Array)
            or not isinstance(display_counts, zarr.Array)
            or not isinstance(display_bin_edges_num, zarr.Array)
            or not isinstance(display_bin_edges_den, zarr.Array)
        ):
            return None
        return (
            counts,
            bin_edges_num,
            bin_edges_den,
            display_counts,
            display_bin_edges_num,
            display_bin_edges_den,
        )

    def create_distribution_2d(
        self,
        sample_id: str,
        feature_id: str,
        counts: np.ndarray,
        bin_edges_num: np.ndarray,
        bin_edges_den: np.ndarray,
        display_counts: np.ndarray,
        display_bin_edges_num: np.ndarray,
        display_bin_edges_den: np.ndarray,
        attributes: dict,
    ) -> None:
        sample_group = self._group.require_group(sample_id)
        dist_group = sample_group.require_group("distribution_2d")
        counts_array = dist_group.create_array(
            f"{feature_id}_counts",
            shape=counts.shape,
            dtype="float64",
            attributes=attributes,
            overwrite=True,
        )
        counts_array[:] = counts
        bin_edges_num_array = dist_group.create_array(
            f"{feature_id}_bin_edges_num",
            shape=bin_edges_num.shape,
            dtype="float64",
            overwrite=True,
        )
        bin_edges_num_array[:] = bin_edges_num
        bin_edges_den_array = dist_group.create_array(
            f"{feature_id}_bin_edges_den",
            shape=bin_edges_den.shape,
            dtype="float64",
            overwrite=True,
        )
        bin_edges_den_array[:] = bin_edges_den
        display_counts_array = dist_group.create_array(
            f"{feature_id}_display_counts",
            shape=display_counts.shape,
            dtype="float64",
            overwrite=True,
        )
        display_counts_array[:] = display_counts
        display_bin_edges_num_array = dist_group.create_array(
            f"{feature_id}_display_bin_edges_num",
            shape=display_bin_edges_num.shape,
            dtype="float64",
            overwrite=True,
        )
        display_bin_edges_num_array[:] = display_bin_edges_num
        display_bin_edges_den_array = dist_group.create_array(
            f"{feature_id}_display_bin_edges_den",
            shape=display_bin_edges_den.shape,
            dtype="float64",
            overwrite=True,
        )
        display_bin_edges_den_array[:] = display_bin_edges_den

    def get_correlation_arrays(
        self, column_name: str
    ) -> tuple[zarr.Array, zarr.Array, zarr.Array, zarr.Array] | None:
        """Returns (pearson, r_squared, mae, mae_p_value) for `column_name`,
        or None if not computed -- including for a column computed by a
        version of this codebase that stored `p_value` instead of `mae`/
        `mae_p_value`, which reads back as "not computed" until
        `compute_correlations` is rerun."""
        correlations_group = self._group.get(CORRELATIONS_GROUP)
        if not isinstance(correlations_group, zarr.Group):
            return None
        column_group = correlations_group.get(quote(column_name, safe=""))
        if not isinstance(column_group, zarr.Group):
            return None
        pearson = column_group.get("pearson")
        r_squared = column_group.get("r_squared")
        mae = column_group.get("mae")
        mae_p_value = column_group.get("mae_p_value")
        if (
            not isinstance(pearson, zarr.Array)
            or not isinstance(r_squared, zarr.Array)
            or not isinstance(mae, zarr.Array)
            or not isinstance(mae_p_value, zarr.Array)
        ):
            return None
        return pearson, r_squared, mae, mae_p_value

    def create_correlation_arrays(
        self,
        column_name: str,
        feature_ids: list[str],
        metric_names: list[str],
        pearson_r: np.ndarray,
        r_squared: np.ndarray,
        mae: np.ndarray,
        mae_p_value: np.ndarray,
    ) -> None:
        correlations_group = self._group.require_group(CORRELATIONS_GROUP)
        column_group = correlations_group.require_group(quote(column_name, safe=""))
        pearson_array = column_group.create_array(
            "pearson",
            shape=pearson_r.shape,
            dtype="float64",
            attributes={
                "feature_ids": feature_ids,
                "metric_names": metric_names,
                "column_name": column_name,
            },
            overwrite=True,
        )
        pearson_array[:] = pearson_r
        r_squared_array = column_group.create_array(
            "r_squared", shape=r_squared.shape, dtype="float64", overwrite=True
        )
        r_squared_array[:] = r_squared
        mae_array = column_group.create_array(
            "mae", shape=mae.shape, dtype="float64", overwrite=True
        )
        mae_array[:] = mae
        mae_p_value_array = column_group.create_array(
            "mae_p_value", shape=mae_p_value.shape, dtype="float64", overwrite=True
        )
        mae_p_value_array[:] = mae_p_value

    def list_correlation_columns(self) -> list[str]:
        """Every external CSV column with computed correlations, in their
        original (unquoted) form."""
        correlations_group = self._group.get(CORRELATIONS_GROUP)
        if not isinstance(correlations_group, zarr.Group):
            return []
        return [unquote(key) for key in correlations_group.keys()]

    def _regression_metric_group(self, column_name: str, metric_name: str) -> zarr.Group | None:
        regressions_group = self._group.get(REGRESSIONS_GROUP)
        if not isinstance(regressions_group, zarr.Group):
            return None
        column_group = regressions_group.get(quote(column_name, safe=""))
        if not isinstance(column_group, zarr.Group):
            return None
        metric_group = column_group.get(quote(metric_name, safe=""))
        return metric_group if isinstance(metric_group, zarr.Group) else None

    def get_regression_arrays(
        self, column_name: str, metric_name: str, feature_id: str
    ) -> tuple[zarr.Array, zarr.Array, zarr.Array] | None:
        """Returns (group_mean_x, group_mean_y, group_std_y) for
        (column_name, metric_name, feature_id), or None if not computed."""
        metric_group = self._regression_metric_group(column_name, metric_name)
        if metric_group is None:
            return None
        feature_group = metric_group.get(feature_id)
        if not isinstance(feature_group, zarr.Group):
            return None
        group_mean_x = feature_group.get("group_mean_x")
        group_mean_y = feature_group.get("group_mean_y")
        group_std_y = feature_group.get("group_std_y")
        if (
            not isinstance(group_mean_x, zarr.Array)
            or not isinstance(group_mean_y, zarr.Array)
            or not isinstance(group_std_y, zarr.Array)
        ):
            return None
        return group_mean_x, group_mean_y, group_std_y

    def create_regression_arrays(
        self,
        column_name: str,
        metric_name: str,
        feature_id: str,
        group_mean_x: np.ndarray,
        group_mean_y: np.ndarray,
        group_std_y: np.ndarray,
        attributes: dict,
    ) -> None:
        # only feature_group is disjoint per cell; the three above it are
        # shared by every compute_regressions shard, so they race the same way
        # bootstrap's do (once per cell here, rather than once per run)
        regressions_group = _require_group_concurrent(self._group, REGRESSIONS_GROUP)
        column_group = _require_group_concurrent(regressions_group, quote(column_name, safe=""))
        metric_group = _require_group_concurrent(column_group, quote(metric_name, safe=""))
        feature_group = metric_group.require_group(feature_id)
        group_mean_x_array = feature_group.create_array(
            "group_mean_x",
            shape=group_mean_x.shape,
            dtype="float64",
            attributes=attributes,
            overwrite=True,
        )
        group_mean_x_array[:] = group_mean_x
        group_mean_y_array = feature_group.create_array(
            "group_mean_y", shape=group_mean_y.shape, dtype="float64", overwrite=True
        )
        group_mean_y_array[:] = group_mean_y
        group_std_y_array = feature_group.create_array(
            "group_std_y", shape=group_std_y.shape, dtype="float64", overwrite=True
        )
        group_std_y_array[:] = group_std_y

    def delete_regression_feature(
        self, column_name: str, metric_name: str, feature_id: str
    ) -> None:
        """Drops one `regressions/<quoted column_name>/<quoted metric_name>/
        <feature_id>/` group if present -- used to prune a feature that's
        gone stale (no longer in a later call's `select_features`, under the
        same metric) without disturbing any other feature or metric already
        stored for that column (see `core.api.compute_regressions`'s
        metric-scoped staleness pruning)."""
        metric_group = self._regression_metric_group(column_name, metric_name)
        if metric_group is None:
            return
        if feature_id in metric_group:
            del metric_group[feature_id]

    def list_regression_features(self, column_name: str, metric_name: str) -> list[str]:
        """Every feature_id with a computed regression for (column_name,
        metric_name), sorted."""
        metric_group = self._regression_metric_group(column_name, metric_name)
        if metric_group is None:
            return []
        return sorted(metric_group.keys())

    def list_regression_metrics(self, column_name: str) -> list[str]:
        """Every metric name with any computed regression for column_name,
        in their original (unquoted) form, sorted -- used to tell whether an
        all-metrics sweep (`compute_regressions(..., metric=None)`) has
        covered a column yet (see `pipeline.correlations`'s own dependency
        logging)."""
        regressions_group = self._group.get(REGRESSIONS_GROUP)
        if not isinstance(regressions_group, zarr.Group):
            return []
        column_group = regressions_group.get(quote(column_name, safe=""))
        if not isinstance(column_group, zarr.Group):
            return []
        return sorted(unquote(key) for key in column_group.keys())

    def list_regression_columns(self) -> list[str]:
        """Every external CSV column with any computed regression, in their
        original (unquoted) form."""
        regressions_group = self._group.get(REGRESSIONS_GROUP)
        if not isinstance(regressions_group, zarr.Group):
            return []
        return [unquote(key) for key in regressions_group.keys()]

    def _applied_regression_metric_group(
        self, sample_id: str, column_name: str, metric_name: str
    ) -> zarr.Group | None:
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        applied_group = sample_group.get(APPLIED_REGRESSIONS_GROUP)
        if not isinstance(applied_group, zarr.Group):
            return None
        column_group = applied_group.get(quote(column_name, safe=""))
        if not isinstance(column_group, zarr.Group):
            return None
        metric_group = column_group.get(quote(metric_name, safe=""))
        return metric_group if isinstance(metric_group, zarr.Group) else None

    def get_applied_regression_arrays(
        self, sample_id: str, column_name: str, metric_name: str, feature_id: str
    ) -> tuple[zarr.Array, zarr.Array, zarr.Array] | None:
        """Returns (predicted, counts, bin_edges) for (sample_id,
        column_name, metric_name, feature_id), or None if not computed."""
        metric_group = self._applied_regression_metric_group(sample_id, column_name, metric_name)
        if metric_group is None:
            return None
        feature_group = metric_group.get(feature_id)
        if not isinstance(feature_group, zarr.Group):
            return None
        predicted = feature_group.get("predicted")
        counts = feature_group.get("counts")
        bin_edges = feature_group.get("bin_edges")
        if (
            not isinstance(predicted, zarr.Array)
            or not isinstance(counts, zarr.Array)
            or not isinstance(bin_edges, zarr.Array)
        ):
            return None
        return predicted, counts, bin_edges

    def create_applied_regression_arrays(
        self,
        sample_id: str,
        column_name: str,
        metric_name: str,
        feature_id: str,
        predicted: np.ndarray,
        counts: np.ndarray,
        bin_edges: np.ndarray,
        attributes: dict,
    ) -> None:
        sample_group = self._group.require_group(sample_id)
        applied_group = sample_group.require_group(APPLIED_REGRESSIONS_GROUP)
        column_group = applied_group.require_group(quote(column_name, safe=""))
        metric_group = column_group.require_group(quote(metric_name, safe=""))
        feature_group = metric_group.require_group(feature_id)
        predicted_array = feature_group.create_array(
            "predicted",
            shape=predicted.shape,
            dtype="float64",
            attributes=attributes,
            overwrite=True,
        )
        predicted_array[:] = predicted
        counts_array = feature_group.create_array(
            "counts", shape=counts.shape, dtype="float64", overwrite=True
        )
        counts_array[:] = counts
        bin_edges_array = feature_group.create_array(
            "bin_edges", shape=bin_edges.shape, dtype="float64", overwrite=True
        )
        bin_edges_array[:] = bin_edges

    def list_applied_regression_features(
        self, sample_id: str, column_name: str, metric_name: str
    ) -> list[str]:
        """Every feature_id with an applied regression for (sample_id,
        column_name, metric_name), sorted."""
        metric_group = self._applied_regression_metric_group(sample_id, column_name, metric_name)
        if metric_group is None:
            return []
        return sorted(metric_group.keys())

    def list_applied_regression_metrics(self, sample_id: str, column_name: str) -> list[str]:
        """Every metric name with an applied regression for (sample_id,
        column_name), in their original (unquoted) form, sorted."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return []
        applied_group = sample_group.get(APPLIED_REGRESSIONS_GROUP)
        if not isinstance(applied_group, zarr.Group):
            return []
        column_group = applied_group.get(quote(column_name, safe=""))
        if not isinstance(column_group, zarr.Group):
            return []
        return sorted(unquote(key) for key in column_group.keys())

    def list_applied_regression_columns(self, sample_id: str) -> list[str]:
        """Every external CSV column with an applied regression for
        sample_id, in their original (unquoted) form."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return []
        applied_group = sample_group.get(APPLIED_REGRESSIONS_GROUP)
        if not isinstance(applied_group, zarr.Group):
            return []
        return [unquote(key) for key in applied_group.keys()]

    def get_bootstrap_arrays(
        self, sample_id: str, bootstrap_id: str
    ) -> tuple[zarr.Array, zarr.Array] | None:
        """Returns (metrics, completed) for an already-created run, or None."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return None
        bootstrap_group = sample_group.get(BOOTSTRAP_GROUP)
        if not isinstance(bootstrap_group, zarr.Group):
            return None
        run_group = bootstrap_group.get(bootstrap_id)
        if not isinstance(run_group, zarr.Group):
            return None
        metrics = run_group.get(BOOTSTRAP_METRICS_NAME)
        completed = run_group.get(BOOTSTRAP_COMPLETED_NAME)
        if not isinstance(metrics, zarr.Array) or not isinstance(completed, zarr.Array):
            return None
        return metrics, completed

    def get_or_create_bootstrap_arrays(
        self,
        sample_id: str,
        bootstrap_id: str,
        n_iterations: int,
        feature_ids: list[str],
        feature_hashes: dict[str, str | None],
        metric_names: list[str],
        method: str,
        resolved_npix: int,
        shuffle: bool,
        base_seed: int,
    ) -> tuple[zarr.Array, zarr.Array]:
        """Idempotent get-or-create-or-extend, delegating to
        `core.runs.checkpoint.get_or_create_checkpoint_arrays` against this
        sample's `bootstrap/<bootstrap_id>/` group. Never overwrites an
        existing run -- the caller (`pipeline.bootstrap.runner.run_bootstrap`)
        is responsible for calling `reset_bootstrap_run` first if
        `feature_hashes` no longer matches what's already stored there."""
        # All three are guarded. bootstrap/<bootstrap_id> are shared by every
        # shard of this run and demonstrably race. <sample_id> only pre-exists
        # because run_bootstrap raises first when the features array is
        # missing -- but that is one caller's precondition holding up another
        # line of a public method, and a direct caller on a fresh store races
        # here exactly like the other two.
        sample_group = _require_group_concurrent(self._group, sample_id)
        bootstrap_group = _require_group_concurrent(sample_group, BOOTSTRAP_GROUP)
        run_group = _require_group_concurrent(bootstrap_group, bootstrap_id)
        arrays = get_or_create_checkpoint_arrays(
            run_group,
            BOOTSTRAP_METRICS_NAME,
            BOOTSTRAP_COMPLETED_NAME,
            n_iterations,
            item_shape=(len(feature_ids), len(metric_names)),
            attributes={
                "feature_ids": feature_ids,
                "feature_hashes": feature_hashes,
                "metric_names": metric_names,
                "method": method,
                "resolved_npix": resolved_npix,
                "shuffle": shuffle,
                "base_seed": base_seed,
            },
        )
        return arrays.results, arrays.completed

    def reset_bootstrap_run(self, sample_id: str, bootstrap_id: str) -> None:
        """Delete `<sample_id>/bootstrap/<bootstrap_id>/` if present, so the
        next `get_or_create_bootstrap_arrays` call creates it fresh (every
        iteration pending again) -- used when a feature's identity hash has
        changed since this run's iterations were computed, since a stale
        `completed` flag there would otherwise keep reporting values
        computed from a peak/expression definition that no longer exists as
        if they were still correct."""
        sample_group = self._group.get(sample_id)
        if not isinstance(sample_group, zarr.Group):
            return
        bootstrap_group = sample_group.get(BOOTSTRAP_GROUP)
        if not isinstance(bootstrap_group, zarr.Group):
            return
        if bootstrap_id in bootstrap_group:
            del bootstrap_group[bootstrap_id]

    def get_bootstrap_progress(self, sample_id: str, bootstrap_id: str) -> tuple[int, int] | None:
        """Returns (n_done, n_total), or None if this run doesn't exist yet."""
        arrays = self.get_bootstrap_arrays(sample_id, bootstrap_id)
        if arrays is None:
            return None
        _, completed = arrays
        return int(np.asarray(completed[:]).sum()), completed.shape[0]
