"""Univariate regression: fit a feature's metric against an external
per-sample scalar property, storing everything needed to both reapply the
fit and redraw its confidence-interval band later -- not just slope/
intercept. This is the **only** place `fit_least_squares` (or any future
regression algorithm) is ever called -- `pipeline.correlations` never fits
anything itself; it only reads back already-stored `FeatureRegression`
results and reshapes them into its four matrices (see that module's own
docstring). `api.compute_regressions` can fit either one selected metric
(today's default, `metric="Mean"`) or every metric in `pipeline.metrics.
METRIC_NAMES` in one sweep (`metric=None`) -- the latter is what gives
`compute_correlations` something to read for every (feature, metric) cell.

Mirrors `pipeline.bootstrap.resampling`'s registry pattern (a second
algorithm later is one function plus one registry entry, never a new
if/elif chain in `core.api`).

`REGRESSION_APPLICATORS` is the fitting registry's counterpart for going the
other direction: given a stored `FeatureRegression` and a sample's raw
per-scan feature values, predict per-scan values of the external property
(e.g. turn an intensity map into a predicted-concentration map). Keyed by
the same algorithm names as `REGRESSION_ALGORITHMS` and dispatched by each
regression's own stored `algorithm`, so a future non-linear algorithm adds
one fit function plus one apply function plus one entry in each registry.
"""

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress, t

from pylibs.core.exceptions import RegressionError
from pylibs.core.pipeline.distributions import Distribution1D, compute_distribution_1d

MIN_SAMPLES_FOR_REGRESSION = 3


INFERENCE_VERSION = 2
"""Which inference rules produced a stored regression.

1: p as the strict fraction of shuffled fits beating the real one (could be
   exactly 0), band and permutations over acquisitions.
2: p as (b + 1) / (N + 1) with ties counted, and -- when grouping is on and
   sample names repeat -- the band from the count-weighted group means at
   n_groups - 2 degrees of freedom, with permutations over whole groups.

Stored on every regression so results from either era can be told apart
without guessing from their values.
"""


def _group_indices(sample_names: list[str], mask: np.ndarray) -> list[np.ndarray]:
    """Row indices of each distinct sample name, over the fit's own finite
    rows, in first-appearance order. One entry per physical sample."""
    order: dict[str, list[int]] = {}
    position = 0
    for name, keep in zip(sample_names, mask, strict=True):
        if keep:
            order.setdefault(name, []).append(position)
            position += 1
    return [np.asarray(rows) for rows in order.values()]


def _fit_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form OLS for many y vectors against one x, vectorized."""
    x_centered = x - x.mean()
    sxx = float(np.sum(x_centered**2))
    y_mean = y.mean(axis=-1)
    slope = (y - y_mean[..., None]) @ x_centered / sxx
    return slope, y_mean - slope * x.mean()


def _grouped_mae_permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_mae: float,
    groups: list[np.ndarray],
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, str, int]:
    """Permutation p-value with whole samples as the unit.

    A property value belongs to a physical sample, not to one ablation
    layer, so the null must move it between samples and carry every
    acquisition of that sample with it. Permuting acquisitions instead
    treats each layer as independent evidence, which is the
    pseudoreplication this exists to avoid: it makes the null far easier to
    beat and the p-value correspondingly smaller.

    Returns `(p_value, mode, n_used)`. When `n_groups!` is at most
    `n_permutations` every arrangement is enumerated ("exact"), and the
    p-value is the exact share of arrangements whose refit MAE is at most
    the real one's, the identity arrangement included -- no `+1` correction
    is needed or applied, because nothing is being sampled. Otherwise
    arrangements are drawn at random ("sampled") and the corrected
    `(b + 1) / (n_permutations + 1)` applies as usual."""
    n_groups = len(groups)
    # one property value per sample, and where each row's value comes from
    per_group = np.array([x[rows][0] for rows in groups])
    row_group = np.empty(x.size, dtype=int)
    for index, rows in enumerate(groups):
        row_group[rows] = index

    total = math.factorial(n_groups)
    if total <= n_permutations:
        arrangements = np.array(list(itertools.permutations(range(n_groups))))
        mode, n_used = "exact", total
    else:
        arrangements = np.stack([rng.permutation(n_groups) for _ in range(n_permutations)])
        mode, n_used = "sampled", n_permutations

    # x moves, y stays: permuting either gives the same null, and moving the
    # shorter per-group vector keeps this one matrix build
    x_perm = per_group[arrangements][:, row_group]  # (n_arrangements, n_rows)
    x_mean = x_perm.mean(axis=1, keepdims=True)
    x_centered = x_perm - x_mean
    sxx = np.sum(x_centered**2, axis=1)
    y_mean = y.mean()
    slope = (x_centered @ (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean[:, 0]
    predicted = slope[:, None] * x_perm + intercept[:, None]
    permuted_mae = np.mean(np.abs(y[None, :] - predicted), axis=1)

    # A tie has to survive being recomputed. The observed MAE came from
    # scipy's fit and these come from the vectorized one above, so the
    # identity arrangement's MAE can differ from it in the last few bits.
    # Comparing exactly would then drop the very arrangement that guarantees
    # a non-zero p-value, and the exact branch would report 0.
    tolerance = abs(observed_mae) * 1e-9 + np.finfo(float).tiny
    b = int(np.count_nonzero(permuted_mae <= observed_mae + tolerance))
    if mode == "exact":
        # the identity arrangement is one of the enumerated ones and ties with
        # itself, so b >= 1 and the p-value has a floor of 1 / n_groups!
        return max(b, 1) / total, mode, n_used
    return (b + 1) / (n_permutations + 1), mode, n_used


def _mae_permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_mae: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Refits OLS on `n_permutations` shuffled pairings of (x, y) and
    returns the corrected permutation p-value

        p = (b + 1) / (n_permutations + 1)

    where `b` counts the permutations whose residual MAE is at most
    `observed_mae` -- ties included, since a permutation that matches the
    real fit is evidence against it, not for it.

    Both the +1 terms and the tie counting matter. The unpermuted pairing is
    itself one of the arrangements being tested, so counting it puts a floor
    of `1 / (n_permutations + 1)` on the result: a permutation p-value can
    never be 0, and reporting 0 would claim more resolution than the number
    of permutations can support (Phipson & Smyth 2010). The floor is the
    honest statement "smaller than this test can resolve" -- see
    `format_p_value`, which renders it as an upper bound rather than a value.

    Closed-form OLS (not `scipy.stats.linregress` per iteration -- 10000
    calls x potentially hundreds of correlation cells would be far too
    slow): shuffling y only reorders it, so its own mean/variance -- and
    x's -- are identical across every permutation; only the pairing changes,
    letting slope/intercept for every permutation be computed as one
    vectorized matrix-vector product."""
    n = x.size
    x_centered = x - x.mean()
    sxx = float(np.sum(x_centered**2))
    y_mean = y.mean()

    perm_idx = rng.permuted(np.broadcast_to(np.arange(n), (n_permutations, n)).copy(), axis=1)
    y_perm = y[perm_idx]  # (n_permutations, n)

    slope = (y_perm - y_mean) @ x_centered / sxx  # (n_permutations,)
    intercept = y_mean - slope * x.mean()
    predicted = slope[:, None] * x[None, :] + intercept[:, None]
    permuted_mae = np.mean(np.abs(y_perm - predicted), axis=1)

    b = int(np.count_nonzero(permuted_mae <= observed_mae))
    return (b + 1) / (n_permutations + 1)


def format_p_value(p_value: float, n_permutations: int = 0) -> str:
    """Render a permutation p-value for a figure label or a console line.

    At the floor -- no permutation matched or beat the real fit -- the value
    carries no information beyond "smaller than this test resolves", so it
    is shown as an upper bound, `p < 1.0e-04`, rather than as a number that
    invites being read as an estimate. Every other value prints in normal
    scientific notation.

    `n_permutations = 0` means the count is unknown (a regression stored
    before it was carried on the result), and the plain value is printed."""
    if n_permutations > 0:
        floor = 1.0 / (n_permutations + 1)
        if p_value <= floor * (1.0 + 1e-9):
            return f"p < {floor:.1e}"
    return f"p = {p_value:.2e}"


def format_sample_count(regression) -> str:
    """ "n = 7 samples (21 acquisitions)" when acquisitions were grouped,
    plain "n = 21" when every acquisition is its own sample -- the second
    number only earns its place when the two differ."""
    if getattr(regression, "grouped", False):
        return f"n = {regression.n_groups} samples ({regression.n_acquisitions} acquisitions)"
    return f"n = {regression.n}"


def format_permutation_mode(regression) -> str:
    """How the p-value was obtained: every arrangement enumerated, or a
    random sample of them."""
    used = getattr(regression, "n_permutations_used", 0) or regression.n_permutations
    if getattr(regression, "permutation_mode", "sampled") == "exact":
        return f"exact, {used} permutations"
    return f"sampled, {used}"


@dataclass
class RegressionFit:
    slope: float
    intercept: float
    pearson_r: float  # linregress's own signed correlation coefficient
    r_squared: float
    mae: float  # mean absolute residual of the fit
    # permutation-test p-value on `mae` (see _mae_permutation_p_value), not an
    # analytic test -- (b + 1) / (n_permutations + 1), where b counts the
    # shuffled-and-refit pairings whose residual MAE is at most this fit's
    # own. Never 0; see _mae_permutation_p_value
    p_value: float
    n: int
    x_mean: float
    ssxx: float  # sum((x - x_mean) ** 2) over the fit's own points
    residual_std: float  # sqrt(SSres / (n - 2))
    x_min: float  # quantification limits: the fit's own finite-masked x range
    x_max: float
    # Inference unit. `n` above stays the acquisition count, since the line
    # is fitted over acquisitions; these describe what the band and the
    # p-value treated as independent.
    grouped: bool = False
    n_groups: int = 0
    n_acquisitions: int = 0
    balanced: bool = True
    permutation_mode: str = "sampled"
    n_permutations_used: int = 0
    band_df: int = 0  # degrees of freedom the confidence band uses
    inference_version: int = INFERENCE_VERSION


def grouped_band_parameters(
    x: np.ndarray, y: np.ndarray, groups: list[np.ndarray]
) -> tuple[float, float, float, int]:
    """`(x_mean, ssxx, residual_std, df)` for a confidence band whose unit is
    the physical sample rather than the acquisition.

    The band comes from a least-squares fit of the per-sample mean response
    on the per-sample property value, each sample weighted by how many
    acquisitions it contributed. Because the property value is constant
    within a sample, that weighted fit reproduces the pooled line exactly,
    in balanced and unbalanced designs alike -- so the band stays centred on
    the line actually drawn, while its width is governed by `n_groups - 2`
    degrees of freedom instead of `n_acquisitions - 2`.

    The weighting assumes each sample mean's precision is proportional to
    its acquisition count, which is what pooling acquisitions implies. It
    therefore ignores between-sample variance when counts differ: a sample
    measured ten times is treated as ten times as informative as one
    measured once, which overstates its weight if samples themselves vary.
    With equal counts the weights cancel and this reduces to the ordinary
    unweighted regression on the sample means."""
    weights = np.array([float(rows.size) for rows in groups])
    group_x = np.array([float(np.mean(x[rows])) for rows in groups])
    group_y = np.array([float(np.mean(y[rows])) for rows in groups])
    total = weights.sum()
    x_mean = float((weights * group_x).sum() / total)
    y_mean = float((weights * group_y).sum() / total)
    ssxx = float((weights * (group_x - x_mean) ** 2).sum())
    slope = float((weights * (group_x - x_mean) * (group_y - y_mean)).sum() / ssxx)
    intercept = y_mean - slope * x_mean
    residuals = group_y - (slope * group_x + intercept)
    df = len(groups) - 2
    residual_std = float(np.sqrt(np.sum(weights * residuals**2) / df)) if df > 0 else float("nan")
    return x_mean, ssxx, residual_std, df


def fit_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    sample_names: list[str] | None = None,
) -> RegressionFit | None:
    """Ordinary least squares `y = slope*x + intercept` over the finite
    pairs of `x`/`y`. Returns `None` (not an exception -- mirrors
    `compute_correlation_matrices`'s own per-cell skip guard) if fewer than
    `MIN_SAMPLES_FOR_REGRESSION` pairs are finite, or either side is
    constant over them (slope undefined, r_squared/mae/p_value would be
    NaN).

    `sample_names`, when given and holding repeats, makes the **physical
    sample the unit of inference** rather than the acquisition. Several
    ablation layers of one pellet share a property value and are not
    independent evidence about it, so treating each as its own point is
    pseudoreplication: it narrows the confidence band and shrinks the
    p-value without any more having been measured.

    What that changes, and what it does not:

    * The line, `pearson_r`, `r_squared` and `mae` are **unchanged** --
      still pooled least squares over every acquisition.
    * The confidence band comes from the count-weighted regression on the
      per-sample means, at `n_groups - 2` degrees of freedom (see
      `grouped_band_parameters`). That fit reproduces the pooled line
      exactly, so the band stays centred on the line drawn.
    * The permutation p-value moves property values between whole samples
      (see `_grouped_mae_permutation_p_value`), and is exact rather than
      sampled when `n_groups!` fits within `n_permutations`.

    With no repeated name every group holds one acquisition and all three
    reduce to the ungrouped computation, so passing names costs nothing."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < MIN_SAMPLES_FOR_REGRESSION:
        return None
    x_masked = x[mask]
    y_masked = y[mask]
    if x_masked.std() == 0 or y_masked.std() == 0:
        return None

    result = linregress(x_masked, y_masked)
    x_mean = float(x_masked.mean())
    ssxx = float(np.sum((x_masked - x_mean) ** 2))
    residuals = y_masked - (result.slope * x_masked + result.intercept)
    ss_res = float(np.sum(residuals**2))
    residual_std = float(np.sqrt(ss_res / (n - 2)))
    observed_mae = float(np.mean(np.abs(residuals)))

    groups = _group_indices(list(sample_names), mask) if sample_names is not None else []
    # one acquisition per name is not grouping at all: every quantity below
    # reduces to its ungrouped form, so take the cheaper path and say so
    grouped = bool(groups) and len(groups) < n
    counts = [rows.size for rows in groups]
    band_x_mean, band_ssxx, band_residual_std, band_df = x_mean, ssxx, residual_std, n - 2
    permutation_mode, n_permutations_used = "sampled", n_permutations

    if grouped and len(groups) > MIN_SAMPLES_FOR_REGRESSION - 1:
        band_x_mean, band_ssxx, band_residual_std, band_df = grouped_band_parameters(
            x_masked, y_masked, groups
        )
        mae_p_value, permutation_mode, n_permutations_used = _grouped_mae_permutation_p_value(
            x_masked, y_masked, observed_mae, groups, n_permutations, rng
        )
    else:
        grouped = False
        mae_p_value = _mae_permutation_p_value(
            x_masked, y_masked, observed_mae, n_permutations, rng
        )

    return RegressionFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        pearson_r=float(result.rvalue),
        r_squared=float(result.rvalue) ** 2,
        mae=observed_mae,
        p_value=mae_p_value,
        n=n,
        # the band's own parameters, which are the grouped ones when
        # grouping applies -- confidence_band_half_width needs no change
        x_mean=band_x_mean,
        ssxx=band_ssxx,
        residual_std=band_residual_std,
        x_min=float(x_masked.min()),
        x_max=float(x_masked.max()),
        grouped=grouped,
        n_groups=len(groups) if groups else n,
        n_acquisitions=n,
        balanced=len(set(counts)) <= 1,
        permutation_mode=permutation_mode,
        n_permutations_used=n_permutations_used,
        band_df=band_df,
    )


RegressionAlgorithm = Callable[
    [np.ndarray, np.ndarray, int, np.random.Generator, list[str] | None], RegressionFit | None
]

REGRESSION_ALGORITHMS: dict[str, RegressionAlgorithm] = {"least_squares": fit_least_squares}


def get_regression_algorithm(name: str) -> RegressionAlgorithm:
    algorithm = REGRESSION_ALGORITHMS.get(name)
    if algorithm is None:
        raise RegressionError(
            f"Unknown regression algorithm {name!r}; registered algorithms: "
            f"{sorted(REGRESSION_ALGORITHMS)}"
        )
    return algorithm


@dataclass
class SampleGroupStats:
    sample_names: list[str]  # distinct, sorted
    group_mean_x: np.ndarray
    group_mean_y: np.ndarray
    group_std_y: np.ndarray


def sample_group_stats(sample_names: list[str], x: np.ndarray, y: np.ndarray) -> SampleGroupStats:
    """Groups the finite-masked `(x, y)` pairs -- the same ones a fit would
    use -- by physical `sample_name`, for plotting only: one point per
    physical sample (mean_x, mean_y, std_y across that sample's own files/
    ablation layers), matching the reference report's calibration-curve
    figure rather than one point per file."""
    mask = np.isfinite(x) & np.isfinite(y)
    groups: dict[str, list[tuple[float, float]]] = {}
    for name, xi, yi, keep in zip(sample_names, x, y, mask, strict=True):
        if keep:
            groups.setdefault(name, []).append((float(xi), float(yi)))

    names = sorted(groups)
    group_mean_x = np.array([np.mean([xi for xi, _ in groups[name]]) for name in names])
    group_mean_y = np.array([np.mean([yi for _, yi in groups[name]]) for name in names])
    group_std_y = np.array([np.std([yi for _, yi in groups[name]]) for name in names])
    return SampleGroupStats(
        sample_names=names,
        group_mean_x=group_mean_x,
        group_mean_y=group_mean_y,
        group_std_y=group_std_y,
    )


@dataclass
class FeatureRegression:
    column_name: str
    feature_id: str
    metric_name: str
    algorithm: str
    slope: float
    intercept: float
    pearson_r: float
    r_squared: float
    mae: float
    p_value: float
    n: int
    x_mean: float
    ssxx: float
    residual_std: float
    x_min: float
    x_max: float
    sample_names: list[str]
    group_mean_x: np.ndarray
    group_mean_y: np.ndarray
    group_std_y: np.ndarray
    # how many permutations produced `p_value`, so its floor is recoverable
    # at display time. 0 means a regression stored before this was carried.
    n_permutations: int = 0
    # what the band and the p-value treated as independent (see
    # fit_least_squares). `n` remains the acquisition count, since the line
    # is fitted over acquisitions.
    grouped: bool = False
    n_groups: int = 0
    n_acquisitions: int = 0
    balanced: bool = True
    permutation_mode: str = "sampled"
    n_permutations_used: int = 0
    band_df: int = 0
    inference_version: int = 0


def compute_feature_regression(
    column_name: str,
    feature_id: str,
    metric_name: str,
    algorithm: str,
    sample_names: list[str],
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    group_by_sample_name: bool = True,
) -> FeatureRegression | None:
    """Fits `algorithm` (a registered name, see `get_regression_algorithm`)
    over `(x, y)`, then groups the same points by `sample_names` for
    plotting. Returns `None` if the fit itself couldn't be computed
    (insufficient or degenerate data).

    `group_by_sample_name` (default True) additionally makes the physical
    sample the unit of inference -- see `fit_least_squares`. It is a no-op
    when no name repeats."""
    fit = get_regression_algorithm(algorithm)(
        x, y, n_permutations, rng, sample_names if group_by_sample_name else None
    )
    if fit is None:
        return None
    plot_points = sample_group_stats(sample_names, x, y)
    return FeatureRegression(
        column_name=column_name,
        feature_id=feature_id,
        metric_name=metric_name,
        algorithm=algorithm,
        slope=fit.slope,
        intercept=fit.intercept,
        pearson_r=fit.pearson_r,
        r_squared=fit.r_squared,
        mae=fit.mae,
        p_value=fit.p_value,
        n=fit.n,
        x_mean=fit.x_mean,
        ssxx=fit.ssxx,
        residual_std=fit.residual_std,
        x_min=fit.x_min,
        x_max=fit.x_max,
        sample_names=plot_points.sample_names,
        group_mean_x=plot_points.group_mean_x,
        group_mean_y=plot_points.group_mean_y,
        group_std_y=plot_points.group_std_y,
        n_permutations=n_permutations,
        grouped=fit.grouped,
        n_groups=fit.n_groups,
        n_acquisitions=fit.n_acquisitions,
        balanced=fit.balanced,
        permutation_mode=fit.permutation_mode,
        n_permutations_used=fit.n_permutations_used,
        band_df=fit.band_df,
        inference_version=fit.inference_version,
    )


def confidence_band_half_width(
    x_query: np.ndarray | float,
    n: int,
    x_mean: float,
    ssxx: float,
    residual_std: float,
    confidence: float = 0.95,
    df: int | None = None,
) -> np.ndarray | float:
    """Half-width of the `confidence` confidence band for a simple linear
    regression's fitted line, at `x_query`, reproducible purely from the
    fit's own stored `n`/`x_mean`/`ssxx`/`residual_std` -- no per-scan data
    needed.

    `n` and the degrees of freedom are separate inputs because grouped
    inference separates them. `n` is the total weight behind the fit, which
    is the acquisition count: `residual_std` is per unit weight and `ssxx`
    is the weighted sum of squares, so the standard error carries `1 / n`
    either way. The degrees of freedom are the independent points, which is
    the number of physical samples when acquisitions were grouped (see
    `grouped_band_parameters`). Passing `n - 2` for both would widen the
    band's `1/n` term and narrow its `t` multiplier at the same time.

    `df=None` means `n - 2`, the ungrouped case and what a regression stored
    before 1.0.1 needs."""
    degrees = n - 2 if df is None else df
    t_crit = float(t.ppf(1 - (1 - confidence) / 2, df=degrees))
    return t_crit * residual_std * np.sqrt(1 / n + (x_query - x_mean) ** 2 / ssxx)


def apply_least_squares(regression: FeatureRegression, raw_values: np.ndarray) -> np.ndarray:
    """Inverts `y = slope*x + intercept` to predict `x` (the external
    property) from raw per-scan `y` values (the feature's own intensity at
    each scan/pixel). NaNs in `raw_values` simply propagate to NaN in the
    output -- never raised here, left for `compute_distribution_1d`'s own
    NaN-drop to handle downstream."""
    return (raw_values - regression.intercept) / regression.slope


RegressionApplicator = Callable[[FeatureRegression, np.ndarray], np.ndarray]

REGRESSION_APPLICATORS: dict[str, RegressionApplicator] = {"least_squares": apply_least_squares}


def get_regression_applicator(name: str) -> RegressionApplicator:
    applicator = REGRESSION_APPLICATORS.get(name)
    if applicator is None:
        raise RegressionError(
            f"Unknown regression algorithm {name!r}; registered applicators: "
            f"{sorted(REGRESSION_APPLICATORS)}"
        )
    return applicator


@dataclass
class AppliedRegression:
    sample_id: str
    column_name: str
    feature_id: str
    metric_name: str
    algorithm: str
    predicted: np.ndarray
    distribution: Distribution1D
    x_min: float  # quantification limits: the underlying fit's own x range,
    x_max: float  # not the predicted array's own min/max -- see compute_applied_regression


def compute_applied_regression(
    sample_id: str, regression: FeatureRegression, raw_values: np.ndarray
) -> AppliedRegression:
    """Applies `regression`'s registered algorithm-specific inverse to a
    sample's raw per-scan feature values, predicting the external property
    at every scan/pixel, then computes fresh 1D distribution stats over the
    predicted array (`fit_gaussian=False` -- this may run across many
    samples in bulk, so the slower curve fit is skipped, mirroring
    bootstrap's own per-iteration stats). Carries `regression`'s own
    quantification limits (`x_min`/`x_max`, the fit's finite-masked x range)
    forward -- not derived from `predicted`'s own range, which may exceed
    them wherever the model is extrapolating."""
    applicator = get_regression_applicator(regression.algorithm)
    predicted = applicator(regression, raw_values)
    distribution = compute_distribution_1d(predicted, fit_gaussian=False)
    return AppliedRegression(
        sample_id=sample_id,
        column_name=regression.column_name,
        feature_id=regression.feature_id,
        metric_name=regression.metric_name,
        algorithm=regression.algorithm,
        predicted=predicted,
        distribution=distribution,
        x_min=regression.x_min,
        x_max=regression.x_max,
    )
