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

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress, t

from pylibs.core.exceptions import RegressionError
from pylibs.core.pipeline.distributions import Distribution1D, compute_distribution_1d

MIN_SAMPLES_FOR_REGRESSION = 3


def _mae_permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    observed_mae: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Refits OLS on `n_permutations` shuffled pairings of (x, y) and
    returns the fraction whose residual MAE is strictly lower than
    `observed_mae`. Closed-form OLS (not `scipy.stats.linregress` per
    iteration -- 10000 calls x potentially hundreds of correlation cells
    would be far too slow): shuffling y only reorders it, so its own
    mean/variance -- and x's -- are identical across every permutation;
    only the pairing changes, letting slope/intercept for every permutation
    be computed as one vectorized matrix-vector product."""
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

    return float(np.mean(permuted_mae < observed_mae))


@dataclass
class RegressionFit:
    slope: float
    intercept: float
    pearson_r: float  # linregress's own signed correlation coefficient
    r_squared: float
    mae: float  # mean absolute residual of the fit
    # permutation-test p-value on `mae` (see _mae_permutation_p_value), not an
    # analytic test -- fraction of shuffled-and-refit pairings whose residual
    # MAE is strictly lower than this fit's own
    p_value: float
    n: int
    x_mean: float
    ssxx: float  # sum((x - x_mean) ** 2) over the fit's own points
    residual_std: float  # sqrt(SSres / (n - 2))
    x_min: float  # quantification limits: the fit's own finite-masked x range
    x_max: float


def fit_least_squares(
    x: np.ndarray, y: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> RegressionFit | None:
    """Ordinary least squares `y = slope*x + intercept` over the finite
    pairs of `x`/`y`. Returns `None` (not an exception -- mirrors
    `compute_correlation_matrices`'s own per-cell skip guard) if fewer than
    `MIN_SAMPLES_FOR_REGRESSION` pairs are finite, or either side is
    constant over them (slope undefined, r_squared/mae/p_value would be
    NaN). `p_value` is a permutation-test p-value on the fit's own residual
    MAE (see `_mae_permutation_p_value`), computed with `n_permutations`
    shuffled refits driven by `rng`."""
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
    mae_p_value = _mae_permutation_p_value(x_masked, y_masked, observed_mae, n_permutations, rng)

    return RegressionFit(
        slope=float(result.slope),
        intercept=float(result.intercept),
        pearson_r=float(result.rvalue),
        r_squared=float(result.rvalue) ** 2,
        mae=observed_mae,
        p_value=mae_p_value,
        n=n,
        x_mean=x_mean,
        ssxx=ssxx,
        residual_std=residual_std,
        x_min=float(x_masked.min()),
        x_max=float(x_masked.max()),
    )


RegressionAlgorithm = Callable[
    [np.ndarray, np.ndarray, int, np.random.Generator], RegressionFit | None
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


def group_by_sample_name(sample_names: list[str], x: np.ndarray, y: np.ndarray) -> SampleGroupStats:
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
) -> FeatureRegression | None:
    """Fits `algorithm` (a registered name, see `get_regression_algorithm`)
    over `(x, y)`, then groups the same points by `sample_names` for
    plotting. Returns `None` if the fit itself couldn't be computed
    (insufficient or degenerate data)."""
    fit = get_regression_algorithm(algorithm)(x, y, n_permutations, rng)
    if fit is None:
        return None
    grouped = group_by_sample_name(sample_names, x, y)
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
        sample_names=grouped.sample_names,
        group_mean_x=grouped.group_mean_x,
        group_mean_y=grouped.group_mean_y,
        group_std_y=grouped.group_std_y,
    )


def confidence_band_half_width(
    x_query: np.ndarray | float,
    n: int,
    x_mean: float,
    ssxx: float,
    residual_std: float,
    confidence: float = 0.95,
) -> np.ndarray | float:
    """Half-width of the `confidence` confidence band for a simple linear
    regression's fitted line, at `x_query`, reproducible purely from the
    fit's own stored `n`/`x_mean`/`ssxx`/`residual_std` -- no per-scan data
    needed. `df = n - 2` (two fitted parameters, slope and intercept)."""
    t_crit = float(t.ppf(1 - (1 - confidence) / 2, df=n - 2))
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
