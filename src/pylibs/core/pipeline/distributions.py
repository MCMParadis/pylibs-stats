"""Distribution containers: `Distribution1D` (a single feature's per-scan
values) and `Distribution2D` (a ratio feature's numerator/denominator joint
distribution) -- siblings, not parent/child, despite the similar names; a
ratio feature gets *both* (its own `Distribution1D` plus a `Distribution2D`
for its numerator/denominator pair), a bare-peak feature only the former.

All the actual metric math lives in `pipeline.metrics`, and only there --
`compute_distribution_1d`/`compute_distribution_2d` below build the
histogram(s) and the small "everything a metric might need" context objects
(`metrics.MetricContext1D`/`MetricContext2D`), then loop `metrics.METRICS_1D`/
`METRICS_2D` to fill each object's `metrics: dict[str, float]` field -- see
`pipeline.metrics`'s own module docstring for how to add a new one. Computed
once (by `steps.distribution_1d.Distribution1DStep`/
`steps.distribution_2d.Distribution2DStep`) and cached in the results store;
report generation and correlations only ever read these back, never
recompute from raw values.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from pylibs.core.exceptions import DistributionError
from pylibs.core.logging import get_logger
from pylibs.core.pipeline import metrics
from pylibs.core.pipeline.metrics import MetricContext1D, MetricContext2D

logger = get_logger(__name__)


def gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def joint_histogram(
    x1: np.ndarray, x2: np.ndarray, n_bins: int = 150
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D histogram of `x1` against `x2` (dropping any pair where either
    side is NaN/inf), shape (n_bins, n_bins). Returns (counts, bin_edges_x1,
    bin_edges_x2)."""
    finite = np.isfinite(x1) & np.isfinite(x2)
    return np.histogram2d(x1[finite], x2[finite], bins=n_bins)


def quantile_skewness(values: np.ndarray) -> float:
    """Robust skewness measure in `[-1, 1]` -- `((P90-P50) - (P50-P10)) /
    (P90-P10)`: 0 for a symmetric distribution, positive for right-skew,
    negative for left-skew. Cheap (percentiles only, O(n log n)) and free of
    the medcouple's O(n^2) cost and tricky tie-handling at the median, at
    the cost of being a somewhat coarser measure (only P10/P50/P90, not
    every pairwise comparison). `0.0` for a degenerate `P90 == P10`
    (effectively constant data) or an empty `values`."""
    if values.size == 0:
        return 0.0
    p10, p50, p90 = np.percentile(values, [10.0, 50.0, 90.0])
    spread = p90 - p10
    if spread == 0:
        return 0.0
    return float(((p90 - p50) - (p50 - p10)) / spread)


def skew_adjusted_bounds(values: np.ndarray, whisker: float = 1.5) -> tuple[float, float]:
    """Skew-adjusted boxplot whisker bounds (Hubert & Vandervieren 2008-
    style exponential fence), using `quantile_skewness` (`sk`) in place of
    the medcouple:

    - `sk >= 0` (right-skewed): `lower = Q1 - whisker*e^(-4*sk)*IQR`,
      `upper = Q3 + whisker*e^(3*sk)*IQR`
    - `sk < 0` (left-skewed): `lower = Q1 - whisker*e^(-3*sk)*IQR`,
      `upper = Q3 + whisker*e^(4*sk)*IQR`

    so the fence on the distribution's longer/skewed side widens
    exponentially with the skew magnitude while the other side narrows --
    `sk == 0` collapses to the plain symmetric Tukey fence
    (`whisker*IQR` on each side). The fence values are then narrowed to the
    actual data min/max *within* them (matplotlib's own boxplot whisker
    convention -- a whisker only ever reaches a real data point, never the
    bare fence formula's own value), which also means a non-negative
    feature's lower bound can never come out negative, since no real data
    point can. A robust alternative to a mean +/- n_sig*std window, far less
    sensitive to a skewed tail or a handful of extreme outliers (which can
    otherwise blow that out). Used by both `compute_distribution_1d` (its
    histogram window/color scale) and `compute_distribution_2d` (each side's
    own marginal window for panel (B)'s joint-density plot).
    `values.size == 0` (e.g. every scan came out non-finite --
    a degenerate regression fit's near-zero slope can send every predicted
    value to +/-inf) returns `(0.0, 0.0)` rather than raising, matching
    `metrics.mean`/`std`'s own empty-input convention."""
    if values.size == 0:
        return 0.0, 0.0
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = q3 - q1
    sk = quantile_skewness(values)
    if sk >= 0:
        lower_fence = q1 - whisker * math.exp(-4 * sk) * iqr
        upper_fence = q3 + whisker * math.exp(3 * sk) * iqr
    else:
        lower_fence = q1 - whisker * math.exp(-3 * sk) * iqr
        upper_fence = q3 + whisker * math.exp(4 * sk) * iqr
    within = values[(values >= lower_fence) & (values <= upper_fence)]
    if within.size == 0:
        return float(values.min()), float(values.max())
    return float(within.min()), float(within.max())


def _none_to_nan(values: dict[str, float | None]) -> dict[str, float]:
    """Every metric adapter may return `None` for a degenerate/undefined
    case (see `metrics.kl_divergence_normal`/`pearson_correlation`) -- the
    stored `metrics` dict always holds plain floats (`nan` for "undefined"),
    matching how correlations/bootstrap already treat a missing value;
    `Distribution1D`/`Distribution2D`'s own `@property` wrappers convert
    back to `None` for the handful of call sites that want Optional
    semantics."""
    return {name: (math.nan if value is None else value) for name, value in values.items()}


@dataclass
class Distribution1D:
    counts: np.ndarray
    bin_edges: np.ndarray
    metrics: dict[str, float]  # keyed by pipeline.metrics.METRIC_NAMES_1D
    zero_count: int  # structural -- not a correlatable metric
    fitted_params: tuple[float, float, float] | None  # (amplitude, mu, sigma), or None

    @property
    def mean(self) -> float:
        return self.metrics["Mean"]

    @property
    def median(self) -> float:
        return self.metrics["Median"]

    @property
    def mode(self) -> float:
        return self.metrics["Mode"]

    @property
    def std(self) -> float:
        return self.metrics["Standard deviation"]

    @property
    def gini(self) -> float:
        return self.metrics["Gini index"]

    @property
    def shannon_entropy(self) -> float:
        return self.metrics["1D Shannon entropy"]

    @property
    def kl_divergence_normal(self) -> float | None:
        value = self.metrics["c-KL divergence (vs. normal)"]
        return None if math.isnan(value) else value

    @property
    def bin_centers(self) -> np.ndarray:
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def expected_gaussian(self, x: np.ndarray | None = None) -> np.ndarray:
        """The Gaussian implied by this distribution's own mean/std, scaled
        to the histogram's area -- what the data would look like if it were
        normally distributed. `x` defaults to `bin_centers` (the histogram's
        own bins); pass a denser array (e.g. a `linspace` over a zoomed plot
        window) for a smooth curve independent of the histogram's own bin
        count/range."""
        if x is None:
            x = self.bin_centers
        total = self.counts.sum()
        if self.std <= 0 or total == 0:
            return np.zeros_like(x)
        bin_width = self.bin_edges[1] - self.bin_edges[0] if self.bin_edges.size > 1 else 0.0
        amplitude = total * bin_width / (self.std * np.sqrt(2 * np.pi))
        return gaussian(x, amplitude, self.mean, self.std)

    def fitted_gaussian(self, x: np.ndarray | None = None) -> np.ndarray | None:
        """The Gaussian actually fit to the histogram via curve_fit, or None
        if fitting didn't converge (e.g. too little/degenerate data). `x`
        defaults to `bin_centers`, same as `expected_gaussian`."""
        if self.fitted_params is None:
            return None
        if x is None:
            x = self.bin_centers
        return gaussian(x, *self.fitted_params)


@dataclass
class Distribution2D:
    numerator_expression: str
    denominator_expression: str
    counts: np.ndarray  # shape (n_bins, n_bins) -- natural (unscaled) range
    bin_edges_num: np.ndarray  # shape (n_bins + 1,)
    bin_edges_den: np.ndarray  # shape (n_bins + 1,)
    metrics: dict[str, float]  # keyed by pipeline.metrics.METRIC_NAMES_2D
    shannon_entropy: float  # structural -- raw (unnormalized), not in METRIC_NAMES_2D itself
    display_counts: np.ndarray  # shape (display_bins, display_bins) -- skew-adjusted, for panel (B)
    display_bin_edges_num: np.ndarray  # shape (display_bins + 1,); natural units, may be negative
    display_bin_edges_den: np.ndarray  # ditto, for the denominator
    numerator_bounds: tuple[float, float]  # skew_adjusted_bounds(numerator) -- see that function
    denominator_bounds: tuple[float, float]  # ditto, for the denominator

    @property
    def gini(self) -> float:
        return self.metrics["2D Gini index"]

    @property
    def shannon_entropy_boundary(self) -> float:
        return self.metrics["2D Shannon entropy"]

    @property
    def pearson_correlation(self) -> float | None:
        value = self.metrics["Pearson-CR"]
        return None if math.isnan(value) else value

    @property
    def kl_divergence_gaussians(self) -> float | None:
        value = self.metrics["c-KL divergence"]
        return None if math.isnan(value) else value

    @property
    def kl_divergence_histograms(self) -> float:
        return self.metrics["d-KL divergence"]


def compute_distribution_1d(
    values: np.ndarray,
    n_bins: int = 150,
    fit_gaussian: bool = True,
    whisker: float = 1.5,
    window: str = "skew",
) -> Distribution1D:
    """Histogram + every `METRIC_NAMES_1D` metric for `values`, dropping any
    NaN/inf first (e.g. from an out-of-range peak in a composite
    expression).

    `window` chooses what range the bins span:

    * `"skew"` (default) -- the robust fence described below, which keeps a
      skewed tail or a few extreme outliers from flattening everything else
      into one bin.
    * `"full"` -- the finite data's own min to max, every value binned and
      none dropped. For a quantity whose range is meaningful rather than
      estimated (a probability on [0, 1], say), the rare extremes can be the
      interesting ones, and the fence is exactly what discards them.

    Only the histogram changes. Every metric computed from raw values is
    identical either way; see the list below for which are which.

    Under `"skew"` the histogram is windowed to `skew_adjusted_bounds(finite,
    whisker)` -- a skew-adjusted, exponential boxplot-whisker-style fence
    (default 1.5x IQR, widened/narrowed per side based on `quantile_
    skewness`), narrowed to the real data min/max within it -- a robust
    alternative to a mean+/-std window that isn't thrown off by a skewed
    tail or a handful of extreme outliers. Values outside the window are
    dropped, not clipped into the edge bins, same rationale as
    `compute_distribution_2d`'s own windowed `display_counts` -- then split
    into exactly `n_bins` (default 150) equal-width bins spanning that
    window. Zero-valued pixels are kept (not specially excluded). `Mean`/
    `Median`/`Standard deviation` (and `Gini index`/`c-KL divergence (vs.
    normal)`, computed from raw values, not bins) are unaffected and still
    reflect the full, unwindowed data; only bin-derived values (`Mode`,
    `Count at mean/median/mode`, `1D Shannon entropy`, the Gaussian fit, and
    what panels (A)/(C) of the report plot) follow the windowed histogram
    instead.

    `fit_gaussian=False` skips the `curve_fit` call entirely (leaving
    `fitted_params` as None) -- set by bootstrap iterations, which never
    read `fitted_params`/`fitted_gaussian()` (only report figures do) and
    would otherwise pay curve_fit's cost, and risk its convergence
    warnings, once per iteration per feature for nothing.

    The fit itself is bounded (`method="trf"`, amplitude >= 0, `mu` within
    `[vmin, vmax]`, `sigma` within `(0, vmax - vmin]`) and seeded from the
    windowed histogram's own peak (`counts.max()`/its own bin center/
    `windowed`'s std), not the full data's `mean`/`std` -- for a skewed
    feature those are a poor proxy for where the *windowed* peak actually
    sits, and unbounded `curve_fit` (the original approach) either fails to
    converge outright against the narrower window, or "succeeds" with a
    `mu` nonsensically outside the window entirely."""
    finite = values[np.isfinite(values)]
    zero_count = int(np.sum(finite == 0))

    mean = metrics.mean(finite)
    median = metrics.median(finite)
    std = metrics.std(finite)

    if window == "skew":
        vmin, vmax = skew_adjusted_bounds(finite, whisker)
    elif window == "full":
        # min to max of the finite values, so nothing is dropped. Empty input
        # gives (0.0, 0.0), matching skew_adjusted_bounds' own empty case.
        vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    else:
        raise DistributionError(f"window must be 'skew' or 'full', got {window!r}")
    windowed = finite[(finite >= vmin) & (finite <= vmax)]
    counts, bin_edges = np.histogram(windowed, bins=n_bins, range=(vmin, vmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mode = metrics.mode(counts, bin_centers, fallback=mean)

    context = MetricContext1D(x=finite, counts=counts, mean=mean, std=std)
    computed = _none_to_nan({name: fn(context) for name, fn in metrics.METRICS_1D.items()})
    computed.update(
        {
            "Mean": mean,
            "Median": median,
            "Mode": mode,
            "Standard deviation": std,
            "Maximum value": metrics.maximum(finite),
            "Minimum value": metrics.minimum(finite),
            "Count at mean": metrics.count_at_value(counts, bin_edges, mean),
            "Count at median": metrics.count_at_value(counts, bin_edges, median),
            "Count at mode": metrics.count_at_value(counts, bin_edges, mode),
        }
    )

    fitted_params: tuple[float, float, float] | None = None
    if fit_gaussian and std > 0 and finite.size and vmax > vmin:
        try:
            peak_idx = int(np.argmax(counts))
            p0 = [float(counts[peak_idx]), float(bin_centers[peak_idx]), float(np.std(windowed))]
            fit_bounds = ([0.0, vmin, 1e-9], [np.inf, vmax, vmax - vmin])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                params, _ = curve_fit(
                    gaussian, bin_centers, counts, p0=p0, bounds=fit_bounds, method="trf"
                )
            for warning in caught:
                logger.debug("Gaussian fit warning: %s", warning.message)
            fitted_params = (float(params[0]), float(params[1]), float(params[2]))
        except RuntimeError:
            logger.debug("Gaussian fit did not converge -- skipping fitted curve")

    return Distribution1D(
        counts=counts,
        bin_edges=bin_edges,
        metrics=computed,
        zero_count=zero_count,
        fitted_params=fitted_params,
    )


def compute_distribution_2d(
    numerator: np.ndarray,
    denominator: np.ndarray,
    numerator_expression: str,
    denominator_expression: str,
    n_bins: int = 150,
    display_bins: int = 150,
    whisker: float = 1.5,
) -> Distribution2D:
    """Joint histogram + every `METRIC_NAMES_2D` metric for a ratio
    feature's numerator and denominator raw values (see
    `features.expression.split_ratio`). Panel (B)'s display window (see
    `numerator_bounds`/`denominator_bounds` below) is `skew_adjusted_bounds`
    applied independently to each side, same skew-adjusted exponential
    boxplot-whisker fence as `compute_distribution_1d`'s own histogram
    window -- not a fixed mean+/-n_sig*std window."""
    counts, bin_edges_num, bin_edges_den = joint_histogram(numerator, denominator, n_bins)
    flat_counts = counts.flatten()
    nonzero_counts = flat_counts[flat_counts > 0]

    finite_num = numerator[np.isfinite(numerator)]
    finite_den = denominator[np.isfinite(denominator)]
    counts_num, _ = np.histogram(finite_num, bins=n_bins)
    counts_den, _ = np.histogram(finite_den, bins=n_bins)

    mean_num, std_num = metrics.mean(finite_num), metrics.std(finite_num)
    mean_den, std_den = metrics.mean(finite_den), metrics.std(finite_den)

    context = MetricContext2D(
        flat_counts=flat_counts,
        nonzero_counts=nonzero_counts,
        counts_num=counts_num.astype(np.float64),
        counts_den=counts_den.astype(np.float64),
        mean_num=mean_num,
        std_num=std_num,
        mean_den=mean_den,
        std_den=std_den,
    )
    computed = _none_to_nan({name: fn(context) for name, fn in metrics.METRICS_2D.items()})

    num_bound_min, num_bound_max = skew_adjusted_bounds(finite_num, whisker)
    den_bound_min, den_bound_max = skew_adjusted_bounds(finite_den, whisker)

    # Keep a scan only if BOTH sides are finite and within their own
    # skew-adjusted fence -- dropped, not clipped, so a handful of extreme
    # outliers don't pile up in (and distort) the windowed histogram's edge
    # bins.
    within_window = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator >= num_bound_min)
        & (numerator <= num_bound_max)
        & (denominator >= den_bound_min)
        & (denominator <= den_bound_max)
    )
    windowed_num = numerator[within_window]
    windowed_den = denominator[within_window]
    display_counts, display_bin_edges_num, display_bin_edges_den = np.histogram2d(
        windowed_num,
        windowed_den,
        bins=display_bins,
        range=[[num_bound_min, num_bound_max], [den_bound_min, den_bound_max]],
    )

    return Distribution2D(
        numerator_expression=numerator_expression,
        denominator_expression=denominator_expression,
        counts=counts,
        bin_edges_num=bin_edges_num,
        bin_edges_den=bin_edges_den,
        metrics=computed,
        shannon_entropy=metrics.shannon_entropy(flat_counts),
        display_counts=display_counts,
        display_bin_edges_num=display_bin_edges_num,
        display_bin_edges_den=display_bin_edges_den,
        numerator_bounds=(num_bound_min, num_bound_max),
        denominator_bounds=(den_bound_min, den_bound_max),
    )
