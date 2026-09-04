"""Every metric this project computes over a feature's distribution(s) --
1D (a single feature's per-scan values) and 2D (a ratio feature's numerator/
denominator joint distribution) -- lives here, and only here.

To add a metric: write a pure function below (in terms of raw values, a
histogram's counts, or a Gaussian's (mean, std) -- whatever the metric
naturally needs, so it stays directly callable/testable on its own, e.g.
`gini_index(some_array)` in a notebook, no distribution object required),
add a one-line adapter to `METRICS_1D`/`METRICS_2D` pulling whatever pieces
of `MetricContext1D`/`MetricContext2D` it needs, and add its display name to
`METRIC_NAMES_1D`/`METRIC_NAMES_2D`. Nothing in `distributions.py`,
`store.py`, or `api.py` needs touching after that -- `distributions.
compute_distribution_1d`/`compute_distribution_2d` build the context once
(the histogram, mean, std -- everything every metric might need) and loop
the registry generically; the result lands in `Distribution1D`/
`Distribution2D`'s `metrics: dict[str, float]` field, which is what
`feature_metric_values` (below), storage, correlations, and bootstrap all
read from.

The exception is "descriptive metrics" (mean/median/mode/std/max/min/
count-at-X, below): mode needs the histogram already built, and "count at
mode" needs mode's own value first, so they're inherently order-dependent
rather than independently computable from one shared context. They're still
plain functions here -- just called explicitly, in dependency order, by
`distributions.py`'s orchestrators instead of through the registry.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pylibs.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pure metric math -- each function is directly callable/testable on its own
# minimal natural input, independent of any distribution object.
# ---------------------------------------------------------------------------


def gini_index(values: np.ndarray) -> float:
    """Gini index of inequality, 0 (perfectly uniform) to ~1 (concentrated
    in a few values). Defined for non-negative values; an all-zero array is
    treated as perfectly uniform (gini 0), not a division by zero.

    Used for both 1D (raw per-scan values -- "how unequal is intensity
    across scans") and 2D (non-zero joint-histogram bin counts -- "how
    concentrated is the joint density across bins") -- same formula, two
    different populations; see `METRICS_1D`/`METRICS_2D` for which array
    each gets. The 2D case can never go negative (histogram counts are
    always >= 0), but a 1D feature that can itself be negative (e.g. a
    log-ratio, negative whenever the ratio is < 1) can drive the formula
    negative -- not a real Gini coefficient at that point, so that's logged
    (INFO -- file-only by default, not console; see `core.logging`) and
    reported as NaN rather than a misleading negative number."""
    if values.size == 0:
        return 1.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    total = sorted_values.sum()
    if total == 0:
        return 0.0
    weighted_sum = np.sum(np.arange(1, n + 1) * sorted_values)
    result = float(2.0 * weighted_sum / (n * total) - (n + 1.0) / n)
    if result < 0:
        logger.info(
            "gini_index computed a negative value (%.6g) -- only defined for "
            "non-negative input, so this usually means the underlying feature can "
            "itself be negative (e.g. a log-ratio below 1); returning NaN instead.",
            result,
        )
        return math.nan
    return result


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy (bits) of a histogram's bin counts (1D or flattened
    2D), not normalized -- see `normalized_shannon_entropy` for the
    boundary-normalized 2D metric."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def boundary_shannon_entropy(total: float) -> float:
    """The maximum possible Shannon entropy (bits) for `total` items spread
    as uniformly as possible over `total` categories -- the normalizing
    constant for `normalized_shannon_entropy`."""
    if total <= 0:
        return 0.0
    p = 1.0 / total
    return float(-total * (p * np.log2(p)))


def normalized_shannon_entropy(counts: np.ndarray) -> float:
    """`shannon_entropy(counts)` scaled to 0-1 by `boundary_shannon_entropy`
    -- the "2D Shannon entropy" metric (the 1D metric uses raw
    `shannon_entropy` unscaled)."""
    entropy = shannon_entropy(counts)
    boundary = boundary_shannon_entropy(float(counts.sum()))
    return entropy / boundary if boundary > 0 else 0.0


def kl_divergence_normal(
    mu: float, sigma: float, mu_ref: float = 0.0, sigma_ref: float = 1.0
) -> float | None:
    """Continuous KL divergence KL(N(mu, sigma) || N(mu_ref, sigma_ref)), or
    None if either distribution is degenerate (sigma <= 0). Defaults to
    comparing against a standard normal (the 1D "vs. normal" metric); pass
    another distribution's own (mu, sigma) for the 2D numerator-vs-
    denominator metric -- same function, both directions."""
    if sigma <= 0 or sigma_ref <= 0:
        return None
    return float(
        np.log(sigma_ref / sigma) + (sigma**2 + (mu - mu_ref) ** 2) / (2 * sigma_ref**2) - 0.5
    )


def kl_divergence_histograms(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    smoothing: float = 0.0,
    epsilon: float = 1e-10,
) -> float:
    """Normalized discrete Kullback-Leibler divergence D_KL(p || q) between
    two same-shape histograms, where `p = counts_a / counts_a.sum()` and
    `q = counts_b / counts_b.sum()`.

    Returns `nan` if either histogram's counts sum to <= 0 (before
    `smoothing`, see below) -- an empty histogram means "no data", a
    fundamentally different, undefined case from "these two distributions
    are identical" (which is what a `0.0` result would otherwise claim).

    Summed only over bins where `p_m > 0` (the standard `0 * ln(0) = 0`
    convention). Where `p_m > 0` but `q_m == 0`, KL diverges to infinity;
    `q` is smoothed by `epsilon` -- added to every bin of `counts_b` before
    it's normalized -- so it's always strictly positive, at the cost of a
    small, tunable downward bias versus the true (possibly infinite)
    divergence. `p` is left unsmoothed: `p_m == 0` bins are already handled
    by the summation convention above and don't need it.

    `smoothing` (default 0, i.e. off) optionally adds a constant to both
    histograms' raw *counts* -- before any normalizing -- for numerical
    stability with very sparse histograms; unlike `epsilon`, it also turns
    an all-zero histogram into a valid (uniform) distribution instead of
    `nan`."""
    a = counts_a.astype(np.float64)
    b = counts_b.astype(np.float64)
    if smoothing:
        a = a + smoothing
        b = b + smoothing

    sum_a, sum_b = float(a.sum()), float(b.sum())
    if sum_a <= 0 or sum_b <= 0:
        return float("nan")

    p = a / sum_a
    b_smoothed = b + epsilon
    q = b_smoothed / b_smoothed.sum()

    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation coefficient between `a` and `b`, or None if
    either is constant (correlation is undefined). For the 2D "Pearson-CR"
    metric, `a`/`b` are the numerator's and denominator's own 1D histogram
    bin counts (their distribution *shapes*), not raw per-scan value
    pairs."""
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else 0.0


def median(x: np.ndarray) -> float:
    return float(np.median(x)) if x.size else 0.0


def std(x: np.ndarray) -> float:
    return float(x.std()) if x.size else 0.0


def mode(counts: np.ndarray, bin_centers: np.ndarray, fallback: float) -> float:
    """The bin center with the highest count, or `fallback` (the
    distribution's own mean) if every bin is empty."""
    return float(bin_centers[np.argmax(counts)]) if counts.size and counts.max() > 0 else fallback


def maximum(x: np.ndarray) -> float:
    return float(x.max()) if x.size else 0.0


def minimum(x: np.ndarray) -> float:
    return float(x.min()) if x.size else 0.0


def count_at_value(counts: np.ndarray, bin_edges: np.ndarray, value: float) -> float:
    """`counts[bin]` for the bin containing `value` (clipped to the valid
    index range, since `value` -- e.g. the mean -- can sit exactly on the
    last bin edge)."""
    index = int(np.searchsorted(bin_edges, value, side="right")) - 1
    index = max(0, min(index, len(counts) - 1))
    return float(counts[index])


# ---------------------------------------------------------------------------
# Registry: one shared "everything a metric might need" context per
# dimensionality, built once by distributions.py's orchestrators, plus a
# name -> adapter mapping so METRICS_1D/METRICS_2D can be looped generically.
# ---------------------------------------------------------------------------


@dataclass
class MetricContext1D:
    x: np.ndarray  # finite raw per-scan values
    counts: np.ndarray  # 1D histogram bin counts
    mean: float
    std: float


@dataclass
class MetricContext2D:
    flat_counts: np.ndarray  # full flattened joint histogram (all bins, incl. empty)
    nonzero_counts: np.ndarray  # flat_counts with empty bins excluded (see gini_index's docstring)
    counts_num: np.ndarray  # numerator's own 1D histogram counts
    counts_den: np.ndarray  # denominator's own 1D histogram counts
    mean_num: float
    std_num: float
    mean_den: float
    std_den: float


MetricAdapter1D = Callable[[MetricContext1D], float | None]
MetricAdapter2D = Callable[[MetricContext2D], float | None]

METRICS_1D: dict[str, MetricAdapter1D] = {
    "Gini index": lambda ctx: gini_index(ctx.x),
    "1D Shannon entropy": lambda ctx: shannon_entropy(ctx.counts),
    "c-KL divergence (vs. normal)": lambda ctx: kl_divergence_normal(ctx.mean, ctx.std),
}

METRICS_2D: dict[str, MetricAdapter2D] = {
    "2D Gini index": lambda ctx: gini_index(ctx.nonzero_counts) if ctx.nonzero_counts.size else 0.0,
    "2D Shannon entropy": lambda ctx: normalized_shannon_entropy(ctx.flat_counts),
    "Pearson-CR": lambda ctx: pearson_correlation(ctx.counts_num, ctx.counts_den),
    "c-KL divergence": lambda ctx: kl_divergence_normal(
        ctx.mean_num, ctx.std_num, mu_ref=ctx.mean_den, sigma_ref=ctx.std_den
    ),
    "d-KL divergence": lambda ctx: kl_divergence_histograms(ctx.counts_num, ctx.counts_den),
}


# ---------------------------------------------------------------------------
# Name vocabulary
# ---------------------------------------------------------------------------

# The 12 metrics every feature has (from its Distribution1D).
METRIC_NAMES_1D: list[str] = [
    "Mean",
    "Median",
    "Mode",
    "Standard deviation",
    "Gini index",
    "1D Shannon entropy",
    "Maximum value",
    "Minimum value",
    "Count at mean",
    "Count at median",
    "Count at mode",
    "c-KL divergence (vs. normal)",
]

# The 5 metrics only a ratio feature has (from its Distribution2D).
METRIC_NAMES_2D: list[str] = [
    "2D Gini index",
    "2D Shannon entropy",
    "Pearson-CR",
    "c-KL divergence",
    "d-KL divergence",
]

METRIC_NAMES: list[str] = METRIC_NAMES_1D + METRIC_NAMES_2D


def feature_metric_values(
    distribution_1d: object, distribution_2d: object | None
) -> dict[str, float]:
    """One value per name in `METRIC_NAMES` for a single (feature, sample)
    pair. `distribution_1d`/`distribution_2d` are `distributions.
    Distribution1D`/`Distribution2D` (typed there, not here, to avoid a
    circular import -- both already carry a `metrics: dict[str, float]`
    field keyed exactly by `METRIC_NAMES_1D`/`METRIC_NAMES_2D`, so this is
    just a merge). The five `METRIC_NAMES_2D` keys are `nan` when
    `distribution_2d` is None (the feature isn't a ratio)."""
    values_1d: dict[str, float] = distribution_1d.metrics  # type: ignore[attr-defined]
    if distribution_2d is None:
        return {**values_1d, **dict.fromkeys(METRIC_NAMES_2D, math.nan)}
    values_2d: dict[str, float] = distribution_2d.metrics  # type: ignore[attr-defined]
    return {**values_1d, **values_2d}
