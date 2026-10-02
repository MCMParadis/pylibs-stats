import math

import numpy as np
import pytest

from pylibs.core.exceptions import DistributionError
from pylibs.core.pipeline import metrics as metrics_module
from pylibs.core.pipeline.distributions import (
    compute_distribution_1d,
    compute_distribution_2d,
    joint_histogram,
)
from pylibs.core.pipeline.metrics import gini_index, kl_divergence_histograms


def test_kl_divergence_histograms_identical_is_zero():
    counts = np.array([5.0, 3.0, 2.0, 0.0])
    # not exactly 0.0: q is epsilon-smoothed (see kl_divergence_histograms's
    # docstring) even when counts_a == counts_b, but the perturbation is
    # ~1e-10, far below this tolerance
    assert kl_divergence_histograms(counts, counts) == pytest.approx(0.0, abs=1e-8)


def test_kl_divergence_histograms_known_value():
    # p = [0.75, 0.25], q = [0.25, 0.75]
    # D_KL(p || q) = 0.75*ln(0.75/0.25) + 0.25*ln(0.25/0.75) = 0.5*ln(3)
    counts_a = np.array([3.0, 1.0])
    counts_b = np.array([1.0, 3.0])
    expected = 0.5 * math.log(3)
    assert kl_divergence_histograms(counts_a, counts_b) == pytest.approx(expected, abs=1e-8)


def test_kl_divergence_histograms_empty_histogram_is_nan():
    counts_a = np.array([0.0, 0.0, 0.0])
    counts_b = np.array([1.0, 2.0, 3.0])
    assert math.isnan(kl_divergence_histograms(counts_a, counts_b))
    assert math.isnan(kl_divergence_histograms(counts_b, counts_a))


def test_kl_divergence_histograms_smoothing_avoids_nan_for_empty_histogram():
    counts_a = np.array([0.0, 0.0, 0.0])
    counts_b = np.array([1.0, 2.0, 3.0])
    result = kl_divergence_histograms(counts_a, counts_b, smoothing=1.0)
    assert not math.isnan(result)


def test_gini_index_negative_result_is_nan():
    # sum is negative, driving the formula negative -- not a real Gini
    # coefficient (gini_index is only defined for non-negative input), so
    # it should log it (INFO) and report NaN rather than a misleading
    # negative number
    values = np.array([-1.0, -2.0, -3.0, -10.0])
    assert math.isnan(gini_index(values))


def test_distribution_2d_gini_excludes_empty_bins():
    # most of a 4x4=16-bin joint histogram of these 4 points is empty
    numerator = np.array([1.0, 1.0, 1.0, 5.0])
    denominator = np.array([1.0, 1.0, 1.0, 5.0])
    distribution = compute_distribution_2d(
        numerator, denominator, "num", "den", n_bins=4, display_bins=4
    )

    counts, _, _ = joint_histogram(numerator, denominator, n_bins=4)
    flat = counts.flatten()
    expected = gini_index(flat[flat > 0])
    assert distribution.gini == pytest.approx(expected)
    # sanity: including the empty bins would give a much higher (near-1)
    # gini, dominated by histogram sparsity rather than the data's shape
    assert gini_index(flat) > distribution.gini


def test_distribution_1d_full_window_keeps_a_rare_tail():
    """A bounded quantity whose rare high values are the point: the default
    skew-adjusted fence discards them as outliers, `window="full"` bins the
    whole range and keeps them."""
    values = np.full(10_000, 0.002)
    values[:30] = 0.95  # e.g. the pixels of an image where an element is present

    fenced = compute_distribution_1d(values)
    assert fenced.bin_edges[-1] < 0.95  # the 30 are outside the window entirely
    assert fenced.counts.sum() == 9970

    full = compute_distribution_1d(values, window="full")
    assert (full.bin_edges[0], full.bin_edges[-1]) == (0.002, 0.95)
    assert full.counts.sum() == 10_000
    # metrics computed from the raw values are the same either way
    assert full.mean == fenced.mean and full.gini == fenced.gini


def test_distribution_1d_window_defaults_to_skew_and_rejects_anything_else():
    values = np.concatenate([np.linspace(0.0, 1.0, 200), np.full(3, 50.0)])

    default = compute_distribution_1d(values)
    explicit = compute_distribution_1d(values, window="skew")
    assert np.array_equal(default.counts, explicit.counts)
    assert np.array_equal(default.bin_edges, explicit.bin_edges)

    with pytest.raises(DistributionError, match="'skew' or 'full'"):
        compute_distribution_1d(values, window="minmax")


def test_distribution_1d_property_matches_metrics_dict():
    # the hybrid design: .gini (and friends) are thin properties reading
    # straight out of the generic `metrics` dict -- both must always agree
    values = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 100.0])
    distribution = compute_distribution_1d(values)
    assert distribution.gini == distribution.metrics["Gini index"]
    assert distribution.mean == distribution.metrics["Mean"]
    assert distribution.std == distribution.metrics["Standard deviation"]


def test_new_registry_metric_appears_in_computed_metrics_without_other_changes():
    """Proves the actual extensibility claim: registering a brand-new metric
    only in `metrics.METRICS_1D` makes it show up in `Distribution1D.metrics`
    -- nothing in `distributions.py` needs touching."""
    values = np.array([1.0, 2.0, 3.0, 4.0])

    def double_the_mean(ctx: metrics_module.MetricContext1D) -> float:
        return ctx.mean * 2

    metrics_module.METRICS_1D["Test double mean"] = double_the_mean
    try:
        distribution = compute_distribution_1d(values)
        assert distribution.metrics["Test double mean"] == pytest.approx(2 * values.mean())
    finally:
        del metrics_module.METRICS_1D["Test double mean"]
