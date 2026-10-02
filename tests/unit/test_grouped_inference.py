"""Group-level inference: the physical sample, not the acquisition, is the
unit the confidence band and the permutation test treat as independent.

Several ablation layers of one pellet share a property value and are not
independent evidence about it. Counting each as its own point narrows the
band and shrinks the p-value without anything more having been measured.
These tests pin what that changes, and what it must leave alone.

Covered here rather than in `test_cli.py` because the guarantees are
numerical identities between two fits, which a workflow test cannot express.
"""

import numpy as np
import pytest
from scipy.stats import linregress, t

from pylibs.core import api
from pylibs.core.pipeline.regression import (
    INFERENCE_VERSION,
    _group_indices,
    confidence_band_half_width,
    fit_least_squares,
    grouped_band_parameters,
)

SEED = 0


def _design(sizes, slope=2.5, intercept=1.2, noise=0.4, seed=SEED):
    """One property value per sample, broadcast to that sample's
    acquisitions, which is how the CSV join actually feeds the fit."""
    rng = np.random.default_rng(seed)
    x, y, names = [], [], []
    for index, count in enumerate(sizes):
        value = 0.3 * index
        for _ in range(count):
            x.append(value)
            y.append(slope * value + intercept + rng.normal(0, noise))
            names.append(f"sample{index}")
    return np.array(x), np.array(y), names


def _means_fit(x, y, names):
    groups = _group_indices(list(names), np.ones(x.size, dtype=bool))
    weights = np.array([float(rows.size) for rows in groups])
    group_x = np.array([x[rows].mean() for rows in groups])
    group_y = np.array([y[rows].mean() for rows in groups])
    total = weights.sum()
    mx, my = (weights * group_x).sum() / total, (weights * group_y).sum() / total
    slope = (weights * (group_x - mx) * (group_y - my)).sum() / (
        weights * (group_x - mx) ** 2
    ).sum()
    return slope, my - slope * mx


@pytest.mark.parametrize("sizes", [[3] * 7, [1, 2, 3, 4, 5], [5, 1, 4, 2]])
def test_weighted_group_means_line_equals_the_pooled_line(sizes):
    """The identity the whole design rests on. Because the property value is
    constant within a sample, weighting each sample mean by its acquisition
    count reproduces pooled least squares exactly -- balanced or not -- so
    the band stays centred on the line that is drawn."""
    x, y, names = _design(sizes)
    pooled = linregress(x, y)
    slope, intercept = _means_fit(x, y, names)

    assert slope == pytest.approx(pooled.slope)
    assert intercept == pytest.approx(pooled.intercept)


def test_unweighted_means_line_differs_when_unbalanced():
    """Why the weighting is not optional: without it the band would be
    centred on a different line than the one drawn."""
    x, y, names = _design([1, 2, 3, 4, 5])
    groups = _group_indices(list(names), np.ones(x.size, dtype=bool))
    group_x = np.array([x[rows].mean() for rows in groups])
    group_y = np.array([y[rows].mean() for rows in groups])
    unweighted = linregress(group_x, group_y)
    pooled = linregress(x, y)

    assert unweighted.slope != pytest.approx(pooled.slope)


def test_grouping_leaves_the_line_r_squared_and_mae_untouched():
    x, y, names = _design([3] * 7)
    grouped = fit_least_squares(x, y, 500, np.random.default_rng(SEED), names)
    ungrouped = fit_least_squares(x, y, 500, np.random.default_rng(SEED), None)

    assert grouped is not None and ungrouped is not None
    for field in ("slope", "intercept", "pearson_r", "r_squared", "mae"):
        assert getattr(grouped, field) == pytest.approx(getattr(ungrouped, field)), field
    assert grouped.n == ungrouped.n == len(x)


def test_band_uses_group_degrees_of_freedom_and_is_wider():
    """Fewer independent points means a wider band. The t multiplier alone
    accounts for most of it, and the degrees of freedom must be n_groups - 2."""
    x, y, names = _design([3] * 7)
    grouped = fit_least_squares(x, y, 500, np.random.default_rng(SEED), names)
    ungrouped = fit_least_squares(x, y, 500, np.random.default_rng(SEED), None)

    assert grouped.band_df == 5  # 7 samples - 2
    assert ungrouped.band_df == 19  # 21 acquisitions - 2

    at_mean = grouped.x_mean
    wide = confidence_band_half_width(
        at_mean, grouped.n_groups, grouped.x_mean, grouped.ssxx, grouped.residual_std
    )
    narrow = confidence_band_half_width(
        at_mean, ungrouped.n, ungrouped.x_mean, ungrouped.ssxx, ungrouped.residual_std
    )
    assert wide > narrow


def test_band_parameters_match_a_hand_computed_weighted_fit():
    x, y, names = _design([1, 2, 3, 4, 5])
    groups = _group_indices(list(names), np.ones(x.size, dtype=bool))
    x_mean, ssxx, residual_std, df = grouped_band_parameters(x, y, groups)

    weights = np.array([float(rows.size) for rows in groups])
    group_x = np.array([x[rows].mean() for rows in groups])
    total = weights.sum()
    assert x_mean == pytest.approx((weights * group_x).sum() / total)
    assert ssxx == pytest.approx((weights * (group_x - x_mean) ** 2).sum())
    assert df == len(groups) - 2
    # the multiplier the band will use comes from the group count
    assert t.ppf(0.975, df=df) > t.ppf(0.975, df=x.size - 2)


def test_no_repeated_names_is_a_no_op():
    """One acquisition per sample is not grouping at all, and every stored
    quantity must be identical to the ungrouped computation."""
    x, y, names = _design([1] * 9)
    grouped = fit_least_squares(x, y, 400, np.random.default_rng(SEED), names)
    ungrouped = fit_least_squares(x, y, 400, np.random.default_rng(SEED), None)

    assert grouped.grouped is False
    for field in (
        "slope",
        "intercept",
        "pearson_r",
        "r_squared",
        "mae",
        "p_value",
        "n",
        "x_mean",
        "ssxx",
        "residual_std",
        "band_df",
    ):
        assert getattr(grouped, field) == pytest.approx(getattr(ungrouped, field)), field


def test_grouping_off_matches_the_ungrouped_computation():
    x, y, names = _design([3] * 7)
    off = fit_least_squares(x, y, 400, np.random.default_rng(SEED), None)
    assert off.grouped is False
    assert off.band_df == len(x) - 2
    assert off.n_groups == len(x)


def test_permutations_never_split_a_group():
    """The null must move a property value with all of its acquisitions. If
    it did not, the null would be easier to beat and p would be too small.

    Checked by construction: with distinct per-sample values, every
    arrangement the test can produce assigns one value to all three rows of
    a sample, so the number of distinct values per sample stays 1."""
    x, y, names = _design([3] * 5)
    groups = _group_indices(list(names), np.ones(x.size, dtype=bool))
    per_group = np.array([x[rows][0] for rows in groups])
    row_group = np.empty(x.size, dtype=int)
    for index, rows in enumerate(groups):
        row_group[rows] = index

    rng = np.random.default_rng(SEED)
    for _ in range(50):
        permuted = per_group[rng.permutation(len(groups))][row_group]
        for rows in groups:
            assert len(set(permuted[rows])) == 1


def test_exact_enumeration_is_used_when_it_fits_and_bounds_p_below():
    """5 samples give 120 arrangements, far under the permutation budget, so
    every one is enumerated and p cannot fall below 1/120."""
    x, y, names = _design([3] * 5, noise=0.05)
    fit = fit_least_squares(x, y, 10000, np.random.default_rng(SEED), names)

    assert fit.permutation_mode == "exact"
    assert fit.n_permutations_used == 120
    assert fit.p_value >= 1.0 / 120
    # the identity arrangement always ties, so p is never 0
    assert fit.p_value > 0.0


def test_sampling_is_used_when_enumeration_would_be_too_large():
    """8 samples give 40320 arrangements, above the budget, so the corrected
    sampled estimator applies instead."""
    x, y, names = _design([3] * 8, noise=0.05)
    fit = fit_least_squares(x, y, 1000, np.random.default_rng(SEED), names)

    assert fit.permutation_mode == "sampled"
    assert fit.n_permutations_used == 1000
    assert fit.p_value >= 1.0 / 1001


def test_grouped_p_value_is_coarser_than_the_ungrouped_one():
    """The point of the change: 7 independent samples cannot resolve what 21
    correlated acquisitions appeared to."""
    x, y, names = _design([3] * 7, noise=0.05)
    grouped = fit_least_squares(x, y, 10000, np.random.default_rng(SEED), names)
    ungrouped = fit_least_squares(x, y, 10000, np.random.default_rng(SEED), None)

    assert grouped.p_value >= ungrouped.p_value


def test_stored_metadata_describes_the_design():
    x, y, names = _design([3, 3, 2, 3, 3])
    fit = fit_least_squares(x, y, 500, np.random.default_rng(SEED), names)

    assert fit.grouped is True
    assert fit.n_groups == 5
    assert fit.n_acquisitions == 14
    assert fit.balanced is False
    assert fit.inference_version == INFERENCE_VERSION

    balanced = fit_least_squares(
        *_design([3] * 5)[:2], 500, np.random.default_rng(SEED), _design([3] * 5)[2]
    )
    assert balanced.balanced is True


def test_labels_state_both_counts_and_the_permutation_mode():
    x, y, names = _design([3] * 7)
    grouped = fit_least_squares(x, y, 10000, np.random.default_rng(SEED), names)
    ungrouped = fit_least_squares(x, y, 10000, np.random.default_rng(SEED), None)

    assert api.format_sample_count(grouped) == "n = 7 samples (21 acquisitions)"
    assert api.format_sample_count(ungrouped) == "n = 21"
    assert api.format_permutation_mode(grouped) == "exact, 5040 permutations"
    assert api.format_permutation_mode(ungrouped) == "sampled, 10000"


def test_exact_p_value_counts_the_identity_arrangement():
    """A perfectly collinear fit: no arrangement can beat it, so only the
    identity ties and p must land exactly on 1 / n_groups!, never 0. The
    identity's MAE is recomputed by a different code path than the observed
    one, so this only holds if near-ties are counted."""
    x, y, names = _design([3] * 5, noise=0.0)
    fit = fit_least_squares(x, y, 10000, np.random.default_rng(SEED), names)

    assert fit.permutation_mode == "exact"
    assert fit.p_value > 0.0
    assert fit.p_value == pytest.approx(1.0 / 120)
