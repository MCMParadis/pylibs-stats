"""Cross-sample correlation matrices: pure formatting over already-computed,
already-stored regressions -- **not** its own computation. A correlation
matrix's four values per (feature, metric) cell (Pearson r, R², MAE, MAE's
permutation-test p-value) are exactly what a univariate regression of that
metric against the external property produces (see `pipeline.regression.
fit_least_squares`), so nothing here ever fits anything; `api.
compute_correlations` reads each cell's already-stored `FeatureRegression`
(see `pipeline.regression`/`api.compute_regressions`) and this module just
reshapes those readings into `CorrelationMatrices`. A cell whose regression
hasn't been computed yet (most commonly: `compute_regressions` was only run
for one metric, not an all-metrics sweep) is left `nan` -- `api.
compute_correlations` logs how many, it's never recomputed here.

Unlike the rest of `core.pipeline`, nothing here streams a reader or belongs
to a `Step` -- this operates on a whole project's already-cached per-sample/
per-feature regressions, not on one sample at a time.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrelationMatrices:
    column_name: str
    feature_ids: list[str]
    metric_names: list[str]
    pearson_r: np.ndarray  # shape (n_features, n_metrics)
    r_squared: np.ndarray  # == pearson_r ** 2 elementwise
    mae: np.ndarray  # mean absolute residual of a univariate OLS fit (metric ~ property)
    mae_p_value: np.ndarray  # permutation-test p-value on `mae` -- see fit_least_squares


def compute_correlation_matrices(
    column_name: str,
    feature_ids: list[str],
    metric_names: list[str],
    pearson_r: np.ndarray,
    r_squared: np.ndarray,
    mae: np.ndarray,
    mae_p_value: np.ndarray,
) -> CorrelationMatrices:
    """Packages already-gathered per-(feature, metric) regression results
    (each array shape (n_features, n_metrics), `nan` for a cell with no
    stored regression) into a `CorrelationMatrices` -- the actual per-cell
    store reads happen in `api.compute_correlations`, not here."""
    return CorrelationMatrices(
        column_name=column_name,
        feature_ids=feature_ids,
        metric_names=metric_names,
        pearson_r=pearson_r,
        r_squared=r_squared,
        mae=mae,
        mae_p_value=mae_p_value,
    )
