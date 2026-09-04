"""Exports one bootstrap run's per-iteration metrics to a flat CSV -- a
plain-file counterpart to `reporting/`'s output (PNGs/PDFs), written to
`<project_root>/exports/` (a sibling of `reports/` and `results.zarr`), for
downstream analysis in spreadsheet/plotting tools outside pylibs itself.
Uses the stdlib `csv` module, no pandas, matching `project.sample_properties`'s
existing no-pandas convention.
"""

import csv
from pathlib import Path

import numpy as np

EXPORTS_DIRNAME = "exports"


def export_bootstrap_csv(
    project_root: Path,
    sample_id: str,
    bootstrap_id: str,
    feature_ids: list[str],
    feature_labels: dict[str, str],
    metric_names: list[str],
    values: np.ndarray,  # (n_iterations, n_features, n_metrics)
    completed: np.ndarray,  # bool-like, shape (n_iterations,)
) -> Path:
    """Write one row per iteration to
    `<project_root>/exports/<sample_id>_<bootstrap_id>.csv`: `iteration`,
    `completed`, then one column per (feature, metric) pair, headed
    `'<feature label> - <metric name>'` (`feature_labels` falls back to the
    feature id itself if a feature isn't in the mapping). A not-yet-computed
    iteration is still written as a row (blank metric cells) rather than
    omitted, so row count always equals `n_iterations` regardless of
    progress -- matching the checkpoint arrays' own fixed-length shape."""
    exports_dir = project_root / EXPORTS_DIRNAME
    exports_dir.mkdir(parents=True, exist_ok=True)
    path = exports_dir / f"{sample_id}_{bootstrap_id}.csv"

    header = ["iteration", "completed"] + [
        f"{feature_labels.get(feature_id, feature_id)} - {metric_name}"
        for feature_id in feature_ids
        for metric_name in metric_names
    ]
    n_iterations = values.shape[0]
    n_cells = len(feature_ids) * len(metric_names)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_iterations):
            is_done = bool(completed[i])
            row: list[object] = [i, is_done]
            row.extend(values[i].reshape(-1).tolist() if is_done else [""] * n_cells)
            writer.writerow(row)
    return path
