"""Regression report: one PDF per external CSV column, one calibration-curve
page per regressed feature -- an error-barred point per physical sample, a
dashed fitted line, and a 95% confidence-interval band (see
`pipeline.regression.FeatureRegression`/`confidence_band_half_width`) --
matching the reference paper's calibration-curve figure. `p_value` is a
permutation-test p-value on the fit's own residual MAE (see
`pipeline.regression.fit_least_squares`), not an analytic test, and the
page states both how it was obtained (every arrangement enumerated, or a
sample of them) and what counted as independent: "n = 7 samples (21
acquisitions)" when several acquisitions share a sample name, plain
"n = 21" when none do. The band is drawn from whatever
`x_mean`/`ssxx`/`residual_std` the fit stored, which are the grouped ones
when grouping applied. Uses
matplotlib's built-in PdfPages, like the per-sample diagnostics report in
`pdf.py`.
"""

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from pylibs.core.pipeline.regression import (
    FeatureRegression,
    confidence_band_half_width,
    format_p_value,
    format_permutation_mode,
    format_sample_count,
)
from pylibs.core.reporting.pdf import A4_SIZE, CAPTION_WIDTH, add_page_number, cover_page


def _letters_only(text: str) -> str:
    """Collapse `text` to its letter-only "words" joined by underscores, e.g.
    "c-KL divergence (vs. normal)" -> "c_KL_divergence_vs_normal" -- mirrors
    `api._letters_only_filename`, kept as its own copy here since
    `core.reporting` must not import from `core.api` (the dependency runs
    the other way)."""
    return "_".join(re.findall(r"[A-Za-z]+", text)) or "metric"


CAPTION_FONTSIZE = 12  # matches pdf.py/correlations_report.py
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 9
STATS_FONTSIZE = 9
STATS_GAP = 0.02  # axes-fraction gap between the legend's bottom edge and the stats text

# axes sized to half the page width, and forced square (50:50 aspect ratio) in
# physical inches -- A4_SIZE itself isn't square, so the height fraction is
# derived from the width fraction rather than reused directly
AXES_WIDTH_FRAC = 0.5
AXES_HEIGHT_FRAC = AXES_WIDTH_FRAC * A4_SIZE[0] / A4_SIZE[1]
AXES_LEFT = (1.0 - AXES_WIDTH_FRAC) / 2.0
AXES_BOTTOM = 0.33
CAPTION_Y = AXES_BOTTOM - 0.14  # same gap below the axes as the previous, larger layout


def _regression_figure(
    regression: FeatureRegression,
    feature_label: str,
    column_name: str,
    project_name: str,
) -> Figure:
    """One A4 page: an error-barred point per physical sample (mean ± std of
    `regression.metric_name` across that sample's own files/ablation
    layers), a dashed fitted line, and its 95% confidence-interval band --
    the line's own legend label carries just the fit equation, with R²/
    Pearson r/MAE/p-value as a separate text block anchored just below the
    legend, matching the reference paper's calibration-curve figure. The
    axes are forced to a square (50:50) physical aspect ratio at half the
    page width, rather than filling the page. No caption -- a caller
    building the combined PDF page adds one afterward."""
    fig = plt.figure(figsize=A4_SIZE, facecolor="white")
    ax = fig.add_axes((AXES_LEFT, AXES_BOTTOM, AXES_WIDTH_FRAC, AXES_HEIGHT_FRAC))

    ax.errorbar(
        regression.group_mean_x,
        regression.group_mean_y,
        yerr=regression.group_std_y,
        fmt="o",
        color="black",
        capsize=3,
        markersize=4,
    )

    x_min = float(regression.group_mean_x.min())
    x_max = float(regression.group_mean_x.max())
    x_curve = np.linspace(x_min, x_max, 300)
    y_curve = regression.slope * x_curve + regression.intercept
    half_width = confidence_band_half_width(
        x_curve, regression.n, regression.x_mean, regression.ssxx, regression.residual_std
    )

    equation_label = f"{regression.slope:.2e} · x + {regression.intercept:.2e}"
    ax.plot(x_curve, y_curve, "--", color="black", linewidth=1, label=equation_label)
    ax.fill_between(
        x_curve,
        y_curve - half_width,
        y_curve + half_width,
        color="red",
        alpha=0.2,
        edgecolor="none",
        label="95% Confidence Interval",
    )

    ax.set_xlabel(column_name, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"{regression.metric_name} intensity / a.u.", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        f"{feature_label} -- {regression.metric_name}", fontsize=16, fontweight="bold", pad=15
    )
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    legend = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper left")

    stats_text = (
        f"R²: {regression.r_squared * 100:.2f}%\nr: {regression.pearson_r:.3f}\n"
        f"MAE: {regression.mae:.3g}\n"
        f"{format_p_value(regression.p_value, regression.n_permutations)} "
        f"({format_permutation_mode(regression)})\n"
        f"{format_sample_count(regression)}"
    )
    # anchor just below the legend's own (rendered, not guessed) bottom edge, so
    # it never overlaps regardless of the legend's actual size
    fig.canvas.draw()
    legend_bbox_axes = legend.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(
        legend_bbox_axes.x0,
        legend_bbox_axes.y0 - STATS_GAP,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=STATS_FONTSIZE,
    )

    return fig


def build_regression_report(
    path: Path,
    project_name: str,
    column_name: str,
    regressions: list[FeatureRegression],
    feature_labels: dict[str, str] | None = None,
) -> None:
    """Write a multi-page A4 PDF: a cover page, then one calibration-curve
    page per feature in `regressions` -- a feature regressed under several
    metrics (e.g. from a `compute_regressions(..., metric=None)` all-metrics
    sweep) gets one page per metric, titled "<feature> -- <metric>".
    `feature_labels` maps feature_id to a display name, falling back to the
    feature_id itself when not given.

    Also saves each calibration-curve page as its own PNG (no cover page, no
    caption -- the PDF page still carries its caption) to `<path's
    directory>/img/regressions/<letters-only column_name>/
    <feature_id>_<letters-only metric_name>.png` -- the metric suffix avoids
    collisions once a feature has more than one stored metric, matching
    `report.generate_feature_maps`'s own always-on PNG-per-page convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels_by_id = feature_labels or {}

    safe_column_name = path.stem.removesuffix("_regressions")
    img_dir = path.parent / "img" / "regressions" / safe_column_name
    img_dir.mkdir(parents=True, exist_ok=True)

    title = f"Regressions for {column_name!r} in {project_name}"
    total = len(regressions) + 1
    with PdfPages(path) as pdf:
        fig = cover_page(title)
        add_page_number(fig, 1, total)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        for i, regression in enumerate(regressions, start=1):
            label = labels_by_id.get(regression.feature_id, regression.feature_id)
            fig = _regression_figure(regression, label, column_name, project_name)
            safe_metric_name = _letters_only(regression.metric_name)
            fig.savefig(
                img_dir / f"{regression.feature_id}_{safe_metric_name}.png",
                format="png",
                bbox_inches="tight",
                dpi=300,
            )
            caption = (
                f"Figure {i}. Calibration curve of {label} ({regression.metric_name}) over "
                f"{column_name!r} in {project_name}."
            )
            wrapped_caption = textwrap.fill(caption, width=CAPTION_WIDTH)
            fig.text(0.5, CAPTION_Y, wrapped_caption, ha="center", fontsize=CAPTION_FONTSIZE)
            add_page_number(fig, i + 1, total)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
