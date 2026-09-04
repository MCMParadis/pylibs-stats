"""Applied-regressions report: one PDF per sample, one page per (external
CSV column, feature) pair a stored regression has been applied to (see
`pipeline.regression.AppliedRegression`) -- a spatially-resolved map (A) of
the predicted external property and its own 1D distribution (B), plus a
metrics table, matching the per-sample feature diagnostics report's own A4
page layout (see `pdf._diagnostics_page`/`pdf.metrics_table`). No panel for
diagnostic spectra/joint density -- this is a derived scalar quantity, not
raw spectral data. Uses matplotlib's built-in PdfPages, like the per-sample
diagnostics report in `pdf.py`.
"""

import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from pylibs.core.exceptions import ReportError
from pylibs.core.pipeline.regression import AppliedRegression
from pylibs.core.reporting.figures import _draw_feature_diagnostics, plot_feature_diagnostics
from pylibs.core.reporting.pdf import (
    A4_SIZE,
    CAPTION_WIDTH,
    add_page_number,
    cover_page,
    metrics_table,
    save_metrics_table_image,
)
from pylibs.core.reporting.raster import resolve_physical_extent, to_raster

CAPTION_FONTSIZE = 12  # matches metrics_table()'s own "Table N. ..." title fontsize


def _letters_only(text: str) -> str:
    """Collapse `text` to its letter-only "words" joined by underscores, e.g.
    "c_B / wt%" -> "c_B_wt" -- mirrors `api._letters_only_filename`, kept as
    its own copy here since `core.reporting` must not import from `core.api`
    (the dependency runs the other way)."""
    return "_".join(re.findall(r"[A-Za-z]+", text)) or "column"


def _applied_regression_page(
    applied: AppliedRegression,
    feature_label: str,
    n_rows: int,
    n_cols: int,
    width_img: float | None,
    height_img: float | None,
    figure_number: int,
    table_number: int,
    sample_stem: str,
    mask: np.ndarray | None = None,
) -> Figure:
    """One A4 page: (A) a spatially-resolved map of the predicted external
    property, (B) its 1D distribution, a caption, and a metrics table --
    matching `pdf._diagnostics_page`'s layout exactly. Both panels share the
    fit's own quantification limits (`applied.x_min`/`x_max`) as their color
    scale, rather than `applied.predicted`'s own range -- a pixel predicted
    outside that range is the model extrapolating past where it was
    calibrated, and renders clamped to a distinct color (`extend="both"`).
    No "Expected gaussian distribution" curve -- not a meaningful reference
    for a predicted-property distribution (`show_expected_gaussian=False`)."""
    grid = to_raster(applied.predicted, n_rows, n_cols, mask=mask)
    title = f"Predicted {applied.column_name} from {feature_label} ({applied.metric_name})"
    value_label = f"Predicted {applied.column_name}"

    fig = plt.figure(figsize=A4_SIZE, facecolor="white")
    fig.text(0.5, 0.94, title, ha="center", fontsize=16, fontweight="bold")
    # a taller diagnostics span than _diagnostics_page's own (0.90-0.52) -- with
    # no second (B) panel, (A)'s map and (B)'s distribution stack in a single
    # column, and their own titles/axis labels sit close enough to touch at
    # _diagnostics_page's tighter default when (A)'s raster is a tall aspect
    # ratio (as dataset1's real samples are); the metrics table below is at a
    # fixed position (see metrics_table()), so only the caption above it moves
    _draw_feature_diagnostics(
        fig,
        grid,
        width_img,
        height_img,
        None,
        None,
        applied.distribution,
        top=0.90,
        bottom=0.42,
        value_label=value_label,
        vmin=applied.x_min,
        vmax=applied.x_max,
        extend="both",
        clamp_color="black",
        show_expected_gaussian=False,
    )

    caption = f"Figure {figure_number}. {value_label} for {sample_stem}."
    wrapped_caption = textwrap.fill(caption, width=CAPTION_WIDTH)
    fig.text(0.5, 0.36, wrapped_caption, ha="center", fontsize=CAPTION_FONTSIZE)

    metrics_table(fig, applied.distribution, None, table_number, title)

    return fig


def build_applied_regression_report(
    path: Path,
    project_name: str,
    sample_id: str,
    sample_stem: str,
    applied: list[AppliedRegression],
    params: dict,
    feature_labels: dict[str, str] | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """Write a multi-page A4 PDF: a cover page (titled "Applied regressions
    for '<sample_stem>' in <project_name>" -- `sample_stem` is the display
    stem, e.g. "sample1_1", matching `report.generate_feature_maps`'s own
    cover-page/caption convention), then one page per (column, feature)
    pair in `applied`. `feature_labels` maps feature_id to a display name,
    falling back to the feature_id itself when not given. `params` is the
    raw file's own header metadata (see `ResultsStore.get_raw_params`,
    cached once by `SpectrumStatsStep` -- this function itself never opens
    the raw file); `mask` (likewise cached, see `ResultsStore.get_mask_array`)
    marks which frame cells were actually scanned, for a sample that doesn't
    fill its frame. Raises `ReportError` if `params` has no HeightPixels/
    WidthPixels -- applied-regression maps need a rastered sample (mirrors
    `report.generate_feature_maps`'s own check).

    Also saves, for each (column, feature) pair, a map+distribution PNG (no
    caption, no table -- built via `figures.plot_feature_diagnostics`, the
    same standalone style `report.generate_feature_maps` uses) to
    `<path's directory>/img/applied_regressions/<sample_id>/
    <letters-only column_name>/<feature_id>_<letters-only metric_name>.png`,
    and its metrics table as a separate PNG (no caption) to
    `<...>/<feature_id>_<letters-only metric_name>_table.png` -- the column
    gets its own directory (rather than a filename prefix) so two
    differently-punctuated column names can never collide after
    letters-only sanitization, and the metric suffix avoids collisions once
    a feature has been applied under more than one stored metric."""
    if "HeightPixels" not in params or "WidthPixels" not in params:
        raise ReportError(
            f"Sample {sample_id!r} has no HeightPixels/WidthPixels in its raw file's "
            "params -- applied-regression maps need a rastered sample."
        )
    n_rows = int(params["HeightPixels"])
    n_cols = int(params["WidthPixels"])
    width_img, height_img = resolve_physical_extent(params, n_rows, n_cols)

    path.parent.mkdir(parents=True, exist_ok=True)
    labels_by_id = feature_labels or {}

    img_dir = path.parent / "img" / "applied_regressions" / sample_id
    img_dir.mkdir(parents=True, exist_ok=True)

    title = f"Applied regressions for {sample_stem!r} in {project_name}"
    total = len(applied) + 1
    with PdfPages(path) as pdf:
        fig = cover_page(title)
        add_page_number(fig, 1, total)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        for i, item in enumerate(applied, start=1):
            label = labels_by_id.get(item.feature_id, item.feature_id)
            fig = _applied_regression_page(
                item, label, n_rows, n_cols, width_img, height_img, i, i, sample_stem, mask
            )
            add_page_number(fig, i + 1, total)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

            safe_column_name = _letters_only(item.column_name)
            column_dir = img_dir / safe_column_name
            column_dir.mkdir(parents=True, exist_ok=True)
            safe_metric_name = _letters_only(item.metric_name)

            value_label = f"Predicted {item.column_name}"
            content_fig = plot_feature_diagnostics(
                to_raster(item.predicted, n_rows, n_cols, mask=mask),
                f"Predicted {item.column_name} from {label} ({item.metric_name})",
                width_img=width_img,
                height_img=height_img,
                dist_stats=item.distribution,
                value_label=value_label,
                vmin=item.x_min,
                vmax=item.x_max,
                extend="both",
                clamp_color="black",
                show_expected_gaussian=False,
                bottom=0.42,  # matches _applied_regression_page's own panel geometry
                # exactly -- figsize defaults to A4_SIZE too, so the standalone PNG
                # comes out the same shape/size as the PDF page's own panels
            )
            content_fig.savefig(
                column_dir / f"{item.feature_id}_{safe_metric_name}.png",
                format="png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(content_fig)

            save_metrics_table_image(
                column_dir / f"{item.feature_id}_{safe_metric_name}_table.png",
                item.distribution,
                None,
                label,
            )
