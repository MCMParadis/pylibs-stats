"""Assembles a sample's feature diagnostics into a single multi-page PDF
report, using matplotlib's built-in PdfPages -- no LaTeX or other external
PDF tool required. Every page is a fixed A4 size, with "Page N of M"
pagination in the footer.
"""

import textwrap
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from pylibs.core.pipeline.distributions import Distribution1D, Distribution2D
from pylibs.core.reporting.figures import SpectrumPanelData, _draw_feature_diagnostics

A4_SIZE = (8.27, 11.69)  # inches (210mm x 297mm)
CAPTION_WIDTH = 70  # characters -- keeps captions within the page margins at fontsize 12


class DiagnosticsPage(NamedTuple):
    title: str
    grid: np.ndarray
    width_img: float | None
    height_img: float | None
    spectrum: SpectrumPanelData | None
    joint_dist: Distribution2D | None
    dist_stats: Distribution1D | None


def _logo_path() -> Path:
    return Path(str(resources.files("pylibs.core.reporting") / "data" / "logo.png"))


def cover_page(title: str) -> Figure:
    """A standalone A4 cover page: the pyLIBS logo, "DATA PROCESSING REPORT",
    and `title` -- shared across every pylibs PDF report (not just the
    per-sample diagnostics one), so it's public rather than local to this
    module."""
    fig = plt.figure(figsize=A4_SIZE, facecolor="white")
    logo = mpimg.imread(str(_logo_path()))
    logo_ax = fig.add_axes((0.25, 0.6, 0.5, 0.15))
    logo_ax.imshow(logo)
    logo_ax.axis("off")
    fig.text(0.5, 0.57, "DATA PROCESSING REPORT", ha="center", fontsize=14, color="gray")
    fig.text(0.5, 0.52, title, ha="center", fontsize=18)
    return fig


def _params_page(sample_name: str, params: dict[str, str]) -> Figure:
    fig, ax = plt.subplots(figsize=A4_SIZE, facecolor="white")
    ax.axis("off")
    ax.set_title(sample_name, fontsize=16, fontweight="bold", loc="left")
    if params:
        table_bbox = Bbox.from_bounds(0, 0.65, 1.0, 0.3)
        ax.text(
            table_bbox.x0,
            table_bbox.y1 + 0.01,
            "Table 1. LIBS analysis parameters.",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
        )
        table = ax.table(
            cellText=list(params.items()),
            colWidths=[0.4, 0.6],
            loc="upper left",
            cellLoc="left",
            bbox=table_bbox,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        n_rows = len(params)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("black")
            edges = ("T" if row == 0 else "") + ("B" if row == n_rows - 1 else "")
            cell.visible_edges = edges
            if col == 0:
                cell.set_text_props(fontweight="bold")
    return fig


def metrics_table_rows(
    stats: Distribution1D, dist2d_stats: Distribution2D | None
) -> list[tuple[str, str]]:
    rows = [
        ("Mode", f"{stats.mode:.2e}"),
        ("Mean", f"{stats.mean:.2e}"),
        ("Median", f"{stats.median:.2e}"),
        ("Std. deviation", f"{stats.std:.2e}"),
        ("Gini index", f"{stats.gini:.2e}"),
        ("1D Shannon entropy", f"{stats.shannon_entropy:.2e}"),
        ("Zero pixels", str(stats.zero_count)),
    ]
    if stats.kl_divergence_normal is not None:
        rows.append(("c-KL divergence (vs. normal)", f"{stats.kl_divergence_normal:.2e}"))
    if stats.fitted_params is not None:
        amplitude, mu, sigma = stats.fitted_params
        rows.append(("Fitted amplitude", f"{amplitude:.2e}"))
        rows.append(("Fitted μ", f"{mu:.2e}"))
        rows.append(("Fitted σ", f"{sigma:.2e}"))
    if dist2d_stats is not None:
        if dist2d_stats.pearson_correlation is not None:
            rows.append(("Pearson-CR", f"{dist2d_stats.pearson_correlation:.2e}"))
        if dist2d_stats.kl_divergence_gaussians is not None:
            rows.append(("c-KL divergence", f"{dist2d_stats.kl_divergence_gaussians:.2e}"))
        rows.append(("d-KL divergence", f"{dist2d_stats.kl_divergence_histograms:.2e}"))
        rows.append(("2D Shannon entropy", f"{dist2d_stats.shannon_entropy_boundary:.2e}"))
        rows.append(("2D Gini index", f"{dist2d_stats.gini:.2e}"))
    return rows


def metrics_table(
    fig: Figure,
    stats: Distribution1D,
    dist2d_stats: Distribution2D | None,
    table_number: int,
    title: str,
    bbox: Bbox | None = None,
    show_title: bool = True,
) -> None:
    """Draw a metrics table onto `fig` at `bbox` (figure-fraction
    coordinates), defaulting to the position `_diagnostics_page` reserves
    below its (A)/(B)/(C) panels -- a smaller, standalone figure (e.g. a
    table-only PNG export) can override `bbox` to fill its own bounds
    instead. `show_title=False` omits the "Table N. Metrics for ..." caption
    line, for a standalone export with no page/figure numbering context."""
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    rows = metrics_table_rows(stats, dist2d_stats)
    table_bbox = bbox if bbox is not None else Bbox.from_bounds(0.15, 0.06, 0.7, 0.26)
    if show_title:
        ax.text(
            table_bbox.x0,
            table_bbox.y1 + 0.01,
            f"Table {table_number}. Metrics for {title}.",
            transform=ax.transAxes,
            fontsize=12,
            ha="left",
        )
    table = ax.table(
        cellText=rows,
        colWidths=[0.5, 0.5],
        loc="upper left",
        cellLoc="left",
        bbox=table_bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    n_rows = len(rows)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        edges = ("T" if row == 0 else "") + ("B" if row == n_rows - 1 else "")
        cell.visible_edges = edges
        if col == 0:
            cell.set_text_props(fontweight="bold")


TABLE_IMAGE_BBOX = Bbox.from_bounds(0.05, 0.08, 0.9, 0.8)
TABLE_IMAGE_FIGSIZE = (7.0, 4.0)


def save_metrics_table_image(
    path: Path, stats: Distribution1D, dist2d_stats: Distribution2D | None, title: str
) -> None:
    """Save a standalone, uncaptioned metrics table PNG (see `metrics_table`)
    at `path` -- shared by `report.generate_feature_maps` and
    `applied_regression_report.build_applied_regression_report`'s per-item
    table export, so both stay pixel-for-pixel consistent."""
    fig = plt.figure(figsize=TABLE_IMAGE_FIGSIZE, facecolor="white")
    metrics_table(fig, stats, dist2d_stats, 0, title, bbox=TABLE_IMAGE_BBOX, show_title=False)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _diagnostics_page(
    page: DiagnosticsPage, figure_number: int, table_number: int, sample_name: str
) -> Figure:
    fig = plt.figure(figsize=A4_SIZE, facecolor="white")
    fig.text(0.5, 0.94, page.title, ha="center", fontsize=16, fontweight="bold")
    _draw_feature_diagnostics(
        fig,
        page.grid,
        page.width_img,
        page.height_img,
        page.spectrum,
        page.joint_dist,
        page.dist_stats,
        top=0.90,
        bottom=0.52,
    )

    panels = "(A) feature map"
    has_second_panel = page.spectrum is not None or page.joint_dist is not None
    if page.spectrum is not None:
        panels += ", (B) diagnostic spectra"
    elif page.joint_dist is not None:
        panels += ", (B) 2D joint distribution"
    last_label = "(C)" if has_second_panel else "(B)"
    panels += f", and {last_label} distribution"
    caption = f"Figure {figure_number}. {panels} for {page.title} in {sample_name}."
    wrapped_caption = textwrap.fill(caption, width=CAPTION_WIDTH)
    fig.text(0.5, 0.40, wrapped_caption, ha="center", fontsize=12)

    if page.dist_stats is not None:
        metrics_table(fig, page.dist_stats, page.joint_dist, table_number, page.title)

    return fig


def add_page_number(fig: Figure, page: int, total: int) -> None:
    """Draw a "Page N of M" footer -- shared across every pylibs PDF report,
    so it's public rather than local to this module."""
    fig.text(0.5, 0.03, f"Page {page} of {total}", ha="center", fontsize=9, color="gray")


def build_pdf_report(
    path: Path,
    title: str,
    sample_name: str,
    params: dict[str, str],
    pages: list[DiagnosticsPage],
) -> None:
    """Write a multi-page A4 PDF: a cover page (with logo, titled `title`),
    a parameters table (titled "Table 1..."), then one numbered "Figure
    N..." page per `DiagnosticsPage` in `pages`, combining (A) feature map,
    (B) diagnostic spectra (if available), and (C) distribution, plus a
    "Table N+1..." of its computed metrics -- matching the reference
    report's layout. Every page gets a "Page N of M" footer. `sample_name`
    (the display stem, e.g. "sample1_1") labels the params table and every
    per-feature page's own caption -- distinct from `title`, which is only
    the cover page's own heading."""
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(pages) + 2
    with PdfPages(path) as pdf:
        fig = cover_page(title)
        add_page_number(fig, 1, total)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig = _params_page(sample_name, params)
        add_page_number(fig, 2, total)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        for i, page in enumerate(pages, start=1):
            fig = _diagnostics_page(page, i, i + 1, sample_name)
            add_page_number(fig, i + 2, total)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
