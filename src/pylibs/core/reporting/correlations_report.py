"""Correlations report: one PDF per external CSV column, showing color-scaled
best/worst feature+metric tables for Pearson r, R², MAE, and MAE's
permutation-test p-value -- a pure presentation layer over
`pipeline.correlations.CorrelationMatrices` (see
`core.api.compute_correlations`/`get_correlations`). Uses matplotlib's
built-in PdfPages, like the per-sample diagnostics report in `pdf.py`.
"""

import textwrap
import warnings
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from pylibs.core.pipeline.correlations import CorrelationMatrices
from pylibs.core.reporting.pdf import A4_SIZE, add_page_number, cover_page

TOP_N = 20
CAPTION_WIDTH = 70  # characters -- matches pdf.py, keeps captions within margins at fontsize 12
CAPTION_FONTSIZE = 12  # matches pdf.py's per-sample diagnostics report captions
DATA_FONTSIZE = 5
HEADER_FONTSIZE = 7
ROW_LABEL_FONTSIZE = 7
MAX_FEATURE_LABEL_CHARS = 10
CELL_PADDING = 1.3  # margin around each cell's own text, as a multiplier
ROW_HEIGHT_PADDING = 1.8  # data rows get extra vertical margin vs. CELL_PADDING
HEADER_HEIGHT_PADDING = 1.1  # margin around the header text, tighter than CELL_PADDING
ROW_LABEL_WIDTH_PADDING = 1.6  # auto_set_column_width()'s own margin is too tight for
# the bold mathtext feature_id line, letting its text spill into the first data column
HEADER_GAP_POINTS = 4.0  # small breathing room between header text and row 1
CAPTION_GAP_POINTS = 14.0  # small breathing room between the table and its caption

# (metric label, CorrelationMatrices attribute, colormap, vmin, vmax, is-max-the-"best"-direction)
# vmin/vmax of None means "derive dynamically from the interquartile range
# (Q1, Q3) of the whole table's finite values" (see build_correlations_report)
# -- used for MAE, which has no fixed, metric-independent range unlike the
# other three.
_SPECS: list[tuple[str, str, str, float | None, float | None, bool]] = [
    ("Pearson r", "pearson_r", "RdYlGn", -1.0, 1.0, True),
    ("R²", "r_squared", "RdYlGn", 0.0, 1.0, True),
    ("MAE", "mae", "RdYlGn_r", None, None, False),  # lower MAE is "best"
    ("MAE p-value", "mae_p_value", "RdYlGn_r", 0.0, 1.0, False),  # lower p is "best"
]


def _rank_features(values: np.ndarray, want_max: bool) -> np.ndarray:
    """Row indices of `values` (shape (n_features, n_metrics)), ordered by
    each feature's own most-extreme finite cell -- `nanmax` across that
    feature's metrics if `want_max`, else `nanmin` -- most extreme first. A
    feature with no finite cells at all is excluded entirely."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN (slice|axis) encountered")
        summary = np.nanmax(values, axis=1) if want_max else np.nanmin(values, axis=1)
    finite = np.isfinite(summary)
    order = np.argsort(summary)
    if want_max:
        order = order[::-1]
    return np.array([i for i in order if finite[i]], dtype=int)


def _row_label(feature_id: str, feature_label: str) -> str:
    """'{feature_id}\\n{feature_label}', bolding the id (mathtext) and
    truncating the label to `MAX_FEATURE_LABEL_CHARS` characters (marked
    with a trailing "(...)") so long expressions don't blow up row height."""
    truncated = (
        feature_label
        if len(feature_label) <= MAX_FEATURE_LABEL_CHARS
        else feature_label[:MAX_FEATURE_LABEL_CHARS] + "(...)"
    )
    return f"$\\mathbf{{{feature_id}}}$\n{truncated}"


def _table_figure(
    feature_ids: list[str],
    feature_labels: list[str],
    metric_names: list[str],
    row_indices: np.ndarray,
    values: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
) -> tuple[Figure, Bbox, Bbox | None]:
    """One A4 page: a color-scaled table (rows = the features at
    `row_indices`, in that order; columns = `metric_names`) -- no heading,
    no caption, so this is directly reusable as a standalone table PNG;
    returns the table's own (possibly row-shrunk) bbox so a caller building
    the combined PDF page can add a heading/caption and anchor the latter
    just below it, plus a tight content bbox (figure inches, or None to fall
    back to plain `bbox_inches="tight"`) a caller can pass straight to
    `Figure.savefig(bbox_inches=...)` for a cropped standalone PNG -- the
    page's own full-size invisible axes otherwise makes plain `"tight"`
    crop barely anything.

    Column headers are drawn as independent, rotated text labels above the
    table rather than as the table's own header row. A single long metric
    name (e.g. "c-KL divergence (vs. normal)") rotated 90° needs far more
    vertical room than any individual data row, and matplotlib's Table always
    stretches every row proportionally to fill the given bbox -- so keeping
    the header inside the table let it swallow most of the table's height
    whenever there were only a few features. Decoupling it gives the header
    only the small, fixed amount of space it actually needs."""
    fig = plt.figure(figsize=A4_SIZE, facecolor="white")

    if len(row_indices) == 0:
        fig.text(0.5, 0.5, "No data available.", ha="center", fontsize=11, color="gray")
        return fig, Bbox.from_bounds(0.16, 0.45, 0.7, 0.0), None

    rows = values[row_indices]
    row_labels = [_row_label(feature_ids[i], feature_labels[i]) for i in row_indices]
    # fixed-width scientific notation (e.g. " 9.17e-01", "-1.44e+03") so every
    # data cell renders the same number of characters, regardless of magnitude
    cell_text = [[f"{value: .2e}" if np.isfinite(value) else "" for value in row] for row in rows]

    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.axis("off")
    n_rows, n_cols = rows.shape
    no_renderer = cast(RendererBase, None)

    # explicit bbox (rather than loc="center") so the whole table -- including
    # the row-label column -- is scaled to fit the page instead of overflowing
    # off the left edge when there are many columns. Build a first, throwaway
    # pass with the full page height, purely to measure the cells' own
    # natural (data-driven) sizes before reserving room for the header labels.
    full_bbox = Bbox.from_bounds(0.16, 0.10, 0.82, 0.78)
    table = ax.table(cellText=cell_text, rowLabels=row_labels, cellLoc="center", bbox=full_bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(DATA_FONTSIZE)
    for row in range(n_rows):
        table[row, -1].set_fontsize(ROW_LABEL_FONTSIZE)

    # auto_set_column_width() only *queues* a recompute for the row-label
    # column -- it's applied lazily on the next draw, so force one now
    # before reading back its resulting width
    table.auto_set_column_width([-1])
    fig.canvas.draw()
    row_label_width = table[0, -1].get_width() * ROW_LABEL_WIDTH_PADDING

    # size cells to their own rendered text (rather than an arbitrary split)
    # so every cell is only as big as it needs to be -- get_text_bounds()
    # gives the text's extent in the table's own local coordinate system,
    # the same units cell.set_width()/set_height() expect. Passing renderer=
    # None reuses the renderer cached by the draw() above (get_window_extent's
    # documented fallback), sidestepping the backend-specific get_renderer().
    # The row-label cell's two-line "{feature_id}\n{expression}" text is
    # taller than a single data value, so take whichever needs more room.
    _, _, data_width, data_height = table[0, 0].get_text_bounds(no_renderer)
    _, _, _, row_label_height = table[0, -1].get_text_bounds(no_renderer)
    uniform_width = data_width * CELL_PADDING
    uniform_height = max(data_height, row_label_height) * ROW_HEIGHT_PADDING

    # how much vertical room the header labels need, independent of the data
    # rows: measure the longest one, rotated, at HEADER_FONTSIZE
    longest_header = max(metric_names, key=len)
    header_probe = fig.text(
        0, 0, longest_header, fontsize=HEADER_FONTSIZE, rotation=90, fontweight="bold"
    )
    fig.canvas.draw()
    header_text_height = header_probe.get_window_extent(no_renderer).height / fig.bbox.height
    header_probe.remove()
    gap = HEADER_GAP_POINTS / 72.0 * fig.dpi / fig.bbox.height
    header_area_height = header_text_height * HEADER_HEIGHT_PADDING + gap

    # give the table only as much height as its rows actually need (rather
    # than everything left below the header) -- Table always stretches
    # every row to fill whatever bbox it's given, so a bbox taller than the
    # rows' own natural height would inflate them well past their own text,
    # the same oversizing problem the header used to cause for the rows
    available_height = full_bbox.height - header_area_height
    natural_data_height = n_rows * uniform_height * full_bbox.height
    data_height_total = min(natural_data_height, available_height)

    # rebuild the table in the space left after reserving that header area
    table.remove()
    table_top = full_bbox.y0 + available_height
    table_bbox = Bbox.from_bounds(0.16, table_top - data_height_total, 0.82, data_height_total)
    table = ax.table(cellText=cell_text, rowLabels=row_labels, cellLoc="center", bbox=table_bbox)
    # ax.table(rowLabels=...) always auto-queues auto_set_column_width(-1)
    # internally (matplotlib.table.table(), when rowColours isn't given) --
    # left alone, that silently overrides the explicit set_width() below at
    # the final draw, undoing the pinned width the header positions rely on
    table._autoColumns = []  # type: ignore[attr-defined]
    table.auto_set_font_size(False)
    table.set_fontsize(DATA_FONTSIZE)

    cmap = matplotlib.colormaps[cmap_name]
    norm = Normalize(vmin=vmin, vmax=vmax)
    for row in range(n_rows):
        # row_label_width was measured with this fontsize already applied
        # (see the first pass above) -- pin the width explicitly rather
        # than re-deriving it here, so it can't drift out of sync with the
        # header labels' x-positions, which are computed from that same
        # row_label_width
        table[row, -1].set_fontsize(ROW_LABEL_FONTSIZE)
        table[row, -1].set_width(row_label_width)
        table[row, -1].set_height(uniform_height)
        for col in range(n_cols):
            table[row, col].set_width(uniform_width)
            table[row, col].set_height(uniform_height)
            value = rows[row, col]
            table[row, col].set_facecolor(cmap(norm(value)) if np.isfinite(value) else "#f0f0f0")

    for cell in table.get_celld().values():
        cell.set_edgecolor("none")

    # draw the metric-name headers directly, positioned just above the
    # table -- query each column's *actual* rendered x-position rather than
    # predicting it: Table always internally rescales/repositions cells to
    # exactly fill the given bbox (matplotlib.table.Table._update_positions),
    # in a way that isn't safe to replicate by hand
    fig.canvas.draw()
    header_texts = []
    for col, name in enumerate(metric_names):
        cell_bbox = table[0, col].get_window_extent(no_renderer)
        x = (cell_bbox.x0 + cell_bbox.x1) / 2 / fig.bbox.width
        header_texts.append(
            ax.text(
                x,
                table_top + gap,
                name,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=HEADER_FONTSIZE,
                fontweight="bold",
                transform=ax.transAxes,
            )
        )

    # a standalone PNG export has nothing else on the page once the (now
    # removed) heading/caption are gone, but the full-page invisible `ax`
    # (see `ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))` above) still makes
    # plain `bbox_inches="tight"` crop barely anything -- so compute an
    # explicit content bbox a caller can pass to `savefig` instead, from the
    # *actual* rendered extents of the table (including its row-label
    # column, part of the same Table artist) and the header labels, rather
    # than the table's own nominal bbox -- explicit per-cell set_width()
    # calls above mean the rendered table doesn't necessarily fill that
    # nominal bbox exactly, so deriving the crop from it clipped the
    # row-label column in practice
    fig.canvas.draw()
    content_extent = Bbox.union(
        [table.get_window_extent(no_renderer)]
        + [text.get_window_extent(no_renderer) for text in header_texts]
    )
    pad_px = 8.0
    content_extent = Bbox.from_extents(
        content_extent.x0 - pad_px,
        content_extent.y0 - pad_px,
        content_extent.x1 + pad_px,
        content_extent.y1 + pad_px,
    )
    content_bbox_inches = fig.dpi_scale_trans.inverted().transform_bbox(content_extent)

    return fig, table_bbox, content_bbox_inches


def build_correlations_report(
    path: Path,
    project_name: str,
    matrices: CorrelationMatrices,
    feature_labels: dict[str, str] | None = None,
    top_n: int = TOP_N,
) -> None:
    """Write a multi-page A4 PDF: a cover page, then a best/worst table pair
    each for Pearson r, R², MAE, and MAE's permutation-test p-value (9 pages
    total) -- every table ranks its rows by each feature's own most-extreme
    value among its metrics (see `_rank_features`), and color-scales cells
    over that statistic's natural range (-1..1 for Pearson r, 0..1 for
    R²/MAE p-value). MAE has no such fixed, metric-independent range -- its
    color scale is instead the interquartile range `[Q1, Q3]` of every
    finite MAE value in the whole table (not just the displayed rows): by
    construction exactly half the cells fall inside it with a real
    gradient, a quarter clip to "best", a quarter clip to "worst",
    regardless of how extreme the tail is -- robust to one metric (e.g.
    "c-KL divergence (vs. normal)") having wildly larger units than the
    rest. "best"/"worst" pages for the same column share one scale.
    `feature_labels` maps feature_id to a display name, falling back to the
    feature_id itself when not given.

    Also saves each of the 8 tables as its own 300 dpi, tightly-cropped PNG
    (no cover page, no heading, no caption -- the PDF page still carries
    both) to `<path's directory>/img/correlations/<letters-only
    column_name>/<metric>_<best|worst>.png`, matching
    `report.generate_feature_maps`'s own always-on PNG-per-page convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels_by_id = feature_labels or {}
    labels = [labels_by_id.get(feature_id, feature_id) for feature_id in matrices.feature_ids]

    safe_column_name = path.stem.removesuffix("_correlations")
    img_dir = path.parent / "img" / "correlations" / safe_column_name
    img_dir.mkdir(parents=True, exist_ok=True)

    cover_title = f"Correlations for {matrices.column_name!r} in {project_name}"
    total = 1 + len(_SPECS) * 2
    with PdfPages(path) as pdf:
        fig = cover_page(cover_title)
        add_page_number(fig, 1, total)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        figure_number = 1
        page_number = 2
        for metric_label, attr, cmap_name, spec_vmin, spec_vmax, best_is_max in _SPECS:
            values = getattr(matrices, attr)
            if spec_vmin is None or spec_vmax is None:
                finite = values[np.isfinite(values)]
                if finite.size:
                    # interquartile bounds: purely rank-based, so exactly
                    # half the cells always get a real gradient regardless
                    # of how extreme/heavy-tailed the outliers are -- unlike
                    # mean/std, which one dominant-magnitude metric column
                    # can drag arbitrarily far from the bulk of the data
                    table_vmin, table_vmax = (float(v) for v in np.percentile(finite, [25, 75]))
                else:
                    table_vmin, table_vmax = 0.0, 1.0
            else:
                table_vmin, table_vmax = spec_vmin, spec_vmax
            for which, want_max in (("best", best_is_max), ("worst", not best_is_max)):
                row_indices = _rank_features(values, want_max)[:top_n]
                fig, table_bbox, content_bbox = _table_figure(
                    matrices.feature_ids,
                    labels,
                    matrices.metric_names,
                    row_indices,
                    values,
                    cmap_name,
                    table_vmin,
                    table_vmax,
                )
                fig.savefig(
                    img_dir / f"{attr}_{which}.png",
                    format="png",
                    bbox_inches=content_bbox if content_bbox is not None else "tight",
                    dpi=300,
                )
                fig.text(
                    0.5,
                    0.95,
                    f"{metric_label} -- {which}",
                    ha="center",
                    fontsize=14,
                    fontweight="bold",
                )
                caption = (
                    f"Figure {figure_number}. {which.capitalize()} {len(row_indices)} "
                    f"feature(s) by {metric_label} against {matrices.column_name!r} in "
                    f"{project_name}."
                )
                # anchor just below the table's own (possibly row-shrunk) bottom edge,
                # rather than a fixed page position, so shrinking the table doesn't
                # leave a growing gap above the caption
                caption_gap = CAPTION_GAP_POINTS / 72.0 * fig.dpi / fig.bbox.height
                wrapped_caption = textwrap.fill(caption, width=CAPTION_WIDTH)
                fig.text(
                    0.5,
                    table_bbox.y0 - caption_gap,
                    wrapped_caption,
                    ha="center",
                    va="top",
                    fontsize=CAPTION_FONTSIZE,
                )
                add_page_number(fig, page_number, total)
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
                figure_number += 1
                page_number += 1
