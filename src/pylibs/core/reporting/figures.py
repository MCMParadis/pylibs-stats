"""Feature map figure: a spatial raster plot of one feature's intensity
across a rastered sample.
"""

import matplotlib

matplotlib.use("Agg")  # headless-safe: report generation shouldn't need a display

import warnings
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Transform

from pylibs.core.logging import get_logger
from pylibs.core.pipeline.distributions import Distribution1D, Distribution2D
from pylibs.core.reporting.raster import to_raster

logger = get_logger(__name__)


class _HandlerColormapSwatch(HandlerBase):
    """Legend handler drawing a `cmap`'s own low-to-high gradient (as
    `n_stripes` adjacent thin rectangles -- a legend entry can't show a
    true continuous gradient) as the key icon, instead of a single flat
    color -- for a legend entry standing in for a colormap-shaded plot
    element (e.g. `_draw_distribution`'s histogram bars) where one color
    would misrepresent it. Register on a throwaway proxy artist (e.g. a
    `Patch` used only as a `handler_map` key, never drawn itself) via
    `ax.legend(handles=[...], handler_map={proxy: _HandlerColormapSwatch(cmap)})`."""

    def __init__(self, cmap: Colormap, n_stripes: int = 12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cmap = cmap
        self.n_stripes = n_stripes

    def create_artists(
        self,
        legend: Any,
        orig_handle: Any,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans: Transform,
    ) -> list[Rectangle]:
        stripe_width = width / self.n_stripes
        return [
            Rectangle(
                (xdescent + i * stripe_width, ydescent),
                stripe_width,
                height,
                facecolor=self.cmap(i / (self.n_stripes - 1)),
                edgecolor="none",
                transform=trans,
            )
            for i in range(self.n_stripes)
        ]


# Fixed as a fraction of the *figure* width (not the parent axes' own
# width), so that colorbars on differently sized axes -- e.g. panel (A) and
# the full-width panel (C) -- come out the same visual width.
COLORBAR_WIDTH = 0.0125

# inches (210mm x 297mm) -- duplicated from pdf.py's own A4_SIZE rather than
# imported, since pdf.py already imports from this module and the reverse
# import would be circular. Used by PDF report pages themselves (which must
# stay literal A4 for correct pagination/printing) and as
# plot_feature_diagnostics's own default figsize, so standalone PNGs come
# out the same shape/size as their corresponding PDF page's own panels.
A4_SIZE = (8.27, 11.69)


def _colorbar_x0(fig: Figure, ax: Axes) -> float:
    """The x0 `_make_colorbar_axes` would carve out of `ax` for its colorbar,
    without actually mutating `ax` -- lets a caller align a *different*
    axes' own colorbar to this one ahead of time (see
    `_draw_joint_distribution`'s `cbar_x0` param, shared with panel (C)'s
    own colorbar)."""
    label_clearance = 0.45 / fig.get_size_inches()[0]  # room for the rotated label
    pos = ax.get_position()
    return pos.x1 - label_clearance - COLORBAR_WIDTH


def _make_colorbar_axes(fig: Figure, ax: Axes, pad: float = 0.015) -> Axes:
    """Shrink `ax` to carve out a colorbar-sized axes at its own right edge,
    at a fixed width in *figure* fraction units (not relative to `ax`'s own
    width), so that colorbars on differently sized axes -- e.g. panel (A)
    and the full-width panel (C) -- come out the same visual width. Stays
    self-contained within `ax`'s original box (like matplotlib's own
    `fig.colorbar(..., ax=ax)` would), so it doesn't encroach on a
    neighboring subplot. Must be created before `tight_layout()` would run,
    since manually placed axes aren't tight_layout-compatible -- only used
    by the combined diagnostics figure, which lays itself out with an
    explicit GridSpec instead. Also reserves a fixed *absolute* clearance
    (independent of figure size) after the colorbar for its rotated label
    text, so a narrower page (e.g. an A4 PDF page) doesn't crowd the label
    into a neighboring subplot the way a purely figure-fraction pad would."""
    cax_x0 = _colorbar_x0(fig, ax)
    pos = ax.get_position()
    ax.set_position((pos.x0, pos.y0, cax_x0 - pad - pos.x0, pos.height))
    return fig.add_axes((cax_x0, pos.y0, COLORBAR_WIDTH, pos.height), label="_colorbar_cax")


def _add_colorbar(
    fig: Figure,
    ax: Axes,
    mappable: ScalarMappable,
    clabel: str,
    cax: Axes | None = None,
    extend: str = "neither",
) -> None:
    """Attach a colorbar to `ax` (default matplotlib sizing, tight_layout
    friendly) or to `cax` if given (see `_make_colorbar_axes`). `extend`
    (matplotlib's own vocabulary: "neither"/"both"/"min"/"max") draws a
    triangular wedge at either end to flag values clamped past the
    colormap's own over/under colors (see `_draw_feature_map`)."""
    cbar = fig.colorbar(mappable, cax=cax, ax=ax if cax is None else None, extend=extend)
    cast(ScalarFormatter, cbar.formatter).set_powerlimits((0, 0))  # always e.g. 1e4
    cbar.update_ticks()
    cbar.set_label(clabel, rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=11)


def _draw_feature_map(
    fig: Figure,
    ax: Axes,
    grid: np.ndarray,
    title: str,
    width_img: float | None = None,
    height_img: float | None = None,
    clabel: str = "Normalized intensity / a.u.",
    correction: float = 1.9,
    cax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: str = "neither",
    clamp_color: str | None = None,
) -> None:
    """Draw `grid` (shape (n_rows, n_cols)) onto `ax` as a raster map. Shared
    by `plot_feature_map` (standalone PNG) and the PDF report's feature
    pages, so both look identical.

    `vmin`/`vmax`, when both given, are used as-is (no washout clipping) --
    lets a caller share one explicit scale across this map and a companion
    distribution panel (see `_draw_feature_diagnostics`). Otherwise the scale
    is self-computed: `vmin` is the true min, `vmax` is capped at
    `correction * 2 * mean` so a handful of saturated pixels don't wash out
    the rest of the map. Pixels outside `[vmin, vmax]` clamp to the
    colormap's own boundary color by default (`clamp_color=None`,
    matplotlib's own behavior for an unset `under`/`over`) -- visually
    distinct from `bad="white"` (masked/NaN pixels, a different, unrelated
    concept -- e.g. an out-of-spectral-range feature). `clamp_color` (e.g.
    `"black"`) forces a single, visually distinct color for *every*
    out-of-range pixel instead -- meant for a scale whose bounds are
    themselves meaningful (e.g. a regression's quantification limits, where
    any clamped pixel means the model is extrapolating past where it was
    fit), not the plain windowing-for-display case. `extend`
    ("neither"/"both"/"min"/"max") only controls the colorbar's own
    triangular wedge, not pixel color -- see `_draw_feature_diagnostics` for
    how it's auto-detected from whether any pixel is actually clamped."""
    n_rows, n_cols = grid.shape

    if vmin is None or vmax is None:
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        vmean = float(np.nanmean(grid))
        vmax = min(vmax, correction * 2 * vmean)

    cmap = matplotlib.colormaps["turbo"].with_extremes(
        bad="white", under=clamp_color, over=clamp_color
    )

    has_physical_extent = width_img is not None and height_img is not None
    extent: tuple[float, float, float, float] = (
        (0.0, width_img, 0.0, height_img)
        if width_img is not None and height_img is not None
        else (0.0, float(n_cols), 0.0, float(n_rows))
    )

    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", extent=extent)

    ax.set_xlabel("x / mm" if has_physical_extent else "x / px", fontsize=12)
    ax.set_ylabel("y / mm" if has_physical_extent else "y / px", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(labelsize=11)

    _add_colorbar(fig, ax, im, clabel, cax=cax, extend=extend)


def plot_feature_map(
    values: np.ndarray,
    n_rows: int,
    n_cols: int,
    title: str,
    width_img: float | None = None,
    height_img: float | None = None,
    clabel: str = "Normalized intensity / a.u.",
    correction: float = 1.9,
    figsize: tuple[float, float] = (7, 5),
    mask: np.ndarray | None = None,
) -> Figure:
    """Build a spatial raster figure of `values` (shape (n_rows * n_cols,), or
    (mask.sum(),) when `mask` is given -- see `raster.to_raster`). The color
    scale is capped at `correction * 2 * mean` so a handful of saturated
    pixels don't wash out the rest of the map. Axes are in mm if
    `width_img`/`height_img` are given, otherwise in pixels."""
    grid = to_raster(values, n_rows, n_cols, mask=mask)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _draw_feature_map(fig, ax, grid, title, width_img, height_img, clabel, correction)
    fig.tight_layout()
    return fig


def _diagnostic_spectra_window(
    wavelengths: np.ndarray, peak_wavelength: float, window: float
) -> tuple[np.ndarray, bool]:
    """Mask selecting `[peak_wavelength - window, peak_wavelength + 3*window]`
    -- asymmetric so the peak sits at 1/4 of the selected range once it
    becomes the x-axis limits. Falls back to the full spectrum (and reports
    `False`) if nothing falls in that window."""
    mask = (wavelengths >= peak_wavelength - window) & (wavelengths <= peak_wavelength + 3 * window)
    if mask.any():
        return mask, True
    return np.ones_like(wavelengths, dtype=bool), False


def _draw_diagnostic_spectra(
    ax: Axes,
    wavelengths: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    mean: np.ndarray,
    peak_wavelength: float,
    title: str,
    window: float = 7.0,
    peak_scale_window: float = 5.0,
    y_fill: float = 2 / 3,
) -> None:
    """Draw the average/min/max spectrum onto `ax`, with a dashed line
    marking the peak. The x-axis is zoomed to `[peak_wavelength - window,
    peak_wavelength + 3*window]`, positioning the peak at 1/4 of the way
    across (leaving more room to show the tail on the other side); the
    y-axis is scaled so the tallest of the three spectra within
    `peak_wavelength +/- peak_scale_window` (default 5nm) fills `y_fill`
    (default 2/3) of the axis height -- not the tallest point anywhere in
    the (wider) *display* window above, which can belong to an unrelated
    nearby line that has nothing to do with this feature's own peak; a
    small span rather than the exact `peak_wavelength` sample accounts for
    the true peak center drifting a little from calibration/noise. Falls
    back to the full spectrum, auto-scaled, if nothing falls in the display
    window. Shared by `plot_diagnostic_spectra` (standalone PNG) and the PDF
    report's feature pages, so both look identical."""
    mask, windowed = _diagnostic_spectra_window(wavelengths, peak_wavelength, window)

    ax.plot(wavelengths[mask], mean[mask], color="black", linewidth=1, label="Average spectrum")
    ax.plot(wavelengths[mask], minimum[mask], color="blue", linewidth=1, label="Minimum spectrum")
    ax.plot(wavelengths[mask], maximum[mask], color="red", linewidth=1, label="Maximum spectrum")
    ax.axvline(peak_wavelength, color="gray", linestyle="--", linewidth=1)

    if windowed:
        ax.set_xlim(peak_wavelength - window, peak_wavelength + 3 * window)
        peak_mask = np.abs(wavelengths - peak_wavelength) <= peak_scale_window
        if not peak_mask.any():  # peak_scale_window narrower than the sample spacing
            peak_mask[int(np.argmin(np.abs(wavelengths - peak_wavelength)))] = True
        peak_height = float(
            max(mean[peak_mask].max(), minimum[peak_mask].max(), maximum[peak_mask].max())
        )
        if peak_height > 0:
            ax.set_ylim(0, peak_height / y_fill)

    ax.set_xlabel("Wavelength / nm", fontsize=12)
    ax.set_ylabel("Normalized intensity / a.u.", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.tick_params(labelsize=11)
    # solid background -- an unrelated, taller line elsewhere in the window can now get
    # clipped at the top of the plot (see peak_height above), right where this legend sits
    ax.legend(
        fontsize=7,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )


def plot_diagnostic_spectra(
    wavelengths: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    mean: np.ndarray,
    peak_wavelength: float,
    title: str,
    window: float = 7.0,
    figsize: tuple[float, float] = (7, 5),
) -> Figure:
    """Build the average/min/max spectrum figure for one peak, zoomed to
    `peak_wavelength +/- window` nm (falls back to the full spectrum if
    nothing falls in that window)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _draw_diagnostic_spectra(
        ax, wavelengths, minimum, maximum, mean, peak_wavelength, title, window
    )
    fig.tight_layout()
    return fig


def _draw_distribution(
    fig: Figure,
    ax: Axes,
    stats: Distribution1D,
    title: str,
    clabel: str = "Normalized intensity / a.u.",
    y_fill: float = 1 / 1.5,
    cax: Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extend: str = "neither",
    clamp_color: str | None = None,
    show_expected_gaussian: bool = True,
) -> None:
    """Draw a colored histogram of `stats` onto `ax`, with the expected
    (mean/std) and fitted Gaussian curves overlaid and a legend entry
    (a color patch) giving bin width and zero-pixel count. The x-axis is
    always zoomed to the histogram's own `[bin_edges[0], bin_edges[-1]]`
    range -- exactly what's actually binned, so the *full* distribution is
    always visible, regardless of what `vmin`/`vmax` (the color scale, see
    below) happen to be; the y-axis is scaled so the tallest curve/bar fills
    `y_fill` (default `1/1.5`, i.e. ~1.5x headroom above the tallest
    bar/curve -- room for the legend, which otherwise can overlap the
    tallest bars) of the axis height. `stats` is precomputed (by
    Distribution1DStep) and read back from the results store -- this
    function never recomputes from raw values. Shared by `plot_distribution`
    (standalone PNG) and the PDF report's feature pages, so both look
    identical.

    `vmin`/`vmax`, when both given, replace `stats.bin_edges[0]/[-1]` as the
    color scale *only* -- deliberately decoupled from the visible x-window
    above (unlike `_draw_feature_map`, which has no separate "view" concept
    from its color scale): a caller like the applied-regression report
    passes the fit's own quantification limits here, which can be much
    narrower/wider than this particular sample's own predicted-value spread,
    and still wants the *whole* distribution shown (matching what the same
    feature's own plain distribution would look like), with only the bars
    outside `[vmin, vmax]` singled out -- by default clamped to the
    colormap's own boundary color, or to `clamp_color` (e.g. `"black"`) if
    given, same as an out-of-range map pixel (see `_draw_feature_map`'s own
    docstring). `extend` only controls the colorbar's own wedge (see
    `_draw_feature_diagnostics`).

    `show_expected_gaussian=False` hides the mean/std-implied curve (e.g. not
    meaningful for an applied regression's predicted-property distribution)
    -- it's still computed either way, since its peak also bounds the y-axis."""
    bin_edges = stats.bin_edges
    bin_centers = stats.bin_centers
    bin_width = bin_edges[1] - bin_edges[0] if bin_edges.size > 1 else 1.0
    view_min = float(bin_edges[0])
    view_max = float(bin_edges[-1])

    if vmin is None or vmax is None:
        vmin = view_min
        vmax = view_max
    cmap = matplotlib.colormaps["turbo"].with_extremes(under=clamp_color, over=clamp_color)
    norm = Normalize(vmin=vmin, vmax=vmax)
    bar_colors = cmap(norm(bin_centers))

    ax.bar(bin_centers, stats.counts, width=bin_width, color=bar_colors, edgecolor="none")

    # evaluate the Gaussian curves on a dense grid spanning the *visible*
    # x-window (the histogram's own bin range, not the color scale), not the
    # histogram's own (coarser) bin_centers -- a narrow window can otherwise
    # only have a handful of bins in view, making the "smooth" curve look
    # like a jagged polyline through just those few points
    if view_max > view_min:
        curve_x = np.linspace(view_min, view_max, 300)
    else:
        curve_x = bin_centers

    expected = stats.expected_gaussian(curve_x)
    if show_expected_gaussian:
        ax.plot(
            curve_x,
            expected,
            color="black",
            linewidth=1,
            label="Expected gaussian distribution",
        )
    fitted = stats.fitted_gaussian(curve_x)
    if fitted is not None:
        ax.plot(
            curve_x,
            fitted,
            color="red",
            linewidth=1,
            label="Fitted gaussian distribution",
        )

    if view_max > view_min:
        ax.set_xlim(view_min, view_max)

        peak_height = float(stats.counts.max())
        peak_height = max(peak_height, float(expected.max()))
        if fitted is not None:
            peak_height = max(peak_height, float(fitted.max()))
        if peak_height > 0:
            ax.set_ylim(0, peak_height / y_fill)

    ax.set_xlabel(clabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.ticklabel_format(axis="both", style="scientific", scilimits=(0, 0))
    ax.tick_params(labelsize=11)

    # a legend-only proxy patch (never drawn on the axes) standing in for
    # the histogram itself -- carries the two pieces of info specific to
    # *this plot's own rendering* that aren't already in the companion
    # metrics table (pdf.metrics_table_rows has Mode/Mean/Median/Std/Gini/
    # Shannon/Zero pixels already). Its own facecolor is never drawn --
    # _HandlerColormapSwatch (registered below) replaces it with a
    # low-to-high gradient swatch of `cmap` itself, the same colors the
    # bars use, rather than one flat color.
    histogram_handle = Patch(
        label=f"Bin width: {bin_width:.2e}  |  Zero pixels: {stats.zero_count}"
    )
    handles, _ = ax.get_legend_handles_labels()
    handles.append(histogram_handle)
    ax.legend(
        handles=handles,
        handler_map={histogram_handle: _HandlerColormapSwatch(cmap)},
        fontsize=9,
        frameon=False,
        loc="upper right",
    )

    _add_colorbar(fig, ax, ScalarMappable(cmap=cmap, norm=norm), clabel, cax=cax, extend=extend)


def plot_distribution(
    stats: Distribution1D,
    title: str,
    clabel: str = "Normalized intensity / a.u.",
    figsize: tuple[float, float] = (7, 5),
) -> Figure:
    """Build the distribution figure for a feature from its precomputed
    `stats` (see `Distribution1DStep`): a colored histogram (turbo
    colormap) with expected/fitted Gaussian curves and summary statistics
    (mode, mean, median, std, gini index, Shannon entropy, zero count)."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _draw_distribution(fig, ax, stats, title, clabel)
    fig.tight_layout()
    return fig


def _marginal_bar_plot(
    ax: Axes,
    bin_edges: np.ndarray,
    counts: np.ndarray,
    cmap: str = "turbo",
    vertical: bool = False,
) -> None:
    """Draw a marginal 1D histogram as color-mapped bars along `bin_edges`
    (natural units, may include negative values) -- matching the (C)
    distribution panel's own bar coloring, rather than a smooth gradient-
    filled area under a curve. `vertical=True` draws the bars along the
    y-axis instead, for a right-hand marginal subplot. The peak reaches
    `1/1.5` of the axis range, same headroom fraction as panel (C)'s own
    `y_fill` default, for visual consistency between the two. `antialiased
    =False` plus an `edgecolor` matching each bar's own `facecolor` (instead
    of `edgecolor="none"`) -- this panel's marginals are much narrower than
    panel (C)'s own (identically-styled) histogram for the same bin count,
    so without both, matplotlib's default edge antialiasing (and the hairline
    unfilled seam between adjacent bars it otherwise leaves) blends each thin
    bar against the white background and the whole plot reads as faded/
    translucent, with white gaps between bars, rather than solid color."""
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0] if bin_edges.size > 1 else 1.0
    norm = Normalize(vmin=float(bin_edges[0]), vmax=float(bin_edges[-1]))
    bar_colors = matplotlib.colormaps[cmap](norm(bin_centers))
    peak = float(counts.max()) if counts.size else 0.0
    headroom = 1.5
    if not vertical:
        ax.bar(
            bin_centers,
            counts,
            width=bin_width,
            color=bar_colors,
            edgecolor=bar_colors,
            linewidth=0.4,
            antialiased=False,
        )
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.set_ylim(0, peak * headroom if peak > 0 else 1.0)
    else:
        ax.barh(
            bin_centers,
            counts,
            height=bin_width,
            color=bar_colors,
            edgecolor=bar_colors,
            linewidth=0.4,
            antialiased=False,
        )
        ax.set_ylim(bin_edges[0], bin_edges[-1])
        ax.set_xlim(0, peak * headroom if peak > 0 else 1.0)


def _draw_joint_distribution(
    fig: Figure,
    subplot_spec: SubplotSpec,
    stats: Distribution2D,
    title: str,
    cbar_x0: float,
) -> None:
    """Draw a joint-density panel for a ratio feature's numerator vs.
    denominator: the raw (unsmoothed, uninterpolated) `stats.display_counts`
    raster -- windowed to each side's own skew-adjusted boxplot-whisker fence
    (see `compute_distribution_2d`/`distributions.skew_adjusted_bounds`) --
    with an external colorbar (same rotated-label format as panels (A)/(C),
    see `_add_colorbar`) and color-mapped bar-plot marginal 1D distributions
    on top/right, matching the reference report's panel (B) for ratio
    features. `cbar_x0` (figure fraction, passed in by the caller -- see
    `_colorbar_x0`) upper-bounds how wide this panel is allowed to grow, so
    its own colorbar -- placed tight against the right marginal, not at
    `cbar_x0` itself, see below -- never overflows past panel (C)'s own
    colorbar."""
    MARGINAL_RATIO = 4.2  # both marginals' thickness relative to the main square's side
    inner = subplot_spec.subgridspec(
        2,
        2,
        width_ratios=(MARGINAL_RATIO, 1),
        height_ratios=(1, MARGINAL_RATIO),
        wspace=0.02,
        hspace=0.11,
    )
    ax_main = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0])
    ax_right = fig.add_subplot(inner[1, 1])

    num_bound_min, num_bound_max = stats.numerator_bounds
    den_bound_min, den_bound_max = stats.denominator_bounds

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        im = ax_main.imshow(
            stats.display_counts,
            origin="lower",
            extent=(den_bound_min, den_bound_max, num_bound_min, num_bound_max),
            interpolation="none",  # the raw raster -- no interpolation between pixels
            aspect="auto",  # a data/box aspect constraint here shrinks ax_main to fit
            # whenever its GridSpec cell isn't already that exact shape,
            # opening a gap to ax_top/ax_right (see the explicit square
            # resize below, which avoids that by construction instead)
            cmap="turbo",
        )
    for w in caught:
        logger.info("%s: %s", title, w.message)
    ax_main.set_xlabel(stats.denominator_expression, fontsize=12)
    ax_main.set_ylabel(stats.numerator_expression, fontsize=12)
    ax_main.ticklabel_format(axis="both", style="scientific", scilimits=(0, 0))
    ax_main.tick_params(labelsize=11)

    # Force ax_main square (50:50) without matplotlib's own aspect machinery
    # (aspect="equal"/set_box_aspect both shrink-to-fit and, per above, can
    # reopen a gap or -- combined with sharex/sharey -- even raise at draw
    # time): measure its actual rendered size in figure-inches after a real
    # draw, then explicitly resize+reposition all three boxes by hand,
    # anchored to ax_main's top edge (fixed) and -- unlike the marginals'
    # own thickness/gap below -- a *solved* left edge (see `new_x0` below),
    # not simply this panel's own natural gridspec start.
    fig.canvas.draw()
    fig_width_in, fig_height_in = fig.get_size_inches()
    main_pos = ax_main.get_position()

    marginal_gap_in = 0.05  # same physical gap on both sides of the imshow (see below)
    cbar_gap = 0.006
    side_max_height = main_pos.height * fig_height_in
    side_max_width = ((cbar_x0 - cbar_gap - main_pos.x0) * fig_width_in - marginal_gap_in) / (
        1 + 1 / MARGINAL_RATIO
    )
    side = min(side_max_height, side_max_width)
    new_width = side / fig_width_in
    new_height = side / fig_height_in
    # Solve for the left edge that puts the tight colorbar (see its own
    # placement below) exactly at `cbar_x0` -- aligned with panel (C)'s own
    # colorbar by construction, not just when width happens to be the
    # binding constraint above. This is what actually widens the (A)-(B)
    # gap when panel (B) is height-constrained (the common case): the whole
    # block shifts right just far enough for the alignment to hold.
    right_new_width = (side / MARGINAL_RATIO) / fig_width_in
    new_x0 = cbar_x0 - cbar_gap - right_new_width - marginal_gap_in / fig_width_in - new_width
    new_y0 = main_pos.y1 - new_height
    ax_main.set_position((new_x0, new_y0, new_width, new_height))
    fig.canvas.draw()  # tick labels must reflect the just-applied square position

    # Pin each axis's scientific-notation offset text (e.g. "1e3") at its own
    # axis's actual end (the x-axis's right edge; the y-axis's top edge) --
    # not the last *visible* tick label, which can sit well short of the
    # axis's true end whenever the data doesn't happen to reach a round-
    # number tick (e.g. a "4" tick with the axis actually extending well
    # past it): anchoring to that tick instead of the axis itself leaves the
    # exponent glued mid-axis with a large gap to the axis's real end. Along
    # the *other* axis it still aligns with the tick label row/column (the
    # x offset text's own vertical center; the y offset text's own right
    # edge), which needs an actual tick label's rendered position -- there's
    # no equivalent axis-relative shortcut for that dimension. Draw a plain
    # replacement text (a fresh artist, in figure coordinates so it survives
    # the resize above) rather than repositioning the real offsetText
    # objects -- those get their own transform/position reset by
    # Axis.draw() on every redraw (e.g. savefig's own internal draw pass),
    # which would silently discard a direct `.set_position()` override.
    ax_main.xaxis.get_offset_text().set_visible(False)
    ax_main.yaxis.get_offset_text().set_visible(False)
    inv = fig.transFigure.inverted()
    x_pad = 0.03 / fig_width_in  # ~3px at the 100dpi draw() renderer, in figure fraction
    y_pad = 0.03 / fig_height_in
    xlim = ax_main.get_xlim()
    x_ticklabels = [  # the tick locator can emit ticks just past xlim/ylim (e.g. a round
        t  # "6000" past a 5453 upper limit) -- still real, positioned Text objects, just
        for loc, t in zip(ax_main.get_xticks(), ax_main.get_xticklabels(), strict=True)  # clipped
        if t.get_text() and xlim[0] <= loc <= xlim[1]  # from being drawn, so still needed to
    ]  # find a real tick label for vertical centering, even though x no longer comes from it
    if x_ticklabels and (x_offset := ax_main.xaxis.get_major_formatter().get_offset()):
        bbox = x_ticklabels[-1].get_window_extent()
        _, y = inv.transform((bbox.x1, (bbox.y0 + bbox.y1) / 2))
        x = new_x0 + new_width + x_pad
        fig.text(x, y, x_offset, transform=fig.transFigure, fontsize=9, ha="left", va="center")
    ylim = ax_main.get_ylim()
    y_ticklabels = [
        t
        for loc, t in zip(ax_main.get_yticks(), ax_main.get_yticklabels(), strict=True)
        if t.get_text() and ylim[0] <= loc <= ylim[1]
    ]
    if y_ticklabels and (y_offset := ax_main.yaxis.get_major_formatter().get_offset()):
        bbox = y_ticklabels[-1].get_window_extent()
        # right-aligned to the tick labels' own right edge (closest to the axis) --
        # left-aligned would grow the text rightward, into the imshow itself
        x, _ = inv.transform((bbox.x1, bbox.y1))
        y = new_y0 + new_height + y_pad
        fig.text(x, y, y_offset, transform=fig.transFigure, fontsize=9, ha="right", va="bottom")

    # Both marginals get the *same* absolute thickness, derived from the final
    # square's own side -- rather than each keeping its independently
    # gridspec-computed height/width, which (since this panel's cell isn't
    # itself square) wouldn't match each other in real inches. Both also sit
    # `marginal_gap_in` from the imshow -- the same physical gap on both
    # sides, converted through the relevant figure dimension for each.
    top_new_height = (side / MARGINAL_RATIO) / fig_height_in
    ax_top.set_position(
        (new_x0, new_y0 + new_height + marginal_gap_in / fig_height_in, new_width, top_new_height)
    )
    ax_right.set_position(
        (new_x0 + new_width + marginal_gap_in / fig_width_in, new_y0, right_new_width, new_height)
    )

    numerator_marginal = stats.display_counts.sum(axis=1)
    denominator_marginal = stats.display_counts.sum(axis=0)

    _marginal_bar_plot(ax_top, stats.display_bin_edges_den, denominator_marginal)
    _marginal_bar_plot(ax_right, stats.display_bin_edges_num, numerator_marginal, vertical=True)
    ax_top.axis("off")
    ax_right.axis("off")

    # a dedicated colorbar axes at `cbar_x0` -- same external, vertical,
    # rotated-label format panels (A)/(C) use (see `_add_colorbar`), tight
    # against the right marginal (by construction: `new_x0` above was solved
    # so that this lands exactly here) and aligned with panel (C)'s own
    # colorbar, both at once.
    cax = fig.add_axes((cbar_x0, new_y0, COLORBAR_WIDTH, new_height), label="_colorbar_cax")
    _add_colorbar(fig, ax_main, im, "Count", cax=cax)

    ax_top.set_title(title, fontsize=14, fontweight="bold", pad=15)


# (wavelengths, minimum, maximum, mean, peak_wavelength)
SpectrumPanelData = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]


def _draw_feature_diagnostics(
    fig: Figure,
    grid: np.ndarray,
    width_img: float | None,
    height_img: float | None,
    spectrum: SpectrumPanelData | None,
    joint_dist: Distribution2D | None,
    dist_stats: Distribution1D | None,
    top: float = 0.90,
    bottom: float = 0.08,
    value_label: str = "Normalized intensity / a.u.",
    vmin: float | None = None,
    vmax: float | None = None,
    extend: str | None = None,
    clamp_color: str | None = None,
    show_expected_gaussian: bool = True,
) -> None:
    """Draw (A) feature map, (B) diagnostic spectra (if `spectrum` is given)
    or a joint-density plot (if `joint_dist` is given -- for a ratio
    feature; mutually exclusive with `spectrum`), and (C) distribution (if
    `dist_stats` is given) onto `fig`, arranged as three sub-panels the way
    the reference report does: (A)/(B) side by side, (C) spanning the full
    width below. `top`/`bottom` constrain how much of the figure's height
    the panels occupy, leaving room above for a suptitle and below for a
    caption. `value_label` labels panel (A)'s colorbar and panel (C)'s
    x-axis -- defaults to raw intensity, but a derived quantity (e.g. a
    regression's predicted external property) can override it. Panel (B)'s
    column gets extra width over panel (A)'s when it's a joint-density plot
    (not a diagnostic-spectra one) -- `_draw_joint_distribution` forces itself
    square from `min(row height, column width)`, so the wider column lets it
    grow instead of being capped by whichever dimension is smaller.

    `vmin`/`vmax` are shared by panel (A)'s map and panel (C)'s distribution
    *color scale only* -- panel (C)'s own visible x-window always shows the
    *full* histogram (`dist_stats.bin_edges[0]`/`[-1]`) regardless of
    `vmin`/`vmax` (see `_draw_distribution`'s own docstring). When `vmin`/
    `vmax` aren't given and `dist_stats` is present, they default to that
    same `dist_stats.bin_edges[0]`/`[-1]` -- the histogram's own skew-
    adjusted, exponential boxplot-whisker window (see
    `compute_distribution_1d`/`skew_adjusted_bounds`, not the values' true
    min/max), so panel (A) automatically gets the same washout-guarding
    window as panel (C)'s histogram, with no caller changes needed. A caller
    with its own scale in mind (e.g. a regression's quantification limits)
    passes `vmin`/`vmax` explicitly to override this default -- panel (A)'s
    color scale (and out-of-range clamping) follows that override exactly,
    while panel (C) still shows its own full distribution, with only the
    portion outside `[vmin, vmax]` clamped/flagged.

    `extend` ("neither"/"both"/"min"/"max") flags the colorbar's wedge(s).
    `None` (default) auto-detects it from `dist_stats`'s own already-computed
    `Minimum value`/`Maximum value` metrics (the true, unwindowed data
    extremes) against the resolved `vmin`/`vmax` -- a wedge only appears on
    an end that actually has real data clamped past it. A caller with a
    scale that's deliberately narrower than the data on purpose regardless of
    what's actually out of range (e.g. a regression's quantification limits)
    passes `extend` explicitly to override this.

    `clamp_color` (e.g. `"black"`), when given, forces a single distinct
    color for every out-of-range pixel/bar instead of the default boundary-
    color clamp -- see `_draw_feature_map`'s own docstring for when that's
    wanted (e.g. applied-regression's quantification limits).

    `show_expected_gaussian` passes straight through to `_draw_distribution`."""
    has_second_panel = spectrum is not None or joint_dist is not None
    if has_second_panel:
        width_ratios = (0.72, 1.28) if joint_dist is not None else (1, 1)
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1, 1.15],
            width_ratios=width_ratios,
            top=top,
            bottom=bottom,
            hspace=0.5,
            wspace=0.25,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_c = fig.add_subplot(gs[1, :])
        last_label = "(C)"

        if spectrum is not None:
            ax_b = fig.add_subplot(gs[0, 1])
            # slightly wider than tall (not square), regardless of this GridSpec
            # cell's own proportions -- extra width gives the legend more
            # room to clear the signal for single-transition spectra
            ax_b.set_box_aspect(0.8)
            wavelengths, minimum, maximum, mean, peak_wavelength = spectrum
            _draw_diagnostic_spectra(
                ax_b, wavelengths, minimum, maximum, mean, peak_wavelength, "(B)"
            )
        elif joint_dist is not None:
            # ax_c's colorbar isn't carved out until _draw_distribution runs, below --
            # but its position (hence its eventual colorbar x0) is already fixed by the
            # gridspec, so panel (B)'s own colorbar can align to it ahead of time.
            _draw_joint_distribution(
                fig, gs[0, 1], joint_dist, "(B)", cbar_x0=_colorbar_x0(fig, ax_c)
            )
    else:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], top=top, bottom=bottom, hspace=0.5)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_c = fig.add_subplot(gs[1, 0])
        last_label = "(B)"

    if (vmin is None or vmax is None) and dist_stats is not None:
        vmin = float(dist_stats.bin_edges[0])
        vmax = float(dist_stats.bin_edges[-1])

    if extend is None:
        if dist_stats is not None and vmin is not None and vmax is not None:
            below = dist_stats.metrics["Minimum value"] < vmin
            above = dist_stats.metrics["Maximum value"] > vmax
            if below and above:
                extend = "both"
            elif below:
                extend = "min"
            elif above:
                extend = "max"
            else:
                extend = "neither"
        else:
            extend = "neither"

    _draw_feature_map(
        fig,
        ax_a,
        grid,
        "(A)",
        width_img,
        height_img,
        clabel=value_label,
        cax=_make_colorbar_axes(fig, ax_a),
        vmin=vmin,
        vmax=vmax,
        extend=extend,
        clamp_color=clamp_color,
    )

    if dist_stats is not None:
        _draw_distribution(
            fig,
            ax_c,
            dist_stats,
            last_label,
            clabel=value_label,
            cax=_make_colorbar_axes(fig, ax_c),
            vmin=vmin,
            vmax=vmax,
            extend=extend,
            clamp_color=clamp_color,
            show_expected_gaussian=show_expected_gaussian,
        )
    else:
        ax_c.axis("off")


def plot_feature_diagnostics(
    grid: np.ndarray,
    title: str,
    width_img: float | None = None,
    height_img: float | None = None,
    spectrum: SpectrumPanelData | None = None,
    joint_dist: Distribution2D | None = None,
    dist_stats: Distribution1D | None = None,
    figsize: tuple[float, float] = A4_SIZE,
    top: float = 0.90,
    bottom: float = 0.08,
    value_label: str = "Normalized intensity / a.u.",
    vmin: float | None = None,
    vmax: float | None = None,
    extend: str | None = None,
    clamp_color: str | None = None,
    show_expected_gaussian: bool = True,
) -> Figure:
    """Build the combined diagnostics figure for one feature: (A) a feature
    map, (B) diagnostic spectra (if `spectrum` is given -- only meaningful
    for a bare-peak feature) or a joint-density plot (if `joint_dist` is
    given -- only meaningful for a ratio feature), and (C) a distribution
    histogram (if `dist_stats` is given), matching the reference report's
    layout. `value_label` labels panel (A)'s colorbar and panel (C)'s
    x-axis -- defaults to raw intensity, overridable for a derived
    quantity (e.g. a regression's predicted external property). `vmin`/
    `vmax` share one explicit color scale between panels (A) and (C);
    `extend` flags the colorbar's wedge(s), auto-detected from the data
    when left as `None`; `clamp_color` (e.g. `"black"`) forces a single
    distinct color for out-of-range pixels/bars instead of the default
    boundary-color clamp -- see `_draw_feature_diagnostics`.
    `show_expected_gaussian=False` hides panel (C)'s mean/std-implied curve.

    `figsize` defaults to the same `A4_SIZE` the PDF report pages use, and
    `top`/`bottom` are forwarded straight to `_draw_feature_diagnostics` --
    a caller building a standalone PNG that should match a specific PDF
    page's own panel geometry (e.g. `pdf._diagnostics_page`'s `bottom=0.52`)
    passes the same values here, so the two only differ in that the PDF
    page also draws a caption/table below `bottom`, while this figure
    leaves that region empty (typically then tightly cropped away)."""
    fig = plt.figure(figsize=figsize, facecolor="white")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    _draw_feature_diagnostics(
        fig,
        grid,
        width_img,
        height_img,
        spectrum,
        joint_dist,
        dist_stats,
        top=top,
        bottom=bottom,
        value_label=value_label,
        vmin=vmin,
        vmax=vmax,
        extend=extend,
        clamp_color=clamp_color,
        show_expected_gaussian=show_expected_gaussian,
    )
    return fig
