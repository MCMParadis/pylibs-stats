"""Reconstructs a spatial raster (2D image) from a flat per-scan feature
array. LIBS rasters are scanned serpentine (boustrophedon): the stage scans
one row left-to-right, then the next row right-to-left, and so on, so every
other row needs to be reversed to reconstruct true spatial order.

A sample that doesn't fill its frame -- a round one in a square frame, say --
records fewer scans than the frame has cells, so there's nothing to reshape.
Its raw file carries a `mask` saying which cells were actually scanned (see
`ResultsStore.get_mask_array`, cached from `PixelAssignmentMatrix`); pass it
and each scan is placed at its own cell, in the same serpentine order, with
unscanned cells left NaN. Without a mask -- the norm for a full rectangular
`.libs` raster -- the plain reshape is used, unchanged.
"""

import numpy as np

from pylibs.core.exceptions import ReportError
from pylibs.core.logging import get_logger

logger = get_logger(__name__)


def serpentine_positions(mask: np.ndarray) -> np.ndarray:
    """The (row, col) of every set cell in `mask`, in serpentine scan order:
    rows top to bottom, each row left-to-right except odd rows, which run
    right-to-left. Shape (mask.sum(), 2).

    Only the *scanned* cells of a row are ordered, so a row whose scanned
    span is an arbitrary subset (any binary mask, not just a disc) still
    reads out in the direction the stage travelled."""
    rows, cols = np.where(mask)
    # np.where already yields row-major order, so within each row the columns
    # are ascending -- reversing the odd rows is all that's left to do
    flip = (rows % 2).astype(bool)
    order = np.lexsort((np.where(flip, -cols, cols), rows))
    return np.stack([rows[order], cols[order]], axis=1)


def to_raster(
    values: np.ndarray, n_rows: int, n_cols: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """Place `values` (shape (n,)) onto a (n_rows, n_cols) raster image,
    undoing the serpentine scan order.

    Without `mask`, the scan is assumed to fill the frame: reshape and reverse
    every other row. With `mask` (shape (n_rows, n_cols), True where a scan
    was taken), `values` are placed at its set cells in serpentine order
    instead and every unscanned cell is left NaN -- which is what a partially
    scanned sample needs, since it has fewer values than the frame has cells."""
    if mask is None:
        if values.shape[0] != n_rows * n_cols:
            raise ReportError(
                f"Can't reshape {values.shape[0]} values into a {n_rows}x{n_cols} raster"
            )
        grid = values.reshape(n_rows, n_cols).copy()
        grid[1::2] = grid[1::2, ::-1]
        return grid

    if mask.shape != (n_rows, n_cols):
        raise ReportError(
            f"Mask is {mask.shape[0]}x{mask.shape[1]} but the raster is {n_rows}x{n_cols}"
        )
    n_scanned = int(mask.sum())
    if values.shape[0] != n_scanned:
        raise ReportError(
            f"Can't place {values.shape[0]} values into a mask marking {n_scanned} scanned "
            f"cell(s) of a {n_rows}x{n_cols} raster"
        )
    positions = serpentine_positions(mask)
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    grid[positions[:, 0], positions[:, 1]] = values
    return grid


def frame_shape(params: dict, n_values: int, sample_id: str = "") -> tuple[int, int]:
    """The `(n_rows, n_cols)` a full scan of `n_values` fills, from the raw
    file's `HeightPixels`/`WidthPixels`.

    Some instruments write those two as the number of *intervals* between
    pixels rather than the number of pixels, so a 400x400 scan is recorded as
    399x399. When the header's own product doesn't account for the scans but
    `(h + 1) * (w + 1)` does, the larger frame is the real one. The two can
    never both match: `h*w = (h+1)(w+1)` has no positive solution.

    Called once per sample, at ingest (`SpectrumStatsStep`), which stores the
    resolved pair so no later consumer has to repeat the guess -- see that
    step for why this belongs there rather than at every read. Reinterpreting
    a header is logged, since nothing else would record that the file said one
    thing and pylibs concluded another.

    Not for a partially scanned sample: its scan count is smaller than its
    frame by construction, so its `PixelAssignmentMatrix` is the authority on
    the shape and this would only raise.
    """
    if "HeightPixels" not in params or "WidthPixels" not in params:
        raise ReportError("Raw file params have no HeightPixels/WidthPixels")
    try:
        n_rows, n_cols = int(params["HeightPixels"]), int(params["WidthPixels"])
    except (TypeError, ValueError) as exc:
        raise ReportError(
            "HeightPixels/WidthPixels are not whole numbers: "
            f"{params['HeightPixels']!r}x{params['WidthPixels']!r}"
        ) from exc
    if n_rows * n_cols == n_values:
        return n_rows, n_cols
    if (n_rows + 1) * (n_cols + 1) == n_values:
        logger.info(
            "%sheader says %dx%d but %d scan(s) fill %dx%d -- reading those as the intervals "
            "between pixels, not the pixels",
            f"Sample {sample_id!r}: " if sample_id else "",
            n_rows,
            n_cols,
            n_values,
            n_rows + 1,
            n_cols + 1,
        )
        return n_rows + 1, n_cols + 1
    raise ReportError(
        f"{n_values} values fill neither a {n_rows}x{n_cols} raster nor a "
        f"{n_rows + 1}x{n_cols + 1} one"
    )


def resolve_physical_extent(
    params: dict, n_rows: int, n_cols: int
) -> tuple[float | None, float | None]:
    """(width_img, height_img) in mm: direct `WidthIMG`/`HeightIMG` raw-file
    params if present, else `Resolution` (mm per pixel) * n_cols/n_rows if
    `Resolution` is present, else `(None, None)` -- callers fall back to
    pixel-unit axis labels in that case (see `figures._draw_feature_map`).
    Each axis is resolved independently, so a file with e.g. only `WidthIMG`
    direct and `Resolution` for height still works."""
    width_img = params.get("WidthIMG")
    height_img = params.get("HeightIMG")
    resolution = params.get("Resolution")
    if width_img is None and resolution is not None:
        width_img = resolution * n_cols
    if height_img is None and resolution is not None:
        height_img = resolution * n_rows
    return width_img, height_img
