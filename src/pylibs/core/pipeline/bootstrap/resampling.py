"""Pluggable bootstrap resampling strategies: given a numpy.random.Generator
already seeded deterministically for one iteration, choose which of a
sample's scan positions (0..n_scans-1) make up that iteration's resampled
subset. `RESAMPLING_STRATEGIES` is the whole extension surface -- a third
method is one function plus one registry entry, never a new if/elif branch
in `compute.py`.
"""

import math
from collections.abc import Callable

import numpy as np

from pylibs.core.exceptions import BootstrapError
from pylibs.core.reporting.raster import to_raster

# (rng, n_scans, npix, n_rows, n_cols, mask) -> array of scan indices, shape
# (npix,). Every strategy takes the whole set and ignores what it doesn't need
# -- `compute.py` passes all of them positionally-by-keyword, so a strategy
# that omitted one would fail only for the sample that happens to have it.
ResamplingStrategy = Callable[..., np.ndarray]


def resample_random(
    rng: np.random.Generator,
    n_scans: int,
    npix: int,
    n_rows: int | None = None,
    n_cols: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Classic bootstrap draw: `npix` scan positions independently and
    uniformly drawn *with replacement* from [0, n_scans) -- matches legacy
    `getindexes_random`, a true bootstrap resample, not a subset. Needs no
    raster geometry at all, so `n_rows`/`n_cols`/`mask` are ignored: drawing
    from [0, n_scans) already only ever picks a real, scanned position."""
    return rng.integers(0, n_scans, size=npix)


def resample_local(
    rng: np.random.Generator,
    n_scans: int,
    npix: int,
    n_rows: int | None = None,
    n_cols: int | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """A spatially contiguous square block of `npix` *distinct* scan
    positions (no replacement), anchored at a uniformly random in-bounds
    corner. Reuses `core.reporting.raster.to_raster` to get the serpentine-
    corrected (row, col) -> flat-scan-index grid, rather than reimplementing
    the legacy `Mat2Cube`. Computes the valid anchor range directly, rather
    than legacy's retry-until-in-bounds loop -- uniform over valid anchors
    either way, simpler, no pathological retry case.

    For a sample that doesn't fill its frame (`mask` given -- e.g. a round
    one), anchors are restricted to squares lying *entirely* within the
    scanned region, so a block is always exactly `npix` real scans rather
    than a rim block padded with unscanned cells. Which anchors qualify is
    again computed directly, via a summed-area table over the mask."""
    if n_rows is None or n_cols is None:
        raise BootstrapError("method='local' requires the sample's raster dimensions")
    side = math.isqrt(npix)
    if side * side != npix:
        raise BootstrapError(f"method='local' requires npix to be a perfect square, got {npix}")
    if side > n_rows or side > n_cols:
        raise BootstrapError(f"npix block side {side} doesn't fit in a {n_rows}x{n_cols} raster")
    grid = to_raster(np.arange(n_scans, dtype=float), n_rows, n_cols, mask=mask)

    if mask is None:
        anchor_row = int(rng.integers(0, n_rows - side + 1))
        anchor_col = int(rng.integers(0, n_cols - side + 1))
    else:
        anchor_row, anchor_col = _draw_unmasked_anchor(rng, mask, side)
    block = grid[anchor_row : anchor_row + side, anchor_col : anchor_col + side]
    return block.reshape(-1).astype(np.int64)


def _draw_unmasked_anchor(rng: np.random.Generator, mask: np.ndarray, side: int) -> tuple[int, int]:
    """A uniformly random anchor whose `side`x`side` block is fully scanned.

    A summed-area table gives every candidate block's scanned-cell count in
    one pass, so the qualifying anchors are known exactly -- no rejection
    loop, which on a disc would retry forever once `side` approaches the
    disc's own width."""
    integral = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)
    # inclusion-exclusion: scanned cells in every side x side block at once
    block_counts = (
        integral[side:, side:]
        - integral[:-side, side:]
        - integral[side:, :-side]
        + integral[:-side, :-side]
    )
    anchors = np.argwhere(block_counts == side * side)
    if anchors.size == 0:
        largest = _largest_full_square(mask)
        raise BootstrapError(
            f"No fully-scanned {side}x{side} block exists in this sample's scanned region "
            f"(largest that fits is {largest}x{largest}, i.e. npix up to {largest * largest})"
        )
    row, col = anchors[int(rng.integers(0, len(anchors)))]
    return int(row), int(col)


def _largest_full_square(mask: np.ndarray) -> int:
    """The side of the largest fully-scanned square in `mask` -- only for
    naming a usable npix in the error above."""
    best = 0
    sizes = np.zeros(mask.shape, dtype=np.int64)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            sizes[r, c] = (
                1
                if r == 0 or c == 0
                else 1 + min(sizes[r - 1, c], sizes[r, c - 1], sizes[r - 1, c - 1])
            )
            best = max(best, int(sizes[r, c]))
    return best


RESAMPLING_STRATEGIES: dict[str, ResamplingStrategy] = {
    "random": resample_random,
    "local": resample_local,
}


def get_resampling_strategy(method: str) -> ResamplingStrategy:
    strategy = RESAMPLING_STRATEGIES.get(method)
    if strategy is None:
        raise BootstrapError(
            f"Unknown bootstrap method {method!r}; registered methods: "
            f"{sorted(RESAMPLING_STRATEGIES)}"
        )
    return strategy
