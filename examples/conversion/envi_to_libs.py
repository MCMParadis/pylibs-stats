"""Convert an ENVI hyperspectral cube to `.libs`.

    pip install spectral          # optional, only for this example
    python examples/conversion/envi_to_libs.py path/to/cube.hdr

`spectral` is NOT a pylibs dependency. It is used here only to read the ENVI
header and cube; pylibs itself never sees the ENVI format. Any other reader
that yields a NumPy array works the same way.

An ENVI cube is (lines, samples, bands): a spatial row, a spatial column, and
the spectral axis. `.libs` wants (n_scans, n_pixels) with one row per pixel,
so the two spatial axes are flattened row-major -- which is plain raster
order, the writer's default, so odd rows are reversed on the way in.

The cube is read band-interleaved through the library's own memory map, and
handed to the writer one raster row at a time, so peak memory is one row
rather than the whole cube.
"""

import sys
from pathlib import Path

import numpy as np

from pylibs.core import api


def row_blocks(cube, n_rows: int, rows_per_block: int = 4):
    """Whole raster rows at a time. Blocks must not split a row: the writer
    reverses odd rows block by block to produce the serpentine order."""
    for start in range(0, n_rows, rows_per_block):
        block = np.asarray(cube[start : start + rows_per_block, :, :], dtype="float64")
        yield block.reshape(-1, block.shape[-1])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: envi_to_libs.py path/to/cube.hdr")
    try:
        import spectral
    except ImportError:
        raise SystemExit(
            "this example needs the optional `spectral` package: pip install spectral"
        ) from None

    header = Path(sys.argv[1])
    cube = spectral.open_image(str(header))
    n_rows, n_cols, n_bands = cube.shape
    wavelengths = np.asarray(cube.bands.centers, dtype="float64")
    if wavelengths.size != n_bands:
        raise SystemExit(f"{header} gives {wavelengths.size} band centres for {n_bands} bands")

    out = api.write_libs(
        header.with_suffix(".libs"),
        row_blocks(cube.open_memmap(writable=False), n_rows),
        wavelengths,
        grid=(n_rows, n_cols),
        n_scans=n_rows * n_cols,
        step=1.0,
        units="px",
        metadata={"Instrument": "ENVI import", "SourceFile": header.name},
    )
    print(f"Wrote {out}: {n_rows}x{n_cols} pixels, {n_bands} bands")


if __name__ == "__main__":
    main()
