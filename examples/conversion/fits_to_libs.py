"""Convert a FITS spectral cube to `.libs`.

    pip install astropy           # optional, only for this example
    python examples/conversion/fits_to_libs.py path/to/cube.fits

`astropy` is NOT a pylibs dependency. It is used here only to read the FITS
file; pylibs itself never sees the FITS format.

A FITS spectral cube is conventionally (spectral, y, x) -- the spectral axis
first, which is the reverse of ENVI. The wavelength axis is reconstructed from
the WCS keywords on the spectral axis (CRVAL3, CDELT3, CRPIX3), since a FITS
cube rarely carries an explicit wavelength array.

Adjust the axis order and the unit scaling to match your own files; this is a
worked starting point, not a universal FITS reader.
"""

import sys
from pathlib import Path

import numpy as np

from pylibs.core import api


def wavelength_axis(header, n_bands: int) -> np.ndarray:
    """Wavelengths in nm from the spectral axis's WCS keywords."""
    crval = float(header.get("CRVAL3", 0.0))
    cdelt = float(header.get("CDELT3", header.get("CD3_3", 1.0)))
    crpix = float(header.get("CRPIX3", 1.0))
    axis = crval + (np.arange(n_bands) + 1 - crpix) * cdelt
    unit = str(header.get("CUNIT3", "nm")).strip().lower()
    scales = {"m": 1e9, "um": 1e3, "micron": 1e3, "nm": 1.0, "angstrom": 0.1, "a": 0.1}
    scale = scales.get(unit, 1.0)
    return axis * scale


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: fits_to_libs.py path/to/cube.fits")
    try:
        from astropy.io import fits
    except ImportError:
        raise SystemExit(
            "this example needs the optional `astropy` package: pip install astropy"
        ) from None

    path = Path(sys.argv[1])
    with fits.open(path, memmap=True) as hdus:
        hdu = hdus[0]
        if hdu.data.ndim != 3:
            raise SystemExit(f"{path} holds a {hdu.data.ndim}D array, expected a 3D cube")
        n_bands, n_rows, n_cols = hdu.data.shape
        wavelengths = wavelength_axis(hdu.header, n_bands)

        def row_blocks(rows_per_block: int = 4):
            # move the spectral axis last and flatten the two spatial axes,
            # one group of whole raster rows at a time
            for start in range(0, n_rows, rows_per_block):
                block = np.asarray(hdu.data[:, start : start + rows_per_block, :], dtype="float64")
                yield np.moveaxis(block, 0, -1).reshape(-1, n_bands)

        out = api.write_libs(
            path.with_suffix(".libs"),
            row_blocks(),
            wavelengths,
            grid=(n_rows, n_cols),
            n_scans=n_rows * n_cols,
            step=1.0,
            units="px",
            metadata={"Instrument": "FITS import", "SourceFile": path.name},
        )
    print(f"Wrote {out}: {n_rows}x{n_cols} pixels, {n_bands} bands")


if __name__ == "__main__":
    main()
