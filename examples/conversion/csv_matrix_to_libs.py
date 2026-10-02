"""Convert a CSV spectral matrix to `.libs`.

    python examples/conversion/csv_matrix_to_libs.py

No optional dependencies: the CSV reader is part of pylibs.

Expected layout -- one row per pixel, two coordinate columns, then one column
per wavelength, with the header giving each column's wavelength:

    x,y,200.5,200.9,201.3
    0.0,0.0,1712,1690,1705
    0.1,0.0,1698,1702,1711

The coordinates say where each pixel sits, so the grid spacing and the raster
shape are both derived from them and no --grid is needed.
"""

from pathlib import Path

import numpy as np

from pylibs.core import api

OUT_DIR = Path(__file__).resolve().parent / "out"


def make_example_csv(path: Path, n_rows: int = 6, n_cols: int = 8, n_pixels: int = 64) -> None:
    """A small synthetic scan, so the example runs with no external data."""
    wavelengths = np.linspace(200.0, 210.0, n_pixels)
    rng = np.random.default_rng(0)
    lines = ["x,y," + ",".join(f"{w:.4f}" for w in wavelengths)]
    for row in range(n_rows):
        for col in range(n_cols):
            spectrum = rng.uniform(9.0, 11.0, n_pixels)
            spectrum[np.argmin(np.abs(wavelengths - 205.0))] += 100.0 + row * 5 + col
            lines.append(
                f"{col * 0.1:.3f},{row * 0.1:.3f}," + ",".join(f"{v:.4f}" for v in spectrum)
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "example_matrix.csv"
    make_example_csv(csv_path)

    spectra, wavelengths, coordinates = api.load_csv_matrix(csv_path)
    out = api.write_libs(
        OUT_DIR / "from_csv.libs",
        spectra,
        wavelengths,
        coordinates=coordinates,
        units="mm",
        metadata={"Instrument": "CSV export", "SourceFile": csv_path.name},
    )
    print(f"Wrote {out}")

    reader = api.import_libs_file(out)
    print(f"  {reader.n_scans} scans x {len(reader.wavelengths)} wavelengths")
    print(f"  grid: {reader.params['HeightPixels']}x{reader.params['WidthPixels']}")


if __name__ == "__main__":
    main()
