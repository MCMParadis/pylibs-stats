"""Writer for the open `.libs` format -- the ingestion counterpart to
`readers.libs_reader`, and the one place a `.libs` file is ever produced.

A `.libs` file is a plain, uncompressed npz archive holding `data`
(n_scans, n_pixels), `wavelengths` (n_pixels,) and a JSON-encoded `params`
dict. See `docs/libs_format.md` for the written specification. Anything a
user can get into a NumPy array -- ENVI, FITS, a CSV spectral matrix, an
instrument export -- becomes a processable sample by passing it through
here; nothing about the format is proprietary and nothing here needs a
private package.

Two details carry the real weight.

**Scan order.** Rows of `data` are stored in serpentine order: rows top to
bottom, each row left to right except odd rows, which run right to left.
That is how the stage travels, and `reporting.raster` undoes it when
placing values back on a grid. Data arriving from another format is almost
always in plain raster order instead, so `order="raster"` (the default)
reverses every odd row on the way in. Getting this wrong mirrors alternate
rows of every map, which is why `order` is explicit rather than assumed.

**Memory.** The archive members are stored uncompressed, so each one can be
written straight into the zip entry: the `.npy` header first, then the
spectra block by block. Peak memory is one block, not the whole cube, which
is what lets a scan larger than RAM be converted. The array is never
materialised, and no temporary file is involved.
"""

import json
import math
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format

from pylibs.core.exceptions import ReaderError
from pylibs.core.logging import get_logger

logger = get_logger(__name__)

DATA_MEMBER = "data.npy"
WAVELENGTHS_MEMBER = "wavelengths.npy"
PARAMS_MEMBER = "params.npy"

# float64 is what every bundled file uses; float32 halves the size and runs
# through the pipeline unchanged (metrics promote as needed), so it is
# offered for large conversions where the source was never float64 anyway.
SUPPORTED_DTYPES = ("float64", "float32")

# Written from the grid/step arguments, so a caller's own `params` may not
# also set them -- two disagreeing sources of the same geometry is exactly
# the bug this writer exists to prevent.
DERIVED_PARAM_KEYS = frozenset(
    {"HeightPixels", "WidthPixels", "WidthIMG", "HeightIMG", "Resolution", "NbPixels", "NbLambda"}
)


def _as_blocks(spectra, n_pixels: int, dtype: str) -> Iterator[np.ndarray]:
    """Normalise every accepted spectra input to a stream of 2D blocks.

    An array or memmap is yielded in slices rather than whole, so a memmap
    is never faulted in entirely; an iterable is passed through as given."""
    if isinstance(spectra, np.ndarray):
        step = max(1, 1_000_000 // max(1, n_pixels))
        for start in range(0, spectra.shape[0], step):
            yield np.ascontiguousarray(spectra[start : start + step], dtype=dtype)
        return
    for block in spectra:
        yield np.ascontiguousarray(np.atleast_2d(np.asarray(block)), dtype=dtype)


def _flip_odd_rows(block: np.ndarray, n_cols: int, row_offset: int) -> np.ndarray:
    """Convert one raster-ordered block to serpentine order.

    `row_offset` is how many complete raster rows precede this block, which
    is what decides whether the block's first row is an odd one. Blocks are
    required to hold whole rows, so each row can be reversed independently
    and the result is identical to reversing the whole array at once."""
    reshaped = block.reshape(-1, n_cols, block.shape[-1])
    out = reshaped.copy()
    odd = (np.arange(reshaped.shape[0]) + row_offset) % 2 == 1
    out[odd] = reshaped[odd, ::-1]
    return out.reshape(block.shape)


def _grid_from_coordinates(coordinates: np.ndarray) -> tuple[tuple[int, int], tuple[float, float]]:
    """(ny, nx) and (dy, dx) from per-pixel (x, y) positions on a regular
    lattice. Raises if the positions are not on one -- this writer stores a
    raster, and an irregular point cloud has no raster to store."""
    xs = np.unique(coordinates[:, 0])
    ys = np.unique(coordinates[:, 1])
    if xs.size < 1 or ys.size < 1:
        raise ReaderError("coordinates must contain at least one distinct x and y value")
    if coordinates.shape[0] != xs.size * ys.size:
        raise ReaderError(
            f"coordinates describe {xs.size} x {ys.size} = {xs.size * ys.size} lattice "
            f"position(s) but {coordinates.shape[0]} spectra were given -- this writer stores "
            "a full rectangular raster, so every grid position needs exactly one spectrum"
        )
    steps = []
    for axis in (ys, xs):
        if axis.size == 1:
            steps.append(0.0)
            continue
        diffs = np.diff(axis)
        if not np.allclose(diffs, diffs[0]):
            raise ReaderError(
                "coordinates are not on a regular lattice: spacing varies between "
                f"{diffs.min():g} and {diffs.max():g}"
            )
        steps.append(float(diffs[0]))
    return (int(ys.size), int(xs.size)), (steps[0], steps[1])


def _validate(
    wavelengths: np.ndarray, n_scans: int, grid: tuple[int, int], dtype: str, order: str
) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise ReaderError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")
    if order not in ("raster", "serpentine"):
        raise ReaderError(f"order must be 'raster' or 'serpentine', got {order!r}")
    if wavelengths.ndim != 1 or wavelengths.size == 0:
        raise ReaderError(
            f"wavelengths must be a non-empty 1D array, got shape {wavelengths.shape}"
        )
    if not np.all(np.isfinite(wavelengths)):
        raise ReaderError("wavelengths must be finite -- found NaN or infinity")
    if not np.all(np.diff(wavelengths) > 0):
        bad = int(np.argmin(np.diff(wavelengths))) + 1
        raise ReaderError(
            "wavelengths must be strictly increasing; the axis does not increase at index "
            f"{bad} ({wavelengths[bad - 1]:g} then {wavelengths[bad]:g})"
        )
    n_rows, n_cols = grid
    if n_rows < 1 or n_cols < 1:
        raise ReaderError(f"grid must be positive, got {grid}")
    if n_rows * n_cols != n_scans:
        raise ReaderError(
            f"grid {n_rows}x{n_cols} holds {n_rows * n_cols} position(s) but {n_scans} "
            "spectra were given"
        )


def _build_params(
    wavelengths: np.ndarray,
    n_scans: int,
    grid: tuple[int, int],
    step: tuple[float, float] | None,
    units: str,
    mask: np.ndarray | None,
    metadata: dict | None,
    params: dict | None,
    path: Path,
) -> dict:
    """The `params` dict as it will be stored.

    `params` is passed through unchanged apart from the geometry fields this
    writer derives itself; `metadata` is merged on top and may not touch
    them at all. Keeping the two separate is what lets a `.ELE`-derived dict
    round-trip while a hand-written one stays impossible to contradict."""
    n_rows, n_cols = grid
    out: dict[str, Any] = {}
    if params:
        clashing = sorted(DERIVED_PARAM_KEYS.intersection(params))
        out.update({k: v for k, v in params.items() if k not in DERIVED_PARAM_KEYS})
        if clashing:
            logger.info(
                "Ignoring geometry key(s) %s in the supplied params -- they are derived from "
                "the grid and step arguments instead",
                clashing,
            )
    if metadata:
        forbidden = sorted(DERIVED_PARAM_KEYS.intersection(metadata))
        if forbidden:
            raise ReaderError(
                f"metadata may not set geometry key(s) {forbidden} -- pass grid/coordinates and "
                "step instead, so the stored geometry cannot contradict the data"
            )
        out.update(metadata)

    out.update(
        {
            "HeightPixels": int(n_rows),
            "WidthPixels": int(n_cols),
            "NbPixels": int(n_scans),
            "NbLambda": int(wavelengths.size),
        }
    )
    if step is not None:
        dy, dx = step
        if dy > 0 and dx > 0 and math.isclose(dy, dx):
            out["Resolution"] = float(dx)
        if dx > 0:
            out["WidthIMG"] = float(dx * n_cols)
        if dy > 0:
            out["HeightIMG"] = float(dy * n_rows)
        out["SpatialUnits"] = units
    if mask is not None:
        out["PixelAssignmentMatrix"] = np.asarray(mask).astype(int).tolist()
    out.setdefault("filename", path.name)
    out.setdefault("stem", path.stem)
    return out


def _write_archive(
    path: Path,
    blocks: Iterator[np.ndarray],
    total: int,
    n_pixels: int,
    dtype: str,
    wavelengths: np.ndarray,
    params: dict,
    n_cols: int = 0,
    flip_odd_rows: bool = False,
) -> None:
    """Write the three members, streaming `blocks` straight into the entry.

    The only place a `.libs` archive is produced. `force_zip64` because the
    member's size is not known when its entry opens, and a scan larger than
    4 GiB is exactly what this path exists for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rows_written = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        with archive.open(DATA_MEMBER, "w", force_zip64=True) as handle:
            npy_format.write_array_header_2_0(
                handle,
                {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": (total, n_pixels)},
            )
            for block in blocks:
                if block.shape[1] != n_pixels:
                    raise ReaderError(
                        f"a spectra block has {block.shape[1]} wavelength column(s), expected "
                        f"{n_pixels}"
                    )
                if not np.all(np.isfinite(block)):
                    raise ReaderError(
                        "spectra must be finite -- found NaN or infinity in the block at row "
                        f"{written}"
                    )
                if flip_odd_rows and n_cols > 1:
                    if block.shape[0] % n_cols:
                        raise ReaderError(
                            f"a raster-ordered block holds {block.shape[0]} spectra, which is "
                            f"not a whole number of {n_cols}-wide rows -- odd rows are reversed "
                            "block by block, so a block must not split a row"
                        )
                    block = _flip_odd_rows(block, n_cols, rows_written)
                    rows_written += block.shape[0] // n_cols
                written += block.shape[0]
                if written > total:
                    raise ReaderError(f"spectra yielded more than the declared {total} row(s)")
                handle.write(block.tobytes())
        if written != total:
            raise ReaderError(f"spectra yielded {written} row(s), expected {total}")
        with archive.open(WAVELENGTHS_MEMBER, "w", force_zip64=True) as handle:
            npy_format.write_array(handle, wavelengths)
        with archive.open(PARAMS_MEMBER, "w", force_zip64=True) as handle:
            npy_format.write_array(handle, np.array(json.dumps(params)))


def write_libs_file(path: Path, data: np.ndarray, wavelengths: np.ndarray, params: dict) -> None:
    """Write a `.libs` file from a complete array and a complete params dict,
    storing both exactly as given.

    The raw primitive under `write_libs`: no validation, no geometry derived,
    no scan-order conversion. Use `write_libs` for ingestion -- this is for a
    caller that already holds a finished params dict, and for tests that need
    a file with deliberately sparse metadata."""
    data = np.ascontiguousarray(data)
    _write_archive(
        Path(path),
        iter([data]),
        int(data.shape[0]),
        int(data.shape[1]),
        str(data.dtype),
        np.ascontiguousarray(wavelengths),
        params,
    )


def write_libs(
    path: Path,
    spectra,
    wavelengths,
    *,
    grid: tuple[int, int] | None = None,
    coordinates=None,
    step: tuple[float, float] | float | None = None,
    units: str = "mm",
    mask=None,
    metadata: dict | None = None,
    params: dict | None = None,
    dtype: str = "float64",
    order: str = "raster",
    n_scans: int | None = None,
) -> Path:
    """Write `spectra` and `wavelengths` to `path` as a `.libs` archive.

    See the module docstring for the format and for why `order` matters.
    Returns `path`.
    """
    path = Path(path).expanduser()
    wavelengths = np.ascontiguousarray(np.asarray(wavelengths, dtype="float64").ravel())
    n_pixels = int(wavelengths.size)

    if (grid is None) == (coordinates is None):
        raise ReaderError("exactly one of grid or coordinates is required")

    if coordinates is not None:
        coordinates = np.asarray(coordinates, dtype="float64")
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ReaderError(f"coordinates must have shape (n_scans, 2), got {coordinates.shape}")
        if not np.all(np.isfinite(coordinates)):
            raise ReaderError("coordinates must be finite -- found NaN or infinity")
        grid, derived_step = _grid_from_coordinates(coordinates)
        if step is None:
            step = derived_step
        # positions already say where each spectrum sits, so the caller's own
        # ordering is what the file should preserve
        order = "serpentine"

    if isinstance(spectra, np.ndarray):
        if spectra.ndim != 2:
            raise ReaderError(f"spectra must be 2D (n_scans, n_pixels), got shape {spectra.shape}")
        if spectra.shape[1] != n_pixels:
            raise ReaderError(
                f"spectra has {spectra.shape[1]} wavelength column(s) but wavelengths has "
                f"{n_pixels}"
            )
        total = int(spectra.shape[0])
        if n_scans is not None and n_scans != total:
            raise ReaderError(f"n_scans={n_scans} contradicts the array's own {total} row(s)")
    else:
        if n_scans is None:
            raise ReaderError(
                "n_scans is required when spectra is an iterable of blocks -- the archive's "
                "array header is written before the blocks arrive, so the final shape has to "
                "be known up front"
            )
        total = int(n_scans)

    if coordinates is not None and coordinates.shape[0] != total:
        raise ReaderError(f"coordinates has {coordinates.shape[0]} row(s) for {total} spectra")

    assert grid is not None
    if isinstance(step, int | float):
        step = (float(step), float(step))
    _validate(wavelengths, total, grid, dtype, order)

    n_rows, n_cols = grid
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (n_rows, n_cols):
            raise ReaderError(f"mask must have shape {(n_rows, n_cols)}, got {mask.shape}")

    stored_params = _build_params(
        wavelengths, total, grid, step, units, mask, metadata, params, path
    )

    _write_archive(
        path,
        _as_blocks(spectra, n_pixels, dtype),
        total,
        n_pixels,
        dtype,
        wavelengths,
        stored_params,
        n_cols=n_cols,
        flip_odd_rows=order == "raster",
    )

    logger.info(
        "Wrote %s: %d scan(s) x %d wavelength(s), %dx%d raster, dtype %s",
        path,
        total,
        n_pixels,
        n_rows,
        n_cols,
        dtype,
    )
    return path


def write_libs_from_blocks(
    path: Path, blocks: Iterable[np.ndarray], wavelengths, n_scans: int, **kwargs
) -> Path:
    """`write_libs` for an explicit block iterator -- a thin alias that makes
    the required `n_scans` impossible to forget."""
    return write_libs(path, blocks, wavelengths, n_scans=n_scans, **kwargs)


# ----------------------------------------------------------------- inputs ---
#
# The two file layouts the CLI and DSL accept. Both live here rather than in
# an interface, so the facade exposes one conversion path and the CLI stays a
# thin wrapper. No new dependencies: `.npy`/`.npz` is NumPy, the CSV reader is
# stdlib plus one NumPy parse.


def load_csv_matrix(path: Path, x_column: str = "x", y_column: str = "y"):
    """Read a CSV spectral matrix: one row per pixel, coordinate columns,
    then one column per wavelength.

    The header's first cells name the coordinate columns; every remaining
    header cell is the wavelength that column holds, as a number. So

        x,y,200.5,200.9,201.3
        0.0,0.0,1712,1690,1705
        0.1,0.0,1698,1702,1711

    is a two-pixel scan at three wavelengths. Returns
    `(spectra, wavelengths, coordinates)`.

    Coordinate columns may appear in any position and are matched by name;
    everything else must parse as a float, which is what makes a stray
    non-numeric column an error here rather than a puzzle later."""
    import csv

    path = Path(path).expanduser()
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ReaderError(f"{path} is empty") from None
        rows = [row for row in reader if row]
    if not rows:
        raise ReaderError(f"{path} has a header but no data rows")

    lowered = [cell.strip().lower() for cell in header]
    try:
        x_index = lowered.index(x_column.lower())
        y_index = lowered.index(y_column.lower())
    except ValueError:
        raise ReaderError(
            f"{path} has no {x_column!r} and {y_column!r} column(s); its header starts {header[:4]}"
        ) from None

    spectral_indices = [i for i in range(len(header)) if i not in (x_index, y_index)]
    if not spectral_indices:
        raise ReaderError(f"{path} has coordinate columns but no wavelength columns")
    try:
        wavelengths = np.array([float(header[i]) for i in spectral_indices])
    except ValueError as exc:
        raise ReaderError(
            f"{path}'s header must give a wavelength for every non-coordinate column: {exc}"
        ) from None

    width = len(header)
    for number, row in enumerate(rows, start=2):
        if len(row) != width:
            raise ReaderError(
                f"{path} line {number} has {len(row)} field(s), the header has {width}"
            )
    try:
        values = np.array([[float(cell) for cell in row] for row in rows])
    except ValueError as exc:
        raise ReaderError(f"{path} holds a non-numeric value: {exc}") from None

    return values[:, spectral_indices], wavelengths, values[:, [x_index, y_index]]


def load_array(path: Path, key: str | None = None) -> np.ndarray:
    """Read one array from a `.npy` file, or one member of a `.npz`.

    `key` names the member; a single-member `.npz` needs none."""
    path = Path(path).expanduser()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            if key is None:
                if len(names) != 1:
                    raise ReaderError(
                        f"{path} holds {len(names)} arrays ({names}) -- name the one to use"
                    )
                key = names[0]
            if key not in names:
                raise ReaderError(f"{path} has no array {key!r}; it holds {names}")
            return np.asarray(archive[key])
    return np.load(path, allow_pickle=False, mmap_mode="r")
