"""Convert a proprietary `.ELE` file to the open, self-contained `.libs`
format -- lets a dataset be shared, committed, or used in tests without
requiring the private `libs_importer` package.
"""

from pathlib import Path

from pylibs.core.exceptions import ReaderError
from pylibs.core.io.reader import get_reader
from pylibs.core.io.writers.libs_writer import write_libs


def convert_ele_to_libs(
    ele_path: Path,
    libs_path: Path,
    capdata: int | None = None,
    batchsize: int | None = None,
) -> None:
    """Read all of `ele_path` (streamed in `batchsize`-sized batches through
    the private `libs_importer` package) and write it to `libs_path` in the
    `.libs` archive format.

    Streams: batches go straight into the archive as they are read, so peak
    memory is one batch rather than the whole sample (see
    `io.writers.libs_writer`). The source's own `params` are carried across
    unchanged apart from the geometry fields the writer derives.
    """
    ele_path = Path(ele_path)
    if ele_path.suffix.lower() != ".ele":
        raise ReaderError(f"Not a .ELE file: {ele_path}")

    reader = get_reader(ele_path, capdata=capdata, batchsize=batchsize)
    params = dict(reader.params)
    n_scans = reader.n_scans
    # the source already stores its scans in the stage's own serpentine
    # order, so they pass through untouched; the grid comes from the file's
    # own params, falling back to a single row when it carries none
    n_rows = int(params.get("HeightPixels") or 1)
    n_cols = int(params.get("WidthPixels") or n_scans)
    if n_rows * n_cols != n_scans:
        n_rows, n_cols = 1, n_scans
    write_libs(
        Path(libs_path),
        reader.iter_window(batchsize),
        reader.wavelengths,
        grid=(n_rows, n_cols),
        params=params,
        order="serpentine",
        n_scans=n_scans,
    )
