"""Convert a proprietary `.ELE` file to the open, self-contained `.libs`
format -- lets a dataset be shared, committed, or used in tests without
requiring the private `libs_importer` package.
"""

from pathlib import Path

import numpy as np

from pylibs.core.exceptions import ReaderError
from pylibs.core.io.reader import get_reader
from pylibs.core.io.readers.libs_reader import write_libs_file


def convert_ele_to_libs(
    ele_path: Path,
    libs_path: Path,
    capdata: int | None = None,
    batchsize: int | None = None,
) -> None:
    """Read all of `ele_path` (streamed in `batchsize`-sized batches through
    the private `libs_importer` package) and write it to `libs_path` in the
    `.libs` archive format.

    The whole sample is held in memory before writing -- `.libs` is a single
    npz archive, not an incrementally-written format -- so this is meant for
    producing small, shareable samples, not for production-scale conversions.
    """
    ele_path = Path(ele_path)
    if ele_path.suffix.lower() != ".ele":
        raise ReaderError(f"Not a .ELE file: {ele_path}")

    reader = get_reader(ele_path, capdata=capdata, batchsize=batchsize)
    batches = list(reader.iter_window(batchsize))
    data = np.concatenate(batches, axis=0) if batches else np.empty((0, len(reader.wavelengths)))
    wavelengths = reader.wavelengths
    params = reader.params
    write_libs_file(Path(libs_path), data=data, wavelengths=wavelengths, params=params)
