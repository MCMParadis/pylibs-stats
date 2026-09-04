"""The backend boundary: any LIBS file backend (built-in or proprietary)
implements this Protocol to plug into pylibs without pylibs depending on it.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from pylibs.core.exceptions import ReaderError


class LibsReader(Protocol):
    @property
    def wavelengths(self) -> np.ndarray:
        """Shape (n_pix,)."""
        ...

    @property
    def params(self) -> dict:
        """Flat instrument/acquisition metadata."""
        ...

    @property
    def n_scans(self) -> int:
        """Total number of spectra available."""
        ...

    def iter_window(self, batch_size: int | None = None) -> Iterator[np.ndarray]:
        """Yield batches of shape (n, n_pix), n <= batch_size. If `batch_size` is
        omitted, uses the reader's own default (or a single batch of everything)."""
        ...


# Every extension some reader below can open, lowercased. The single source of
# truth for "is this file a sample?" -- `project.add_samples_from_dir`'s
# directory scan filters on it, so a new backend becomes visible to the scanner
# by being registered here rather than in a second, parallel list.
SUPPORTED_EXTENSIONS = frozenset({".libs", ".ele"})


def get_reader(path: Path, capdata: int | None = None, batchsize: int | None = None) -> LibsReader:
    """Return the `LibsReader` for `path`, chosen by its file extension.

    `capdata` caps the total number of spectra the reader reports/yields.
    `batchsize` sets the default batch size used by `iter_window` when it's
    called without an explicit `batch_size`.
    """
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ReaderError(f"No reader available for file extension {suffix!r}: {path}")
    if suffix == ".libs":
        from pylibs.core.io.readers.libs_reader import LibsFileReader

        return LibsFileReader(path, capdata=capdata, batchsize=batchsize)
    if suffix == ".ele":
        from pylibs.core.io.readers.ele_reader import EleFileReader

        return EleFileReader(path, capdata=capdata, batchsize=batchsize)
    # unreachable unless SUPPORTED_EXTENSIONS gains an entry without a branch here
    raise ReaderError(f"No reader wired up for supported extension {suffix!r}: {path}")
