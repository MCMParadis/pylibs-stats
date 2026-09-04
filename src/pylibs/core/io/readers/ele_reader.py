"""Adapter over the private `libs_importer` package (.ELE files). `libs_importer`
is never a hard dependency of pylibs -- the import is attempted lazily, here,
so pylibs works fully without it (just can't open .ELE files).

Translating that package's own failures into `ReaderError` is this adapter's
job too: nothing above `core.io` should have to know `libs_importer`'s
exception vocabulary, and an interface that only catches `PylibsError` (every
CLI command does) would otherwise dump a traceback at the user instead of a
one-line message. `libs_importer` raises `EleParseError` for a corrupt file --
including one whose scanned-cell mask disagrees with its own scan count, which
it refuses to open rather than let it mis-place every value downstream.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pylibs.core.exceptions import ReaderError


def _import_ele_stream() -> tuple[type, tuple[type[BaseException], ...]]:
    """`libs_importer`'s `EleStream` plus the exception types it raises for an
    unreadable file. Falls back to the base `Exception` on a version that
    predates those types, so this adapter still translates rather than leaking
    whatever the older package raised."""
    try:
        from libs_importer import EleStream
    except ImportError as exc:
        raise ReaderError(
            "Reading .ELE files requires the private `libs_importer` package, "
            "which isn't installed. Install it (e.g. `pip install pylibs[ele]`) "
            "to enable this."
        ) from exc

    try:
        from libs_importer import EleParseError, EleVersionError

        return EleStream, (EleParseError, EleVersionError)
    except ImportError:
        return EleStream, (Exception,)


@contextmanager
def _as_reader_error(path: Path, errors: tuple[type[BaseException], ...]) -> Iterator[None]:
    """Re-raise `libs_importer`'s own read failures as `ReaderError`, keeping
    its message (which names the actual defect) and chaining the original."""
    try:
        yield
    except ReaderError:
        raise
    except errors as exc:
        raise ReaderError(f"Can't read {path}: {exc}") from exc


class EleFileReader:
    def __init__(self, path: Path, capdata: int | None = None, batchsize: int | None = None):
        ele_stream, self._errors = _import_ele_stream()

        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise ReaderError(f"No such file: {self.path}")
        self._EleStream = ele_stream
        self._capdata = capdata
        self._batchsize = batchsize
        with _as_reader_error(self.path, self._errors):
            self._stream = ele_stream(self.path, capdata=capdata if capdata is not None else -1)

    @property
    def wavelengths(self) -> np.ndarray:
        return self._stream.wavelengths

    @property
    def params(self) -> dict:
        return self._stream.params

    @property
    def n_scans(self) -> int:
        total = self._stream.nr_scan
        return min(total, self._capdata) if self._capdata is not None else total

    def iter_window(self, batch_size: int | None = None) -> Iterator[np.ndarray]:
        bs = batch_size if batch_size is not None else self._batchsize
        with _as_reader_error(self.path, self._errors):
            with self._EleStream(
                self.path,
                batchsize=bs if bs is not None else -1,
                capdata=self._capdata if self._capdata is not None else -1,
            ) as stream:
                yield from stream
