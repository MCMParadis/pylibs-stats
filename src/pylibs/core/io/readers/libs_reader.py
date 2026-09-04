"""Built-in reader for the open `.libs` format: an npz archive holding
`data` (n_scan, n_pix), `wavelengths` (n_pix,), and a JSON-encoded `params`
dict. Intended for the small public sample dataset, not large production
scans (those go through a proprietary LibsReader backend with real
seek-based streaming) -- this reader loads the whole (small) archive once
and serves `iter_window` from the in-memory array.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from pylibs.core.exceptions import ReaderError


class LibsFileReader:
    def __init__(self, path: Path, capdata: int | None = None, batchsize: int | None = None):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise ReaderError(f"No such file: {self.path}")
        with np.load(self.path, allow_pickle=False) as npz:
            data = npz["data"]
            self._wavelengths = npz["wavelengths"]
            self._params = json.loads(str(npz["params"]))
        self._data = data[:capdata] if capdata is not None else data
        self._batchsize = batchsize

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @property
    def params(self) -> dict:
        return self._params

    @property
    def n_scans(self) -> int:
        return int(self._data.shape[0])

    def iter_window(self, batch_size: int | None = None) -> Iterator[np.ndarray]:
        bs = batch_size if batch_size is not None else self._batchsize
        if not bs or bs <= 0:
            bs = self._data.shape[0]
        for start in range(0, self._data.shape[0], bs):
            yield self._data[start : start + bs]


def write_libs_file(path: Path, data: np.ndarray, wavelengths: np.ndarray, params: dict) -> None:
    """Write a `.libs` sample file. For tests and the small public sample
    dataset -- not a general-purpose ingestion path."""
    with open(path, "wb") as f:
        np.savez(f, data=data, wavelengths=wavelengths, params=np.array(json.dumps(params)))
