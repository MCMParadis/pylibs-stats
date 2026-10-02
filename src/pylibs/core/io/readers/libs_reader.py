"""Built-in reader for the open `.libs` format: an npz archive holding
`data` (n_scan, n_pix), `wavelengths` (n_pix,), and a JSON-encoded `params`
dict. See `docs/libs_format.md` for the written specification and
`io.writers.libs_writer` for the matching writer.

The members are stored uncompressed, so `data` is memory-mapped in place at
its offset inside the archive rather than read into memory: `iter_window`
then faults in one batch at a time and a scan larger than RAM streams from
disk, the way the `.ELE` reader does. `wavelengths` and `params` are small
and are read eagerly. A file whose `data` member turns out to be compressed
(nothing in this project writes one, but `numpy.savez_compressed` would)
falls back to loading the array, so such a file still reads correctly.
"""

import json
import struct
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format

from pylibs.core.exceptions import ReaderError

DATA_MEMBER = "data.npy"
_LOCAL_HEADER_SIZE = 30


def _memmap_member(path: Path, member: str) -> np.ndarray | None:
    """Memory-map one stored (uncompressed) `.npy` member of a zip archive.

    Returns `None` when the member is compressed, so the caller can fall
    back to reading it. The payload offset comes from the archive's real
    local header rather than a reconstructed one, since a zip64 entry's
    extra field changes its length."""
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError:
            raise ReaderError(f"{path} has no {member!r} member -- is it a .libs file?") from None
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        with open(path, "rb") as handle:
            handle.seek(info.header_offset)
            name_len, extra_len = struct.unpack("<HH", handle.read(_LOCAL_HEADER_SIZE)[26:30])
        with archive.open(member) as handle:
            version = npy_format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
            else:
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
            header_size = handle.tell()
    offset = info.header_offset + _LOCAL_HEADER_SIZE + name_len + extra_len + header_size
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


class LibsFileReader:
    def __init__(self, path: Path, capdata: int | None = None, batchsize: int | None = None):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise ReaderError(f"No such file: {self.path}")
        with np.load(self.path, allow_pickle=False) as npz:
            self._wavelengths = npz["wavelengths"]
            self._params = json.loads(str(npz["params"]))
            data = _memmap_member(self.path, DATA_MEMBER)
            if data is None:
                data = npz[DATA_MEMBER.removesuffix(".npy")]
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
            # np.asarray materialises just this batch out of the memory map,
            # so a consumer holding one batch never pins the whole scan
            yield np.asarray(self._data[start : start + bs])
