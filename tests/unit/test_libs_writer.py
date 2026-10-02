"""The `.libs` writer: the ingestion path a user reaches for when their data
arrived in some other format.

Covered here rather than in `test_cli.py` because most of what matters is a
property of the written file rather than of a command: that it round-trips,
that it is indistinguishable from a native sample, that a chunked write and
an in-memory one produce the same bytes, and that the raster-to-serpentine
conversion puts values back in the right place on a map. The CLI and DSL
wrappers are exercised at the end.
"""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from pylibs.core import api
from pylibs.core.exceptions import PylibsError
from pylibs.core.io.readers.libs_reader import LibsFileReader
from pylibs.core.reporting.raster import to_raster
from pylibs.interfaces.cli.app import app

runner = CliRunner()

NY, NX, N_PIXELS = 4, 5, 8


def _wavelengths(n: int = N_PIXELS) -> np.ndarray:
    return np.linspace(200.0, 210.0, n)


def _raster_spectra(ny: int = NY, nx: int = NX, n_pixels: int = N_PIXELS) -> np.ndarray:
    """One spectrum per pixel, every channel carrying that pixel's own
    row*10+col, so a mirrored row is visible in the values themselves."""
    return np.array([[float(row * 10 + col)] * n_pixels for row in range(ny) for col in range(nx)])


def _read(path):
    reader = LibsFileReader(path)
    return np.concatenate(list(reader.iter_window(3))), reader.wavelengths, reader.params


def test_round_trip_preserves_spectra_wavelengths_and_metadata(tmp_path):
    spectra = _raster_spectra()
    wavelengths = _wavelengths()
    path = api.write_libs(
        tmp_path / "s.libs",
        spectra,
        wavelengths,
        grid=(NY, NX),
        step=0.1,
        order="serpentine",  # store exactly what is given
        metadata={"Instrument": "test bench", "Note": "round trip"},
    )

    stored, stored_wavelengths, params = _read(path)

    assert np.array_equal(stored, spectra)
    assert np.array_equal(stored_wavelengths, wavelengths)
    assert params["Instrument"] == "test bench"
    assert params["Note"] == "round trip"
    assert (params["HeightPixels"], params["WidthPixels"]) == (NY, NX)
    assert params["Resolution"] == pytest.approx(0.1)
    assert params["WidthIMG"] == pytest.approx(0.1 * NX)
    assert params["HeightIMG"] == pytest.approx(0.1 * NY)


def test_raster_input_is_stored_serpentine_and_maps_back_unmirrored(tmp_path):
    """The one that catches a mirrored map: written in plain row-major order,
    every odd row must come back in its original left-to-right order once
    `to_raster` has undone the serpentine storage."""
    spectra = _raster_spectra()
    path = api.write_libs(tmp_path / "s.libs", spectra, _wavelengths(), grid=(NY, NX), step=0.1)

    stored, _, _ = _read(path)

    # stored: odd rows reversed
    assert list(stored[:NX, 0]) == [0, 1, 2, 3, 4]
    assert list(stored[NX : 2 * NX, 0]) == [14, 13, 12, 11, 10]
    # mapped back: every row ascending again
    grid = to_raster(stored[:, 0], NY, NX)
    for row in range(NY):
        assert list(grid[row]) == [row * 10 + col for col in range(NX)]


def test_serpentine_input_is_stored_unchanged(tmp_path):
    spectra = _raster_spectra()
    path = api.write_libs(
        tmp_path / "s.libs", spectra, _wavelengths(), grid=(NY, NX), order="serpentine"
    )
    stored, _, _ = _read(path)
    assert np.array_equal(stored, spectra)


def test_chunked_write_matches_in_memory_write_member_for_member(tmp_path):
    spectra = _raster_spectra()
    wavelengths = _wavelengths()

    whole = api.write_libs(tmp_path / "whole.libs", spectra, wavelengths, grid=(NY, NX), step=0.1)
    # blocks of two whole rows, the way a converter would stream them
    blocks = [spectra[i : i + 2 * NX] for i in range(0, NY * NX, 2 * NX)]
    chunked = api.write_libs(
        tmp_path / "chunked.libs",
        iter(blocks),
        wavelengths,
        grid=(NY, NX),
        step=0.1,
        n_scans=NY * NX,
    )

    # Not byte-for-byte across the whole file: a zip entry records the moment
    # it was written, and `params` carries each file's own name. What has to
    # match is the data and the axis, byte for byte, and every other param.
    import zipfile

    with zipfile.ZipFile(whole) as a, zipfile.ZipFile(chunked) as b:
        assert a.namelist() == b.namelist()
        assert a.read("data.npy") == b.read("data.npy")
        assert a.read("wavelengths.npy") == b.read("wavelengths.npy")

    whole_params, chunked_params = _read(whole)[2], _read(chunked)[2]
    for params in (whole_params, chunked_params):
        for naming in ("filename", "stem"):
            params.pop(naming)
    assert whole_params == chunked_params
    assert np.array_equal(_read(whole)[0], _read(chunked)[0])


def test_chunked_write_from_a_memmap_matches(tmp_path):
    spectra = _raster_spectra()
    memmap_path = tmp_path / "cube.npy"
    np.save(memmap_path, spectra)
    memmap = np.load(memmap_path, mmap_mode="r")

    whole = api.write_libs(tmp_path / "a.libs", spectra, _wavelengths(), grid=(NY, NX))
    mapped = api.write_libs(tmp_path / "b.libs", memmap, _wavelengths(), grid=(NY, NX))

    assert np.array_equal(_read(whole)[0], _read(mapped)[0])


def test_coordinates_derive_the_grid_and_step(tmp_path):
    spectra = _raster_spectra()
    coordinates = np.array(
        [[col * 0.2, row * 0.5] for row in range(NY) for col in range(NX)], dtype=float
    )

    path = api.write_libs(tmp_path / "s.libs", spectra, _wavelengths(), coordinates=coordinates)

    _, _, params = _read(path)
    assert (params["HeightPixels"], params["WidthPixels"]) == (NY, NX)
    assert params["WidthIMG"] == pytest.approx(0.2 * NX)
    assert params["HeightIMG"] == pytest.approx(0.5 * NY)
    # spacing differs per axis, so no single Resolution is claimed
    assert "Resolution" not in params


def test_float32_is_stored_and_read_back_as_float32(tmp_path):
    path = api.write_libs(
        tmp_path / "s.libs", _raster_spectra(), _wavelengths(), grid=(NY, NX), dtype="float32"
    )
    stored, _, _ = _read(path)
    assert stored.dtype == np.float32


def test_params_pass_through_but_cannot_contradict_the_geometry(tmp_path):
    """A converted file's own params ride along; the geometry fields are
    always the writer's, since they must match the data that was written."""
    path = api.write_libs(
        tmp_path / "s.libs",
        _raster_spectra(),
        _wavelengths(),
        grid=(NY, NX),
        params={"AnalysisDate": "5/9/2024", "HeightPixels": 999, "nDetectors": 2},
    )
    _, _, params = _read(path)
    assert params["AnalysisDate"] == "5/9/2024"
    assert params["nDetectors"] == 2
    assert params["HeightPixels"] == NY


def test_metadata_may_not_set_geometry_fields(tmp_path):
    with pytest.raises(PylibsError, match="geometry key"):
        api.write_libs(
            tmp_path / "s.libs",
            _raster_spectra(),
            _wavelengths(),
            grid=(NY, NX),
            metadata={"WidthPixels": 3},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"grid": (3, 3)}, "holds 9 position"),
        ({"grid": (NY, NX), "dtype": "int16"}, "dtype must be"),
        ({"grid": (NY, NX), "order": "snake"}, "order must be"),
        ({}, "exactly one of grid or coordinates"),
    ],
)
def test_validation_errors_are_specific(tmp_path, kwargs, message):
    with pytest.raises(PylibsError, match=message):
        api.write_libs(tmp_path / "s.libs", _raster_spectra(), _wavelengths(), **kwargs)


def test_non_increasing_wavelengths_are_rejected(tmp_path):
    wavelengths = _wavelengths().copy()
    wavelengths[3], wavelengths[4] = wavelengths[4], wavelengths[3]
    with pytest.raises(PylibsError, match="strictly increasing"):
        api.write_libs(tmp_path / "s.libs", _raster_spectra(), wavelengths, grid=(NY, NX))


def test_non_finite_spectra_are_rejected(tmp_path):
    spectra = _raster_spectra()
    spectra[7, 2] = np.nan
    with pytest.raises(PylibsError, match="finite"):
        api.write_libs(tmp_path / "s.libs", spectra, _wavelengths(), grid=(NY, NX))


def test_mismatched_coordinate_count_is_rejected(tmp_path):
    coordinates = np.array([[0.0, 0.0], [0.1, 0.0]])
    with pytest.raises(PylibsError, match=r"2 row\(s\) for 20 spectra"):
        api.write_libs(
            tmp_path / "s.libs", _raster_spectra(), _wavelengths(), coordinates=coordinates
        )


def test_coordinates_off_a_regular_lattice_are_rejected(tmp_path):
    """A point cloud has no raster to store, so it is refused rather than
    silently snapped to one."""
    coordinates = np.array(
        [[col * 0.1, row * 0.1] for row in range(NY) for col in range(NX)], dtype=float
    )
    coordinates[3, 0] += 0.037  # one pixel off the lattice
    with pytest.raises(PylibsError, match="lattice|regular"):
        api.write_libs(
            tmp_path / "s.libs", _raster_spectra(), _wavelengths(), coordinates=coordinates
        )


def test_a_block_may_not_split_a_raster_row(tmp_path):
    spectra = _raster_spectra()
    blocks = [spectra[i : i + 3] for i in range(0, NY * NX, 3)]  # 3 does not divide NX=5
    with pytest.raises(PylibsError, match="whole number of"):
        api.write_libs(
            tmp_path / "s.libs", iter(blocks), _wavelengths(), grid=(NY, NX), n_scans=NY * NX
        )


def test_iterator_input_requires_n_scans(tmp_path):
    with pytest.raises(PylibsError, match="n_scans is required"):
        api.write_libs(
            tmp_path / "s.libs", iter([_raster_spectra()]), _wavelengths(), grid=(NY, NX)
        )


# --- the written file is a real sample ---------------------------------------


def _peak_table(tmp_path):
    path = tmp_path / "peaks.csv"
    path.write_text(
        "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\n"
        "A,A203,202.857,,,,1\nB,B207,207.143,,,,1\n"
    )
    return path


def test_a_converted_file_registers_and_runs_the_whole_pipeline(tmp_path):
    n_pixels = 64
    wavelengths = np.linspace(200.0, 210.0, n_pixels)
    rng = np.random.default_rng(0)
    spectra = rng.uniform(9.0, 11.0, size=(NY * NX, n_pixels))
    for target in (202.857, 207.143):
        spectra[:, np.argmin(np.abs(wavelengths - target))] += np.arange(NY * NX) + 100.0

    libs_path = api.write_libs(
        tmp_path / "converted.libs", spectra, wavelengths, grid=(NY, NX), step=0.1
    )

    root = tmp_path / "proj"
    api.create_project("demo", root=root)
    api.use_peak_table("demo", _peak_table(tmp_path), project_root=root)
    api.add_sample("demo", libs_path, project_root=root)
    api.add_feature("demo", "A203", project_root=root)
    api.add_feature("demo", "A203/B207", project_root=root)

    result = api.run_pipeline("demo", "sample001", project_root=root)

    assert result.n_scans == NY * NX
    assert result.steps_run == ["features", "spectrum_stats", "distribution_1d", "distribution_2d"]
    assert api.get_distribution_1d("demo", "sample001", "feature001", project_root=root) is not None
    assert api.get_distribution_2d("demo", "sample001", "feature002", project_root=root) is not None


def test_memory_mapped_reader_matches_a_full_load(tmp_path):
    """The reader maps `data` in place rather than loading the archive. The
    values it serves must be exactly those `numpy.load` would return."""
    spectra = _raster_spectra()
    path = api.write_libs(tmp_path / "s.libs", spectra, _wavelengths(), grid=(NY, NX), step=0.1)

    with np.load(path, allow_pickle=False) as archive:
        reference = archive["data"]
        reference_wavelengths = archive["wavelengths"]
        reference_params = json.loads(str(archive["params"]))

    streamed, wavelengths, params = _read(path)

    assert np.array_equal(streamed, reference)
    assert streamed.dtype == reference.dtype
    assert np.array_equal(wavelengths, reference_wavelengths)
    assert params == reference_params
    # capdata still truncates the same way
    capped = LibsFileReader(path, capdata=6)
    assert capped.n_scans == 6
    assert np.array_equal(np.concatenate(list(capped.iter_window(4))), reference[:6])


# --- CLI and DSL wrappers -----------------------------------------------------


def test_cli_writes_from_npy_arrays(tmp_path):
    spectra, wavelengths = _raster_spectra(), _wavelengths()
    np.save(tmp_path / "spectra.npy", spectra)
    np.save(tmp_path / "wavelengths.npy", wavelengths)
    out = tmp_path / "out.libs"

    result = runner.invoke(
        app,
        [
            "write-libs",
            str(out),
            "--spectra",
            str(tmp_path / "spectra.npy"),
            "--wavelengths",
            str(tmp_path / "wavelengths.npy"),
            "--grid",
            f"{NY}x{NX}",
            "--step",
            "0.1",
        ],
    )

    assert result.exit_code == 0, result.output
    stored, _, params = _read(out)
    assert stored.shape == (NY * NX, N_PIXELS)
    assert params["WidthPixels"] == NX


def test_cli_writes_from_an_npz_archive(tmp_path):
    np.savez(tmp_path / "bundle.npz", cube=_raster_spectra(), axis=_wavelengths())
    out = tmp_path / "out.libs"

    result = runner.invoke(
        app,
        [
            "write-libs",
            str(out),
            "--spectra",
            str(tmp_path / "bundle.npz"),
            "--npz-key",
            "cube",
            "--wavelengths",
            str(tmp_path / "bundle.npz"),
            "--grid",
            f"{NY}x{NX}",
        ],
    )

    # the wavelengths archive holds two arrays and was given no key
    assert result.exit_code == 1
    assert "name the one to use" in result.output


def test_cli_writes_from_a_csv_matrix(tmp_path):
    wavelengths = _wavelengths(4)
    csv_path = tmp_path / "matrix.csv"
    lines = ["x,y," + ",".join(f"{w:g}" for w in wavelengths)]
    for row in range(NY):
        for col in range(NX):
            values = [str(float(row * 10 + col))] * len(wavelengths)
            lines.append(f"{col * 0.1},{row * 0.1}," + ",".join(values))
    csv_path.write_text("\n".join(lines) + "\n")
    out = tmp_path / "out.libs"

    result = runner.invoke(app, ["write-libs", str(out), "--csv", str(csv_path)])

    assert result.exit_code == 0, result.output
    stored, stored_wavelengths, params = _read(out)
    assert np.allclose(stored_wavelengths, wavelengths)
    assert (params["HeightPixels"], params["WidthPixels"]) == (NY, NX)
    assert stored.shape == (NY * NX, len(wavelengths))


def test_cli_rejects_a_grid_that_is_not_rows_by_cols(tmp_path):
    np.save(tmp_path / "s.npy", _raster_spectra())
    np.save(tmp_path / "w.npy", _wavelengths())
    result = runner.invoke(
        app,
        [
            "write-libs",
            str(tmp_path / "o.libs"),
            "--spectra",
            str(tmp_path / "s.npy"),
            "--wavelengths",
            str(tmp_path / "w.npy"),
            "--grid",
            "20",
        ],
    )
    assert result.exit_code == 1
    assert "ROWSxCOLS" in result.output


def test_csv_loader_rejects_a_missing_coordinate_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("a,b,200.0,201.0\n1,2,3,4\n")
    with pytest.raises(PylibsError, match="no 'x' and 'y' column"):
        api.load_csv_matrix(csv_path)


def test_csv_loader_rejects_a_non_numeric_header(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("x,y,200.0,label\n0,0,3,4\n")
    with pytest.raises(PylibsError, match="wavelength for every"):
        api.load_csv_matrix(csv_path)


def test_dsl_verb_writes_a_file(tmp_path):
    script = tmp_path / "convert.pylibs"
    script.write_text(
        "import numpy as np\n"
        f"spectra = np.zeros(({NY * NX}, {N_PIXELS})) + 1.0\n"
        f"wavelengths = np.linspace(200.0, 210.0, {N_PIXELS})\n"
        f"WRITELIBS({str(tmp_path / 'dsl.libs')!r}, spectra, wavelengths, "
        f"grid=({NY}, {NX}), step=0.1)\n"
    )
    result = runner.invoke(app, ["run", str(script)])
    assert result.exit_code == 0, result.output
    stored, _, params = _read(tmp_path / "dsl.libs")
    assert stored.shape == (NY * NX, N_PIXELS)
    assert params["Resolution"] == pytest.approx(0.1)
