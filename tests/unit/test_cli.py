"""CLI integration tests: one test per workflow (a happy-path sequence of
commands) or per critical error boundary, rather than one test per command --
correctness of the underlying computations is covered at the core.api layer
(or trusted via these end-to-end runs); these tests exist to catch a broken
CLI wrapper (wrong argument forwarding, wrong exit code, wrong output text).
"""

import json
import logging
import shutil
import sys
import types

import numpy as np
import pytest
from typer.testing import CliRunner

from pylibs.core import api
from pylibs.core.io.writers.libs_writer import write_libs_file
from pylibs.core.logging import LOG_FILENAME
from pylibs.core.pipeline.metrics import METRIC_NAMES_2D
from pylibs.core.pipeline.regression import _mae_permutation_p_value, fit_least_squares
from pylibs.interfaces.cli.app import app

runner = CliRunner()

_ELE_WAVELENGTHS = np.linspace(200.0, 210.0, 5)
_ELE_DATA = np.arange(20.0).reshape(4, 5)


class _FakeEleStream:
    def __init__(self, path, capdata=-1, batchsize=-1):
        self.wavelengths = _ELE_WAVELENGTHS
        self.params = {}
        self.nr_scan = _ELE_DATA.shape[0]
        self._batchsize = batchsize

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def __iter__(self):
        bs = self._batchsize if self._batchsize and self._batchsize > 0 else _ELE_DATA.shape[0]
        for start in range(0, _ELE_DATA.shape[0], bs):
            yield _ELE_DATA[start : start + bs]


@pytest.fixture
def fake_libs_importer(monkeypatch):
    module = types.ModuleType("libs_importer")
    module.EleStream = _FakeEleStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libs_importer", module)


_PEAK_CSV_HEADER = "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\n"


def _use_peak_table(tmp_path, root, csv_body, name="peaks.csv"):
    """Write a peak table CSV and register it with `project`, which every
    later peak-resolving command reads out of the project itself. Returns the
    source path, so a test that edits it mid-run can rewrite and re-register."""
    peak_table = tmp_path / name
    peak_table.write_text(_PEAK_CSV_HEADER + csv_body)
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(peak_table), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    return peak_table


def _make_processed_project(tmp_path, root):
    """A project with one bare feature and 3 rastered, processed samples
    (S1/S2/S3 with known intensities) plus a matching properties CSV --
    the shared starting point for correlations/regressions/applied-
    regressions/report workflows below."""
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    for index, (name, height) in enumerate([("S1", 1.0), ("S2", 2.0), ("S3", 3.0)], start=1):
        wavelengths = np.array([200.0, 300.0, 400.0, 500.0])
        data = np.full((5, 4), 10.0)
        data[:, 1] = height + np.arange(5) * 0.01
        libs_path = tmp_path / f"{name}.libs"
        write_libs_file(
            libs_path,
            data=data,
            wavelengths=wavelengths,
            params={"HeightPixels": 1, "WidthPixels": 5},
        )
        runner.invoke(
            app,
            [
                "project",
                "add-sample",
                "demo",
                str(libs_path),
                "--sample-name",
                name,
                "--root",
                str(root),
            ],
        )
        sample_id = f"sample{index:03d}"  # registration order is deterministic/sequential
        runner.invoke(
            app,
            [
                "project",
                "run-pipeline",
                "demo",
                sample_id,
                "--root",
                str(root),
            ],
        )
    csv_path = tmp_path / "props.csv"
    csv_path.write_text("sample_name,concentration\nS1,1.0\nS2,2.0\nS3,3.0\n")
    return csv_path


def _setup_bootstrap_project(tmp_path, root, n_scans=20):
    wavelengths = np.linspace(200.0, 210.0, 200)
    data = np.full((n_scans, len(wavelengths)), 10.0)
    peak_idx = np.argmin(np.abs(wavelengths - 205.0))
    data[:, peak_idx] = np.arange(n_scans) + 100.0
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(libs_path, data=data, wavelengths=wavelengths, params={})
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    peak_table = _use_peak_table(
        tmp_path,
        root,
        "A,A205,205.0,,,,1\n",
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A205",
            "--root",
            str(root),
        ],
    )
    # a ratio feature too (feature002), so bootstrap's numerator/denominator-
    # from-the-store code path (Distribution2DStep's own via run-pipeline, and
    # run_bootstrap's own) both get exercised, not just the bare-peak path
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A205 / A205",
            "--root",
            str(root),
        ],
    )
    runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    return peak_table


def test_project_create_command(tmp_path, monkeypatch):
    root = tmp_path / "proj"

    result = runner.invoke(
        app, ["project", "create", "demo", "--root", str(root), "--description", "old"]
    )
    assert result.exit_code == 0, result.output
    assert (root / "registry.json").exists()

    # re-creating at the same root updates the existing project in place
    runner.invoke(app, ["project", "create", "demo", "--root", str(root), "--description", "new"])
    data = json.loads((root / "registry.json").read_text())
    assert data["config"]["description"] == "new"

    # no --root defaults to ./projects/<name>
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["project", "create", "other"])
    assert (tmp_path / "projects" / "other" / "registry.json").exists()


def test_project_lifecycle_happy_path(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    result = runner.invoke(
        app,
        [
            "project",
            "add-sample",
            "demo",
            str(libs_path),
            "--sample-name",
            "S1",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "sample001" in result.output

    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    result = runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "feature001" in result.output

    result = runner.invoke(app, ["project", "list-samples", "demo", "--root", str(root)])
    assert "sample001" in result.output and "S1" in result.output

    result = runner.invoke(app, ["project", "list-features", "demo", "--root", str(root)])
    assert "feature001" in result.output and "A300" in result.output

    result = runner.invoke(
        app, ["project", "inspect-sample", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "2 scans" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 feature(s) for 2 scans" in result.output

    result = runner.invoke(
        app, ["project", "get-results", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "mean=11" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "generate-feature-report",
            "demo",
            "sample001",
            "--root",
            str(root),
            "--save-pdf",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "img" / "features" / "sample001" / "feature001.png").exists()
    assert (root / "reports" / "img" / "features" / "sample001" / "feature001_table.png").exists()
    assert (root / "reports" / "sample001_features.pdf").exists()


def test_project_lifecycle_error_boundaries(tmp_path):
    root = tmp_path / "proj"  # project never created

    for args in [
        ["project", "add-sample", "demo", "data/sample01.csv"],
        ["project", "add-feature", "demo", "peak_a / peak_b"],
        ["project", "add-element-features", "demo", "A"],
        ["project", "add-samples-from-dir", "demo", "."],
        ["project", "list-samples", "demo"],
        ["project", "list-features", "demo"],
        ["project", "inspect-sample", "demo", "sample001"],
        ["project", "run-pipeline", "demo", "sample001"],
        ["project", "get-results", "demo", "sample001"],
    ]:
        result = runner.invoke(app, [*args, "--root", str(root)])
        assert result.exit_code == 1, f"{args} should fail against a nonexistent project"
        # assert the reason, not just the code: add-feature/add-element-features
        # also need a registered peak table, and when they checked for that
        # first they still exited 1 -- but for the wrong reason, which this
        # loop could not see
        assert "No project found" in result.output, f"{args}: {result.output}"

    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])

    # ...and with the project created but no peak table registered, the two
    # that resolve peak ids now fail on that instead
    for args in [
        ["project", "add-feature", "demo", "A300"],
        ["project", "add-element-features", "demo", "A"],
    ]:
        result = runner.invoke(app, [*args, "--root", str(root)])
        assert result.exit_code == 1, f"{args}: {result.output}"
        assert "No peak table registered" in result.output, f"{args}: {result.output}"
    result = runner.invoke(
        app, ["project", "run-pipeline", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 1  # sample never registered

    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.zeros((2, 4)),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={},
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    result = runner.invoke(
        app, ["project", "generate-feature-report", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 1  # pipeline never run


def _element_peak_table(tmp_path, root):
    """A table with a 3-peak element (one hyphen-suffixed id, one flagged
    default), a 1-peak element, and an element with no flagged default at
    all -- the three shapes add-element-features has to tell apart."""
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n"
        "A,A300-2,300.0,,,,0\n"
        "A,A400,400.0,,,,0\n"
        "B,B200,200.0,,,,1\n"
        "C,C500,500.0,,,,0\n",
        name="element_peaks.csv",
    )


def test_add_element_features_workflow(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    _element_peak_table(tmp_path, root)

    result = runner.invoke(
        app,
        [
            "project",
            "add-element-features",
            "demo",
            "A",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    # every peak for A, in the table's own row order -- including the
    # hyphen-suffixed id, which the expression parser reads as one token
    assert ["A300", "A300-2", "A400"] == [
        line.rsplit(": ", 1)[1] for line in result.output.strip().splitlines()
    ]

    # re-running registers nothing new -- same feature ids, no duplicates
    repeat = runner.invoke(
        app,
        [
            "project",
            "add-element-features",
            "demo",
            "A",
            "--root",
            str(root),
        ],
    )
    assert repeat.exit_code == 0, repeat.output
    assert repeat.output == result.output

    # several elements in one invocation
    result = runner.invoke(
        app,
        [
            "project",
            "add-element-features",
            "demo",
            "B",
            "C",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "B200" in result.output and "C500" in result.output

    result = runner.invoke(app, ["project", "list-features", "demo", "--root", str(root)])
    for expression in ["A300", "A300-2", "A400", "B200", "C500"]:
        assert expression in result.output
    assert "feature005" in result.output and "feature006" not in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "5 feature(s) for 2 scans" in result.output


def test_add_element_features_only_default_and_unknown_element(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _element_peak_table(tmp_path, root)

    result = runner.invoke(
        app,
        [
            "project",
            "add-element-features",
            "demo",
            "A",
            "--only-default",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "A300" in result.output
    assert "A300-2" not in result.output and "A400" not in result.output

    # C exists but has no flagged default, and Zz doesn't exist at all: both
    # register nothing, and neither is fatal (the distinction is on the log)
    for args in [
        ["project", "add-element-features", "demo", "C", "--only-default"],
        ["project", "add-element-features", "demo", "Zz"],
    ]:
        result = runner.invoke(app, [*args, "--root", str(root)])
        assert result.exit_code == 0, result.output
        assert "No features registered." in result.output

    # ...and that distinction really is on the project log, at INFO -- below
    # the console handler's own threshold, so a standing element list doesn't
    # shout on every run about the ones this table has no lines for
    log = (root / "pylibs.log").read_text()
    assert "No peaks for element 'Zz'" in log
    assert "No peak flagged as the default for element 'C'" in log
    assert " ERROR " not in log

    # a bad element among good ones doesn't cost the good ones
    result = runner.invoke(
        app,
        [
            "project",
            "add-element-features",
            "demo",
            "Zz",
            "B",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "B200" in result.output

    result = runner.invoke(app, ["project", "list-features", "demo", "--root", str(root)])
    assert "A300" in result.output and "B200" in result.output
    assert "A400" not in result.output


def _sample_dir(tmp_path):
    """A directory of two physical samples' ablation layers, plus the two
    things a scan has to leave alone: a non-sample file, and a nested one."""
    directory = tmp_path / "raw"
    (directory / "nested").mkdir(parents=True)
    for name in ["s1_1.libs", "s1_2.libs", "s2_1.libs"]:
        write_libs_file(
            directory / name,
            data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
            wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
            params={"WidthPixels": 2, "HeightPixels": 1},
        )
    (directory / "s3_1.ELE").touch()  # registration never opens the file
    (directory / "concentrations.csv").write_text("sample_name,c\ns1,1.0\n")
    (directory / "nested" / "s9_1.libs").touch()
    return directory


def _registry_samples(root):
    return json.loads((root / "registry.json").read_text())["samples"]


def test_add_samples_from_dir_workflow(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    directory = _sample_dir(tmp_path)

    result = runner.invoke(
        app, ["project", "add-samples-from-dir", "demo", str(directory), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output

    samples = _registry_samples(root)
    # sorted path order, so ids are reproducible; the .ELE counts, the CSV
    # doesn't, and the nested file is out of reach without --recursive
    assert [s["sample_stem"] for s in samples.values()] == ["s1_1", "s1_2", "s2_1", "s3_1"]
    assert [s["sample_name"] for s in samples.values()] == ["s1", "s1", "s2", "s3"]
    assert [s["name_parts"] for s in samples.values()] == [
        {"layer": "1"},
        {"layer": "2"},
        {"layer": "1"},
        {"layer": "1"},
    ]

    # re-running registers no duplicates
    repeat = runner.invoke(
        app, ["project", "add-samples-from-dir", "demo", str(directory), "--root", str(root)]
    )
    assert repeat.exit_code == 0, repeat.output
    assert len(_registry_samples(root)) == 4

    result = runner.invoke(
        app,
        [
            "project",
            "add-samples-from-dir",
            "demo",
            str(directory),
            "--recursive",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "s9_1" in result.output
    assert len(_registry_samples(root)) == 5

    # a directory-registered sample processes like any other
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 feature(s) for 2 scans" in result.output


def test_add_samples_from_dir_scheme_and_error_boundaries(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    directory = tmp_path / "raw"
    directory.mkdir()
    for name in ["sA_temp300-run2_1.libs", "sA_temp300-run2_2.libs", "calibration.libs"]:
        (directory / name).touch()

    # a richer scheme: every field but sample_name lands in name_parts
    result = runner.invoke(
        app,
        [
            "project",
            "add-samples-from-dir",
            "demo",
            str(directory),
            "--scheme",
            "{sample_name}_{param}-{info1}_{layer}",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    samples = _registry_samples(root)
    assert [s["sample_name"] for s in samples.values()] == ["sA", "sA"]
    assert list(samples.values())[0]["name_parts"] == {
        "param": "temp300",
        "info1": "run2",
        "layer": "1",
    }
    # 'calibration' can't match that scheme -- skipped, not fatal, and it
    # didn't stop its two siblings from registering
    assert len(samples) == 2

    for args in [
        ["project", "add-samples-from-dir", "demo", str(directory), "--scheme", "{layer}"],
        ["project", "add-samples-from-dir", "demo", str(tmp_path / "nope")],
    ]:
        result = runner.invoke(app, [*args, "--root", str(root)])
        assert result.exit_code == 1, result.output
        assert "Error:" in result.output

    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app, ["project", "add-samples-from-dir", "demo", str(empty), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "No samples registered." in result.output


def _disc_mask(size, radius):
    """A round scanned region in a square frame -- the shape a real .ELE soil
    sample has, small enough to assert on cell by cell."""
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size - 1) / 2.0
    return ((yy - centre) ** 2 + (xx - centre) ** 2) <= radius**2


def _write_masked_sample(path, mask, n_pix=4):
    """A .libs sample carrying `mask` as its PixelAssignmentMatrix, with one
    scan per set cell -- the .libs side of what libs_importer hands over for
    a partially-scanned .ELE."""
    n_scans = int(mask.sum())
    data = np.zeros((n_scans, n_pix))
    data[:, 1] = np.arange(n_scans) + 1.0  # a per-scan ramp, so placement is visible
    write_libs_file(
        path,
        data=data,
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={
            "HeightPixels": int(mask.shape[0]),
            "WidthPixels": int(mask.shape[1]),
            "Resolution": 0.1,
            "PixelAssignmentMatrix": mask.astype(int).tolist(),
        },
    )
    return n_scans


def test_to_raster_masked_placement_and_guards():
    """The reconstruction itself: serpentine order, NaN outside, and the two
    ways a caller can hand over a mask that can't place its values."""
    from pylibs.core.exceptions import ReportError
    from pylibs.core.reporting.raster import to_raster

    # a 3x3 frame with the middle row fully scanned and one cell above/below
    mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    grid = to_raster(np.arange(1.0, 6.0), 3, 3, mask=mask)

    # row 0 left-to-right, row 1 RIGHT-to-left (serpentine), row 2 left-to-right
    assert grid[0, 1] == 1.0
    assert [grid[1, 2], grid[1, 1], grid[1, 0]] == [2.0, 3.0, 4.0]
    assert grid[2, 1] == 5.0
    assert np.isnan([grid[0, 0], grid[0, 2], grid[2, 0], grid[2, 2]]).all()

    # an all-True mask must reproduce the plain reshape exactly, so the
    # unmasked .libs default can't drift from the masked path
    values = np.arange(12.0)
    assert np.array_equal(
        to_raster(values, 3, 4), to_raster(values, 3, 4, mask=np.ones((3, 4), bool))
    )

    with pytest.raises(ReportError, match="marking 5 scanned"):
        to_raster(np.arange(4.0), 3, 3, mask=mask)
    with pytest.raises(ReportError, match="Mask is"):
        to_raster(np.arange(5.0), 4, 4, mask=mask)


def test_frame_shape_counts_pixels_or_intervals():
    """Some headers count the intervals between pixels rather than the pixels."""
    from pylibs.core.exceptions import ReportError
    from pylibs.core.reporting.raster import frame_shape

    assert frame_shape({"HeightPixels": 3, "WidthPixels": 4}, 12) == (3, 4)
    # a 400x400 .ELE scan whose header says 399x399
    assert frame_shape({"HeightPixels": 399, "WidthPixels": 399}, 160000) == (400, 400)
    with pytest.raises(ReportError, match="neither a 3x4 raster nor a 4x5"):
        frame_shape({"HeightPixels": 3, "WidthPixels": 4}, 13)
    with pytest.raises(ReportError, match="no HeightPixels"):
        frame_shape({"WidthPixels": 4}, 12)


def test_interval_style_header_is_settled_at_ingest(tmp_path):
    """Some instruments write HeightPixels/WidthPixels as the intervals
    between pixels, so a 4x5 scan arrives as 3x4. The frame is resolved once,
    when the sample is processed, and the stored params carry the real one --
    so every consumer (maps, applied-regression maps, `local` bootstrap
    resampling) reads a trustworthy value instead of re-deriving it."""
    root = tmp_path / "proj"
    wavelengths = np.linspace(200.0, 210.0, 60)
    n_scans = 20  # a 4x5 frame
    data = np.full((n_scans, len(wavelengths)), 10.0)
    data[:, np.argmin(np.abs(wavelengths - 205.0))] = np.arange(n_scans) + 100.0
    libs_path = tmp_path / "sample.libs"
    write_libs_file(
        libs_path,
        data=data,
        wavelengths=wavelengths,
        params={"HeightPixels": 3, "WidthPixels": 4},  # intervals, not pixels
    )
    peak_table = tmp_path / "peaks.csv"
    peak_table.write_text(
        "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\nA,A205,205.0,,,,1\n"
    )

    api.create_project("demo", root=root)
    api.use_peak_table("demo", peak_table, project_root=root)
    api.add_sample("demo", libs_path, project_root=root)
    api.add_feature("demo", "A205", project_root=root)
    api.run_pipeline("demo", "sample001", project_root=root)

    store = api.ResultsStore(root / "results.zarr")
    params = store.get_raw_params("sample001")
    assert (params["HeightPixels"], params["WidthPixels"]) == (4, 5)
    # the header's own values are kept, so the correction is auditable
    assert (params["HeightPixelsHeader"], params["WidthPixelsHeader"]) == (3, 4)

    # and the report that used to raise "Can't reshape 20 values" now renders
    result = runner.invoke(
        app, ["project", "generate-feature-report", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output


def test_a_mask_outranks_a_disagreeing_header(tmp_path):
    """The mask carries one set cell per scan, which is checkable; the header
    is the part known to be unreliable. So a mask that disagrees with the
    header defines the frame rather than being dropped for disagreeing."""
    root = tmp_path / "proj"
    wavelengths = np.linspace(200.0, 210.0, 60)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True  # 6 scanned cells in a 4x5 frame
    n_scans = int(mask.sum())
    data = np.full((n_scans, len(wavelengths)), 10.0)
    data[:, np.argmin(np.abs(wavelengths - 205.0))] = np.arange(n_scans) + 100.0
    libs_path = tmp_path / "sample.libs"
    write_libs_file(
        libs_path,
        data=data,
        wavelengths=wavelengths,
        params={
            "HeightPixels": 3,
            "WidthPixels": 4,
            "PixelAssignmentMatrix": mask.astype(int).tolist(),
        },
    )
    peak_table = tmp_path / "peaks.csv"
    peak_table.write_text(
        "element,id,peak,continuum,tol_peak,tol_continuum,_isdefault\nA,A205,205.0,,,,1\n"
    )

    api.create_project("demo", root=root)
    api.use_peak_table("demo", peak_table, project_root=root)
    api.add_sample("demo", libs_path, project_root=root)
    api.add_feature("demo", "A205", project_root=root)
    api.run_pipeline("demo", "sample001", project_root=root)

    store = api.ResultsStore(root / "results.zarr")
    params = store.get_raw_params("sample001")
    assert (params["HeightPixels"], params["WidthPixels"]) == (4, 5)
    # the mask survived rather than being discarded for disagreeing
    assert store.get_mask_array("sample001") is not None


def test_masked_sample_report_workflow(tmp_path):
    """A round sample reports where it used to raise 'Can't reshape N values'."""
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    mask = _disc_mask(21, 9.5)
    n_scans = _write_masked_sample(tmp_path / "disc.libs", mask)
    assert n_scans < mask.size  # the whole point: fewer scans than frame cells

    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app, ["project", "add-sample", "demo", str(tmp_path / "disc.libs"), "--root", str(root)]
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output

    # the mask was cached as its own array, since set_raw_params drops it
    store = api.ResultsStore(root / "results.zarr")
    stored = store.get_mask_array("sample001")
    assert stored is not None and np.array_equal(stored, mask)

    result = runner.invoke(
        app, ["project", "generate-feature-report", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "img" / "features" / "sample001" / "feature001.png").exists()


def test_mask_ignored_when_it_cannot_place_scans(tmp_path):
    """A mask whose set-cell count disagrees with n_scans is dropped at store
    time rather than silently mis-placing every value."""
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    mask = _disc_mask(9, 4.0)
    write_libs_file(
        tmp_path / "bad.libs",
        data=np.zeros((int(mask.sum()) - 3, 4)),  # three scans short of the mask
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={
            "HeightPixels": 9,
            "WidthPixels": 9,
            "PixelAssignmentMatrix": mask.astype(int).tolist(),
        },
    )
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app, ["project", "add-sample", "demo", str(tmp_path / "bad.libs"), "--root", str(root)]
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert api.ResultsStore(root / "results.zarr").get_mask_array("sample001") is None


def test_local_bootstrap_block_stays_inside_the_mask():
    """Every 'local' block on a round sample is npix real, distinct scans --
    never a rim block padded with unscanned cells."""
    from pylibs.core.exceptions import BootstrapError
    from pylibs.core.pipeline.bootstrap.resampling import resample_local

    mask = _disc_mask(21, 9.5)
    n_scans = int(mask.sum())
    rng = np.random.default_rng(0)
    for _ in range(50):
        positions = resample_local(rng, n_scans, 9, 21, 21, mask=mask)
        assert positions.shape == (9,)
        assert len(set(positions.tolist())) == 9
        assert positions.min() >= 0 and positions.max() < n_scans

    # a block bigger than the disc's own largest full square can't be placed
    with pytest.raises(BootstrapError, match="No fully-scanned"):
        resample_local(rng, n_scans, 21 * 21, 21, 21, mask=mask)


def test_ratio_feature_distribution_2d(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.full((6, 4), 1.0),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={},
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\nB,B400,400.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300 / B400",
            "--root",
            str(root),
        ],
    )
    runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )

    result = runner.invoke(
        app,
        ["project", "get-distribution-2d", "demo", "sample001", "feature001", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert "A300 / B400" in result.output


def test_correlations_workflow(tmp_path):
    root = tmp_path / "proj"
    csv_path = _make_processed_project(tmp_path, root)

    # compute-correlations never fits anything itself -- it reads back
    # already-stored regressions, so an all-metrics sweep must run first
    result = runner.invoke(
        app,
        [
            "project",
            "compute-regressions",
            "demo",
            str(csv_path),
            "--root",
            str(root),
            "--all-metrics",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app, ["project", "compute-correlations", "demo", str(csv_path), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "concentration" in result.output

    result = runner.invoke(app, ["project", "list-correlations", "demo", "--root", str(root)])
    assert "concentration" in result.output

    result = runner.invoke(
        app, ["project", "get-correlations", "demo", "concentration", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "feature001" in result.output

    result = runner.invoke(
        app, ["project", "get-correlations", "demo", "does_not_exist", "--root", str(root)]
    )
    assert result.exit_code == 1

    result = runner.invoke(
        app,
        ["project", "generate-correlations-report", "demo", "concentration", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "concentration_correlations.pdf").exists()

    result = runner.invoke(
        app,
        ["project", "generate-correlations-report", "demo", "does_not_exist", "--root", str(root)],
    )
    assert result.exit_code == 1


def test_regressions_workflow(tmp_path):
    root = tmp_path / "proj"
    csv_path = _make_processed_project(tmp_path, root)

    result = runner.invoke(
        app,
        [
            "project",
            "compute-regressions",
            "demo",
            str(csv_path),
            "--root",
            str(root),
            "--select-features",
            "A300",
            "--metric",
            "Mean",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "concentration" in result.output

    result = runner.invoke(app, ["project", "list-regressions", "demo", "--root", str(root)])
    assert "concentration" in result.output

    result = runner.invoke(
        app, ["project", "get-regression-results", "demo", "concentration", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "slope=" in result.output

    result = runner.invoke(
        app, ["project", "get-regression-results", "demo", "does_not_exist", "--root", str(root)]
    )
    assert result.exit_code == 1

    result = runner.invoke(
        app,
        ["project", "generate-regression-report", "demo", "concentration", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "concentration_regressions.pdf").exists()

    result = runner.invoke(
        app,
        ["project", "generate-regression-report", "demo", "does_not_exist", "--root", str(root)],
    )
    assert result.exit_code == 1


def _make_unprocessed_multi_sample_project(tmp_path, root, n_samples=3):
    """Like `_make_processed_project`, but with `n_samples` samples
    registered and *not yet run* -- the starting point for
    run-pipeline-batch tests."""
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\nB,B400,400.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300 / B400",
            "--root",
            str(root),
        ],
    )
    sample_ids = []
    for index in range(1, n_samples + 1):
        libs_path = tmp_path / f"sample{index}.libs"
        write_libs_file(
            libs_path,
            data=np.full((5, 4), 1.0) + index,
            wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
            params={},
        )
        runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
        sample_ids.append(f"sample{index:03d}")
    return sample_ids


def test_run_pipeline_batch_workflow(tmp_path):
    root = tmp_path / "proj"
    sample_ids = _make_unprocessed_multi_sample_project(tmp_path, root, n_samples=3)

    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline-batch",
            "demo",
            "--root",
            str(root),
            "--n-processes",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3/3 sample(s) processed" in result.output

    for sample_id in sample_ids:
        results = api.get_results("demo", sample_id, project_root=root)
        assert results.feature_ids == ["feature001", "feature002"]
        distribution = api.get_distribution_1d("demo", sample_id, "feature001", project_root=root)
        assert distribution.metrics  # every Distribution1DStep worker actually wrote its result

    # idempotent: a second batch call has nothing left to do
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline-batch",
            "demo",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3/3 sample(s) processed" in result.output


def test_run_pipeline_n_processes_matches_sequential(tmp_path):
    """Distribution1DStep/Distribution2DStep results must not depend on
    n_processes -- each feature's histogram+metrics is computed
    independently of the others."""
    root_seq = tmp_path / "seq"
    sample_ids = _make_unprocessed_multi_sample_project(tmp_path, root_seq, 1)
    sample_id = sample_ids[0]
    runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            sample_id,
            "--root",
            str(root_seq),
        ],
    )

    root_par = tmp_path / "par"
    _make_unprocessed_multi_sample_project(tmp_path, root_par, 1)
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            sample_id,
            "--root",
            str(root_par),
            "--n-processes",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    for feature_id in ["feature001", "feature002"]:
        seq = api.get_distribution_1d("demo", sample_id, feature_id, project_root=root_seq)
        par = api.get_distribution_1d("demo", sample_id, feature_id, project_root=root_par)
        assert seq.metrics == par.metrics

    seq_2d = api.get_distribution_2d("demo", sample_id, "feature002", project_root=root_seq)
    par_2d = api.get_distribution_2d("demo", sample_id, "feature002", project_root=root_par)
    assert seq_2d.metrics == par_2d.metrics


def test_n_processes_auto_detect_completes(tmp_path):
    """n_processes<=0 resolves to the local CPU count (core.runs.executor.
    resolve_n_processes) -- just check it runs successfully, not the exact
    process count."""
    root = tmp_path / "proj"
    sample_ids = _make_unprocessed_multi_sample_project(tmp_path, root, 1)
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            sample_ids[0],
            "--root",
            str(root),
            "--n-processes",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_compute_regressions_n_processes_matches_sequential(tmp_path):
    """The regression fitting sweep must not depend on n_processes -- each
    (column, metric, feature) cell is seeded independently."""
    root = tmp_path / "proj"
    csv_path = _make_processed_project(tmp_path, root)

    result_seq = runner.invoke(
        app,
        [
            "project",
            "compute-regressions",
            "demo",
            str(csv_path),
            "--root",
            str(root),
            "--all-metrics",
            "--n-permutations",
            "200",
            "--random-seed",
            "7",
        ],
    )
    assert result_seq.exit_code == 0, result_seq.output
    sequential = {
        r.metric_name: (r.slope, r.intercept, r.mae, r.p_value, r.pearson_r)
        for r in api.get_regression_results("demo", "concentration", project_root=root)
    }

    root_par = tmp_path / "proj_par"
    _make_processed_project(tmp_path, root_par)
    result_par = runner.invoke(
        app,
        [
            "project",
            "compute-regressions",
            "demo",
            str(csv_path),
            "--root",
            str(root_par),
            "--all-metrics",
            "--n-permutations",
            "200",
            "--random-seed",
            "7",
            "--n-processes",
            "2",
        ],
    )
    assert result_par.exit_code == 0, result_par.output
    parallel = {
        r.metric_name: (r.slope, r.intercept, r.mae, r.p_value, r.pearson_r)
        for r in api.get_regression_results("demo", "concentration", project_root=root_par)
    }
    assert sequential == parallel


def test_compute_regressions_worker_sharding_covers_all_cells(tmp_path):
    """--worker-index/--n-workers statically shard the full (column, metric,
    feature) cell range across separately-invoked processes (e.g. SLURM
    array-job tasks). Each shard run in isolation covers a disjoint subset
    of cells (a degenerate fit -- e.g. a metric that ties across all 3
    samples -- is skipped regardless of sharding, so which exact metrics
    fit isn't asserted here); running every shard together must land the
    exact union of what each shard covers alone, with no cell computed
    twice, same partitioning contract as run-pipeline-batch's own
    sharding."""

    def _fitted_metrics(root) -> set[str]:
        return {
            r.metric_name
            for r in api.get_regression_results("demo", "concentration", project_root=root)
        }

    def _run_shard(root, worker_index: int) -> None:
        csv_path = _make_processed_project(tmp_path, root)
        result = runner.invoke(
            app,
            [
                "project",
                "compute-regressions",
                "demo",
                str(csv_path),
                "--root",
                str(root),
                "--all-metrics",
                "--worker-index",
                str(worker_index),
                "--n-workers",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output

    root0 = tmp_path / "shard0"
    _run_shard(root0, 0)
    fitted_0 = _fitted_metrics(root0)

    root1 = tmp_path / "shard1"
    _run_shard(root1, 1)
    fitted_1 = _fitted_metrics(root1)

    assert fitted_0.isdisjoint(fitted_1)  # no cell computed by both shards
    assert fitted_0 | fitted_1  # at least one shard found something to fit

    root_both = tmp_path / "both_shards"
    _run_shard(root_both, 0)
    _run_shard(root_both, 1)
    assert _fitted_metrics(root_both) == fitted_0 | fitted_1


def test_n_processes_auto_detect_bootstrap_and_regressions(tmp_path):
    """n_processes<=0 resolves to the local CPU count (core.runs.executor.
    resolve_n_processes) for run-bootstrap and compute-regressions too, not
    just run-pipeline (see test_n_processes_auto_detect_completes) -- just
    check both complete successfully, not the exact process count."""
    root = tmp_path / "bootstrap_proj"
    _setup_bootstrap_project(tmp_path, root)
    runner.invoke(
        app,
        ["project", "add-bootstrap-config", "demo", "random", "--npix", "5", "--root", str(root)],
    )
    result = runner.invoke(
        app,
        [
            "project",
            "run-bootstrap",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
            "--n-processes",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output

    root_reg = tmp_path / "regression_proj"
    csv_path = _make_processed_project(tmp_path, root_reg)
    result = runner.invoke(
        app,
        [
            "project",
            "compute-regressions",
            "demo",
            str(csv_path),
            "--root",
            str(root_reg),
            "--n-processes",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output


def test_applied_regressions_workflow(tmp_path):
    root = tmp_path / "proj"
    csv_path = _make_processed_project(tmp_path, root)

    # applying before ever computing a regression is a hard error
    result = runner.invoke(
        app, ["project", "apply-regressions", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 1
    # ...and so is reading results/reporting before ever applying
    result = runner.invoke(
        app,
        ["project", "get-applied-regression-results", "demo", "sample001", "--root", str(root)],
    )
    assert result.exit_code == 1

    runner.invoke(
        app, ["project", "compute-regressions", "demo", str(csv_path), "--root", str(root)]
    )
    result = runner.invoke(
        app, ["project", "apply-regressions", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "concentration" in result.output

    result = runner.invoke(
        app, ["project", "list-applied-regressions", "demo", "sample001", "--root", str(root)]
    )
    assert "concentration" in result.output

    result = runner.invoke(
        app,
        ["project", "get-applied-regression-results", "demo", "sample001", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert "feature001" in result.output

    result = runner.invoke(
        app,
        ["project", "generate-applied-regression-report", "demo", "sample001", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "sample001_applied_regressions.pdf").exists()
    img_dir = root / "reports" / "img" / "applied_regressions" / "sample001" / "concentration"
    assert (img_dir / "feature001_Mean.png").exists()
    assert (img_dir / "feature001_Mean_table.png").exists()


def test_apply_regressions_zero_slope_no_crash(tmp_path):
    """A symmetric x=[1,2,3]/y=[a,b,a] fit gives an OLS slope of exactly
    0.0 (not skipped as degenerate, since y isn't constant) -- applying it
    divides every raw pixel value by that zero slope, producing an
    all-non-finite predicted array. This used to crash inside
    skew_adjusted_bounds/quantile_skewness (np.percentile on an empty
    finite-values array) before they were fixed to return (0.0, 0.0) on
    empty input -- regression test for that fix."""
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    wavelengths = np.array([200.0, 300.0, 400.0, 500.0])
    for index, (name, height) in enumerate([("S1", 1.0), ("S2", 5.0), ("S3", 1.0)], start=1):
        data = np.full((5, 4), 10.0)
        data[:, 1] = height + np.arange(5) * 0.01
        libs_path = tmp_path / f"{name}.libs"
        write_libs_file(
            libs_path,
            data=data,
            wavelengths=wavelengths,
            params={"HeightPixels": 1, "WidthPixels": 5},
        )
        runner.invoke(
            app,
            [
                "project",
                "add-sample",
                "demo",
                str(libs_path),
                "--sample-name",
                name,
                "--root",
                str(root),
            ],
        )
        sample_id = f"sample{index:03d}"
        runner.invoke(
            app,
            [
                "project",
                "run-pipeline",
                "demo",
                sample_id,
                "--root",
                str(root),
            ],
        )
    csv_path = tmp_path / "props.csv"
    csv_path.write_text("sample_name,concentration\nS1,1.0\nS2,2.0\nS3,3.0\n")

    result = runner.invoke(
        app, ["project", "compute-regressions", "demo", str(csv_path), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    regression = api.get_regression_results("demo", "concentration", project_root=root)[0]
    assert regression.slope == 0.0  # exact zero -- the condition this test guards against

    result = runner.invoke(
        app, ["project", "apply-regressions", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output


def _make_two_feature_project(tmp_path, root):
    """Like `_make_processed_project`, but with two bare features (A300,
    B400) so a --select test can regress each under its own metric."""
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\nB,B400,400.0,,,,1\n",
    )
    for expression in ("A300", "B400"):
        runner.invoke(
            app,
            [
                "project",
                "add-feature",
                "demo",
                expression,
                "--root",
                str(root),
            ],
        )
    for index, (name, height) in enumerate([("S1", 1.0), ("S2", 2.0), ("S3", 3.0)], start=1):
        wavelengths = np.array([200.0, 300.0, 400.0, 500.0])
        data = np.full((5, 4), 10.0)
        data[:, 1] = height + np.arange(5) * 0.01
        data[:, 2] = (height * 2) + np.arange(5) * 0.01
        libs_path = tmp_path / f"{name}.libs"
        write_libs_file(
            libs_path,
            data=data,
            wavelengths=wavelengths,
            params={"HeightPixels": 1, "WidthPixels": 5},
        )
        runner.invoke(
            app,
            [
                "project",
                "add-sample",
                "demo",
                str(libs_path),
                "--sample-name",
                name,
                "--root",
                str(root),
            ],
        )
        sample_id = f"sample{index:03d}"
        runner.invoke(
            app,
            [
                "project",
                "run-pipeline",
                "demo",
                sample_id,
                "--root",
                str(root),
            ],
        )
    csv_path = tmp_path / "props.csv"
    csv_path.write_text("sample_name,concentration\nS1,1.0\nS2,2.0\nS3,3.0\n")
    return csv_path


def test_regression_report_select(tmp_path):
    root = tmp_path / "proj"
    csv_path = _make_two_feature_project(tmp_path, root)

    # two narrow compute-regressions calls, mirroring the pre-all-metrics-
    # sweep workflow: A300 only under Mean, B400 only under Median
    for expression, metric in (("A300", "Mean"), ("B400", "Median")):
        result = runner.invoke(
            app,
            [
                "project",
                "compute-regressions",
                "demo",
                str(csv_path),
                "--root",
                str(root),
                "--select-features",
                expression,
                "--metric",
                metric,
            ],
        )
        assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "project",
            "get-regression-results",
            "demo",
            "concentration",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--select",
            "B400:Median",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "feature001 (Mean)" in result.output
    assert "feature002 (Median)" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "generate-regression-report",
            "demo",
            "concentration",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--select",
            "B400:Median",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "concentration_regressions.pdf").exists()

    # --select and --metric are mutually exclusive
    result = runner.invoke(
        app,
        [
            "project",
            "generate-regression-report",
            "demo",
            "concentration",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--metric",
            "Mean",
        ],
    )
    assert result.exit_code == 1


def test_applied_regressions_select(tmp_path):
    root = tmp_path / "proj"
    csv_path = _make_two_feature_project(tmp_path, root)
    for expression, metric in (("A300", "Mean"), ("B400", "Median")):
        runner.invoke(
            app,
            [
                "project",
                "compute-regressions",
                "demo",
                str(csv_path),
                "--root",
                str(root),
                "--select-features",
                expression,
                "--metric",
                metric,
            ],
        )

    result = runner.invoke(
        app,
        [
            "project",
            "apply-regressions",
            "demo",
            "sample001",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--select",
            "B400:Median",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "project",
            "get-applied-regression-results",
            "demo",
            "sample001",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--select",
            "B400:Median",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "feature001 (Mean)" in result.output
    assert "feature002 (Median)" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "generate-applied-regression-report",
            "demo",
            "sample001",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--select",
            "B400:Median",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports" / "sample001_applied_regressions.pdf").exists()

    # --select and --select-features are mutually exclusive
    result = runner.invoke(
        app,
        [
            "project",
            "apply-regressions",
            "demo",
            "sample001",
            "--root",
            str(root),
            "--select",
            "A300:Mean",
            "--select-features",
            "A300",
        ],
    )
    assert result.exit_code == 1


def test_bootstrap_workflow(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])

    # an unregistered resampling method is rejected up front
    result = runner.invoke(
        app,
        ["project", "add-bootstrap-config", "demo", "bogus", "--npix", "5", "--root", str(root)],
    )
    assert result.exit_code == 1

    # running before a pipeline has ever been run is a hard error
    runner.invoke(
        app,
        ["project", "add-bootstrap-config", "demo", "random", "--npix", "5", "--root", str(root)],
    )
    result = runner.invoke(
        app, ["project", "run-bootstrap", "demo", "sample001", "bootstrap001", "--root", str(root)]
    )
    assert result.exit_code == 1

    root = tmp_path / "proj2"
    _setup_bootstrap_project(tmp_path, root)

    result = runner.invoke(
        app,
        [
            "project",
            "add-bootstrap-config",
            "demo",
            "random",
            "--npix",
            "5",
            "--n-iterations",
            "3",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bootstrap001" in result.output

    result = runner.invoke(app, ["project", "list-bootstrap-configs", "demo", "--root", str(root)])
    assert "bootstrap001" in result.output

    result = runner.invoke(
        app,
        ["project", "extend-bootstrap-config", "demo", "bootstrap001", "5", "--root", str(root)],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "project",
            "run-bootstrap",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
            "--save-csv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "5/5 done" in result.output
    assert (root / "exports" / "sample001_bootstrap001.csv").exists()

    result = runner.invoke(
        app,
        ["project", "bootstrap-progress", "demo", "sample001", "bootstrap001", "--root", str(root)],
    )
    assert "5/5 done" in result.output

    result = runner.invoke(
        app, ["project", "bootstrap-status", "demo", "bootstrap001", "--root", str(root)]
    )
    assert "sample001" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "get-bootstrap-results",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "feature001" in result.output
    # feature002 is the ratio feature "A205 / A205" -- confirms run_bootstrap
    # computed its 2D metrics from ratio_components read out of the store,
    # not by re-streaming/re-evaluating the raw file itself
    assert "feature002" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "export-bootstrap-csv",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output


def test_pack_workflow(tmp_path):
    root = tmp_path / "proj"

    # packing before a pipeline has ever been run is a hard error -- no
    # results.zarr to pack yet
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    result = runner.invoke(app, ["project", "pack", "demo", "--root", str(root)])
    assert result.exit_code == 1

    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["project", "pack", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output
    archive = root / "results.zarr.zip"
    manifest = root / "results.zarr.zip.manifest.json"
    assert archive.exists()
    assert manifest.exists()
    assert str(archive) in result.output

    result = runner.invoke(app, ["project", "verify-pack", str(archive)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output

    result = runner.invoke(app, ["project", "verify-pack", str(tmp_path / "bogus.zip")])
    assert result.exit_code == 1

    result = runner.invoke(app, ["runs", "index", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert str(archive) in result.output


def test_snapshot_workflow(tmp_path):
    root = tmp_path / "proj"
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )
    runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )

    result = runner.invoke(app, ["project", "pack-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output
    # back-to-back, with no sleep: snapshot timestamps carry milliseconds, so
    # two packs in the same second no longer resolve to one filename. This
    # used to need a time.sleep(1) -- without it, the second pack silently
    # truncated the first and this test saw one archive, not two.
    result = runner.invoke(app, ["project", "pack-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output

    snapshot_dir = root / "snapshots"
    archives = sorted(snapshot_dir.glob("*.zarr.zip"))
    assert len(archives) == 2  # two distinct timestamped snapshots, neither overwritten

    result = runner.invoke(app, ["project", "list-snapshots", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output
    for archive in archives:
        assert str(archive) in result.output

    # results.zarr already exists -- refuses without --force
    result = runner.invoke(app, ["project", "use-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 1

    result = runner.invoke(app, ["project", "use-snapshot", "demo", "--root", str(root), "--force"])
    assert result.exit_code == 0, result.output
    assert "replaced existing" in result.output
    assert (root / ".active_snapshot.json").exists()

    # no results.zarr at all -- succeeds without --force
    shutil.rmtree(root / "results.zarr")
    result = runner.invoke(app, ["project", "use-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert (root / "results.zarr").exists()

    # the peak table travels in the manifest alongside registry.json, so a
    # machine that received only the archive can still resolve peak ids
    peak_table_path = root / "peak_table.csv"
    # CRLF on purpose: the manifest stores the table as text, so a text-mode
    # read/write would strip the \r and quietly corrupt what was packed
    peak_table_path.write_bytes(peak_table_path.read_bytes().replace(b"\n", b"\r\n"))
    runner.invoke(app, ["project", "pack-snapshot", "demo", "--root", str(root)])
    original_table = peak_table_path.read_bytes()
    peak_table_path.unlink()
    shutil.rmtree(root / "results.zarr")
    result = runner.invoke(app, ["project", "use-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert peak_table_path.read_bytes() == original_table

    # ...and a locally-diverged table is treated exactly like a diverged
    # registry.json: refused without --force, overwritten with it
    peak_table_path.write_text(_PEAK_CSV_HEADER + "Z,Z999,999.0,,,,1\n")
    shutil.rmtree(root / "results.zarr")
    result = runner.invoke(app, ["project", "use-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 1
    result = runner.invoke(app, ["project", "use-snapshot", "demo", "--root", str(root), "--force"])
    assert result.exit_code == 0, result.output
    assert peak_table_path.read_bytes() == original_table

    result = runner.invoke(
        app, ["project", "use-snapshot", "demo", "--root", str(root), "--timestamp", "bogus"]
    )
    assert result.exit_code == 1


def test_pack_snapshot_disambiguates_a_colliding_timestamp(tmp_path):
    """Two snapshots that resolve to the SAME filename must both survive.

    Millisecond timestamps make an accidental collision rare, but rare is not
    impossible -- two processes can still land on one millisecond. So the
    guarantee is enforced rather than assumed: the archive is created with
    `pack_results(exclusive=True)`, and a name that already exists is retried
    under a `-1`/`-2` suffix instead of being truncated.

    Passing one explicit timestamp twice forces exactly that collision,
    which is the case the millisecond resolution alone cannot cover. Against
    the previous version (mode `"w"`, no existence check) the second pack
    overwrote the first and this found a single archive."""
    root = tmp_path / "proj"
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(tmp_path, root, "A,A300,300.0,,,,1\n")
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    runner.invoke(app, ["project", "add-feature", "demo", "A300", "--root", str(root)])
    runner.invoke(app, ["project", "run-pipeline", "demo", "sample001", "--root", str(root)])

    from pylibs.core.io.snapshot import pack_snapshot_results

    first = pack_snapshot_results(root, "demo", "20260101-000000-000")
    second = pack_snapshot_results(root, "demo", "20260101-000000-000")

    assert first.out_path != second.out_path, "the colliding snapshot overwrote the first"
    assert first.out_path.name == "20260101-000000-000.zarr.zip"
    assert second.out_path.name == "20260101-000000-000-1.zarr.zip"
    assert len(sorted((root / "snapshots").glob("*.zarr.zip"))) == 2

    # both are listed, and both are independently readable -- an overwritten
    # archive would still be *listed* via its leftover manifest, so the
    # round trip through use-snapshot is what actually proves it survived
    result = runner.invoke(app, ["project", "list-snapshots", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert str(first.out_path) in result.output
    assert str(second.out_path) in result.output

    for archive in (first.out_path, second.out_path):
        result = runner.invoke(
            app,
            [
                "project",
                "use-snapshot",
                "demo",
                "--root",
                str(root),
                "--timestamp",
                archive.name.removesuffix(".zarr.zip"),
                "--force",
            ],
        )
        assert result.exit_code == 0, f"{archive.name}: {result.output}"


def _make_packed_project(tmp_path, root):
    """A minimal processed project with one packed snapshot."""
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 20.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    _use_peak_table(tmp_path, root, "A,A300,300.0,,,,1\n")
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])
    runner.invoke(app, ["project", "add-feature", "demo", "A300", "--root", str(root)])
    runner.invoke(app, ["project", "run-pipeline", "demo", "sample001", "--root", str(root)])
    result = runner.invoke(app, ["project", "pack-snapshot", "demo", "--root", str(root)])
    assert result.exit_code == 0, result.output


def test_runs_index_reports_where_an_archive_actually_is(tmp_path):
    """Moving the directory holding an archive must not make the index lie.

    `archive_path` is recorded once, at pack time, and never updated -- and
    since the default project root is relative, it is relative to that run's
    working directory, which is stored nowhere. So it goes stale when the
    project directory is renamed, and (separately) whenever the index is read
    from a different directory than the pack ran in.

    The manifest's own location has neither problem, and the archive sits
    beside it by a convention pack_results enforces. Against the version that
    printed the recorded field, this reported the pre-rename path as if it
    were live."""
    root = tmp_path / "proj"
    _make_packed_project(tmp_path, root)

    # deliberately NOT a name with "proj" as a prefix: the stale-path
    # assertion below is a substring check, and "proj_renamed" contains "proj"
    moved = tmp_path / "elsewhere"
    root.rename(moved)

    result = runner.invoke(app, ["runs", "index", str(moved)])
    assert result.exit_code == 0, result.output
    archive = next((moved / "snapshots").glob("*.zarr.zip"))
    assert str(archive) in result.output, "index did not report the archive's real location"
    assert str(root) not in result.output, "index still reported the stale pre-rename path"
    assert "[MISSING]" not in result.output, "a present archive must not be flagged missing"


def test_runs_index_flags_an_archive_that_is_really_gone(tmp_path):
    """The other half of the same fix: preferring the manifest's own location
    must not paper over an archive that has actually been deleted. Deleting
    the zip while leaving its manifest behind is the case where the derived
    path is right and still points at nothing."""
    root = tmp_path / "proj"
    _make_packed_project(tmp_path, root)

    archive = next((root / "snapshots").glob("*.zarr.zip"))
    archive.unlink()

    result = runner.invoke(app, ["runs", "index", str(root)])
    assert result.exit_code == 0, result.output
    assert "[MISSING]" in result.output, "a deleted archive was reported as if it were present"


def test_use_snapshot_resolves_after_the_project_directory_is_renamed(tmp_path):
    """`use-snapshot` picked the newest snapshot by its recorded
    `archive_path`, so the same staleness broke it functionally, not just
    cosmetically -- and it surfaced as "failed verification", which reads as
    a corrupt archive rather than one that was simply looked for in the wrong
    place."""
    root = tmp_path / "proj"
    _make_packed_project(tmp_path, root)

    # deliberately NOT a name with "proj" as a prefix: the stale-path
    # assertion below is a substring check, and "proj_renamed" contains "proj"
    moved = tmp_path / "elsewhere"
    root.rename(moved)

    result = runner.invoke(
        app, ["project", "use-snapshot", "demo", "--root", str(moved), "--force"]
    )
    assert result.exit_code == 0, result.output
    assert "failed verification" not in result.output
    assert (moved / "results.zarr").exists()


def test_feature_identity_hash_workflow(tmp_path):
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    runner.invoke(app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)])

    peak_table = _use_peak_table(
        tmp_path,
        root,
        "A,A300,300.0,,,,1\n",
    )
    runner.invoke(
        app,
        [
            "project",
            "add-feature",
            "demo",
            "A300",
            "--root",
            str(root),
        ],
    )

    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ran: features, spectrum_stats, distribution_1d" in result.output
    assert "skipped (already computed): distribution_2d" in result.output  # no ratio features

    before = api.get_results("demo", "sample001", project_root=root).get("feature001")

    # rerun unchanged -- nothing needs recomputing at all
    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ran:" not in result.output
    assert "skipped (already computed): features, spectrum_stats, distribution_1d" in result.output

    # same feature_id, same expression -- but the peak table's own definition of
    # A300 moved from 300nm to 400nm, so its identity hash changes
    peak_table.write_text(_PEAK_CSV_HEADER + "A,A300,400.0,,,,1\n")

    # ...except the project holds a *copy*, so editing the source alone changes
    # nothing until it is re-registered. This is the whole copy-semantics
    # contract, and the reason use-peak-table has to be called again.
    result = runner.invoke(
        app, ["project", "run-pipeline", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "ran:" not in result.output

    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(peak_table), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "Replaced the previous peak table" in result.output

    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    # both features.py AND distribution_1d re-ran -- confirms the cross-step
    # ordering fix: distribution_1d's own is_done() must see FeatureStep's
    # freshly-updated hash from this same call, not the pre-rerun state
    assert "ran: features, distribution_1d" in result.output
    assert "skipped (already computed): spectrum_stats, distribution_2d" in result.output

    after = api.get_results("demo", "sample001", project_root=root).get("feature001")
    assert not np.array_equal(before, after)


def test_bootstrap_reset_on_feature_identity_change(tmp_path):
    root = tmp_path / "proj"
    peak_table = _setup_bootstrap_project(tmp_path, root)

    runner.invoke(
        app,
        [
            "project",
            "add-bootstrap-config",
            "demo",
            "random",
            "--npix",
            "5",
            "--n-iterations",
            "3",
            "--root",
            str(root),
        ],
    )
    result = runner.invoke(
        app,
        [
            "project",
            "run-bootstrap",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3/3 done" in result.output
    assert "3 newly computed this run" in result.output

    # rerun unchanged -- everything's already done, nothing newly computed
    result = runner.invoke(
        app,
        [
            "project",
            "run-bootstrap",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3/3 done" in result.output
    assert "0 newly computed this run" in result.output

    # A205's peak table entry moves -- re-register the edited table, reprocess
    # the pipeline (as the normal workflow requires; bootstrap reads its
    # feature hashes off FeatureStep's stored output, never off a table, so it
    # can only notice once that output is refreshed), then rerun bootstrap:
    # the whole run must reset and fully recompute, not silently stay
    # "3/3 done" with stale values
    peak_table.write_text(_PEAK_CSV_HEADER + "A,A205,206.0,,,,1\n")
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(peak_table), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "project",
            "run-pipeline",
            "demo",
            "sample001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "project",
            "run-bootstrap",
            "demo",
            "sample001",
            "bootstrap001",
            "--root",
            str(root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "3/3 done" in result.output
    assert "3 newly computed this run" in result.output  # reset -- all recomputed, not 0


def test_convert_ele_to_libs_command(tmp_path, fake_libs_importer):
    ele_path = tmp_path / "sample.ele"
    ele_path.touch()
    libs_path = tmp_path / "sample.libs"

    result = runner.invoke(app, ["convert-ele-to-libs", str(ele_path), str(libs_path)])

    assert result.exit_code == 0, result.output
    assert libs_path.exists()

    result = runner.invoke(app, ["convert-ele-to-libs", str(libs_path), str(tmp_path / "out.libs")])
    assert result.exit_code == 1  # source isn't a .ELE file


def test_ele_read_failure_is_reported_as_a_pylibs_error(tmp_path, monkeypatch):
    """libs_importer raises its own EleParseError for a corrupt file -- e.g.
    one whose scanned-cell mask disagrees with its scan count. The adapter has
    to translate that, or every CLI command (which only catches PylibsError)
    dumps a traceback instead of a one-line message."""

    class _EleParseError(Exception):
        pass

    def _explode(*args, **kwargs):
        raise _EleParseError("mask.sum() == nr_scan is violated; the file is corrupt")

    module = types.ModuleType("libs_importer")
    module.EleStream = _explode  # type: ignore[attr-defined]
    module.EleParseError = _EleParseError  # type: ignore[attr-defined]
    module.EleVersionError = _EleParseError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libs_importer", module)

    root = tmp_path / "proj"
    ele_path = tmp_path / "broken.ELE"
    ele_path.touch()
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    runner.invoke(app, ["project", "add-sample", "demo", str(ele_path), "--root", str(root)])

    result = runner.invoke(
        app, ["project", "inspect-sample", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 1
    assert "Error: Can't read" in result.output
    assert "the file is corrupt" in result.output  # the backend's own message survives
    assert "Traceback" not in result.output


def test_run_dsl_script_command(tmp_path):
    root = tmp_path / "proj"
    peak_table = tmp_path / "peaks.csv"
    peak_table.write_text(_PEAK_CSV_HEADER + "A,A300,300.0,,,,1\n")
    script = tmp_path / "script.py"
    # USEPEAKTABLE/ADDFEATURE as well as CREATEPROJECT: VERBS is a
    # hand-maintained dict, so a verb missing from it is only caught by
    # actually executing it through the runner
    script.write_text(
        f'CREATEPROJECT("demo", root={str(root)!r})\n'
        f'USEPEAKTABLE("demo", {str(peak_table)!r}, project_root={str(root)!r})\n'
        f'ADDFEATURE("demo", "A300", project_root={str(root)!r})\n'
        'print("done")\n'
    )

    result = runner.invoke(app, ["run", str(script)])

    assert result.exit_code == 0, result.output
    assert "done" in result.output
    assert (root / "registry.json").exists()
    # the verb copied the table in, and the feature resolved against it
    assert (root / "peak_table.csv").read_text() == peak_table.read_text()
    assert "A300" in (root / "registry.json").read_text()


def test_debug_flag_raises_console_and_file_log_level(tmp_path):
    root = tmp_path / "proj"

    runner.invoke(app, ["--debug", "project", "create", "demo", "--root", str(root)])
    logger = logging.getLogger("pylibs")
    assert logger.level == logging.DEBUG
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    console_handlers = [h for h in logger.handlers if h not in file_handlers]
    assert any(h.level == logging.DEBUG for h in console_handlers)
    assert any(h.level == logging.DEBUG for h in file_handlers)

    # applies before dispatch even for a command that never touches a project
    runner.invoke(app, ["--debug", "convert-ele-to-libs", str(tmp_path / "x.libs"), "/dev/null"])
    assert logger.level == logging.DEBUG

    # without the flag, both fall back to their quieter defaults
    runner.invoke(app, ["project", "create", "other", "--root", str(tmp_path / "proj2")])
    assert logger.level == logging.INFO
    console_handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    assert any(h.level == logging.WARNING for h in console_handlers)


def test_all_metrics_sweep_keeps_routine_skips_off_the_console(tmp_path):
    """An all-metrics sweep skips every METRIC_NAMES_2D metric of every
    non-ratio feature -- unfittable by construction, not a data problem, and
    on a real project thousands of lines. It belongs in the project log with
    a one-line console summary, matching how compute_correlations already
    treats its own missing cells."""
    root = tmp_path / "proj"
    csv_path = _make_processed_project(tmp_path, root)

    result = runner.invoke(
        app,
        [
            "project",
            "compute-regressions",
            "demo",
            str(csv_path),
            "--root",
            str(root),
            "--all-metrics",
            "--n-permutations",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output

    # A300 is a bare peak, so no 2D metric exists for it. The console says how
    # many were skipped and where to read them, but never lists them.
    assert "fit(s) -- full list in" in result.output  # pinned: 4 lines start "Skipped"
    assert LOG_FILENAME in result.output
    for metric_name in METRIC_NAMES_2D:
        assert metric_name not in result.output

    # ...while the log keeps every one, classified rather than all reported as
    # "insufficient data".
    log_text = (root / LOG_FILENAME).read_text()
    for metric_name in METRIC_NAMES_2D:
        assert metric_name in log_text
    assert "metric not computed for this feature" in log_text


def test_use_peak_table_workflow(tmp_path):
    """Registering a peak table copies it into the project, and every
    peak-resolving command reads that copy rather than a flag."""
    root = tmp_path / "proj"
    runner.invoke(app, ["project", "create", "demo", "--root", str(root)])
    installed = root / "peak_table.csv"

    # a sample can be registered without any table (add-sample resolves no peak
    # ids), which is also what lets the run-pipeline cases below fail on the
    # table rather than on a missing sample
    libs_path = tmp_path / "sample01.libs"
    write_libs_file(
        libs_path,
        data=np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]),
        wavelengths=np.array([200.0, 300.0, 400.0, 500.0]),
        params={"WidthPixels": 2, "HeightPixels": 1},
    )
    result = runner.invoke(
        app, ["project", "add-sample", "demo", str(libs_path), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output

    # nothing resolves a peak id before a table is registered, and every such
    # command says the same thing. generate-feature-report needs a table too
    # but can't be checked here -- it looks for stored results first, so on an
    # unprocessed sample it reports that instead. It gets its own test below,
    # against a sample that *has* been processed.
    for args in (
        ["project", "add-feature", "demo", "A300"],
        ["project", "add-element-features", "demo", "A"],
        ["project", "run-pipeline", "demo", "sample001"],
        ["project", "run-pipeline-batch", "demo"],
    ):
        result = runner.invoke(app, [*args, "--root", str(root)])
        assert result.exit_code == 1, result.output
        assert "No peak table registered" in result.output

    source = tmp_path / "peaks.csv"
    source.write_text(_PEAK_CSV_HEADER + "A,A300,300.0,,,,1\nB,B400,400.0,,,,1\n")
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(source), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    # the count and elements are echoed so registering the wrong CSV is visible
    assert "2 peaks: A, B" in result.output
    assert "Replaced" not in result.output
    assert installed.read_text() == source.read_text()

    # ...and a feature now resolves
    result = runner.invoke(app, ["project", "add-feature", "demo", "A300", "--root", str(root)])
    assert result.exit_code == 0, result.output

    # re-registering byte-identical content is not a "replacement": nothing
    # about it can change an identity hash, so warning that features will be
    # reprocessed would be a lie
    same_content = tmp_path / "same.csv"
    same_content.write_text(source.read_text())
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(same_content), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "Replaced" not in result.output

    # CRLF survives the copy byte for byte -- real peak tables are CRLF, and
    # text-mode I/O would silently strip the \r
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes((_PEAK_CSV_HEADER + "A,A300,300.0,,,,1\n").replace("\n", "\r\n").encode())
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(crlf), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert installed.read_bytes() == crlf.read_bytes()
    runner.invoke(app, ["project", "use-peak-table", "demo", str(source), "--root", str(root)])

    # editing the source alone does nothing -- the project holds a copy
    source.write_text(_PEAK_CSV_HEADER + "A,A300,301.0,,,,1\n")
    assert "300.0" in installed.read_text()

    # re-registering replaces it, and says so
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(source), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "Replaced the previous peak table" in result.output
    assert "301.0" in installed.read_text()

    # the replacement is reported exactly once on the console, and recorded
    # in the project log. It used to be logged at WARNING too, which printed
    # the same fact twice -- stdout from the CLI, stderr from the logger.
    assert result.output.count("Replaced the previous peak table") == 1
    assert "Replaced the peak table" in (root / "pylibs.log").read_text()

    # registering the project's own copy is a safe no-op, not a truncating
    # self-copy -- a natural thing to type after editing it in place
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(installed), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "301.0" in installed.read_text()

    # a bad source is rejected before anything is copied, so the previously
    # registered table survives every one of these
    good = installed.read_text()
    bad_sources = {
        "missing.csv": None,
        "header_only.csv": _PEAK_CSV_HEADER,
        "no_continuum.csv": "element,id,peak\nA,A300,300.0\n",
        "not_a_number.csv": _PEAK_CSV_HEADER + "A,A300,not_a_number,,,,1\n",
    }
    for name, text in bad_sources.items():
        path = tmp_path / name
        if text is not None:
            path.write_text(text)
        result = runner.invoke(
            app, ["project", "use-peak-table", "demo", str(path), "--root", str(root)]
        )
        assert result.exit_code == 1, f"{name}: {result.output}"
        assert installed.read_text() == good, name

    # and registering against a project that doesn't exist is a project error,
    # not a silently-created directory with a CSV in it
    absent = tmp_path / "absent"
    result = runner.invoke(
        app, ["project", "use-peak-table", "demo", str(source), "--root", str(absent)]
    )
    assert result.exit_code == 1
    assert "No project found" in result.output
    assert not absent.exists()


def test_generate_feature_report_needs_the_projects_peak_table(tmp_path):
    """A processed sample whose peak table has gone missing must fail loudly.

    generate-feature-report resolves bare-peak features against the table to
    decide which get a diagnostic-spectra panel, so it needs one -- but it
    checks for stored results first, which is why the no-table loop in
    test_use_peak_table_workflow can't reach this path. Deleting the table from
    an already-processed project is the case that does: it is also what a user
    hits after transferring results.zarr and registry.json by hand and
    forgetting the third file."""
    root = tmp_path / "proj"
    _make_processed_project(tmp_path, root)

    installed = root / "peak_table.csv"
    assert installed.exists()
    installed.unlink()

    result = runner.invoke(
        app, ["project", "generate-feature-report", "demo", "sample001", "--root", str(root)]
    )

    assert result.exit_code == 1, result.output
    assert "No peak table registered" in result.output
    # and it failed before writing anything, rather than half-rendering
    assert not (root / "reports").exists()

    # re-registering makes it work again -- the failure is about the missing
    # file, not about the sample's stored results
    _use_peak_table(tmp_path, root, "A,A300,300.0,,,,1\n", name="restored.csv")
    result = runner.invoke(
        app, ["project", "generate-feature-report", "demo", "sample001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert (root / "reports").exists()


# --- permutation p-value: corrected estimator --------------------------------
# A permutation p-value is (b + 1) / (N + 1), counting the unpermuted pairing
# itself, so it can never be 0. These exercise the estimator and the display
# rule for its floor directly, since a CLI workflow can only reach them
# through whatever the data happens to produce.


def test_permutation_p_value_is_never_zero_and_counts_ties():
    """Perfectly collinear data: no permutation can beat the real fit, so b
    counts only ties and the result sits exactly on the floor."""
    x = np.arange(12, dtype=float)
    y = 3.0 * x + 1.0
    n_permutations = 199

    fit = fit_least_squares(x, y, n_permutations, np.random.default_rng(0))

    assert fit is not None
    assert fit.p_value > 0.0
    assert fit.p_value == pytest.approx(1.0 / (n_permutations + 1))


def test_permutation_p_value_counts_ties_against_the_fit():
    """Two points always fit exactly, so every permutation's residual MAE is
    0.0 -- identical to the unpermuted one. Counting ties makes every
    permutation evidence against the fit and p reaches 1.0; a strict "<"
    would discard all of them and report the floor instead, which is the
    opposite conclusion.

    Goes at the estimator rather than through `fit_least_squares`, which
    rejects degenerate input before it gets this far."""
    x = np.array([0.0, 1.0])
    y = np.array([3.0, 7.0])
    n_permutations = 99

    p_value = _mae_permutation_p_value(x, y, 0.0, n_permutations, np.random.default_rng(0))

    assert p_value == pytest.approx(1.0)


def test_permutation_p_value_endpoints_follow_the_corrected_formula():
    """Both ends of (b + 1) / (N + 1), pinned by an observed MAE no
    permutation can reach and one every permutation reaches."""
    rng_args = (
        np.arange(8, dtype=float),
        np.arange(8, dtype=float) * 2.0 + 1.0,
    )
    n_permutations = 49

    nothing_beats_it = _mae_permutation_p_value(
        *rng_args, 0.0, n_permutations, np.random.default_rng(0)
    )
    everything_beats_it = _mae_permutation_p_value(
        *rng_args, np.inf, n_permutations, np.random.default_rng(0)
    )

    assert nothing_beats_it == pytest.approx(1.0 / (n_permutations + 1))
    assert everything_beats_it == pytest.approx(1.0)


def test_permutation_p_value_is_a_fraction_for_unrelated_data():
    """Noise unrelated to x lands strictly between the floor and 1, so the
    estimator is not merely clamped at its bounds."""
    rng = np.random.default_rng(7)
    x = np.arange(40, dtype=float)
    y = rng.normal(size=40)
    n_permutations = 999

    fit = fit_least_squares(x, y, n_permutations, np.random.default_rng(1))

    assert fit is not None
    assert 1.0 / (n_permutations + 1) < fit.p_value < 1.0


def test_p_value_at_the_floor_is_shown_as_an_upper_bound():
    """At the floor the value means "smaller than this test resolves", so it
    prints as a bound rather than as an estimate."""
    assert api.format_p_value(1.0 / 10001, 10000) == "p < 1.0e-04"
    assert api.format_p_value(1.0 / 1001, 1000) == "p < 1.0e-03"
    # anything above the floor prints as a value
    assert api.format_p_value(0.0432, 10000) == "p = 4.32e-02"
    # an unknown permutation count cannot locate the floor, so it prints plain
    assert api.format_p_value(1.0 / 10001, 0) == "p = 1.00e-04"
