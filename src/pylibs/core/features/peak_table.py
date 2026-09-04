"""Peak table: element/id/peak/continuum lookup used by feature expressions.

There is no bundled table -- a project registers its own, once, via
`api.use_peak_table`/`USEPEAKTABLE`, which *copies* the CSV to
`<project_root>/peak_table.csv`. The copy is what every later operation
reads (`project_peak_table_path`/`load_project_peak_table`), so a project is
self-contained: it survives `pack_snapshot`/`use_snapshot` onto another
machine, and can't silently change meaning because the source file moved or
was edited. Presence of that file *is* the registration -- nothing records a
path in `registry.json`, which would be a second source of truth able to
disagree with the first.

`peak` and `continuum` are wavelengths (nm). An empty `continuum` means no
continuum subtraction for that peak. Ids follow the convention
`<element><floored peak wavelength>[-<n>]`, e.g. `Fe266`, `Fe266-1`,
`Fe266-2` for multiple transitions at the same floored wavelength.

`tol_peak`/`tol_continuum` are optional per-peak windows (nm) used when
extracting intensities: empty or 0 means "use the single nearest pixel",
a positive value means "use the max (peak) or min (continuum) over every
pixel within that tolerance".

`_isdefault` flags an element's one canonical peak, used only to narrow
`peaks_for_element` (and so `api.add_element_features`) to a single line per
element. It's optional in both senses: a table may omit the column entirely,
and a table may leave some elements with no flagged peak at all -- either way
every entry is simply non-default, which callers have to treat as "nothing to
add", not as an unknown element.

Loading is strict, because `install_peak_table` is the gate a user-supplied
file passes through: a malformed row or an empty table raises `PeakTableError`
rather than a bare `KeyError`/`ValueError`, so interfaces catching
`PylibsError` report it properly instead of showing a traceback -- and so a
header-only file can't register cleanly and then make every feature id
"unknown" (which `validate_expression` only logs).
"""

import csv
from pathlib import Path

from pydantic import BaseModel

from pylibs.core.atomic import atomic_write_bytes
from pylibs.core.exceptions import PeakTableError

PEAK_TABLE_FILENAME = "peak_table.csv"


class PeakEntry(BaseModel):
    element: str
    id: str
    peak: float
    continuum: float | None
    tol_peak: float = 0.0
    tol_continuum: float = 0.0
    is_default: bool = False


PeakTable = dict[str, PeakEntry]


class PeakTableResult(BaseModel):
    """What `install_peak_table` did -- reported by the CLI/DSL so registering
    the wrong CSV is visible immediately (a `(3 peaks: X)` line where you
    expected 254 is the tell)."""

    project_root: Path
    path: Path
    source: Path
    n_peaks: int
    elements: list[str]
    replaced_existing: bool


def _entry_from_row(row: dict[str, str]) -> PeakEntry:
    continuum = row["continuum"].strip()
    tol_peak = row.get("tol_peak", "").strip()
    tol_continuum = row.get("tol_continuum", "").strip()
    return PeakEntry(
        element=row["element"],
        id=row["id"],
        peak=float(row["peak"]),
        continuum=float(continuum) if continuum else None,
        tol_peak=float(tol_peak) if tol_peak else 0.0,
        tol_continuum=float(tol_continuum) if tol_continuum else 0.0,
        is_default=row.get("_isdefault", "").strip() == "1",
    )


def load_peak_table(path: Path) -> PeakTable:
    """Load a peak table CSV into a dict keyed by peak id.

    Raises `PeakTableError` if the file is missing, if any row is malformed
    (a missing required column, or a wavelength that isn't a number), or if
    the table holds no rows at all."""
    path = Path(path)
    if not path.exists():
        raise PeakTableError(f"No such peak table: {path.resolve()}")

    table: PeakTable = {}
    with open(path, newline="") as f:
        # start=2: row 1 is the header, so this is the reader's own line number
        for line_number, row in enumerate(csv.DictReader(f), start=2):
            try:
                entry = _entry_from_row(row)
            except (KeyError, TypeError) as exc:
                raise PeakTableError(
                    f"Malformed peak table {path.resolve()}: row {line_number} is missing "
                    f"column {exc}. Required columns: element, id, peak, continuum."
                ) from exc
            except ValueError as exc:
                raise PeakTableError(
                    f"Malformed peak table {path.resolve()}: row {line_number} has a "
                    f"non-numeric wavelength ({exc})."
                ) from exc
            table[entry.id] = entry

    if not table:
        raise PeakTableError(
            f"Peak table {path.resolve()} has no rows -- a header-only table would leave "
            f"every peak id unknown."
        )
    return table


def project_peak_table_path(project_root: Path) -> Path:
    """Where a project keeps its registered peak table."""
    return Path(project_root) / PEAK_TABLE_FILENAME


def load_project_peak_table(project_root: Path) -> PeakTable:
    """The project's own registered peak table. The single place "no peak
    table registered" is raised, so the message can't drift between the api,
    the pipeline runner and batch workers."""
    path = project_peak_table_path(project_root)
    if not path.exists():
        raise PeakTableError(
            f"No peak table registered for the project at {project_root} "
            f"({PEAK_TABLE_FILENAME} missing) -- register one first: "
            f"`pylibs project use-peak-table <project> <path/to/peak_table.csv>`, "
            f"or USEPEAKTABLE(<project>, <path/to/peak_table.csv>) in a DSL script."
        )
    return load_peak_table(path)


def install_peak_table(project_root: Path, source: Path) -> PeakTableResult:
    """Copy `source` to `<project_root>/peak_table.csv`, replacing any table
    already registered.

    The source is loaded and validated *first*, so a malformed file never
    lands in the project. The write goes through `core.atomic`, so a sharded
    `run_pipeline_batch` reading the destination concurrently sees the whole
    old file or the whole new one, never a half-written CSV.

    Re-registering the project's own copy (natural after editing it in place)
    is a no-op rather than `shutil.SameFileError` or a truncating write that
    would destroy the only copy."""
    source = Path(source)
    table = load_peak_table(source)
    destination = project_peak_table_path(project_root)

    payload = source.read_bytes()
    same_file = destination.exists() and source.resolve() == destination.resolve()
    # `replaced_existing` drives a warning about features being reprocessed, so
    # it has to mean "the definitions actually changed", not merely "a file was
    # already there": re-registering byte-identical content changes no identity
    # hash and reprocesses nothing.
    replaced_existing = (
        destination.exists() and not same_file and destination.read_bytes() != payload
    )

    if not same_file:
        atomic_write_bytes(destination, payload)

    return PeakTableResult(
        project_root=Path(project_root),
        path=destination,
        source=source,
        n_peaks=len(table),
        elements=elements_in_table(table),
        replaced_existing=replaced_existing,
    )


def peaks_for_element(
    table: PeakTable, element: str, only_default: bool = False
) -> list[PeakEntry]:
    """Every peak in `table` belonging to `element` (matched case-insensitively,
    but exactly -- 'ca' finds 'Ca' and never 'CaF'), in the table's own row
    order rather than sorted, so a curated ordering survives.

    `only_default` narrows that to the element's `_isdefault`-flagged peak.
    An empty result means either an unknown element or no flagged peak; the
    two are different mistakes, so telling them apart is the caller's job
    (see `api.add_element_features`)."""
    wanted = element.strip().casefold()
    return [
        entry
        for entry in table.values()
        if entry.element.casefold() == wanted and (entry.is_default or not only_default)
    ]


def elements_in_table(table: PeakTable) -> list[str]:
    """Every distinct element name in `table`, sorted -- for error messages
    naming what the table does know."""
    return sorted({entry.element for entry in table.values()})
