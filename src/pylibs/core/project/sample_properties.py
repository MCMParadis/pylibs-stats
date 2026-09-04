"""Sample-properties CSV loading: an external table of per-sample scalar
properties (e.g. concentration, physical properties) used to correlate
against a project's registered features (see `pipeline.correlations`).
Loaded with the stdlib `csv` module -- no pandas dependency, matching
`features.peak_table`'s existing CSV-loading convention.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pylibs.core.exceptions import CorrelationError
from pylibs.core.logging import get_logger

logger = get_logger(__name__)

SAMPLE_NAME_COLUMN = "sample_name"


@dataclass
class SampleProperties:
    sample_names: list[str]
    column_names: list[str]  # numeric columns only, CSV header order
    skipped_columns: list[str]  # non-numeric, dropped with a logged warning
    values: np.ndarray  # shape (len(sample_names), len(column_names))


def _parse_column(rows: list[dict[str, str]], column: str) -> np.ndarray | None:
    """Parse `column` across `rows` as floats (blank cells become nan), or
    None if any non-blank cell fails to parse -- the whole column is dropped
    in that case, not just the offending cell."""
    parsed = []
    for row in rows:
        raw = row[column].strip()
        if not raw:
            parsed.append(math.nan)
            continue
        try:
            parsed.append(float(raw))
        except ValueError:
            return None
    return np.array(parsed, dtype=np.float64)


def load_sample_properties(csv_path: Path) -> SampleProperties:
    """Load `csv_path`: a table of external per-sample scalar properties, one
    row per sample, with a `sample_name` column joined against each
    registered sample's `SampleConfig.sample_name`. Every other column is
    parsed as float (blank cells become NaN); a column with any non-blank,
    non-numeric cell is dropped entirely (logged as a warning).

    Raises `CorrelationError` if the file doesn't exist, has no
    `sample_name` header, has two rows sharing the same sample_name
    (ambiguous join), or ends up with zero numeric columns."""
    path = Path(csv_path)
    if not path.exists():
        raise CorrelationError(f"No such sample-properties file: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if SAMPLE_NAME_COLUMN not in fieldnames:
            raise CorrelationError(
                f"{path} has no {SAMPLE_NAME_COLUMN!r} column -- "
                "every row must be joined to a sample by name."
            )
        rows = list(reader)

    sample_names = [row[SAMPLE_NAME_COLUMN] for row in rows]
    duplicates = sorted({name for name in sample_names if sample_names.count(name) > 1})
    if duplicates:
        raise CorrelationError(
            f"{path} has more than one row for sample_name(s) {duplicates!r} -- "
            "can't join unambiguously."
        )

    candidate_columns = [name for name in fieldnames if name != SAMPLE_NAME_COLUMN]
    column_names: list[str] = []
    skipped_columns: list[str] = []
    parsed_columns: list[np.ndarray] = []
    for column in candidate_columns:
        parsed = _parse_column(rows, column)
        if parsed is None:
            # info, not warning: a properties CSV carrying text columns is
            # ordinary, and callers already surface the skipped set themselves
            # (compute-regressions echoes result.skipped_columns), so warning
            # here reported the same fact a second time, on stderr
            logger.info("Column %r in %s isn't numeric for every row -- skipping it.", column, path)
            skipped_columns.append(column)
        else:
            column_names.append(column)
            parsed_columns.append(parsed)

    if not column_names:
        raise CorrelationError(f"{path} has no numeric columns to correlate.")

    return SampleProperties(
        sample_names=sample_names,
        column_names=column_names,
        skipped_columns=skipped_columns,
        values=np.column_stack(parsed_columns),
    )
