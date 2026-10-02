class PylibsError(Exception):
    """Base class for all pylibs errors."""


class ProjectError(PylibsError):
    """Raised for invalid project operations (e.g. creating over an existing project)."""


class ReaderError(PylibsError):
    """Raised when a sample's raw data file can't be read."""


class PeakTableError(PylibsError):
    """Raised when a peak table can't be loaded."""


class FilenameSchemeError(PylibsError):
    """Raised when a sample-filename scheme template is malformed (unbalanced
    braces, a repeated field, or no `{sample_name}` field) -- not for a
    filename that simply doesn't match a valid scheme, which logs instead."""


class ExpressionError(PylibsError):
    """Raised when a feature expression is malformed (bad syntax, unsupported
    operator or function) -- not for unknown peak ids, which log instead."""


class ResultsError(PylibsError):
    """Raised when a sample has no pipeline results yet (run_pipeline first)."""


class DistributionError(PylibsError):
    """Raised for an invalid distribution option (e.g. an unknown histogram
    windowing strategy)."""


class ReportError(PylibsError):
    """Raised when a report can't be generated (e.g. missing raster metadata)."""


class CorrelationError(PylibsError):
    """Raised for invalid cross-sample correlation input (a malformed sample-
    properties CSV, an ambiguous sample_name join) or when reading back
    correlations that haven't been computed yet."""


class BootstrapError(PylibsError):
    """Raised for invalid bootstrap configuration (unknown method, both or
    neither of npix/pct given, a non-perfect-square npix for method='local',
    an out-of-range worker_index/n_workers) or an attempt to extend a run to
    a target no greater than its current one."""


class RegressionError(PylibsError):
    """Raised for invalid regression input (unknown method/algorithm/metric,
    a select_features expression that matches no registered feature, no
    usable samples at all) or when reading back regressions that haven't
    been computed yet."""


class PackError(PylibsError):
    """Raised when packing/verifying a results.zarr archive fails (no
    results.zarr to pack, or a corrupted/incomplete archive on verify)."""


class SnapshotError(PylibsError):
    """Raised when using a packed snapshot fails: no snapshot found (by
    timestamp, or at all), a corrupt archive, or refusing to overwrite an
    existing results.zarr without force=True."""
