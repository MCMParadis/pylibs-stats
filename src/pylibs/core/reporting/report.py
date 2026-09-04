"""Feature report generation: one combined diagnostics PNG per registered
feature -- (A) a feature map, (B) diagnostic spectra (for features that are
exactly one bare peak id) or a joint-density plot (for a ratio feature), and
(C) a distribution histogram -- with an optional multi-page PDF bundling
every feature's diagnostics together, one page each.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from pylibs.core.exceptions import ReportError
from pylibs.core.features.peak_table import PeakEntry
from pylibs.core.pipeline.distributions import Distribution1D, Distribution2D
from pylibs.core.reporting.figures import SpectrumPanelData, plot_feature_diagnostics
from pylibs.core.reporting.pdf import DiagnosticsPage, build_pdf_report, save_metrics_table_image
from pylibs.core.reporting.raster import resolve_physical_extent, to_raster


class SpectrumStatsData(NamedTuple):
    wavelengths: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray


class FeatureMapsResult(BaseModel):
    sample_id: str
    image_paths: list[Path]
    table_paths: list[Path] = []
    pdf_path: Path | None = None


def _fmt_number(value: object) -> str:
    """Format a possibly float-imprecise number (e.g. 3.0500000000000003) for
    display, without changing integers or non-numeric values."""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _format_analysis_date(raw: str) -> str:
    """Reformat a raw instrument timestamp (e.g. "5/9/2024 11:56:21 AM") into
    the same "YYYY-MM-DD HH:MM:SS" style as "Report date" -- labeled as
    local time, since it's the instrument's own clock, not UTC. Falls back
    to `raw` unchanged if it doesn't match the expected format (readers
    other than the one this was observed from may format it differently)."""
    try:
        parsed = datetime.strptime(raw, "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        return raw
    return parsed.strftime("%Y-%m-%d %H:%M:%S") + " (local time)"


def _params_table(
    params: dict,
    wavelengths: np.ndarray,
    n_scans: int,
    sample_name: str,
    width_img: float | None,
    height_img: float | None,
) -> dict[str, str]:
    table = {"File name": sample_name}
    if params.get("AnalysisDate"):
        table["Analysis date"] = _format_analysis_date(str(params["AnalysisDate"]))
    if params.get("AnalysisTime"):
        table["Analysis duration"] = str(params["AnalysisTime"])
    table["Report date"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    table["Wavelength range"] = f"{wavelengths.min():.0f} nm to {wavelengths.max():.0f} nm"
    if params.get("Resolution") is not None:
        table["Resolution / mm"] = _fmt_number(params["Resolution"])
    if width_img is not None:
        table["Image width / mm"] = _fmt_number(width_img)
    if height_img is not None:
        table["Image height / mm"] = _fmt_number(height_img)
    table["Pixel count"] = str(n_scans)
    return table


def generate_feature_maps(
    sample_id: str,
    sample_name: str,
    project_name: str,
    params: dict,
    wavelengths: np.ndarray,
    n_scans: int,
    feature_ids: list[str],
    feature_names: dict[str, str],
    get_values: Callable[[str], np.ndarray],
    reports_dir: Path,
    peak_entries: dict[str, PeakEntry | None] | None = None,
    spectrum_stats: SpectrumStatsData | None = None,
    get_distribution_1d: Callable[[str], Distribution1D | None] | None = None,
    get_distribution_2d: Callable[[str], Distribution2D | None] | None = None,
    save_pdf: bool = False,
    mask: np.ndarray | None = None,
) -> FeatureMapsResult:
    """Save one combined diagnostics PNG per feature id to
    `<reports_dir>/img/features/<sample_id>/<feature_id>.png`: (A) a feature
    map, (B) diagnostic spectra or a joint-density plot, and (C) a
    distribution histogram, matching the reference report's layout.
    `get_values(feature_id)` returns that feature's per-scan values (shape
    (n_scans,)).

    Panel (B) shows diagnostic spectra for a feature_id whose `peak_entries`
    entry isn't None (i.e. its expression is exactly one bare peak id) when
    `spectrum_stats` is given, or a joint-density plot when
    `get_distribution_2d(feature_id)` isn't None (i.e. it's a ratio feature,
    precomputed by Distribution2DStep) -- the two are mutually exclusive.
    Panel (C), and a separate metrics-table PNG at
    `<reports_dir>/img/features/<sample_id>/<feature_id>_table.png` (see
    `pdf.save_metrics_table_image`), only appear when
    `get_distribution_1d(feature_id)` isn't None (precomputed by
    Distribution1DStep).

    If `save_pdf` is set, also bundles a cover page (titled "Feature
    diagnostics for '<sample_name>' in <project_name>" -- `sample_name` is
    the display stem, e.g. "sample1_1", matching every per-feature page's
    own caption), a parameters table, and every feature's diagnostics into
    `<reports_dir>/<sample_id>_features.pdf` -- no LaTeX involved, just
    matplotlib's own PDF backend.

    `params`/`wavelengths`/`n_scans` are the raw file's own header metadata
    (see `ResultsStore.get_raw_params`/`get_wavelengths_array`, cached once
    by `SpectrumStatsStep`) -- this function itself never opens the raw
    file. `mask` (likewise cached, see `ResultsStore.get_mask_array`) marks
    which frame cells were actually scanned, for a sample that doesn't fill
    its frame; without one the scan is assumed to fill it."""
    if "HeightPixels" not in params or "WidthPixels" not in params:
        raise ReportError(
            f"Sample {sample_id!r} has no HeightPixels/WidthPixels in its raw file's "
            "params -- feature maps need a rastered sample."
        )
    n_rows = int(params["HeightPixels"])
    n_cols = int(params["WidthPixels"])
    width_img, height_img = resolve_physical_extent(params, n_rows, n_cols)
    peak_entries = peak_entries or {}

    out_dir = reports_dir / "img" / "features" / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    table_paths = []
    pdf_pages: list[DiagnosticsPage] = []
    for feature_id in feature_ids:
        values = get_values(feature_id)
        title = feature_names.get(feature_id, feature_id)
        grid = to_raster(values, n_rows, n_cols, mask=mask)

        spectrum: SpectrumPanelData | None = None
        peak_entry = peak_entries.get(feature_id)
        if peak_entry is not None and spectrum_stats is not None:
            spectrum = (
                spectrum_stats.wavelengths,
                spectrum_stats.minimum,
                spectrum_stats.maximum,
                spectrum_stats.mean,
                peak_entry.peak,
            )

        dist_stats = get_distribution_1d(feature_id) if get_distribution_1d else None
        joint_dist = get_distribution_2d(feature_id) if get_distribution_2d else None

        fig = plot_feature_diagnostics(
            grid,
            title,
            width_img=width_img,
            height_img=height_img,
            spectrum=spectrum,
            joint_dist=joint_dist,
            dist_stats=dist_stats,
            bottom=0.52,  # matches pdf._diagnostics_page's own panel geometry exactly --
            # figsize defaults to A4_SIZE too, so the standalone PNG comes out
            # the same shape/size as the PDF page's own panels
        )
        path = out_dir / f"{feature_id}.png"
        fig.savefig(path, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        image_paths.append(path)

        if dist_stats is not None:
            table_path = out_dir / f"{feature_id}_table.png"
            save_metrics_table_image(table_path, dist_stats, joint_dist, title)
            table_paths.append(table_path)

        if save_pdf:
            pdf_pages.append(
                DiagnosticsPage(
                    title, grid, width_img, height_img, spectrum, joint_dist, dist_stats
                )
            )

    pdf_path = None
    if save_pdf:
        pdf_path = reports_dir / f"{sample_id}_features.pdf"
        table = _params_table(params, wavelengths, n_scans, sample_name, width_img, height_img)
        title = f"Feature diagnostics for {sample_name!r} in {project_name}"
        build_pdf_report(pdf_path, title, sample_name, table, pdf_pages)

    return FeatureMapsResult(
        sample_id=sample_id, image_paths=image_paths, table_paths=table_paths, pdf_path=pdf_path
    )
