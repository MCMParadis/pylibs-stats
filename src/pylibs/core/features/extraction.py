"""Peak intensity extraction: given a batch of spectra and a peak table
entry, compute the peak's intensity per spectrum -- the max value within
`tol_peak` nm of `peak`, minus the min value within `tol_continuum` nm of
`continuum` (floored at 0) if a continuum is set. A tolerance of 0 (or
unset) means "use the single nearest pixel" instead of a window.
"""

import numpy as np

from pylibs.core.features.peak_table import PeakEntry
from pylibs.core.logging import get_logger

logger = get_logger(__name__)


def _window_or_nearest(wavelengths: np.ndarray, target: float, tol: float) -> np.ndarray:
    if tol > 0:
        (indices,) = np.where(np.abs(wavelengths - target) <= tol)
        if indices.size:
            return indices
    return np.array([int(np.argmin(np.abs(wavelengths - target)))])


def extract_peak(data: np.ndarray, wavelengths: np.ndarray, entry: PeakEntry) -> np.ndarray:
    """Extract `entry`'s intensity for each spectrum in `data` (shape (n, n_pix)).

    Returns an array of shape (n,). If `entry.peak` or `entry.continuum` falls
    outside `wavelengths`' range, logs an error and returns NaN for every
    spectrum in the batch instead of raising."""
    wl_min, wl_max = float(wavelengths.min()), float(wavelengths.max())

    if not wl_min <= entry.peak <= wl_max:
        logger.error(
            "Peak %r at %.3f nm is out of the sample's wavelength range (%.3f-%.3f nm)",
            entry.id,
            entry.peak,
            wl_min,
            wl_max,
        )
        return np.full(data.shape[0], np.nan)

    peak_idx = _window_or_nearest(wavelengths, entry.peak, entry.tol_peak)
    values = data[:, peak_idx].max(axis=1)

    if entry.continuum is not None:
        if not wl_min <= entry.continuum <= wl_max:
            logger.error(
                "Continuum for peak %r at %.3f nm is out of the sample's "
                "wavelength range (%.3f-%.3f nm)",
                entry.id,
                entry.continuum,
                wl_min,
                wl_max,
            )
            return np.full(data.shape[0], np.nan)

        continuum_idx = _window_or_nearest(wavelengths, entry.continuum, entry.tol_continuum)
        baseline = data[:, continuum_idx].min(axis=1)
        values = np.maximum(values - baseline, 0)

    return values
