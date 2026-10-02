"""SpectrumStatsStep: computes the min, max, and mean spectrum across every
scan in a sample, streaming batches through its reader, and writes the
result to the results store as a (3, n_pix) array. This backs the
diagnostic-spectra panel in the PDF report.

Also caches the raw file's `wavelengths` array, its per-detector channel
masks (if present, as a proper `detectors` array -- see `ResultsStore.
create_detectors_array`), and its scalar `params` (see `set_raw_params`) --
all already-loaded attributes on the same `reader` this step has open
regardless, so this costs nothing extra, and it's what lets
`get_spectrum_stats`/`generate_feature_report`/
`generate_applied_regression_report`/bootstrap's "local" resampling method
read this header metadata back from the store instead of ever reopening the
raw file themselves.

Not every raw file's `params` necessarily has a `Detectors` key, or spells
its wavelength-calibration key `slambda` (that key isn't itself used for
anything -- the actual wavelength axis always comes from `reader.
wavelengths` independently) -- both are logged as a warning, not an error,
since the pipeline doesn't depend on either.
"""

import numpy as np

from pylibs.core.exceptions import ReportError
from pylibs.core.io.reader import LibsReader
from pylibs.core.io.store import ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.project.registry import Registry
from pylibs.core.reporting.raster import frame_shape

logger = get_logger(__name__)


class SpectrumStatsStep:
    name = "spectrum_stats"
    produces = "spectrum_stats"
    requires: str | None = None
    needs_reader = True

    def is_done(self, store: ResultsStore, sample_id: str, registry: Registry) -> bool:
        return store.has_spectrum_stats(sample_id)

    def run(
        self, reader: LibsReader | None, sample_id: str, store: ResultsStore, registry: Registry
    ) -> None:
        assert reader is not None, "SpectrumStatsStep.needs_reader is True"
        n_pix = len(reader.wavelengths)
        running_min = np.full(n_pix, np.inf)
        running_max = np.full(n_pix, -np.inf)
        running_sum = np.zeros(n_pix)
        n_scans = 0

        for batch in reader.iter_window():
            running_min = np.minimum(running_min, batch.min(axis=0))
            running_max = np.maximum(running_max, batch.max(axis=0))
            running_sum += batch.sum(axis=0)
            n_scans += batch.shape[0]

        mean = running_sum / n_scans if n_scans else running_sum
        store.create_spectrum_stats_array(sample_id, running_min, running_max, mean)
        store.create_wavelengths_array(sample_id, reader.wavelengths)

        # the frame is settled once, here, against this sample's own scan
        # count, and the resolved pair is what gets stored -- every later
        # consumer (feature and applied-regression maps, `local` bootstrap
        # resampling) then reads one trustworthy value instead of re-deriving
        # it from a header that may not be trustworthy
        mask = self._usable_mask(reader, sample_id, n_scans)
        store.set_raw_params(
            sample_id, self._resolved_params(reader.params, sample_id, n_scans, mask)
        )
        if mask is not None:
            store.create_mask_array(sample_id, mask)
            logger.debug(
                "Sample %r: cached a %dx%d scanned-cell mask (%d of %d cells scanned)",
                sample_id,
                mask.shape[0],
                mask.shape[1],
                int(mask.sum()),
                mask.size,
            )

        if "Detectors" in reader.params:
            store.create_detectors_array(sample_id, reader.params["Detectors"])
        else:
            logger.warning(
                "Sample %r's raw params have no 'Detectors' key -- per-detector channel data "
                "won't be available",
                sample_id,
            )
        if "slambda" not in reader.params:
            logger.warning(
                "Sample %r's raw params have no 'slambda' key -- the raw file's own "
                "wavelength-calibration metadata wasn't found under that name (this doesn't "
                "affect the wavelength axis itself, read from the file's own wavelengths data "
                "independently)",
                sample_id,
            )

    @staticmethod
    def _usable_mask(reader: LibsReader, sample_id: str, n_scans: int) -> np.ndarray | None:
        """The raw file's scanned-cell mask, if it has one that can place
        this sample's scans.

        A partially-scanned sample (e.g. a round one in a square frame)
        records fewer scans than its frame has cells, and
        `PixelAssignmentMatrix` is what says which cells those were --
        without it `raster.to_raster` can only reshape, which such a sample
        can't satisfy. Absent for a full rectangular raster, which is the
        norm for `.libs`: nothing stored, no complaint.

        The one test applied is exactly one set cell per scan. A mask failing
        that can't say where each scan landed, so it is logged and dropped.
        Its *shape* is deliberately not checked against the header: the mask
        carries one set cell per scan, which is independently verifiable,
        while the header is the thing known to be unreliable (see
        `raster.frame_shape`). So the mask defines the frame when present,
        rather than being discarded for disagreeing with it.
        """
        raw_mask = reader.params.get("PixelAssignmentMatrix")
        if raw_mask is None:
            return None
        mask = np.asarray(raw_mask).astype(bool)
        if mask.ndim != 2:
            logger.warning(
                "Sample %r's PixelAssignmentMatrix is %dD, not a 2D frame -- ignoring it",
                sample_id,
                mask.ndim,
            )
            return None

        n_set = int(mask.sum())
        if n_set != n_scans:
            logger.warning(
                "Sample %r's PixelAssignmentMatrix marks %d scanned cell(s) but the file has %d "
                "scan(s) -- ignoring it, since it can't say where each scan landed",
                sample_id,
                n_set,
                n_scans,
            )
            return None
        return mask

    @staticmethod
    def _resolved_params(
        params: dict, sample_id: str, n_scans: int, mask: np.ndarray | None
    ) -> dict:
        """`params` with `HeightPixels`/`WidthPixels` set to the frame this
        sample's scans actually fill.

        Some instruments write those two as the number of intervals between
        pixels rather than the number of pixels, so a 400x400 scan arrives as
        399x399 (see `raster.frame_shape`). Settling it here, once, is what
        keeps every consumer honest: a report would raise on the mismatch,
        but `local` bootstrap resampling would quietly draw blocks from a
        wrongly shaped raster, and nothing downstream could tell.

        A mask, when present, is the authority -- it holds one set cell per
        scan. Otherwise the header is checked against the scan count and
        reinterpreted if needed. A header that cannot be reconciled either
        way is left exactly as it was: this step must not fail a sample whose
        other results are perfectly usable, and the consumers that need a
        frame raise their own errors with their own context.

        The original pair is preserved as `HeightPixelsHeader`/
        `WidthPixelsHeader` whenever it is overridden, so the correction is
        auditable rather than silent.
        """
        resolved = dict(params)
        if mask is not None:
            n_rows, n_cols = int(mask.shape[0]), int(mask.shape[1])
        else:
            try:
                n_rows, n_cols = frame_shape(params, n_scans, sample_id)
            except ReportError as exc:
                logger.warning(
                    "Sample %r: could not settle the raster frame (%s) -- storing the raw "
                    "header unchanged; anything needing a frame will report it itself",
                    sample_id,
                    exc,
                )
                return resolved

        header = (params.get("HeightPixels"), params.get("WidthPixels"))
        if header != (n_rows, n_cols):
            logger.info(
                "Sample %r: storing the raster frame as %dx%d (its raw header said %sx%s)",
                sample_id,
                n_rows,
                n_cols,
                *header,
            )
            if header != (None, None):
                resolved["HeightPixelsHeader"], resolved["WidthPixelsHeader"] = header
        resolved["HeightPixels"], resolved["WidthPixels"] = n_rows, n_cols
        return resolved
