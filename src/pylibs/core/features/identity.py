"""A feature's true identity isn't just its expression string -- a peak id
like `Fe438` is a lookup key into the peak table, and the peak table itself
can change (a corrected wavelength, a widened tolerance) without the
expression or the feature_id changing at all. `feature_identity_hash`
folds both into one value, so `is_done()` checks across the pipeline
(`FeatureStep`, `Distribution1DStep`/`Distribution2DStep`, bootstrap) can
detect "this feature means something different now" and reprocess it in
place, even though its feature_id and expression look unchanged.
"""

import hashlib
import json

from pylibs.core.features.expression import referenced_peak_ids
from pylibs.core.features.peak_table import PeakTable


def feature_identity_hash(expression: str, peak_table: PeakTable) -> str:
    """A short, deterministic hash of `expression` plus the resolved
    element/peak/continuum/tol_peak/tol_continuum of every peak id it
    references (or `None`, for a token not in `peak_table` -- an unknown
    peak id becoming known, or vice versa, is itself an identity change)."""
    payload = {
        "expression": expression,
        "peaks": {
            token: (
                {
                    "element": entry.element,
                    "peak": entry.peak,
                    "continuum": entry.continuum,
                    "tol_peak": entry.tol_peak,
                    "tol_continuum": entry.tol_continuum,
                }
                if (entry := peak_table.get(token)) is not None
                else None
            )
            for token in referenced_peak_ids(expression)
        },
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.blake2b(canonical.encode(), digest_size=16).hexdigest()
