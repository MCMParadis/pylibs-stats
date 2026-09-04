"""Step contract: a Step knows what it produces in the results store and how
to compute it, streaming a sample's reader if it needs to. The runner skips
a step whose output already matches what's currently registered.

A sample's *raw data* still can't be reprocessed in place -- a new project
(a new results store) is required if the `.libs`/`.ELE` file itself
changes. But a feature/peak *definition* change (see `features.identity.
feature_identity_hash`) is now detected by `is_done()` -- which is why it
takes `registry`, not just `store` -- and reprocessed in place, at every
affected layer (`FeatureStep`, `Distribution1DStep`/`Distribution2DStep`,
and `pipeline.bootstrap.runner`, which resets a stale run's checkpoint
rather than silently leaving wrong values marked done).
"""

from typing import Protocol

from pylibs.core.io.reader import LibsReader
from pylibs.core.io.store import ResultsStore
from pylibs.core.project.registry import Registry


class Step(Protocol):
    name: str
    produces: str
    requires: str | None
    """Name of another Step whose output this one depends on, or None. The
    runner doesn't topologically sort -- it relies on `core.pipeline.runner`
    listing steps in an order that already satisfies every `requires`."""

    needs_reader: bool
    """Whether `run()` actually streams the raw reader (True for
    `FeatureStep`/`SpectrumStatsStep`) or only reads back already-stored
    results (False for `Distribution1DStep`/`Distribution2DStep`) -- lets
    the runner avoid opening the raw file at all when only a step that
    doesn't need it is stale (e.g. reprocessing a sample on a machine
    without access to the original raw-data mount)."""

    def is_done(self, store: ResultsStore, sample_id: str, registry: Registry) -> bool:
        """Whether this step's output already exists for `sample_id` AND
        matches what's currently registered (feature set, and each
        feature's identity hash where relevant)."""
        ...

    def run(
        self, reader: LibsReader | None, sample_id: str, store: ResultsStore, registry: Registry
    ) -> None:
        """Write this step's output to `store`, streaming `reader` if
        `needs_reader` is True (guaranteed non-None then; steps with
        `needs_reader = False` ignore it)."""
        ...
