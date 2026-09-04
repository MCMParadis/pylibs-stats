"""Named, timestamped project snapshots -- the practical counterpart to
`core.io.pack`'s single-canonical `pack_project`, for a workflow where a
project is processed on one machine and worked on (reported, extended) on
another: `pack_snapshot_results` never overwrites a prior snapshot -- a name
collision is disambiguated with a `-1`/`-2` suffix, not resolved by
truncating what is already there -- and
`use_snapshot_results` extracts one back into `<project_root>/results.zarr`
-- and restores `<project_root>/registry.json` and `<project_root>/
peak_table.csv` from the manifest's own embedded copies, since every
`core.api` function (`Project.load`) needs the first to exist before it'll
touch `results.zarr` at all, and anything resolving a peak id needs the
second -- to resume working from it on a machine that only ever received the
single archive file.

Snapshots live under `<project_root>/snapshots/<UTC timestamp>.zarr.zip`
plus their `.manifest.json` sidecar (see `core.io.pack.pack_results`).
"""

import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from pylibs.core.atomic import atomic_write_bytes, atomic_write_text
from pylibs.core.exceptions import SnapshotError
from pylibs.core.features.peak_table import PEAK_TABLE_FILENAME
from pylibs.core.io.pack import PackManifest, PackResult, list_manifests, pack_results, verify_pack
from pylibs.core.io.store import RESULTS_DIRNAME
from pylibs.core.logging import get_logger
from pylibs.core.project.registry import REGISTRY_FILENAME

logger = get_logger(__name__)

SNAPSHOTS_DIRNAME = "snapshots"
# millisecond resolution (%f is microseconds; new_timestamp trims the last 3
# digits). Seconds alone were not enough: two snapshots packed in the same
# second resolved to one filename, and the second silently destroyed the
# first. Milliseconds keep the name readable while making an accidental
# collision rare -- but "rare" is not "impossible", which is what
# _MAX_COLLISION_RETRIES and pack_results(exclusive=True) below are for.
_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S-%f"
_ACTIVE_SNAPSHOT_FILENAME = ".active_snapshot.json"
_MAX_COLLISION_RETRIES = 100


class SnapshotUseResult(BaseModel):
    project_root: Path
    snapshot_path: Path
    manifest: PackManifest
    replaced_existing: bool


def pack_snapshot_results(project_root: Path, project_name: str, timestamp: str) -> PackResult:
    """Pack `<project_root>/results.zarr` into
    `<project_root>/snapshots/<timestamp>.zarr.zip` -- a new archive every
    call, never *overwriting* a prior snapshot (unlike `pack_project`'s
    single-canonical `results.zarr.zip`).

    The guarantee is enforced, not merely assumed from the timestamp being
    unique. The archive is created with `pack_results(exclusive=True)`, so a
    name that already exists raises `FileExistsError` instead of truncating
    what is there; this retries under `<timestamp>-1`, `-2`, ... until it
    finds a free name. Two snapshots packed in the same millisecond -- by one
    caller in a tight loop, or by two processes at once -- therefore both
    survive, under distinct names, rather than one silently replacing the
    other."""
    project_root = Path(project_root)
    snapshots_dir = project_root / SNAPSHOTS_DIRNAME
    for attempt in range(_MAX_COLLISION_RETRIES):
        name = timestamp if attempt == 0 else f"{timestamp}-{attempt}"
        out_path = snapshots_dir / f"{name}.zarr.zip"
        try:
            return pack_results(
                project_root / RESULTS_DIRNAME, out_path, project_name, exclusive=True
            )
        except FileExistsError:
            # another snapshot already claimed this name (same millisecond,
            # or a racing sibling process) -- try the next suffix
            continue
    raise SnapshotError(
        f"Could not find a free snapshot name for timestamp {timestamp!r} under "
        f"{snapshots_dir} after {_MAX_COLLISION_RETRIES} attempts."
    )


def _resolve_snapshot_path(project_root: Path, timestamp: str | None) -> Path:
    snapshots_dir = project_root / SNAPSHOTS_DIRNAME
    if timestamp is not None:
        path = snapshots_dir / f"{timestamp}.zarr.zip"
        if not path.exists():
            raise SnapshotError(f"No snapshot {timestamp!r} found at {path}")
        return path

    manifests = list_manifests(snapshots_dir)
    if not manifests:
        raise SnapshotError(f"No snapshots found under {snapshots_dir}")
    # resolved, not the recorded `archive_path`: that is relative to the
    # working directory of the run that packed it, so resolving it from
    # anywhere else pointed at nothing and surfaced as "failed verification"
    # -- reading as corruption, when the archive was intact all along
    return manifests[-1].resolved_archive_path


def use_snapshot_results(
    project_root: Path, timestamp: str | None, force: bool = False
) -> SnapshotUseResult:
    """Extract a snapshot (the most recent one, if `timestamp` is None) into
    `<project_root>/results.zarr`, and restore `<project_root>/registry.json`
    and `<project_root>/peak_table.csv` from the manifest's own embedded
    copies if they aren't already there (or aren't already identical to
    them) -- the peak table travels so that reporting/reprocessing on the
    receiving machine can still resolve peak ids. Refuses (`SnapshotError`)
    if `results.zarr` already exists, or either restored file exists with
    different content, unless `force=True` -- extracting a zip on top of an existing
    zarr store's directory tree would merge old and new files at the chunk
    level rather than cleanly replace it, and there's no way to tell whether
    either existing artifact reflects local work that hasn't been saved
    anywhere else. On `force=True`, the existing `results.zarr` is deleted
    first (this repo's first directory-tree deletion -- gated behind an
    explicit flag for exactly that reason) and `registry.json`/
    `peak_table.csv` are overwritten."""
    project_root = Path(project_root)
    snapshot_path = _resolve_snapshot_path(project_root, timestamp)

    verification = verify_pack(snapshot_path)
    if not verification.ok:
        reason = verification.bad_member or verification.error or "unknown error"
        raise SnapshotError(f"Snapshot {snapshot_path} failed verification: {reason}")

    manifest_path = snapshot_path.with_suffix(snapshot_path.suffix + ".manifest.json")
    manifest = PackManifest.model_validate_json(manifest_path.read_text())

    results_root = project_root / RESULTS_DIRNAME
    registry_path = project_root / REGISTRY_FILENAME
    registry_conflict = registry_path.exists() and (
        json.loads(registry_path.read_text()) != manifest.registry
    )
    peak_table_path = project_root / PEAK_TABLE_FILENAME
    peak_table_conflict = (
        manifest.peak_table is not None
        and peak_table_path.exists()
        and peak_table_path.read_bytes().decode("utf-8") != manifest.peak_table
    )
    replaced_existing = results_root.exists()
    if (replaced_existing or registry_conflict or peak_table_conflict) and not force:
        raise SnapshotError(
            f"{results_root} already exists, and/or {registry_path} or {peak_table_path} "
            "exists with different content -- pack your current state first "
            "(pack_snapshot/PACKSNAPSHOT), or pass force=True to discard it."
        )
    if results_root.exists():
        shutil.rmtree(results_root)

    with zipfile.ZipFile(snapshot_path, "r") as zf:
        zf.extractall(results_root)
    if not registry_path.exists() or registry_conflict:
        atomic_write_text(registry_path, json.dumps(manifest.registry, indent=2))
    if manifest.peak_table is not None:
        if not peak_table_path.exists() or peak_table_conflict:
            # bytes, not write_text: preserve CRLF exactly as packed
            atomic_write_bytes(peak_table_path, manifest.peak_table.encode("utf-8"))
    elif not peak_table_path.exists():
        # a snapshot packed before peak tables travelled -- say so now, rather
        # than letting the next report fail with "no peak table registered"
        # and no clue why
        logger.warning(
            "Snapshot %s carries no peak table and none is registered at %s -- register one "
            "with `pylibs project use-peak-table` / USEPEAKTABLE before reporting or "
            "reprocessing.",
            snapshot_path,
            peak_table_path,
        )

    active_path = project_root / _ACTIVE_SNAPSHOT_FILENAME
    active_path.write_text(manifest.model_dump_json(indent=2))

    logger.info(
        "Extracted snapshot %s (packed %s) into %s%s",
        snapshot_path,
        manifest.packed_at_utc,
        results_root,
        " (replaced existing)" if replaced_existing else "",
    )
    return SnapshotUseResult(
        project_root=project_root,
        snapshot_path=snapshot_path,
        manifest=manifest,
        replaced_existing=replaced_existing,
    )


def new_timestamp() -> str:
    """UTC timestamp in this module's snapshot-filename format, to
    millisecond resolution -- exposed so callers (the `core.api` facade)
    don't need to know the format string, just that repeated calls produce
    sortable names that are *almost always* distinct.

    Almost, not always: two calls within the same millisecond return the
    same string, and nothing here can prevent that for two separate
    processes. Uniqueness of the resulting *file* is enforced downstream by
    `pack_snapshot_results`, which creates the archive exclusively and
    retries under a suffix on collision -- do not rely on this function
    alone for it."""
    # %f is microseconds; keep 3 digits so the filename stays readable
    return datetime.now(UTC).strftime(_TIMESTAMP_FORMAT)[:-3]
