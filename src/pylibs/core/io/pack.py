"""Pack a project's `results.zarr` directory tree into a single `.zarr.zip`
archive -- still randomly readable via `zarr.storage.ZipStore`, no unpack
step needed -- plus a `.manifest.json` provenance sidecar (git SHA/dirty
flag, the project's own `registry.json` snapshot, hostname, sizes,
checksum). Fixes the file-count/inode blowup from `core.runs.checkpoint`'s
deliberate chunk-size-1 bootstrap arrays (one file per iteration, by
design -- see that module) without touching the write path at all: this
only ever runs against an already-written store, after the fact.

Non-destructive: `pack_results` never deletes `source_root` -- delete it
yourself once `verify_pack` confirms the archive is good.
"""

import hashlib
import json
import socket
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import zarr
from pydantic import BaseModel, Field

from pylibs.core.exceptions import PackError
from pylibs.core.features.peak_table import PEAK_TABLE_FILENAME
from pylibs.core.logging import get_logger
from pylibs.core.project.registry import REGISTRY_FILENAME

logger = get_logger(__name__)

# A manifest is always written beside its archive, as `<archive>.manifest.json`
# (see pack_results). That convention is what lets a manifest found on disk say
# where its archive really is, regardless of what `archive_path` recorded.
MANIFEST_SUFFIX = ".manifest.json"


class PackManifest(BaseModel):
    project_name: str
    archive_path: Path
    packed_at_utc: datetime
    git_sha: str | None
    git_dirty: bool | None
    hostname: str
    source_size_bytes: int
    n_files_packed: int
    archive_size_bytes: int
    archive_sha256: str
    registry: dict
    # The project's registered peak table, as raw CSV text -- so a snapshot
    # restored on another machine can still resolve peak ids. Text, not a
    # parsed dict: identity hashes are computed from these values, so a
    # byte-for-byte round trip avoids any risk of silent reformatting.
    # The default is load-bearing: manifests written before this field
    # existed are still model_validate'd by list_manifests/use_snapshot.
    peak_table: str | None = None

    # Where this manifest was actually found, set by `list_manifests`.
    # Runtime-only (`exclude=True`), so it never reaches the JSON on disk and
    # the manifest schema is unchanged -- this records where the file *is*,
    # which is not a property of the pack and must not be written into one.
    found_at: Path | None = Field(default=None, exclude=True)

    @property
    def resolved_archive_path(self) -> Path:
        """Where this archive actually is, preferring the manifest's own
        location over the `archive_path` recorded at pack time.

        `archive_path` is whatever the caller passed to `pack_results`, and
        the default project root (`api.DEFAULT_PROJECTS_DIR`) is relative --
        so it is usually relative to the *current working directory at pack
        time*, which is recorded nowhere. Reading it back from a different
        directory yields a path that does not resolve even though nothing
        moved; renaming the directory holding the archive breaks it too.

        The manifest's own path has neither problem: `list_manifests` found
        the file, so it exists, and the archive sits beside it under a
        convention this module enforces. Falls back to the stored field when
        `found_at` is unset (a manifest built directly rather than listed) or
        does not follow that convention."""
        if self.found_at is not None and self.found_at.name.endswith(MANIFEST_SUFFIX):
            return self.found_at.with_name(self.found_at.name.removesuffix(MANIFEST_SUFFIX))
        return self.archive_path


class PackResult(BaseModel):
    source_root: Path
    out_path: Path
    manifest_path: Path
    n_files_packed: int
    size_bytes: int
    source_size_bytes: int
    sha256: str


class VerifyResult(BaseModel):
    path: Path
    ok: bool
    n_members: int
    n_samples: int
    bad_member: str | None
    error: str | None


def _git_provenance(cwd: Path) -> tuple[str | None, bool | None]:
    """`(git_sha, git_dirty)` for the git repo containing `cwd`, or `(None,
    None)` if `cwd` isn't inside a git repo (or `git` isn't installed) --
    logged, never raised, since provenance is best-effort."""
    try:
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True
        )
    except FileNotFoundError:
        logger.warning("git isn't installed -- packing without git provenance")
        return None, None
    if sha_result.returncode != 0:
        logger.warning("%s isn't inside a git repo -- packing without git provenance", cwd)
        return None, None
    return sha_result.stdout.strip(), bool(status_result.stdout.strip())


def pack_results(
    source_root: Path, out_path: Path, project_name: str, exclusive: bool = False
) -> PackResult:
    """Pack every file under `source_root` (a `results.zarr` directory) into
    `out_path` (a `.zarr.zip` archive, `zipfile.ZIP_STORED` -- zarr's own
    chunk codec already compresses the data, so re-compressing here would
    only burn CPU) plus a `<out_path stem>.manifest.json` sidecar. Raises
    `PackError` if `source_root` doesn't exist.

    `exclusive=False` (the default) overwrites `out_path` if it already
    exists -- what `pack_project` wants, since its archive is the project's
    single canonical one and re-packing is how you refresh it.
    `exclusive=True` opens the archive with mode `"x"` instead, so an
    existing `out_path` raises `FileExistsError` rather than being
    truncated. `core.io.snapshot.pack_snapshot_results` uses that to make a
    timestamp collision fail loudly and be retried under a disambiguated
    name, instead of silently destroying the snapshot already there."""
    source_root = Path(source_root)
    out_path = Path(out_path)
    if not source_root.is_dir():
        raise PackError(f"No results.zarr to pack at {source_root}")

    files = sorted(f for f in source_root.rglob("*") if f.is_file())
    source_size_bytes = sum(f.stat().st_size for f in files)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "x" if exclusive else "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, arcname=str(f.relative_to(source_root)))

    archive_size_bytes = out_path.stat().st_size
    archive_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    registry_path = source_root.parent / REGISTRY_FILENAME
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    peak_table_path = source_root.parent / PEAK_TABLE_FILENAME
    # decode from bytes rather than read_text(): text mode applies universal
    # newline translation, which would silently strip the \r from a CRLF table
    # (peak tables in the wild are CRLF) and make the "byte-for-byte" round
    # trip a lie
    peak_table = peak_table_path.read_bytes().decode("utf-8") if peak_table_path.exists() else None
    git_sha, git_dirty = _git_provenance(Path.cwd())

    manifest = PackManifest(
        project_name=project_name,
        archive_path=out_path,
        packed_at_utc=datetime.now(UTC),
        git_sha=git_sha,
        git_dirty=git_dirty,
        hostname=socket.gethostname(),
        source_size_bytes=source_size_bytes,
        n_files_packed=len(files),
        archive_size_bytes=archive_size_bytes,
        archive_sha256=archive_sha256,
        registry=registry,
        peak_table=peak_table,
    )
    manifest_path = out_path.with_suffix(out_path.suffix + MANIFEST_SUFFIX)
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    logger.info(
        "Packed %d files (%d bytes) from %s into %s (%d bytes)",
        len(files),
        source_size_bytes,
        source_root,
        out_path,
        archive_size_bytes,
    )
    return PackResult(
        source_root=source_root,
        out_path=out_path,
        manifest_path=manifest_path,
        n_files_packed=len(files),
        size_bytes=archive_size_bytes,
        source_size_bytes=source_size_bytes,
        sha256=archive_sha256,
    )


def verify_pack(path: Path) -> VerifyResult:
    """Sanity-check a `.zarr.zip` produced by `pack_results`: validate every
    member's checksum (`zipfile.ZipFile.testzip`), then confirm it opens as
    a structurally intact zarr store and spot-read each top-level (sample)
    group's own arrays. Never raises -- any failure (missing file, not a
    zip, a bad member, a corrupted zarr tree) lands in `bad_member`/`error`,
    with `ok=False`."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            n_members = len(zf.namelist())
            bad_member = zf.testzip()

        if bad_member is not None:
            return VerifyResult(
                path=path,
                ok=False,
                n_members=n_members,
                n_samples=0,
                bad_member=bad_member,
                error=None,
            )

        store = zarr.storage.ZipStore(str(path), mode="r")
        group = zarr.open_group(store=store, mode="r")
        sample_keys = list(group.keys())
        for key in sample_keys:
            sample_group = group[key]
            if isinstance(sample_group, zarr.Group):
                for _, array in sample_group.arrays():
                    array[...]  # force a real chunk decompress, not just a metadata read
    except Exception as exc:  # noqa: BLE001 -- any failure means the archive is bad, not a bug
        return VerifyResult(
            path=path, ok=False, n_members=0, n_samples=0, bad_member=None, error=str(exc)
        )

    return VerifyResult(
        path=path,
        ok=True,
        n_members=n_members,
        n_samples=len(sample_keys),
        bad_member=None,
        error=None,
    )


def list_manifests(root: Path) -> list[PackManifest]:
    """Every `*.manifest.json` (written by `pack_results`) found anywhere
    under `root`, parsed and sorted by `packed_at_utc`.

    Each manifest carries `found_at`, its own real location, so callers can
    reach `resolved_archive_path` instead of trusting the `archive_path`
    recorded at pack time -- see that property for why the stored field goes
    stale without anything moving."""
    root = Path(root)
    manifests = []
    for path in sorted(root.rglob(f"*{MANIFEST_SUFFIX}")):
        manifest = PackManifest.model_validate_json(path.read_text())
        manifest.found_at = path
        manifests.append(manifest)
    return sorted(manifests, key=lambda m: m.packed_at_utc)
