#!/bin/bash
# Packing a project's results with the CLI, end to end.
#
# `results.zarr` is a directory tree holding one small file per chunk. That is
# fine on a local disk and awful everywhere else: a real project reaches 100k+
# files, which wrecks inode quotas and makes any copy or transfer crawl. Both
# commands here fix that by consolidating the tree into a single .zarr.zip that
# is *still randomly readable* through zarr's own ZipStore -- there is no
# unpack-before-use step.
#
#   ./examples/pack_project_example.sh
#
# Needs a project that has already been processed. Defaults to the one
# examples/dataset1_example.sh builds:
#
#   PROJECT_NAME=my_project ./examples/pack_project_example.sh
#
# Nothing here destroys anything. The one command that can -- use-snapshot,
# which replaces a working results.zarr -- is only described, unless you opt in
# explicitly with DEMO_USE_SNAPSHOT=1 (see the last section).

set -uo pipefail

# Resolve paths relative to the repo root, not the caller's current directory --
# this script lives at examples/<name>.sh, one level below the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PROJECT_NAME="${PROJECT_NAME:-dataset1_project}"
ROOT="projects/$PROJECT_NAME"
DEMO_USE_SNAPSHOT="${DEMO_USE_SNAPSHOT:-0}"

die() {
    echo "Error: $*" >&2
    exit 1
}

command -v pylibs >/dev/null 2>&1 ||
    die "pylibs is not on PATH -- activate the venv, or run: pip install -e ."
[ -f "$ROOT/registry.json" ] ||
    die "no project at $ROOT -- build one first: ./examples/dataset1_example.sh"
[ -d "$ROOT/results.zarr" ] ||
    die "$ROOT has no results.zarr -- run the pipeline first"

# ------------------------------------------------- 1. the one-off handoff ---

# `pack` writes <project_root>/results.zarr.zip plus a .manifest.json sidecar
# recording the git commit (and whether the tree was dirty), the project's own
# registry.json, member sizes and a checksum -- so the archive says what
# produced it. The original directory is left untouched.
echo "== 1. pack: one archive for a one-off handoff =="
before_files=$(find "$ROOT/results.zarr" -type f | wc -l)
pylibs project pack "$PROJECT_NAME" || die "pack failed"

archive="$ROOT/results.zarr.zip"
[ -f "$archive" ] || die "expected an archive at $archive"
echo
echo "   results.zarr: $before_files files, $(du -sh "$ROOT/results.zarr" | cut -f1)"
echo "   archive:      1 file, $(du -sh "$archive" | cut -f1)"

# Checks every member's checksum against the manifest and confirms the archive
# still opens as a structurally intact zarr store. Worth running on the
# receiving end, not just after writing it.
echo
echo "== 2. verify-pack: is the archive intact? =="
pylibs project verify-pack "$archive" || die "verify-pack failed"

# ------------------------------------------- 2. the iterative alternative ---

# `pack` keeps one canonical archive and overwrites it. For a workflow that
# moves between machines -- process here, report there, reprocess back -- use
# snapshots instead: same format, but written to
# <project_root>/snapshots/<UTC timestamp>.zarr.zip and never overwritten, so
# each is a restorable point in time.
echo
echo "== 3. pack-snapshot: a timestamped, never-overwritten point in time =="
pylibs project pack-snapshot "$PROJECT_NAME" || die "pack-snapshot failed"

echo
echo "== 4. list-snapshots: every snapshot, oldest first =="
pylibs project list-snapshots "$PROJECT_NAME" || die "list-snapshots failed"

# Manifests are what make an archive self-describing, and `runs index` finds
# every one under a directory tree -- handy for answering "what do I actually
# have on this disk, and which commit produced it?"
echo
echo "== 5. runs index: every packed archive found under projects/ =="
pylibs runs index projects || die "runs index failed"

# ---------------------------------------------------- 3. the way back in ---

# On the other machine, `use-snapshot` extracts a snapshot into
# <project_root>/results.zarr so reports and further processing carry on from
# exactly that state. --timestamp picks a specific one; the default is the most
# recent.
#
# It REPLACES the working results.zarr, and refuses to run when one already
# exists unless you pass --force -- pack your current state first if you want
# to keep it. That is why this script only prints the command by default.
echo
echo "== 6. use-snapshot: restore a snapshot as the working results.zarr =="
if [ "$DEMO_USE_SNAPSHOT" = "1" ]; then
    echo "   DEMO_USE_SNAPSHOT=1 -- running it for real (results.zarr will be replaced)"
    pylibs project use-snapshot "$PROJECT_NAME" --force || die "use-snapshot failed"
else
    echo "   Not run: it would replace $ROOT/results.zarr. The command is:"
    echo
    echo "     pylibs project use-snapshot $PROJECT_NAME              # newest snapshot"
    echo "     pylibs project use-snapshot $PROJECT_NAME --timestamp <UTC timestamp>"
    echo "     pylibs project use-snapshot $PROJECT_NAME --force      # over an existing results.zarr"
    echo
    echo "   Re-run this script with DEMO_USE_SNAPSHOT=1 to actually execute it."
fi

echo
echo "== done =="
echo "   archive:   $archive"
echo "   manifest:  $archive.manifest.json"
echo "   snapshots: $ROOT/snapshots/"
