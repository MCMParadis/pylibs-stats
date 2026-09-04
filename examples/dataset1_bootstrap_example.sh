#!/bin/bash
# The CLI counterpart of dataset1_bootstrap_example.py: the same project, the
# same 21 samples and 11 features, the same four bootstrap configs, driven
# entirely through `pylibs project ...` instead of DSL verbs. Either one alone
# produces the same project -- this exists to show the command surface, and to
# be pasted from when scripting a real run outside Python.
#
#   ./examples/dataset1_bootstrap_example.sh
#
# Can be run from any directory. Re-running is safe and cheap: `create`,
# `add-sample` and `add-feature` reuse what is already registered,
# `run-pipeline` skips any step whose output exists, and `run-bootstrap`
# resumes rather than recomputing completed iterations.
#
# Every tunable below can be overridden from the environment without editing
# the file -- handy for a quick smoke test:
#
#   PROJECT_NAME=scratch N_ITERATIONS=2 NPIX=25 ./examples/dataset1_bootstrap_example.sh
#
# The defaults are exactly the .py's values. NPIX must stay a perfect
# square: the `local` method draws a contiguous square block, so it rejects
# anything else (100 = 10x10, 25 = 5x5).

set -uo pipefail

# Resolve paths relative to the repo root, not the caller's current directory --
# this script lives at examples/<name>.sh, one level below the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PROJECT_NAME="${PROJECT_NAME:-dataset1_bootstrap_project}"
DESCRIPTION="B and Ca in graphite -- bootstrap resampling"
PEAK_TABLE="${PEAK_TABLE:-examples/peak_table.csv}"
NPIX="${NPIX:-100}"
N_ITERATIONS="${N_ITERATIONS:-50}"
# Only these samples' iteration metrics are also written out as a flat CSV
# (matching the .py's csv_sample_stems) -- one file per (sample, config).
CSV_SAMPLE_STEMS=("sample1_1")

FEATURES=(
    "B249"
    "B345"
    "Ca393"
    "Ca396"
    "C229"
    "C247"
    "C251"
    "B249/C229"
    "Ca393/C229"
    "(Ca393)/(Ca393+C229)"
    "(B345)/(B345+C229)"
)

die() {
    echo "Error: $*" >&2
    exit 1
}

command -v pylibs >/dev/null 2>&1 ||
    die "pylibs is not on PATH -- activate the venv, or run: pip install -e ."

# ---------------------------------------------------------------- project ---

echo "== creating project $PROJECT_NAME =="
pylibs project create "$PROJECT_NAME" --description "$DESCRIPTION" ||
    die "could not create project"

# Copied into the project; must precede add-feature.
[ -f "$PEAK_TABLE" ] || die "missing $PEAK_TABLE"
pylibs project use-peak-table "$PROJECT_NAME" "$PEAK_TABLE" ||
    die "could not register the peak table"

echo
echo "== registering samples =="
for x in 1 2 3 4 5 6 7; do
    for y in 1 2 3; do
        path="data/dataset1/sample${x}_${y}.libs"
        [ -f "$path" ] || die "missing raw file $path (run this from a checkout with data/)"
        # --sample-name is the correlations join key, deliberately shared by a
        # sample's three ablation layers; the stem stays per-file and unique.
        pylibs project add-sample "$PROJECT_NAME" "$path" --sample-name "sample${x}" ||
            die "could not add $path"
    done
done

echo
echo "== registering features =="
for expression in "${FEATURES[@]}"; do
    pylibs project add-feature "$PROJECT_NAME" "$expression" ||
        die "could not add feature $expression"
done

# --------------------------------------------------------------- pipeline ---

# `list-samples` prints "  sample001 (name=sample1, stem=sample1_1): <path> [status]",
# so this is the shell's stand-in for the .py's LISTSAMPLES loop: one
# "<sample_id> <sample_stem>" pair per line, read back rather than assumed.
mapfile -t SAMPLES < <(
    pylibs project list-samples "$PROJECT_NAME" |
        sed -n 's/^  \([^ ]*\) (name=[^,]*, stem=\([^)]*\)):.*/\1 \2/p'
)
[ "${#SAMPLES[@]}" -gt 0 ] || die "no samples registered -- nothing to process"

echo
echo "== running the pipeline for ${#SAMPLES[@]} samples =="
for entry in "${SAMPLES[@]}"; do
    read -r sample_id _stem <<<"$entry"
    echo "-- $sample_id"
    pylibs project run-pipeline "$PROJECT_NAME" "$sample_id" ||
        die "pipeline failed for $sample_id"
done

# -------------------------------------------------------------- bootstrap ---

# Four configs, in the .py's own order: method varies slowest, shuffle fastest.
# A shuffled run is the permutation null for its unshuffled twin -- same drawn
# positions, values permuted across them first.
echo
echo "== registering bootstrap configs =="
BOOTSTRAP_IDS=()
for method in random local; do
    for shuffle in false true; do
        args=(--npix "$NPIX" --n-iterations "$N_ITERATIONS")
        [ "$shuffle" = "true" ] && args+=(--shuffle)
        # "Bootstrap config 'bootstrap001' ready in project '...': method=..."
        out=$(pylibs project add-bootstrap-config "$PROJECT_NAME" "$method" "${args[@]}") ||
            die "could not register the $method/shuffle=$shuffle config"
        echo "$out"
        bootstrap_id=$(sed -n "s/^Bootstrap config '\([^']*\)'.*/\1/p" <<<"$out")
        [ -n "$bootstrap_id" ] ||
            die "could not read a bootstrap_id out of: $out"
        BOOTSTRAP_IDS+=("$bootstrap_id")
    done
done

echo
echo "== running ${#BOOTSTRAP_IDS[@]} bootstrap configs over ${#SAMPLES[@]} samples =="
for entry in "${SAMPLES[@]}"; do
    read -r sample_id stem <<<"$entry"
    save_csv=""
    for wanted in "${CSV_SAMPLE_STEMS[@]}"; do
        [ "$stem" = "$wanted" ] && save_csv="--save-csv"
    done
    for bootstrap_id in "${BOOTSTRAP_IDS[@]}"; do
        echo "-- $stem ($sample_id) / $bootstrap_id ${save_csv:+[+csv]}"
        args=("$PROJECT_NAME" "$sample_id" "$bootstrap_id")
        [ -n "$save_csv" ] && args+=("$save_csv")
        pylibs project run-bootstrap "${args[@]}" ||
            die "bootstrap $bootstrap_id failed for $sample_id"
    done
done

echo
echo "== done =="
pylibs project list-bootstrap-configs "$PROJECT_NAME"
for bootstrap_id in "${BOOTSTRAP_IDS[@]}"; do
    pylibs project bootstrap-status "$PROJECT_NAME" "$bootstrap_id"
done
