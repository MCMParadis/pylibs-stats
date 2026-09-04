#!/bin/bash
# The CLI counterpart of dataset1_example.py: the same project, the same 21
# samples and 16 features, the same all-metrics sweep, correlations,
# calibration curves and applied quantification maps -- driven entirely through
# `pylibs project ...` instead of DSL verbs. Either one alone produces the same
# project; this exists to show the command surface, and to be pasted from when
# scripting a real run outside Python.
#
#   ./examples/dataset1_example.sh
#
# Can be run from any directory. Re-running is safe: `create`,
# `add-samples-from-dir` and `add-feature` reuse what is already registered,
# and `run-pipeline` skips any step whose output already exists. The
# regression/correlation/report steps do recompute, being pure functions of
# what is already stored.
#
# Every tunable below can be overridden from the environment without editing
# the file. The permutation test dominates the runtime, so a smoke run wants:
#
#   PROJECT_NAME=scratch N_PERMUTATIONS=100 N_PROCESSES=8 ./examples/dataset1_example.sh
#
# The defaults are exactly the .py's values.

set -uo pipefail

# Resolve paths relative to the repo root, not the caller's current directory --
# this script lives at examples/<name>.sh, one level below the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PROJECT_NAME="${PROJECT_NAME:-dataset1_project}"
DESCRIPTION="B and Ca in graphite"
DATA_DIR="data/dataset1"
PEAK_TABLE="${PEAK_TABLE:-examples/peak_table.csv}"
CSV_PATH="$DATA_DIR/dataset1_concentrations.csv"
N_PERMUTATIONS="${N_PERMUTATIONS:-10000}"
N_PROCESSES="${N_PROCESSES:-1}"

# Only these ablation layers get the (slower) per-sample diagnostics PDF and
# applied-regression PDF -- one layer per physical sample, to keep the example
# quick. Every sample is still processed and still has regressions applied.
REPORT_SAMPLE_STEMS=("sample1_1" "sample4_1" "sample7_1")

# Bare peaks, plus ratio features (including two normalized-ratio composites)
# and log-of-ratio/exp-of-ratio composites. The outermost op of a log()/exp()
# composite is not division, so neither gets a diagnostic-spectra or
# joint-density panel -- just (A) map and (C) distribution. Together these
# exercise every feature/report code path.
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
    "log((Ca393)/(Ca393+C229))"
    "exp((Ca393)/(Ca393+C229))"
    "log((B345)/(B345+C229))"
    "log(B249/C229)"
    "log(Ca393/C229)"
)

# A curated, mixed-metric set of (feature, metric) pairs picked out of the
# all-metrics sweep below -- '<expression>:<metric>', the CLI spelling of the
# .py's (expression, metric) tuples. Ca393/B345 on Mean, the ratio feature and
# the exp-of-ratio composite on Median, the normalized-ratio composite and its
# log on Mode. Fully decoupled from what the sweep stored.
CALIBRATION_CURVES=(
    "Ca393:Mean"
    "B345:Mean"
    "Ca393/C229:Median"
    "exp((Ca393)/(Ca393+C229)):Median"
    "(Ca393)/(Ca393+C229):Mode"
    "log((Ca393)/(Ca393+C229)):Mode"
)

die() {
    echo "Error: $*" >&2
    exit 1
}

# --select is repeatable, one flag per pair; built once and reused below.
SELECT_ARGS=()
for pair in "${CALIBRATION_CURVES[@]}"; do
    SELECT_ARGS+=(--select "$pair")
done

wants_report() {
    local stem=$1 wanted
    for wanted in "${REPORT_SAMPLE_STEMS[@]}"; do
        [ "$stem" = "$wanted" ] && return 0
    done
    return 1
}

command -v pylibs >/dev/null 2>&1 ||
    die "pylibs is not on PATH -- activate the venv, or run: pip install -e ."
[ -f "$CSV_PATH" ] || die "missing $CSV_PATH (run this from a checkout with data/)"
[ -f "$PEAK_TABLE" ] || die "missing $PEAK_TABLE"

# ---------------------------------------------------------------- project ---

echo "== creating project $PROJECT_NAME =="
pylibs project create "$PROJECT_NAME" --description "$DESCRIPTION" ||
    die "could not create project"

# Copied into the project, so everything after this resolves peak ids
# against the project's own copy -- no flag is threaded anywhere else.
# Must precede add-feature.
pylibs project use-peak-table "$PROJECT_NAME" "$PEAK_TABLE" ||
    die "could not register the peak table"

# Each sampleX_Y.libs is one ablation layer of physical sample X. The default
# "{sample_name}_{layer}" scheme reads that off the filename, so all 21 files
# share the sample_name that groups them for correlations/regressions later
# ("sample1"), keep their own stem ("sample1_2") for report titles, and carry
# the layer in name_parts. The concentrations CSV isn't a sample file, so the
# scan ignores it.
echo
echo "== registering samples from $DATA_DIR =="
pylibs project add-samples-from-dir "$PROJECT_NAME" "$DATA_DIR" ||
    die "could not register samples"

echo
echo "== registering features =="
for expression in "${FEATURES[@]}"; do
    pylibs project add-feature "$PROJECT_NAME" "$expression" ||
        die "could not add feature $expression"
done

# --------------------------------------------------- pipeline and reports ---

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
    read -r sample_id stem <<<"$entry"
    echo "-- $stem ($sample_id)"
    pylibs project run-pipeline "$PROJECT_NAME" "$sample_id" ||
        die "pipeline failed for $sample_id"
    if wants_report "$stem"; then
        pylibs project generate-feature-report "$PROJECT_NAME" "$sample_id" --save-pdf ||
            die "feature report failed for $sample_id"
    fi
done

# ------------------------------------------------------------ regressions ---

# compute-correlations never fits anything itself -- it only reshapes
# already-stored regressions into (feature x metric) matrices, so the
# all-metrics sweep must run first. It also naturally covers the Mean/Median/
# Mode calibration curves used below, so those need no separate per-metric run.
echo
echo "== all-metrics regression sweep (the slow step) =="
pylibs project compute-regressions "$PROJECT_NAME" "$CSV_PATH" \
    --all-metrics --n-permutations "$N_PERMUTATIONS" --n-processes "$N_PROCESSES" ||
    die "regression sweep failed"

# ----------------------------------------------------------- correlations ---

echo
echo "== correlations against the known B/Ca concentrations =="
pylibs project compute-correlations "$PROJECT_NAME" "$CSV_PATH" ||
    die "could not compute correlations"

# Column names can contain spaces ("c_B / wt%"), so read whole lines, never $1.
mapfile -t CORRELATION_COLUMNS < <(
    pylibs project list-correlations "$PROJECT_NAME" | sed -n 's/^  //p'
)
for column_name in "${CORRELATION_COLUMNS[@]}"; do
    echo "-- $column_name"
    pylibs project generate-correlations-report "$PROJECT_NAME" "$column_name" ||
        die "correlations report failed for $column_name"
done

# ------------------------------------------------------ calibration curves ---

echo
echo "== calibration-curve reports (${#CALIBRATION_CURVES[@]} curated pairs per column) =="
mapfile -t REGRESSION_COLUMNS < <(
    pylibs project list-regressions "$PROJECT_NAME" | sed -n 's/^  //p'
)
for column_name in "${REGRESSION_COLUMNS[@]}"; do
    echo "-- $column_name"
    pylibs project generate-regression-report "$PROJECT_NAME" "$column_name" \
        "${SELECT_ARGS[@]}" ||
        die "regression report failed for $column_name"
done

# ------------------------------------------------- applied quantification ---

# Apply just those same curated pairs to every sample (turning each feature's
# raw intensity map into a predicted-concentration map), but render the PDF
# only for the same handful of samples as above -- applying is cheap, the
# per-sample report PDF is the slower part.
echo
echo "== applying regressions to every sample =="
for entry in "${SAMPLES[@]}"; do
    read -r sample_id stem <<<"$entry"
    echo "-- $stem ($sample_id)"
    pylibs project apply-regressions "$PROJECT_NAME" "$sample_id" "${SELECT_ARGS[@]}" ||
        die "apply-regressions failed for $sample_id"
    if wants_report "$stem"; then
        pylibs project generate-applied-regression-report "$PROJECT_NAME" "$sample_id" ||
            die "applied-regression report failed for $sample_id"
    fi
done

echo
echo "== done -- reports are under projects/$PROJECT_NAME/reports =="
