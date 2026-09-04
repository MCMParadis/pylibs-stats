#!/bin/bash
# Fits every (CSV column, metric, feature) regression cell -- an all-metrics
# sweep against dataset1_concentrations.csv, needed to give
# COMPUTECORRELATIONS a fully populated matrix to read back later -- split
# across 3 genuinely separate OS processes, using `pylibs project
# compute-regressions`'s --worker-index/--n-workers sharding (see
# `pipeline.regression_run.run_regression_sweep`'s docstring -- the same
# mechanism as run-bootstrap's/run-pipeline-batch's own sharding, applied to
# regression cells instead). Requires run_pipeline_parallel.sh (or its SLURM
# equivalent) to have completed first -- every sample needs
# feature+metrics results before any regression can be fit:
#
#   pylibs run examples/dataset1_multiprocess/setup_project.py
#   ./examples/dataset1_multiprocess/run_pipeline_parallel.sh
#   ./examples/dataset1_multiprocess/compute_regressions_parallel.sh
#
# Can be run from any directory. Re-running always refits every cell (a
# regression fit isn't a resumable "pending" operation the way a bootstrap
# iteration or a sample's own pipeline is -- see compute_regressions'
# own docstring), so only run this once per (CSV, seed, permutation count).
#
# See compute_regressions_slurm.sbatch in this same folder for the real
# SLURM-array-job version of this exact mechanism.

set -uo pipefail

# Resolve paths relative to the repo root, not the caller's current
# directory -- this script lives at
# examples/<name>/compute_regressions_parallel.sh, two levels below the
# repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT_NAME="dataset1_multiprocess_project"
ROOT="projects/$PROJECT_NAME"
CSV_PATH="data/dataset1/dataset1_concentrations.csv"
N_WORKERS=3

if [ ! -f "$ROOT/registry.json" ]; then
    echo "Error: no project at $ROOT -- run the setup + pipeline scripts first:" >&2
    echo "  pylibs run examples/dataset1_multiprocess/setup_project.py" >&2
    echo "  ./examples/dataset1_multiprocess/run_pipeline_parallel.sh" >&2
    exit 1
fi

PID_DIR="$ROOT/pids"
LOG_DIR="$ROOT/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

run_worker() {
    local worker_index=$1
    local log_file="$LOG_DIR/regressions_worker_${worker_index}.log"
    echo "[worker $worker_index] starting, PID=$BASHPID, $(date +%T.%3N)"
    if pylibs project compute-regressions "$PROJECT_NAME" "$CSV_PATH" \
        --root "$ROOT" --all-metrics \
        --worker-index "$worker_index" --n-workers "$N_WORKERS" \
        > "$log_file" 2>&1; then
        echo "[worker $worker_index] finished OK, PID=$BASHPID, $(date +%T.%3N)"
    else
        echo "[worker $worker_index] FAILED, PID=$BASHPID -- see $log_file:"
        sed 's/^/    /' "$log_file"
        return 1
    fi
}

any_failed=0
for ((i = 0; i < N_WORKERS; i++)); do
    run_worker "$i" &
    worker_pid=$!
    echo "$worker_pid" > "$PID_DIR/regressions_worker_${i}.pid"
    echo "Launched worker $i as PID $worker_pid (see $LOG_DIR/regressions_worker_${i}.log)"
done

# Waits for every backgrounded worker above -- this is what actually lets
# them run concurrently: nothing blocks between the launches above.
for job in $(jobs -p); do
    wait "$job" || any_failed=1
done

echo
if [ "$any_failed" -ne 0 ]; then
    echo "One or more workers failed -- see above." >&2
    exit 1
fi
pylibs project list-regressions "$PROJECT_NAME" --root "$ROOT"
