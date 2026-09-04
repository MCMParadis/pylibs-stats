#!/bin/bash
# Runs feature+metrics computation for every registered sample, split across
# 3 genuinely separate OS processes (not just threads), using `pylibs
# project run-pipeline-batch`'s --worker-index/--n-workers sharding (see
# `pipeline.runner.run_pipeline_batch`'s docstring -- this is the
# across-sample counterpart to run-bootstrap's own sharding, applied to
# FeatureStep/SpectrumStatsStep/Distribution1DStep/Distribution2DStep
# instead of bootstrap iterations). Run setup_project.py once first:
#
#   pylibs run examples/dataset1_multiprocess/setup_project.py
#   ./examples/dataset1_multiprocess/run_pipeline_parallel.sh
#
# Can be run from any directory. Re-running is safe: any sample already
# fully processed by a previous run is skipped, not recomputed (each Step's
# own is_done() check, same as plain run-pipeline).
#
# See run_pipeline_slurm.sbatch in this same folder for the real
# SLURM-array-job version of this exact mechanism.

set -uo pipefail

# Resolve paths relative to the repo root, not the caller's current
# directory -- this script lives at
# examples/<name>/run_pipeline_parallel.sh, two levels below the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT_NAME="dataset1_multiprocess_project"
ROOT="projects/$PROJECT_NAME"
N_WORKERS=3

if [ ! -f "$ROOT/registry.json" ]; then
    echo "Error: no project at $ROOT -- run the setup script first:" >&2
    echo "  pylibs run examples/dataset1_multiprocess/setup_project.py" >&2
    exit 1
fi

PID_DIR="$ROOT/pids"
LOG_DIR="$ROOT/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

run_worker() {
    local worker_index=$1
    local log_file="$LOG_DIR/pipeline_worker_${worker_index}.log"
    # $BASHPID (not $$) is this specific backgrounded subshell's own PID --
    # $$ would still show the top-level script's PID here.
    echo "[worker $worker_index] starting, PID=$BASHPID, $(date +%T.%3N)"
    if pylibs project run-pipeline-batch "$PROJECT_NAME" \
        --root "$ROOT" --worker-index "$worker_index" --n-workers "$N_WORKERS" \
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
    echo "$worker_pid" > "$PID_DIR/pipeline_worker_${i}.pid"
    echo "Launched worker $i as PID $worker_pid (see $LOG_DIR/pipeline_worker_${i}.log)"
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
pylibs project list-samples "$PROJECT_NAME" --root "$ROOT"
