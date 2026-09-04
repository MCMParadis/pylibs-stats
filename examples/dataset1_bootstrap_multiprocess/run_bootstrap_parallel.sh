#!/bin/bash
# Runs one bootstrap config's iterations split across 2 genuinely separate
# OS processes (not just threads), using `pylibs project run-bootstrap`'s
# --worker-index/--n-workers sharding. Run setup_project.py once first:
#
#   pylibs run examples/dataset1_bootstrap_multiprocess/setup_project.py
#   ./examples/dataset1_bootstrap_multiprocess/run_bootstrap_parallel.sh
#
# Can be run from any directory. Re-running is safe: any iteration already
# done by a previous run is skipped (resumed), not recomputed.
#
# See run_bootstrap_slurm.sbatch in this same folder for the real
# SLURM-array-job version of this exact mechanism.

set -uo pipefail

# Resolve paths relative to the repo root, not the caller's current
# directory -- this script lives at examples/<name>/run_bootstrap_parallel.sh,
# two levels below the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PROJECT_NAME="dataset1_bootstrap_multiprocess_project"
ROOT="projects/$PROJECT_NAME"
SAMPLE_ID="sample001"
BOOTSTRAP_ID="bootstrap001"
N_WORKERS=2

if [ ! -f "$ROOT/registry.json" ]; then
    echo "Error: no project at $ROOT -- run the setup script first:" >&2
    echo "  pylibs run examples/dataset1_bootstrap_multiprocess/setup_project.py" >&2
    exit 1
fi

PID_DIR="$ROOT/pids"
LOG_DIR="$ROOT/logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

run_worker() {
    local worker_index=$1
    local log_file="$LOG_DIR/worker_${worker_index}.log"
    # $BASHPID (not $$) is this specific backgrounded subshell's own PID --
    # $$ would still show the top-level script's PID here.
    echo "[worker $worker_index] starting, PID=$BASHPID, $(date +%T.%3N)"
    if pylibs project run-bootstrap "$PROJECT_NAME" "$SAMPLE_ID" "$BOOTSTRAP_ID" \
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
    echo "$worker_pid" > "$PID_DIR/worker_${i}.pid"
    echo "Launched worker $i as PID $worker_pid (see $LOG_DIR/worker_${i}.log)"
done

# Waits for every backgrounded worker above -- this is what actually lets
# them run concurrently: nothing blocks between the two launches.
for job in $(jobs -p); do
    wait "$job" || any_failed=1
done

echo
if [ "$any_failed" -ne 0 ]; then
    echo "One or more workers failed -- see above." >&2
    exit 1
fi
pylibs project bootstrap-progress "$PROJECT_NAME" "$SAMPLE_ID" "$BOOTSTRAP_ID" --root "$ROOT"
