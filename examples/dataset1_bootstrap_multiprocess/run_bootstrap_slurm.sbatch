#!/bin/bash
# Real SLURM array-job version of run_bootstrap_parallel.sh -- same
# project/sample/bootstrap_id, same underlying mechanism (see that script's
# comments for the local-bash version of this), just launched by SLURM
# instead of `&`/`wait`.
#
# --account and the `module load` line below are placeholders -- this repo
# has no way to know your actual cluster allocation or module system, so
# edit both before submitting. Everything else works as-is once those are
# set.
#
# Prerequisite (run once, NOT as part of this array job -- array tasks only
# ever call run-bootstrap below, never add-bootstrap-config/run-pipeline):
#
#   pylibs run examples/dataset1_bootstrap_multiprocess/setup_project.py
#
# sbatch won't create --output's directory for you, so make it before your
# first submission (setup_project.py above doesn't create it either):
#
#   mkdir -p projects/dataset1_bootstrap_multiprocess_project/logs
#
# Then submit with:
#
#   sbatch examples/dataset1_bootstrap_multiprocess/run_bootstrap_slurm.sbatch
#
#SBATCH --account=<your-account>              # e.g. def-someone -- REQUIRED, edit this
#SBATCH --array=0-3                           # N_WORKERS-1: 4 tasks, one per shard
#SBATCH --cpus-per-task=4                     # cores THIS task's own --n-processes uses
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=bootstrap
#SBATCH --output=projects/dataset1_bootstrap_multiprocess_project/logs/slurm_%A_%a.log

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1

module load python/3.11   # placeholder -- match this to the cluster's module system
source .venv/bin/activate # or wherever your pylibs virtualenv lives

PROJECT_NAME="dataset1_bootstrap_multiprocess_project"
ROOT="projects/$PROJECT_NAME"
SAMPLE_ID="sample001"
BOOTSTRAP_ID="bootstrap001"

# $SLURM_ARRAY_TASK_ID (0..3, one per task) *is* --worker-index;
# $SLURM_ARRAY_TASK_COUNT (4, from --array=0-3 above) *is* --n-workers --
# no translation needed beyond reading the env vars SLURM already sets for
# each task. $SLURM_CPUS_PER_TASK mirrors --cpus-per-task above and becomes
# this task's own in-process --n-processes, so the two parallelism knobs
# compose: 4 array tasks x 4 cores each = 16-way parallelism total.
#
# Every task writes to the same results.zarr; this is safe with zero extra
# locking (core.runs.checkpoint's chunk-size-1 arrays + core.runs.partition's
# static full-range sharding -- see those modules' docstrings and
# tests/integration/test_bootstrap_concurrent.py, which exercises exactly
# this scenario with real separate processes). Re-submitting this same
# array job later (e.g. after a time-limit kill) is also safe: already-
# completed iterations are skipped automatically, never redone.
pylibs project run-bootstrap "$PROJECT_NAME" "$SAMPLE_ID" "$BOOTSTRAP_ID" \
    --root "$ROOT" \
    --worker-index "$SLURM_ARRAY_TASK_ID" \
    --n-workers "$SLURM_ARRAY_TASK_COUNT" \
    --n-processes "$SLURM_CPUS_PER_TASK"
