#!/bin/bash
# Real SLURM array-job version of compute_regressions_parallel.sh -- same
# project, same underlying mechanism (see that script's comments for the
# local-bash version of this), just launched by SLURM instead of `&`/`wait`.
#
# --account and the `module load` line below are placeholders -- this repo
# has no way to know your actual cluster allocation or module system, so
# edit both before submitting. Everything else works as-is once those are
# set.
#
# Prerequisites (run once, NOT as part of this array job -- array tasks only
# ever call compute-regressions below):
#
#   pylibs run examples/dataset1_multiprocess/setup_project.py
#   ./examples/dataset1_multiprocess/run_pipeline_parallel.sh   # or its own sbatch
#
# sbatch won't create --output's directory for you, so make it before your
# first submission:
#
#   mkdir -p projects/dataset1_multiprocess_project/logs
#
# Then submit with:
#
#   sbatch examples/dataset1_multiprocess/compute_regressions_slurm.sbatch
#
# This job reads results run_pipeline_slurm.sbatch produces, so it must not
# start until every task of that job has finished successfully. Prefer
# submit_slurm_chain.sh (in this folder), which submits both jobs together
# with that dependency wired up automatically, over calling sbatch on this
# file directly. To chain manually instead: capture the pipeline job's id
# from sbatch's own "Submitted batch job <id>" output, then submit this file
# with `sbatch --dependency=afterok:<pipeline_job_id>
# compute_regressions_slurm.sbatch` -- against the array's own job id, which
# SLURM treats as "every task in the array", not just one.
#
#SBATCH --account=<your-account>              # e.g. def-someone -- REQUIRED, edit this
#SBATCH --array=0-2                           # N_WORKERS-1: 3 tasks, one per shard
#SBATCH --cpus-per-task=4                     # cores THIS task's own --n-processes uses
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=regressions
#SBATCH --output=projects/dataset1_multiprocess_project/logs/slurm_regressions_%A_%a.log

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1

module load python/3.11   # placeholder -- match this to the cluster's module system
source .venv/bin/activate # or wherever your pylibs virtualenv lives

PROJECT_NAME="dataset1_multiprocess_project"
ROOT="projects/$PROJECT_NAME"
CSV_PATH="data/dataset1/dataset1_concentrations.csv"

# $SLURM_ARRAY_TASK_ID (0..2, one per task) *is* --worker-index;
# $SLURM_ARRAY_TASK_COUNT (3, from --array=0-2 above) *is* --n-workers --
# no translation needed beyond reading the env vars SLURM already sets for
# each task. $SLURM_CPUS_PER_TASK mirrors --cpus-per-task above and becomes
# this task's own in-process --n-processes, parallelizing this shard's own
# (column, metric, feature) cells across an in-process worker pool -- so the
# two knobs compose: 3 array tasks x 4 cores each = 12-way parallelism
# total. Each cell is seeded independently
# (np.random.default_rng([random_seed, cell_index])), so results are
# identical no matter how the sweep is split or how many processes/workers
# run it.
#
# Every task writes to the same results.zarr; this is safe with zero extra
# locking -- every cell writes to its own disjoint
# regressions/<column>/<metric>/<feature_id>/ zarr group, so concurrent
# writers never race (see pipeline.regression_run's own module docstring).
pylibs project compute-regressions "$PROJECT_NAME" "$CSV_PATH" \
    --root "$ROOT" \
    --all-metrics \
    --worker-index "$SLURM_ARRAY_TASK_ID" \
    --n-workers "$SLURM_ARRAY_TASK_COUNT" \
    --n-processes "$SLURM_CPUS_PER_TASK"
