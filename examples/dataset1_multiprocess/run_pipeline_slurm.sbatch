#!/bin/bash
# Real SLURM array-job version of run_pipeline_parallel.sh -- same project,
# same underlying mechanism (see that script's comments for the local-bash
# version of this), just launched by SLURM instead of `&`/`wait`.
#
# --account and the `module load` line below are placeholders -- this repo
# has no way to know your actual cluster allocation or module system, so
# edit both before submitting. Everything else works as-is once those are
# set.
#
# Prerequisite (run once, NOT as part of this array job -- array tasks only
# ever call run-pipeline-batch below, never create-project/add-sample/
# add-feature):
#
#   pylibs run examples/dataset1_multiprocess/setup_project.py
#
# sbatch won't create --output's directory for you, so make it before your
# first submission:
#
#   mkdir -p projects/dataset1_multiprocess_project/logs
#
# Then submit with:
#
#   sbatch examples/dataset1_multiprocess/run_pipeline_slurm.sbatch
#
# compute_regressions_slurm.sbatch (this folder) can only start after every
# task here has finished successfully -- it reads results this job produces.
# To submit both as one dependency chain instead of waiting for this job by
# hand, use submit_slurm_chain.sh (in this folder) instead of calling sbatch
# on this file directly. To chain manually: capture this job's id from
# sbatch's own "Submitted batch job <id>" output, then submit the other with
# `sbatch --dependency=afterok:<id> compute_regressions_slurm.sbatch`.
#
#SBATCH --account=<your-account>              # e.g. def-someone -- REQUIRED, edit this
#SBATCH --array=0-2                           # N_WORKERS-1: 3 tasks, one per shard
#SBATCH --cpus-per-task=4                     # cores THIS task's own --n-processes uses
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=pipeline
#SBATCH --output=projects/dataset1_multiprocess_project/logs/slurm_pipeline_%A_%a.log

set -uo pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1

module load python/3.11   # placeholder -- match this to the cluster's module system
source .venv/bin/activate # or wherever your pylibs virtualenv lives

PROJECT_NAME="dataset1_multiprocess_project"
ROOT="projects/$PROJECT_NAME"

# $SLURM_ARRAY_TASK_ID (0..2, one per task) *is* --worker-index;
# $SLURM_ARRAY_TASK_COUNT (3, from --array=0-2 above) *is* --n-workers --
# no translation needed beyond reading the env vars SLURM already sets for
# each task. $SLURM_CPUS_PER_TASK mirrors --cpus-per-task above and becomes
# this task's own in-process --n-processes, parallelizing this shard's own
# samples across an in-process worker pool -- so the two knobs compose:
# 3 array tasks x 4 cores each = 12-way parallelism total, all fanning out
# across *samples* (each sample's own Distribution1DStep/Distribution2DStep
# always run single-process within that -- see
# `pipeline.runner.run_pipeline_batch`'s docstring for why the two levels
# deliberately don't nest).
#
# Every task writes to the same results.zarr; this is safe with zero extra
# locking -- every sample writes to its own disjoint <sample_id>/... zarr
# group, so concurrent writers never race (see run_pipeline_batch's own
# docstring, and tests/integration/test_run_pipeline_batch_concurrent.py,
# which exercises exactly this scenario with real separate processes).
# Re-submitting this same array job later (e.g. after a time-limit kill) is
# also safe: already-processed samples are skipped automatically.
pylibs project run-pipeline-batch "$PROJECT_NAME" \
    --root "$ROOT" \
    --worker-index "$SLURM_ARRAY_TASK_ID" \
    --n-workers "$SLURM_ARRAY_TASK_COUNT" \
    --n-processes "$SLURM_CPUS_PER_TASK"
