#!/bin/bash
# Submits run_pipeline_slurm.sbatch and compute_regressions_slurm.sbatch as
# one dependency chain, instead of submitting/waiting for each by hand:
# compute_regressions_slurm.sbatch's array is submitted with
# --dependency=afterok:<pipeline_job_id> against the *array's own* job id,
# which SLURM treats as "wait for every task in that array to finish
# successfully" -- not just one task -- so the regression sweep only starts
# once run-pipeline-batch has actually finished every sample. This mirrors
# the hard data dependency between the two stages: compute-regressions reads
# every sample's distribution_1d/distribution_2d, which only exist once
# run-pipeline-batch has processed that sample (see run_pipeline_slurm.sbatch
# and compute_regressions_slurm.sbatch's own comments for the two stages
# individually).
#
# Prerequisite (run once, NOT part of either array job):
#
#   pylibs run examples/dataset1_multiprocess/setup_project.py
#
# Then submit the whole chain with:
#
#   ./examples/dataset1_multiprocess/submit_slurm_chain.sh
#
# Edit --account/the `module load` line in *both* .sbatch files first, same
# as submitting either one standalone -- this script doesn't change either
# file, it just submits them together. Only submits the chain and returns
# immediately; use `squeue -u $USER` or `sacct` to watch progress. If the
# pipeline job fails (or is cancelled), SLURM leaves the dependent
# regressions job permanently pending on a job that will never succeed --
# cancel it yourself with `scancel <regressions_job_id>` rather than leaving
# it queued.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p projects/dataset1_multiprocess_project/logs

pipeline_output=$(sbatch "$SCRIPT_DIR/run_pipeline_slurm.sbatch")
echo "$pipeline_output"
# sbatch prints "Submitted batch job <id>" -- the id is the last field.
pipeline_job_id=$(echo "$pipeline_output" | awk '{print $NF}')
if [ -z "$pipeline_job_id" ]; then
    echo "Error: couldn't parse a job id out of sbatch's output above -- not submitting" \
        "the dependent regressions job." >&2
    exit 1
fi

regressions_output=$(sbatch --dependency="afterok:${pipeline_job_id}" \
    "$SCRIPT_DIR/compute_regressions_slurm.sbatch")
echo "$regressions_output"
regressions_job_id=$(echo "$regressions_output" | awk '{print $NF}')

echo
echo "Chained: regressions job $regressions_job_id will start once every task of" \
    "pipeline job $pipeline_job_id finishes successfully."
