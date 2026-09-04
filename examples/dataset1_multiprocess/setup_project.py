# Multiprocess counterpart to dataset1_example.py: same 7 physical samples x
# 3 ablation layers, same 16 features, same calibration curves -- but the two
# heavy stages (feature+metrics computation, and the regression sweep) are
# each run as a separate, parallelized step instead of inline in one script,
# the way you'd actually run this on a compute cluster (each heavy stage as
# its own job, reporting done locally afterward on the transferred
# results.zarr). Full sequence, from the repo root:
#
#   pylibs run examples/dataset1_multiprocess/setup_project.py
#   ./examples/dataset1_multiprocess/run_pipeline_parallel.sh
#   ./examples/dataset1_multiprocess/compute_regressions_parallel.sh
#   pylibs run examples/dataset1_multiprocess/report_locally.py
#
# See run_pipeline_slurm.sbatch / compute_regressions_slurm.sbatch in this
# same folder for the real SLURM-array-job version of the two parallel
# steps above (same mechanism as dataset1_bootstrap_multiprocess/'s own
# run_bootstrap_slurm.sbatch) -- compute_regressions_slurm.sbatch depends on
# run_pipeline_slurm.sbatch's results, so submit both together with
# submit_slurm_chain.sh (also in this folder) rather than submitting either
# sbatch file directly.
#
# This script only registers the project/samples/features -- it doesn't run
# the pipeline itself (that's the first parallel step above).
CREATEPROJECT(
    "dataset1_multiprocess_project",
    description="B and Ca in graphite -- multiprocess pipeline+regressions demo",
)

# copied into the project, which is what lets every worker process and
# sbatch array task below resolve peak ids with nothing passed to them
USEPEAKTABLE("dataset1_multiprocess_project", "examples/peak_table.csv")

for x in range(1, 8):
    for y in range(1, 4):
        ADDSAMPLE(
            "dataset1_multiprocess_project",
            f"data/dataset1/sample{x}_{y}.libs",
            sample_name=f"sample{x}",
        )

for expression in [
    "B249",
    "B345",
    "Ca393",
    "Ca396",
    "C229",
    "C247",
    "C251",
    "B249/C229",
    "Ca393/C229",
    "(Ca393)/(Ca393+C229)",
    "(B345)/(B345+C229)",
    "log((Ca393)/(Ca393+C229))",
    "exp((Ca393)/(Ca393+C229))",
    "log((B345)/(B345+C229))",
    "log(B249/C229)",
    "log(Ca393/C229)",
]:
    ADDFEATURE("dataset1_multiprocess_project", expression)
