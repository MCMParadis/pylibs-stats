# "Report locally after data transfer": the light, sequential second half of
# the multiprocess workflow in this folder -- everything here just reads
# already-computed results.zarr data (built by run_pipeline_parallel.sh and
# compute_regressions_parallel.sh, or their SLURM equivalents) and renders
# reports, exactly mirroring dataset1_example.py's own second half. Run
# after both parallel steps have finished:
#
#   pylibs run examples/dataset1_multiprocess/report_locally.py
#
# only generate the (slower) per-sample diagnostics PDF for one ablation
# layer per physical sample, to keep this quick to run
report_sample_stems = ["sample1_1", "sample4_1", "sample7_1"]

for sample in LISTSAMPLES("dataset1_multiprocess_project"):
    if sample.sample_stem in report_sample_stems:
        GENERATEFEATUREREPORT("dataset1_multiprocess_project", sample.sample_id, save_pdf=True)

# compute_correlations never fits anything itself -- it only reshapes
# already-stored regressions (computed in parallel by
# compute_regressions_parallel.sh) into (feature x metric) matrices
result = COMPUTECORRELATIONS(
    "dataset1_multiprocess_project", "data/dataset1/dataset1_concentrations.csv"
)
for column_name in result.column_names:
    GENERATECORRELATIONSREPORT("dataset1_multiprocess_project", column_name)

# calibration curves: a curated, mixed-metric set of (feature, metric) pairs
# out of the all-metrics sweep computed in parallel -- same set as
# dataset1_example.py's own, so the two examples' reports are directly
# comparable
calibration_curves = [
    ("Ca393", "Mean"),
    ("B345", "Mean"),
    ("Ca393/C229", "Median"),
    ("exp((Ca393)/(Ca393+C229))", "Median"),
    ("(Ca393)/(Ca393+C229)", "Mode"),
    ("log((Ca393)/(Ca393+C229))", "Mode"),
]
for column_name in LISTREGRESSIONS("dataset1_multiprocess_project"):
    GENERATEREGRESSIONREPORT(
        "dataset1_multiprocess_project", column_name, select=calibration_curves
    )

# apply just those same curated pairs to every sample (turning each
# feature's raw intensity map into a predicted-concentration map), but only
# render the PDF report for the same handful of samples as above
for sample in LISTSAMPLES("dataset1_multiprocess_project"):
    APPLYREGRESSIONS("dataset1_multiprocess_project", sample.sample_id, select=calibration_curves)
    if sample.sample_stem in report_sample_stems:
        GENERATEAPPLIEDREGRESSIONREPORT("dataset1_multiprocess_project", sample.sample_id)
