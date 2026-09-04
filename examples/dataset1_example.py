# dataset1: 7 physical samples (B/Ca in graphite), 3 ablation-layer files each,
# with known B/Ca concentrations in dataset1_concentrations.csv
CREATEPROJECT("dataset1_project", description="B and Ca in graphite")

# the peak table is COPIED into the project, so it is self-contained from
# here on: every later verb resolves peak ids against the project's own
# copy, and a snapshot carries it to another machine
USEPEAKTABLE("dataset1_project", "examples/peak_table.csv")

# each sampleX_Y.libs is one ablation layer of physical sample X -- the default
# "{sample_name}_{layer}" scheme reads that straight off the filename, so all 21
# files share the sample_name that groups them for correlations/regressions
# later ("sample1"), keep their own stem ("sample1_2") for report titles, and
# carry the layer in name_parts. The concentrations CSV in that directory isn't
# a sample file, so the scan ignores it.
ADDSAMPLESFROMDIR("dataset1_project", "data/dataset1")

# bare peaks, plus a few ratio features (including two normalized-ratio
# composites) and log-of-ratio/exp-of-ratio composites (outermost op is
# log()/exp(), not division, so neither gets a diagnostic-spectra or
# joint-density panel -- just (A) map and (C) distribution) to exercise
# every feature/report code path
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
    ADDFEATURE("dataset1_project", expression)

# only generate the (slower) per-sample diagnostics PDF for one ablation
# layer per physical sample, to keep the example quick to run
report_sample_stems = ["sample1_1", "sample4_1", "sample7_1"]

for sample in LISTSAMPLES("dataset1_project"):
    RUNPIPELINE("dataset1_project", sample.sample_id)
    if sample.sample_stem in report_sample_stems:
        GENERATEFEATUREREPORT("dataset1_project", sample.sample_id, save_pdf=True)

# compute_correlations never fits anything itself -- it only reshapes
# already-stored regressions into (feature x metric) matrices, so an
# all-metrics sweep (every registered feature, every metric) must run
# first; this also naturally covers the Mean/Median/Mode calibration
# curves used below, so no separate per-metric calls are needed for those
COMPUTEREGRESSIONS(
    "dataset1_project",
    "data/dataset1/dataset1_concentrations.csv",
    all_metrics=True,
)

# correlate every feature's metrics against the known B/Ca concentrations
result = COMPUTECORRELATIONS("dataset1_project", "data/dataset1/dataset1_concentrations.csv")
for column_name in result.column_names:
    GENERATECORRELATIONSREPORT("dataset1_project", column_name)

# calibration curves: a curated, mixed-metric set of (feature, metric) pairs
# out of the all-metrics sweep above -- Ca393/B345 on Mean, the ratio
# feature and the exp-of-ratio composite on Median, both the
# normalized-ratio composite and its log on Mode -- bundled into one PDF per
# column via `select`, fully decoupled from what the sweep stored
calibration_curves = [
    ("Ca393", "Mean"),
    ("B345", "Mean"),
    ("Ca393/C229", "Median"),
    ("exp((Ca393)/(Ca393+C229))", "Median"),
    ("(Ca393)/(Ca393+C229)", "Mode"),
    ("log((Ca393)/(Ca393+C229))", "Mode"),
]
for column_name in LISTREGRESSIONS("dataset1_project"):
    GENERATEREGRESSIONREPORT("dataset1_project", column_name, select=calibration_curves)

# apply just those same curated pairs to every sample (turning each
# feature's raw intensity map into a predicted-concentration map), but only
# render the PDF report for the same handful of samples as above -- applying
# is cheap, the per-sample report PDF is the slower part
for sample in LISTSAMPLES("dataset1_project"):
    APPLYREGRESSIONS("dataset1_project", sample.sample_id, select=calibration_curves)
    if sample.sample_stem in report_sample_stems:
        GENERATEAPPLIEDREGRESSIONREPORT("dataset1_project", sample.sample_id)
