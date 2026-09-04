CREATEPROJECT(
    "dataset1_bootstrap_project", description="B and Ca in graphite -- bootstrap resampling"
)

# copied into the project -- must come before any ADDFEATURE
USEPEAKTABLE("dataset1_bootstrap_project", "examples/peak_table.csv")

for x in range(1, 8):
    for y in range(1, 4):
        ADDSAMPLE(
            "dataset1_bootstrap_project",
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
]:
    ADDFEATURE("dataset1_bootstrap_project", expression)

samples = LISTSAMPLES("dataset1_bootstrap_project")
for sample in samples:
    RUNPIPELINE("dataset1_bootstrap_project", sample.sample_id)

bootstrap_configs = [
    ADDBOOTSTRAPCONFIG(
        "dataset1_bootstrap_project",
        method,
        npix=100,
        n_iterations=50,
        shuffle=shuffle,
    )
    for method in ["random", "local"]
    for shuffle in [False, True]
]

csv_sample_stems = ["sample1_1"]

for sample in samples:
    for config in bootstrap_configs:
        RUNBOOTSTRAP(
            "dataset1_bootstrap_project",
            sample.sample_id,
            config.bootstrap_id,
            save_csv=sample.sample_stem in csv_sample_stems,
        )
