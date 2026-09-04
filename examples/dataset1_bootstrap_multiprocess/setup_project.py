CREATEPROJECT(
    "dataset1_bootstrap_multiprocess_project",
    description="B in graphite -- multiprocess bootstrap demo",
)

# copied into the project -- the parallel workers read that copy
USEPEAKTABLE("dataset1_bootstrap_multiprocess_project", "examples/peak_table.csv")

ADDSAMPLE(
    "dataset1_bootstrap_multiprocess_project",
    "data/dataset1/sample1_1.libs",
    sample_name="sample1",
)

for expression in ["B249", "C229", "B249/C229"]:
    ADDFEATURE("dataset1_bootstrap_multiprocess_project", expression)

RUNPIPELINE("dataset1_bootstrap_multiprocess_project", "sample001")

# A single config with enough iterations that splitting the work across
# several separate OS processes (see run_bootstrap_parallel.sh) is actually
# worth demonstrating.
ADDBOOTSTRAPCONFIG(
    "dataset1_bootstrap_multiprocess_project",
    "random",
    npix=100,
    n_iterations=200,
)
