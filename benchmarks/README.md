# Parallel-execution benchmark

Measures computation time and speedup against the number of parallel workers,
for the three things `pylibs` parallelises, plus the separately-launched-process
mode used for array jobs.

Everything runs through `pylibs.core.api` and the `pylibs` command line, the way
a user would drive it. No library code is involved, and none was changed to make
this possible.

## Running it

```bash
pip install -e ".[dev]"
python benchmarks/benchmark.py --dry-run          # smallest sample, workers 1 and 2
python benchmarks/benchmark.py                    # the full sweep
python benchmarks/benchmark.py --config benchmarks/config.yaml --out benchmarks/results
```

`--keep` leaves each run's temporary project on disk instead of deleting it.
The exit status is non-zero if the determinism check fails.

## Scenarios

| | what varies | what is measured |
|---|---|---|
| A | in-process workers, one sample, every feature | extraction time, distribution-metrics time, total |
| B | in-process workers, a batch of samples | total |
| C | in-process workers, bootstrap resampling over one sample | total |
| D | independent shard processes, B and C | total, plus each shard's own start and end |

In scenario A the worker count reaches only the two distribution-metric steps.
Feature extraction and spectrum statistics each stream the raw file once and are
strictly sequential, which is why the harness reports them separately: their
share of the single-worker run is the sequential fraction in the Amdahl bound,
measured rather than assumed.

The pipeline scenario C depends on is built before timing starts and is not
included in its time.

Scenario D launches k independent processes, each with one in-process worker and
its own shard index, all writing into the same project store. Total time runs
from the first process starting to the last one exiting. Per-process start and
end times go to `shard_timeline.csv` so load imbalance between shards stays
visible.

## How the timing is kept honest

- **Every timed run starts from a fresh, empty project** in a new temporary
  directory, with the peak table, samples, features and resampling
  configuration registered from scratch. Every pipeline stage skips work it has
  already done, so a second run in the same project would measure nothing. The
  project is deleted afterwards unless `--keep` is passed.
- **Each run happens in its own interpreter**, so import state, the
  project-scoped log handler and allocator warm-up cannot leak between worker
  counts.
- **Threads are pinned to one per worker.** `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` are all
  set to 1 before NumPy is imported anywhere, in the parent and therefore in
  every spawned worker. Without this a BLAS backend would open its own thread
  pool inside each worker and the worker count would no longer mean what the
  x axis says.
- **The file cache is warm.** One untimed warm-up precedes the timed repeats of
  each scenario, and the same acquisitions are read repeatedly throughout, so
  the numbers describe computation rather than first-read latency.
- **Worker counts never exceed the physical core count**, read from `lscpu`.
  Hardware threads on one core share execution units, so counting them would
  make efficiency appear to collapse for reasons unrelated to the software.
- **Stage timing is read from the project's own log**, which is stamped to the
  millisecond. With debug logging on, the runner names each step immediately
  before running it and logs one line after the last finishes, in a fixed
  order, so consecutive stamps bound each step. If any marker is missing the
  harness reports only the total rather than inventing a split.

## Determinism check

Parallelism must not change results. After every run the harness reads the
stored outputs back through the façade, in a fixed order, and compares them
element-wise against the single-worker run of the same scenario: every
distribution metric for scenarios A, B and D over batches, and the full
bootstrap metric array for C and D over resampling.

The maximum absolute difference goes in `results.csv` for every row. It is
expected to be exactly zero. Anything else is printed loudly at the end and
makes the harness exit non-zero, rather than being folded into a tolerance.

## Configuration

All of it lives in `config.yaml`: data paths, the feature list, worker counts,
repeat count, and the bootstrap iteration count and subset size. Nothing is
hard-coded. The defaults use the bundled boron-and-calcium acquisitions and the
peak table in `examples/`, so the benchmark reproduces without private data.

Point `single_sample` at a larger acquisition to raise the per-run workload.
With small acquisitions the process-pool startup cost, paid once per pool and
per worker, can exceed the work being distributed, and speedup curves fall below
one. That is a genuine property of the software worth reporting, but it is a
statement about the workload size, not about scalability in general.

## Outputs

Written to the configured output directory:

- `results.csv`, one row per run: scenario, mode, workers, repeat, total time,
  extraction and metrics time where applicable, sample and feature counts,
  pixel count, bootstrap iteration count, the determinism difference, and a
  timestamp.
- `summary.csv`: median time, speedup and efficiency per scenario and worker
  count, plus the measured sequential fraction and Amdahl bound for scenario A.
- `environment.json`: processor, physical and logical cores, memory, the
  storage backing both the data and the project store, operating system, and
  the Python, NumPy, SciPy and Zarr versions with the `pylibs` commit.
- `speedup.pdf` and `speedup.png`: speedup against workers, single-column width,
  with the ideal line and scenario A's Amdahl bound.
- `table.tex`: a booktabs table of median time, speedup and efficiency.
- `shard_timeline.csv`, when scenario D runs.
