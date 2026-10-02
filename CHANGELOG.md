# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1]

Two performance fixes to parallel execution. Neither changes what is
computed: results are bit-for-bit identical at every worker count, and the
checkpoint invariant that a result reaches disk before its completion flag is
unchanged.

### Fixed

- **Worker startup no longer costs one serial import per worker.** A pool's
  read-only inputs used to travel as `ProcessPoolExecutor` `initargs`, which
  the parent pickles into each new worker's `spawn` pipe one worker at a
  time. A payload above the pipe buffer blocked the parent until that child
  drained it, and the child could not drain it until it had imported the
  package, so startup grew linearly with the worker count. The inputs now go
  out of band through a state file and the pipe payload is a path, so the
  parent never blocks and workers import concurrently. Measured on the
  bundled example data, a bootstrap run of 1000 iterations went from 5.29 s
  at 8 workers to 1.81 s at 16; a 175-feature pipeline run went from 16.56 s
  at 16 workers to 2.32 s.

- **`.libs` reading no longer loads the whole archive.** The members are
  stored uncompressed, so `data` is now memory-mapped at its offset inside
  the file and `iter_window` faults in one batch at a time, the way the
  `.ELE` reader streams. Values, dtype and `capdata` behaviour are unchanged;
  a compressed `data` member still falls back to being read whole.

- **A pipeline run builds one worker pool, not one per step.** The 1D and 2D
  distribution steps now share a single pool for the whole `run_pipeline`
  call, so per-worker startup is paid once rather than twice. Batch
  processing already used a single pool across samples and is unchanged.
  The sequential path still builds no pool at all.

### Changed

- **The permutation p-value can no longer be zero.** `compute_regressions`
  estimated it as the fraction of shuffled refits with a strictly lower
  residual MAE than the real fit, which reports 0 whenever no shuffle beats
  the data. A permutation test cannot resolve a p-value below its own
  permutation count, so that 0 claimed more than the test supports. The
  estimator is now the standard corrected form

      p = (b + 1) / (n_permutations + 1)

  where `b` counts the shuffled refits whose MAE is at most the real one's,
  ties included. The unpermuted pairing is itself one of the arrangements
  being tested, which puts a floor of `1 / (n_permutations + 1)` on the
  result (Phipson & Smyth 2010). At the default 10000 permutations that floor
  is 1.0e-04.

  Every interface prints p through the new `api.format_p_value`, which
  renders a value sitting on the floor as `p < 1.0e-04` rather than as an
  estimate, and everything above it in scientific notation. Nothing else in
  the regression or correlation computation changed.

  **Stored regressions computed before this release are stale.** The
  regression and correlation layers are not hash-guarded, so nothing
  invalidates them automatically. In an existing project, re-run
  `compute_regressions`, then `compute_correlations`, then any report or
  `apply_regressions` that reads them. Old values are recognisable: a stored
  `p_value` of exactly 0.0 can only have come from the previous estimator,
  though a small non-zero value is indistinguishable between the two.

- **Group-level inference, on by default.** Acquisitions sharing a
  `sample_name`, such as the three ablation layers of one pellet, are not
  independent evidence about that pellet's composition. Treating each as its
  own regression point is pseudoreplication: it narrows the confidence band
  and shrinks the p-value without anything more having been measured.

  `compute_regressions` now takes `group_by_sample_name`, defaulting to
  True, with `--no-group-by-sample-name` on the command line. The fitted
  line, Pearson r, R² and MAE are **unchanged**, still pooled least squares
  over every acquisition. What changes is what counts as independent:

  - the confidence band comes from the acquisition-count-weighted regression
    on the per-sample means, at `n_groups - 2` degrees of freedom. Because
    the property value is constant within a sample, that weighted fit
    reproduces the pooled line exactly, in balanced and unbalanced designs
    alike, so the band stays centred on the line drawn. The weighting
    assumes each sample mean's precision is proportional to its acquisition
    count and therefore ignores between-sample variance when counts differ;
    with equal counts it reduces to the unweighted means regression.
  - the permutation test moves property values between whole samples, every
    acquisition of a sample carrying its sample's permuted value.
  - when `n_groups!` fits within `n_permutations`, every arrangement is
    enumerated rather than sampled, and the p-value is the exact share of
    arrangements whose refit MAE is at most the real one's, the identity
    included. The bundled example has 7 samples, so its 5040 arrangements
    are enumerated exactly.

  Grouping is a no-op when no sample name repeats. Seeding is unchanged and
  stays per fit cell, independent of the worker count.

  Each regression now stores `grouped`, `n_groups`, `n_acquisitions`,
  `balanced`, `permutation_mode`, `n_permutations_used`, `band_df` and
  `inference_version`, so results from before and after this release can be
  told apart. Calibration figures and the command-line listing state both
  counts, as "n = 7 samples (21 acquisitions)", and how the p-value was
  obtained. **Regressions stored by 1.0.0 must be recomputed**, as noted for
  the p-value change above.

- **The drawn confidence band now uses `band_df`, as documented.** Grouped
  inference stored `band_df = n_groups - 2` and the documentation said the
  band used Student's t at that figure, but `confidence_band_half_width`
  derived its degrees of freedom from `n - 2` and the report passed the
  acquisition count, so nothing in the drawing path ever read `band_df`. For
  the bundled example the band was drawn with t at 19 degrees of freedom
  instead of 5, making it about 19% too narrow.

  `confidence_band_half_width` now takes the degrees of freedom explicitly.
  The `1 / n` term still uses the acquisition count: `residual_std` is per
  unit weight and `ssxx` is the weighted sum of squares, so the total weight
  behind the fit is every acquisition, and only the degrees of freedom are
  the sample count. The drawn band now equals ordinary least squares on the
  sample means exactly.

  The fitted line, Pearson r, R², MAE and every p-value are unchanged.
  Regressions stored before 1.0.1 carry no `band_df` and keep the previous
  behaviour.

- **The raster frame is settled once, at ingest.** Some instruments write
  `HeightPixels`/`WidthPixels` as the number of intervals between pixels
  rather than the number of pixels, so a 400x400 scan arrives as 399x399.
  `SpectrumStatsStep` now resolves the frame against the sample's own scan
  count and stores the resolved pair, keeping the original as
  `HeightPixelsHeader`/`WidthPixelsHeader` so the correction is auditable.

  Every consumer then reads one trustworthy value. Two of them failed
  silently before: `local` bootstrap resampling drew blocks from a wrongly
  shaped raster, and a valid `PixelAssignmentMatrix` was dropped for
  disagreeing with the header. Feature and applied-regression maps failed
  loudly, with "Can't reshape N values".

  **A mask now outranks a disagreeing header.** It carries one set cell per
  scan, which is checkable against the scan count, while the header is the
  part known to be unreliable, so the mask defines the frame instead of
  being discarded. A header that cannot be reconciled either way is stored
  unchanged with a warning, rather than failing a sample whose other results
  are fine. A project processed before this release must be re-run to gain
  the resolved frame, the same rule masks already follow.

### Added

- `compute_distribution_1d` takes `window`, choosing what range the
  histogram bins span: `"skew"` (the default, unchanged behaviour) uses the
  robust skew-adjusted fence, and `"full"` spans the finite data's own
  minimum to maximum, binning every value and dropping none. For a quantity
  whose range is meaningful rather than estimated, such as a probability on
  [0, 1], the rare extremes can be the interesting ones and the fence is
  exactly what discards them. Metrics computed from raw values are identical
  either way. An unknown value raises the new `DistributionError`.

- `CheckpointWriter` batches finished iterations, writing a batch of result
  rows and only then that batch's completion flags. Each write crosses
  zarr's synchronous wrapper once per call rather than once per iteration.
  An interruption between the two still leaves every iteration in the batch
  pending, never falsely complete. `run_bootstrap` takes a
  `checkpoint_batch_size` parameter, defaulting to 50; passing 1 restores the
  previous write-per-iteration behaviour. Chunk size stays 1, so shard
  processes can still write interleaved indices without contending.

- `WorkerPool`, a reusable `spawn` pool that several `execute` calls can
  submit into. `execute` takes an optional `pool` argument; callers that omit
  it get the previous behaviour of a private pool per call.

- Tests covering both guarantees: that an interrupted flush leaves results
  written and nothing flagged, that batched and unbatched writes produce
  identical arrays, and that one pipeline run constructs exactly one pool
  while the sequential path constructs none.

- `benchmarks/`, a reproducible parallel-execution benchmark over the bundled
  example data. See its README.

- **A public writer for the `.libs` format**, so data in any other format can
  be converted into a processable sample with no proprietary step:
  `api.write_libs`, the `pylibs write-libs` command, and the `WRITELIBS` DSL
  verb. Spectra may be an array, a memory map, or an iterable of row blocks;
  blocks stream straight into the archive, so converting a scan larger than
  memory costs one block of RAM. Geometry comes from a raster shape or from
  per-pixel coordinates, and `order="raster"` (the default) converts plain
  row-major input into the serpentine order the format stores.
  `api.load_csv_matrix` and `api.load_array` read the CSV and NumPy inputs the
  CLI accepts. `float32` storage is supported alongside `float64`.

- **A written specification for the format**, `docs/libs_format.md`: the
  members and their dtypes, the serpentine scan order, every `params` field
  with what it unlocks, and how to read a file with NumPy alone. A README
  section covers supported inputs and conversion.

- `examples/conversion/`, worked ENVI, FITS and CSV-matrix conversions. The
  ENVI and FITS examples use `spectral` and `astropy`, which are optional and
  used only there -- no new dependency reaches `pylibs`.

### Unchanged

- The `spawn` start method, every chunk size, and all computation.

## [1.0.0]

First public release.
