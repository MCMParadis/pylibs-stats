"""Orchestrates one `run_bootstrap` invocation: reads back a sample's
already-computed `features` array (and, for ratio features, its
`ratio_components` numerator/denominator arrays -- see
`ResultsStore.get_ratio_components`; both are computed once by `FeatureStep`,
never re-evaluated from the raw spectrum here), resolves this run's
`resolved_npix` and raster dimensions (for `method="local"`, from the store's
cached `raw_params` -- see `ResultsStore.get_raw_params` -- never by
reopening the raw file), gets-or-creates its checkpoint arrays, and hands its
pending (possibly sharded) iterations to `core.runs.executor.execute`.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import BaseModel

from pylibs.core.config import BootstrapConfig
from pylibs.core.exceptions import BootstrapError, ResultsError
from pylibs.core.features.expression import split_ratio
from pylibs.core.io.store import RESULTS_DIRNAME, ResultsStore
from pylibs.core.logging import get_logger
from pylibs.core.pipeline.bootstrap import compute
from pylibs.core.pipeline.bootstrap.compute import RatioRawData
from pylibs.core.pipeline.bootstrap.csv_export import export_bootstrap_csv
from pylibs.core.pipeline.metrics import METRIC_NAMES
from pylibs.core.project.project import Project
from pylibs.core.runs.checkpoint import (
    DEFAULT_BATCH_SIZE,
    CheckpointArrays,
    CheckpointWriter,
    pending_indices,
)
from pylibs.core.runs.executor import execute
from pylibs.core.runs.partition import plan_shard

logger = get_logger(__name__)


class BootstrapRunResult(BaseModel):
    sample_id: str
    bootstrap_id: str
    n_iterations: int  # current target, post any extension
    n_done: int  # total done after this call, across the whole run (not just this shard)
    n_newly_computed: int  # computed by this invocation
    n_failed: int
    failed_indices: list[int]
    results_path: Path
    csv_path: Path | None = None


@dataclass
class BootstrapResults:
    sample_id: str
    bootstrap_id: str
    feature_ids: list[str]
    metric_names: list[str]
    values: np.ndarray  # shape (n_iterations, n_features, n_metrics)
    completed: np.ndarray  # bool, shape (n_iterations,)


def _resolve_npix(config: BootstrapConfig, n_scans: int) -> int:
    if config.npix is not None:
        npix = config.npix
    else:
        assert config.pct is not None
        npix = math.ceil(n_scans * config.pct / 100)

    if config.method == "local":
        side = math.isqrt(npix)
        if side * side != npix:
            raise BootstrapError(
                f"method='local' requires npix to be a perfect square, got {npix} "
                f"(bootstrap_id={config.bootstrap_id!r})"
            )
    return npix


def _read_ratio_raw(
    store: ResultsStore, sample_id: str, ratios: dict[str, tuple[str, str]]
) -> RatioRawData:
    ratio_raw: RatioRawData = {}
    for feature_id, (num_expr, den_expr) in ratios.items():
        components = store.get_ratio_components(sample_id, feature_id)
        if components is None:
            raise ResultsError(
                f"No stored numerator/denominator for ratio feature {feature_id!r} in sample "
                f"{sample_id!r} -- run the pipeline (RUNPIPELINE) again, e.g. in a new project, "
                "to compute it."
            )
        numerator_array, denominator_array = components
        ratio_raw[feature_id] = (
            num_expr,
            den_expr,
            np.asarray(numerator_array[:]),
            np.asarray(denominator_array[:]),
        )
    return ratio_raw


def run_bootstrap(
    project_root: Path,
    sample_id: str,
    bootstrap_id: str,
    n_processes: int = 1,
    worker_index: int = 0,
    n_workers: int = 1,
    batchsize: int | None = None,
    save_csv: bool = False,
    checkpoint_batch_size: int = DEFAULT_BATCH_SIZE,
) -> BootstrapRunResult:
    """Compute every not-yet-done iteration of `bootstrap_id`'s currently
    registered target for `sample_id` (resuming automatically; picking up an
    extended target automatically). Requires `run_pipeline` (specifically
    `FeatureStep`) to have already run for this sample -- raises
    `ResultsError` otherwise. `batchsize` is unused -- bootstrap never reads
    raw spectra itself (not even for `method="local"`'s raster dimensions,
    read back from the store's cached `raw_params` instead), only
    `FeatureStep`'s already-stored output, so it always reflects whatever
    batch size `run_pipeline` was given; it is kept as a parameter for
    CLI/DSL signature compatibility. There is no peak table here for the same
    reason: the feature identity hashes this compares are read off that
    stored output, never recomputed from a table -- which is why a changed
    peak definition only reaches a bootstrap run once `run_pipeline` has
    refreshed the features array.

    If `save_csv` is set, also exports every iteration's metrics (including
    not-yet-computed ones, as blank cells) to
    `<project_root>/exports/<sample_id>_<bootstrap_id>.csv` -- see
    `export_bootstrap_csv` (also callable standalone, to re-export without
    rerunning)."""
    logger.info("Running bootstrap %r for sample %r", bootstrap_id, sample_id)
    project = Project.load(project_root)
    config = project.get_bootstrap_config(bootstrap_id)
    project.get_sample(sample_id)

    store = ResultsStore(project_root / RESULTS_DIRNAME)
    features_array = store.get_features_array(sample_id)
    if features_array is None:
        raise ResultsError(
            f"No results for sample {sample_id!r} -- run the pipeline first (RUNPIPELINE)."
        )
    all_feature_ids = cast(list[str], features_array.attrs["feature_ids"])
    out_of_range = set(store.get_out_of_range_feature_ids(sample_id))
    if out_of_range:
        logger.info(
            "Excluding %d out-of-range feature(s) from bootstrap %r for sample %r: %s",
            len(out_of_range),
            bootstrap_id,
            sample_id,
            sorted(out_of_range),
        )
    keep_positions = [i for i, fid in enumerate(all_feature_ids) if fid not in out_of_range]
    feature_ids = [all_feature_ids[i] for i in keep_positions]
    feature_values = cast(np.ndarray, features_array[:])[:, keep_positions]
    n_scans = feature_values.shape[0]

    ratios = {
        feature_id: split
        for feature_id in feature_ids
        if feature_id in project.registry.features
        and (split := split_ratio(project.registry.features[feature_id].expression)) is not None
    }

    n_rows: int | None = None
    n_cols: int | None = None
    mask: np.ndarray | None = None
    if config.method == "local":
        params = store.get_raw_params(sample_id)
        if params is None:
            raise ResultsError(
                f"No raw params for sample {sample_id!r} -- run the pipeline first (RUNPIPELINE)."
            )
        n_rows = int(params["HeightPixels"])
        n_cols = int(params["WidthPixels"])
        # a partially-scanned sample places its scans by mask, and its blocks
        # have to stay inside the scanned region (see resample_local)
        mask = store.get_mask_array(sample_id)
    ratio_raw = _read_ratio_raw(store, sample_id, ratios) if ratios else {}

    all_feature_hashes = cast(dict[str, str], features_array.attrs.get("feature_hashes", {}))
    feature_hashes = {fid: all_feature_hashes.get(fid) for fid in feature_ids}

    existing_arrays = store.get_bootstrap_arrays(sample_id, bootstrap_id)
    if existing_arrays is not None:
        stored_feature_hashes = cast(
            dict[str, str], existing_arrays[0].attrs.get("feature_hashes", {})
        )
        if stored_feature_hashes != feature_hashes:
            changed = sorted(
                fid
                for fid in set(feature_hashes) | set(stored_feature_hashes)
                if feature_hashes.get(fid) != stored_feature_hashes.get(fid)
            )
            logger.info(
                "Bootstrap %r for sample %r: feature identity changed since these iterations "
                "were computed (%s) -- resetting the whole run, all %d iteration(s) pending "
                "again.",
                bootstrap_id,
                sample_id,
                changed,
                config.n_iterations,
            )
            store.reset_bootstrap_run(sample_id, bootstrap_id)
            existing_arrays = None

    if existing_arrays is not None:
        resolved_npix = int(cast(int, existing_arrays[0].attrs["resolved_npix"]))
    else:
        resolved_npix = _resolve_npix(config, n_scans)

    metrics_array, completed_array = store.get_or_create_bootstrap_arrays(
        sample_id,
        bootstrap_id,
        config.n_iterations,
        feature_ids,
        feature_hashes,
        METRIC_NAMES,
        config.method,
        resolved_npix,
        config.shuffle,
        config.base_seed,
    )
    arrays = CheckpointArrays(metrics_array, completed_array)
    # Shard the static full range, not pending_indices(arrays) -- see
    # plan_shard's docstring for why sharding a dynamically-shrinking
    # pending list across separately-invoked processes can silently drop
    # iterations. Filtering to this shard's own still-pending indices
    # happens only after the static split.
    shard_indices = plan_shard(list(range(arrays.completed.shape[0])), worker_index, n_workers)
    pending = set(pending_indices(arrays))
    shard_pending = [index for index in shard_indices if index in pending]
    logger.debug(
        "Worker %d/%d: %d/%d iteration(s) of this shard still pending",
        worker_index,
        n_workers,
        len(shard_pending),
        len(shard_indices),
    )

    # batched writes, results before flags (see CheckpointWriter): the store
    # write is parent-side and serial, so it is a fixed cost that no worker
    # count reduces
    writer = CheckpointWriter(arrays, batch_size=checkpoint_batch_size)
    exec_result = execute(
        shard_pending,
        work_fn=compute.compute_iteration,
        on_result=writer.add,
        n_processes=n_processes,
        initializer=compute.init_worker,
        initargs=(
            feature_ids,
            feature_values,
            ratio_raw,
            config.method,
            resolved_npix,
            config.shuffle,
            config.base_seed,
            n_rows,
            n_cols,
            mask,
        ),
    )
    writer.flush()

    n_done = int(np.asarray(arrays.completed[:]).sum())
    logger.info(
        "Bootstrap %r finished for sample %r: %d/%d done (%d newly computed, %d failed)",
        bootstrap_id,
        sample_id,
        n_done,
        config.n_iterations,
        exec_result.n_completed,
        exec_result.n_failed,
    )

    csv_path = None
    if save_csv:
        feature_labels = {
            feature_id: project.registry.features[feature_id].feature_name or feature_id
            for feature_id in feature_ids
            if feature_id in project.registry.features
        }
        csv_path = export_bootstrap_csv(
            project_root,
            sample_id,
            bootstrap_id,
            feature_ids,
            feature_labels,
            METRIC_NAMES,
            np.asarray(arrays.results[:]),
            np.asarray(arrays.completed[:]),
        )

    return BootstrapRunResult(
        sample_id=sample_id,
        bootstrap_id=bootstrap_id,
        n_iterations=config.n_iterations,
        n_done=n_done,
        n_newly_computed=exec_result.n_completed,
        n_failed=exec_result.n_failed,
        failed_indices=exec_result.failed_indices,
        results_path=project_root / RESULTS_DIRNAME,
        csv_path=csv_path,
    )
