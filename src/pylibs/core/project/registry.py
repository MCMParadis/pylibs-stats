import fcntl
import json
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pylibs.core.atomic import atomic_write_text
from pylibs.core.config import BootstrapConfig, FeatureConfig, ProjectConfig, SampleConfig
from pylibs.core.exceptions import BootstrapError, ProjectError

REGISTRY_FILENAME = "registry.json"
REGISTRY_LOCK_FILENAME = "registry.json.lock"


def _next_id(prefix: str, existing_ids: Iterable[str]) -> str:
    """Next sequential id for `prefix` (e.g. 'sample' -> 'sample001'), zero-padded
    to at least 3 digits and growing naturally past 999, 9999, etc."""
    pattern = re.compile(rf"{prefix}(\d+)")
    used_numbers = (
        int(match.group(1))
        for existing_id in existing_ids
        if (match := pattern.fullmatch(existing_id))
    )
    next_number = max(used_numbers, default=0) + 1
    width = max(3, len(str(next_number)))
    return f"{prefix}{next_number:0{width}d}"


@contextmanager
def _locked_registry_dir(root: Path) -> Iterator[None]:
    """Exclusive `flock` over `<root>/registry.json.lock`, held for the
    duration of a read-modify-write cycle against `registry.json`. Needed
    because `Registry.save()` always overwrites the *whole* file from
    whatever this process's own in-memory `Registry` happens to hold --
    without a lock, two separate OS processes each mutating a different
    sample/feature/etc concurrently (e.g. `run_pipeline_batch`'s
    `worker_index`/`n_workers`-sharded invocations, or its own `n_processes`
    in-process pool, each worker being its own OS process) can silently
    clobber each other's update, since whichever process's `save()` lands
    last wins with only its own, possibly-stale snapshot of the registry.

    NOT reentrant: `flock` is keyed to the open file description, so a second
    acquisition from the same process opens a second description and blocks on
    itself. Nothing may call a locking mutator while already holding this --
    see `get_or_create`, which deliberately releases before calling
    `_update_config`."""
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / REGISTRY_LOCK_FILENAME
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


class Registry:
    """The on-disk "database" for a project: its config and registered
    samples/features, with room to grow run records in later slices."""

    def __init__(
        self,
        path: Path,
        config: ProjectConfig,
        samples: dict[str, SampleConfig] | None = None,
        features: dict[str, FeatureConfig] | None = None,
        bootstrap_configs: dict[str, BootstrapConfig] | None = None,
    ):
        self.path = path
        self.config = config
        self.samples = samples if samples is not None else {}
        self.features = features if features is not None else {}
        self.bootstrap_configs = bootstrap_configs if bootstrap_configs is not None else {}

    @classmethod
    def create(cls, root: Path, config: ProjectConfig) -> "Registry":
        """Create a project at `root`, raising if one already exists.

        The exists-check and the write happen under the lock together, so two
        concurrent creators can't both pass the check -- the loser gets the
        error rather than overwriting the winner's registry."""
        path = root / REGISTRY_FILENAME
        with _locked_registry_dir(root):
            if path.exists():
                raise ProjectError(
                    f"A project already exists at {root} ({REGISTRY_FILENAME} present)"
                )
            registry = cls(path, config)
            registry.save()
        return registry

    @classmethod
    def get_or_create(cls, root: Path, config: ProjectConfig) -> "Registry":
        """Get the existing project at `root` (updating `description` if a new
        one is given), or create one if none exists yet."""
        with _locked_registry_dir(root):
            created = not (root / REGISTRY_FILENAME).exists()
            if created:
                cls(root / REGISTRY_FILENAME, config).save()
        # deliberately outside the lock: `_update_config` takes it itself, and
        # `_locked_registry_dir` is not reentrant -- calling it while holding
        # the lock would block on ourselves forever
        registry = cls.load(root)
        if not created:
            registry._update_config(description=config.description)
        return registry

    @classmethod
    def load(cls, root: Path) -> "Registry":
        path = root / REGISTRY_FILENAME
        if not path.exists():
            raise ProjectError(f"No project found at {root} ({REGISTRY_FILENAME} missing)")
        data = json.loads(path.read_text())
        config = ProjectConfig.model_validate(data["config"])
        samples = {
            sample_id: SampleConfig.model_validate(sample_data)
            for sample_id, sample_data in data.get("samples", {}).items()
        }
        features = {
            feature_id: FeatureConfig.model_validate(feature_data)
            for feature_id, feature_data in data.get("features", {}).items()
        }
        bootstrap_configs = {
            bootstrap_id: BootstrapConfig.model_validate(bootstrap_data)
            for bootstrap_id, bootstrap_data in data.get("bootstrap_configs", {}).items()
        }
        return cls(path, config, samples, features, bootstrap_configs)

    def save(self) -> None:
        data = {
            "config": json.loads(self.config.model_dump_json()),
            "samples": {
                sample_id: json.loads(sample.model_dump_json())
                for sample_id, sample in self.samples.items()
            },
            "features": {
                feature_id: json.loads(feature.model_dump_json())
                for feature_id, feature in self.features.items()
            },
            "bootstrap_configs": {
                bootstrap_id: json.loads(bootstrap.model_dump_json())
                for bootstrap_id, bootstrap in self.bootstrap_configs.items()
            },
        }
        # atomic: `Project.load` reads this unlocked, once per sample inside
        # run_pipeline, while sharded siblings write it here. A plain
        # write_text truncates first, and a reader in that window gets a
        # JSONDecodeError -- fatal at the batch level, silently swallowed by
        # core.runs.executor at the per-sample level.
        atomic_write_text(self.path, json.dumps(data, indent=2))

    @contextmanager
    def _locked_update(self) -> Iterator["Registry"]:
        """Re-read this registry from disk under an exclusive lock, hand the
        fresh copy to the caller to mutate, save it, and adopt the result.

        Every mutator goes through this. Mutating `self` and calling `save()`
        directly rewrites the whole file from a snapshot taken whenever this
        `Registry` was loaded, so a concurrent sibling's update is silently
        dropped -- and worse, `_next_id` computed from a stale snapshot hands
        two callers the *same* id. Re-reading inside the lock is what makes
        "register a sample" and "register a feature" safe to run concurrently
        against one project (two `pylibs project add-sample` invocations, two
        DSL scripts), the way `mark_sample_processed` already was.

        A body that raises leaves the file untouched and `self` unchanged: the
        save is on the far side of the yield."""
        root = self.path.parent
        with _locked_registry_dir(root):
            fresh = Registry.load(root)
            yield fresh
            fresh._touch_updated_at()
            fresh.save()
        self.config = fresh.config
        self.samples = fresh.samples
        self.features = fresh.features
        self.bootstrap_configs = fresh.bootstrap_configs

    def add_sample(
        self,
        path: Path,
        sample_name: str | None = None,
        sample_stem: str | None = None,
        name_parts: dict[str, str] | None = None,
    ) -> SampleConfig:
        """Register `path` as a sample (auto-assigning a sequential id like
        'sample001'), or update it if that exact path is already registered.
        `sample_name`/`sample_stem`/`name_parts` are only overwritten when a
        new value is given; on first registration the first two default
        independently to `path`'s stem and the third to empty. Keeping name
        and stem distinct lets `sample_name` be shared across several samples
        (e.g. as a correlations join key) while `sample_stem` stays a unique
        per-sample display label (e.g. for report titles); `name_parts` holds
        whatever else a filename scheme captured (see `filename_scheme`)."""
        with self._locked_update() as fresh:
            existing = next((s for s in fresh.samples.values() if s.path == path), None)
            if existing is not None:
                updates: dict[str, object] = {}
                if sample_name is not None:
                    updates["sample_name"] = sample_name
                if sample_stem is not None:
                    updates["sample_stem"] = sample_stem
                if name_parts is not None:
                    updates["name_parts"] = name_parts
                sample = existing.model_copy(update=updates) if updates else existing
            else:
                sample = SampleConfig(
                    sample_id=_next_id("sample", fresh.samples),
                    path=path,
                    sample_name=sample_name if sample_name is not None else path.stem,
                    sample_stem=sample_stem if sample_stem is not None else path.stem,
                    name_parts=name_parts if name_parts is not None else {},
                )
            fresh.samples[sample.sample_id] = sample
        return sample

    def get_sample(self, sample_id: str) -> SampleConfig:
        sample = self.samples.get(sample_id)
        if sample is None:
            raise ProjectError(f"No sample {sample_id!r} registered in this project")
        return sample

    def mark_sample_processed(self, sample_id: str, results_path: Path) -> SampleConfig:
        """Record where a sample's pipeline results were written, and when.

        Safe to call from several separate OS processes concurrently against
        the same project (e.g. `run_pipeline_batch`'s `worker_index`/
        `n_workers` shards, or its own `n_processes` in-process pool, each
        worker being its own process) as long as each call is for a
        *different* `sample_id`: re-reads the registry fresh from disk under
        an exclusive lock immediately before merging this one sample's
        update in and saving, rather than trusting `self`'s own possibly-
        stale in-memory snapshot (taken whenever this `Registry` was loaded,
        which for a long-running batch worker could be well before a
        sibling process's own concurrent update) -- see
        `_locked_registry_dir`."""
        with self._locked_update() as fresh:
            sample = fresh.get_sample(sample_id).model_copy(
                update={
                    "results_path": results_path,
                    "processed_at_utc": datetime.now(UTC),
                    "processed_at_dst": datetime.now().astimezone(),
                }
            )
            fresh.samples[sample.sample_id] = sample
        return sample

    def add_feature(self, expression: str, feature_name: str | None = None) -> FeatureConfig:
        """Register `expression` as a feature (auto-assigning a sequential id
        like 'feature001'), or update it if that exact expression is already
        registered. `feature_name` is only overwritten when a new value is
        given; on first registration it defaults to `expression`."""
        with self._locked_update() as fresh:
            existing = next(
                (f for f in fresh.features.values() if f.expression == expression), None
            )
            if existing is not None:
                feature = (
                    existing.model_copy(update={"feature_name": feature_name})
                    if feature_name is not None
                    else existing
                )
            else:
                feature = FeatureConfig(
                    feature_id=_next_id("feature", fresh.features),
                    expression=expression,
                    feature_name=feature_name if feature_name is not None else expression,
                )
            fresh.features[feature.feature_id] = feature
        return feature

    def add_bootstrap_config(
        self,
        method: str,
        npix: int | None = None,
        pct: float | None = None,
        n_iterations: int = 100,
        base_seed: int = 0,
        shuffle: bool = False,
    ) -> BootstrapConfig:
        """Register (method, npix-or-pct, base_seed, shuffle) as a bootstrap
        config -- auto-assigning a sequential id like 'bootstrap001' -- or,
        if that exact identity is already registered, grow its
        `n_iterations` in place (a smaller/equal value is a no-op) rather
        than adding a duplicate. A 'local, no-shuffle' sweep and a 'local,
        with-shuffle' sweep are different identities and so get different
        ids, tracked as fully independent, independently resumable runs.
        Does not itself validate `method`/`npix`/`pct` -- see
        `core.api.add_bootstrap_config`."""
        with self._locked_update() as fresh:
            existing = next(
                (
                    b
                    for b in fresh.bootstrap_configs.values()
                    if (b.method, b.npix, b.pct, b.base_seed, b.shuffle)
                    == (method, npix, pct, base_seed, shuffle)
                ),
                None,
            )
            if existing is not None:
                config = (
                    existing.model_copy(update={"n_iterations": n_iterations})
                    if n_iterations > existing.n_iterations
                    else existing
                )
            else:
                config = BootstrapConfig(
                    bootstrap_id=_next_id("bootstrap", fresh.bootstrap_configs),
                    method=method,
                    npix=npix,
                    pct=pct,
                    n_iterations=n_iterations,
                    base_seed=base_seed,
                    shuffle=shuffle,
                )
            fresh.bootstrap_configs[config.bootstrap_id] = config
        return config

    def get_bootstrap_config(self, bootstrap_id: str) -> BootstrapConfig:
        config = self.bootstrap_configs.get(bootstrap_id)
        if config is None:
            raise ProjectError(f"No bootstrap config {bootstrap_id!r} registered in this project")
        return config

    def extend_bootstrap_config(self, bootstrap_id: str, n_iterations: int) -> BootstrapConfig:
        """Raise the registered `n_iterations` target for an existing
        bootstrap config. Raises `BootstrapError` if `n_iterations` isn't
        strictly greater than the current value."""
        with self._locked_update() as fresh:
            # compared against the *fresh* target, so a sibling that already
            # raised it higher isn't silently rolled back
            existing = fresh.get_bootstrap_config(bootstrap_id)
            if n_iterations <= existing.n_iterations:
                raise BootstrapError(
                    f"new n_iterations ({n_iterations}) must be greater than the current value "
                    f"({existing.n_iterations}) for {bootstrap_id!r}"
                )
            config = existing.model_copy(update={"n_iterations": n_iterations})
            fresh.bootstrap_configs[bootstrap_id] = config
        return config

    def _update_config(self, description: str | None = None) -> None:
        """Update `description` (if given) and bump updated_at_utc/updated_at_dst."""
        with self._locked_update() as fresh:
            if description is not None:
                fresh.config = fresh.config.model_copy(update={"description": description})

    def _touch_updated_at(self) -> None:
        self.config = self.config.model_copy(
            update={
                "updated_at_utc": datetime.now(UTC),
                "updated_at_dst": datetime.now().astimezone(),
            }
        )
