from pathlib import Path

from pylibs.core.config import BootstrapConfig, FeatureConfig, ProjectConfig, SampleConfig
from pylibs.core.logging import configure_project_logging
from pylibs.core.project.registry import Registry


class Project:
    def __init__(self, config: ProjectConfig, registry: Registry):
        self.config = config
        self.registry = registry

    @classmethod
    def create(cls, project_name: str, root: Path, description: str | None = None) -> "Project":
        """Get the existing project at `root`, or create one if none exists yet."""
        config = ProjectConfig(name=project_name, root=root, description=description)
        registry = Registry.get_or_create(root, config)
        configure_project_logging(root)
        return cls(registry.config, registry)

    @classmethod
    def load(cls, root: Path) -> "Project":
        registry = Registry.load(root)
        configure_project_logging(root)
        return cls(registry.config, registry)

    def add_sample(
        self,
        path: Path,
        sample_name: str | None = None,
        sample_stem: str | None = None,
        name_parts: dict[str, str] | None = None,
    ) -> SampleConfig:
        return self.registry.add_sample(path, sample_name, sample_stem, name_parts)

    def get_sample(self, sample_id: str) -> SampleConfig:
        return self.registry.get_sample(sample_id)

    def add_feature(self, expression: str, feature_name: str | None = None) -> FeatureConfig:
        return self.registry.add_feature(expression, feature_name)

    def mark_sample_processed(self, sample_id: str, results_path: Path) -> SampleConfig:
        return self.registry.mark_sample_processed(sample_id, results_path)

    def add_bootstrap_config(
        self,
        method: str,
        npix: int | None = None,
        pct: float | None = None,
        n_iterations: int = 100,
        base_seed: int = 0,
        shuffle: bool = False,
    ) -> BootstrapConfig:
        return self.registry.add_bootstrap_config(
            method, npix, pct, n_iterations, base_seed, shuffle
        )

    def get_bootstrap_config(self, bootstrap_id: str) -> BootstrapConfig:
        return self.registry.get_bootstrap_config(bootstrap_id)

    def extend_bootstrap_config(self, bootstrap_id: str, n_iterations: int) -> BootstrapConfig:
        return self.registry.extend_bootstrap_config(bootstrap_id, n_iterations)
