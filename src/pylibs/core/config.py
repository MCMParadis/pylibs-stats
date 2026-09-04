from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_serializer


class ProjectConfig(BaseModel):
    name: str
    root: Path
    created_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at_dst: datetime = Field(
        default_factory=lambda: datetime.now().astimezone(),
        description="created_at_utc, expressed in the creating machine's local time.",
    )
    updated_at_utc: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at_dst: datetime = Field(
        default_factory=lambda: datetime.now().astimezone(),
        description="updated_at_utc, expressed in the creating machine's local time.",
    )
    description: str | None = None

    @field_serializer(
        "created_at_utc", "created_at_dst", "updated_at_utc", "updated_at_dst", when_used="json"
    )
    def _serialize_datetime(self, value: datetime) -> str:
        """Always use an explicit UTC offset (+00:00), never Pydantic's default 'Z'."""
        return value.isoformat()


class SampleConfig(BaseModel):
    sample_id: str
    path: Path
    sample_name: str | None = None
    sample_stem: str | None = None
    # every other field a filename scheme captured (see project.filename_scheme),
    # e.g. {"layer": "2"} -- empty for a sample registered without a scheme
    name_parts: dict[str, str] = {}
    results_path: Path | None = None
    processed_at_utc: datetime | None = None
    processed_at_dst: datetime | None = None

    @field_serializer("processed_at_utc", "processed_at_dst", when_used="json")
    def _serialize_processed_at(self, value: datetime | None) -> str | None:
        """Always use an explicit UTC offset (+00:00), never Pydantic's default 'Z'."""
        return value.isoformat() if value is not None else None


class FeatureConfig(BaseModel):
    feature_id: str
    expression: str
    feature_name: str | None = None


class BootstrapConfig(BaseModel):
    bootstrap_id: str
    method: str
    npix: int | None = None
    pct: float | None = None
    n_iterations: int
    base_seed: int
    shuffle: bool = False
