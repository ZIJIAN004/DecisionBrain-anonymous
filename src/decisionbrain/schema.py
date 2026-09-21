"""Global schema version and shared base class for persisted formats."""

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from .exceptions import UnsupportedSchemaVersionError

CURRENT_SCHEMA_VERSION = "2.0"


class VersionedModel(BaseModel):
    """Shared version constraint for all persisted models."""

    schema_version: str = CURRENT_SCHEMA_VERSION

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    @model_validator(mode="before")
    @classmethod
    def reject_unsupported_schema(cls, value: Any) -> Any:
        if isinstance(value, dict):
            version = value.get("schema_version", CURRENT_SCHEMA_VERSION)
            if version != CURRENT_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"Unsupported schema_version {version!r}; expected {CURRENT_SCHEMA_VERSION!r}"
                )
        return value


__all__ = ["CURRENT_SCHEMA_VERSION", "VersionedModel"]
