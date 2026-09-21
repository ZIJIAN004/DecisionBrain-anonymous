"""Shared DecisionBrain configuration."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single configuration model populated from ``.env`` and process variables."""

    runs_dir: Path
    snapshot_max_file_size_bytes: int = Field(
        gt=0,
        description="Maximum size in bytes of one file copied into an input snapshot.",
    )
    max_active_runs: int = Field(
        gt=0,
        description="Maximum number of active Runs in one API process.",
    )
    debug: bool
    ui_language: Literal["en", "zh"] = "en"

    llm_model_url: str = Field(validation_alias="LLM_MODEL_URL")
    llm_api_key: str = Field(validation_alias="LLM_API_KEY")
    llm_chat_model: str = Field(validation_alias="LLM_CHAT_MODEL")
    opt_reasoning_effort: str = Field(validation_alias="OPT_REASONING_EFFORT")
    opt_llm_timeout: int = Field(gt=0, validation_alias="OPT_LLM_TIMEOUT")
    opt_llm_temperature: float = Field(
        ge=0,
        le=2,
        validation_alias="OPT_LLM_TEMPERATURE",
    )
    opt_llm_max_attempts: int = Field(gt=0, validation_alias="OPT_LLM_MAX_ATTEMPTS")
    opt_llm_retry_base_delay_seconds: float = Field(
        gt=0,
        validation_alias="OPT_LLM_RETRY_BASE_DELAY_SECONDS",
    )

    solver_timeout: int = Field(gt=0, validation_alias="SOLVER_TIMEOUT")
    solver_cpu_budget: int = Field(gt=0, validation_alias="SOLVER_CPU_BUDGET")
    opt_prompts_dir: Path = Field(validation_alias="OPT_PROMPTS_DIR")
    opt_algorithm_manifests_dir: Path = Field(validation_alias="OPT_ALGORITHM_MANIFESTS_DIR")
    opt_workspace_max_output_chars: int = Field(
        gt=0,
        validation_alias="OPT_WORKSPACE_MAX_OUTPUT_CHARS",
    )
    opt_workspace_max_list_entries: int = Field(
        gt=0,
        validation_alias="OPT_WORKSPACE_MAX_LIST_ENTRIES",
    )
    opt_workspace_default_list_entries: int = Field(
        gt=0,
        validation_alias="OPT_WORKSPACE_DEFAULT_LIST_ENTRIES",
    )
    opt_workspace_max_read_bytes: int = Field(
        gt=0,
        validation_alias="OPT_WORKSPACE_MAX_READ_BYTES",
    )
    opt_agent_resume_poll_interval_seconds: float = Field(
        gt=0,
        validation_alias="OPT_AGENT_RESUME_POLL_INTERVAL_SECONDS",
    )

    host: str = Field(validation_alias="HOST")
    port: int = Field(gt=0, validation_alias="PORT")
    doctor_command_timeout_seconds: float = Field(gt=0)

    model_config = SettingsConfigDict(
        env_prefix="DBN_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_workspace_limits(self) -> "Settings":
        if self.opt_workspace_default_list_entries > self.opt_workspace_max_list_entries:
            raise ValueError(
                "OPT_WORKSPACE_DEFAULT_LIST_ENTRIES cannot exceed OPT_WORKSPACE_MAX_LIST_ENTRIES"
            )
        return self

    def to_runtime_config(self) -> dict[str, Any]:
        """Return configuration fields that may be persisted."""
        return {
            "runs_dir": str(self.runs_dir),
            "snapshot_max_file_size_bytes": self.snapshot_max_file_size_bytes,
            "max_active_runs": self.max_active_runs,
            "debug": self.debug,
            "ui_language": self.ui_language,
            "llm_model_url_configured": bool(self.llm_model_url),
            "llm_chat_model": self.llm_chat_model,
            "opt_reasoning_effort": self.opt_reasoning_effort,
            "opt_llm_timeout": self.opt_llm_timeout,
            "opt_llm_temperature": self.opt_llm_temperature,
            "opt_llm_max_attempts": self.opt_llm_max_attempts,
            "opt_llm_retry_base_delay_seconds": self.opt_llm_retry_base_delay_seconds,
            "solver_timeout": self.solver_timeout,
            "solver_cpu_budget": self.solver_cpu_budget,
            "opt_prompts_dir": str(self.opt_prompts_dir),
            "opt_algorithm_manifests_dir": str(self.opt_algorithm_manifests_dir),
            "opt_workspace_max_output_chars": self.opt_workspace_max_output_chars,
            "opt_workspace_max_list_entries": self.opt_workspace_max_list_entries,
            "opt_workspace_default_list_entries": self.opt_workspace_default_list_entries,
            "opt_workspace_max_read_bytes": self.opt_workspace_max_read_bytes,
            "opt_agent_resume_poll_interval_seconds": self.opt_agent_resume_poll_interval_seconds,
            "host": self.host,
            "port": self.port,
            "doctor_command_timeout_seconds": self.doctor_command_timeout_seconds,
        }
