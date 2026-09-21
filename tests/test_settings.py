from pathlib import Path

import pytest
from pydantic import ValidationError

from decisionbrain.config import Settings


def _settings_env(**overrides: str) -> dict[str, str]:
    values = {
        "DBN_RUNS_DIR": "dotenv-runs",
        "DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES": "52428800",
        "DBN_MAX_ACTIVE_RUNS": "5",
        "DBN_DEBUG": "false",
        "LLM_MODEL_URL": "http://example.invalid/chat/completions",
        "LLM_API_KEY": "test-key",
        "LLM_CHAT_MODEL": "test-model",
        "OPT_REASONING_EFFORT": "",
        "OPT_LLM_TIMEOUT": "300",
        "OPT_LLM_TEMPERATURE": "0.2",
        "OPT_LLM_MAX_ATTEMPTS": "4",
        "OPT_LLM_RETRY_BASE_DELAY_SECONDS": "2",
        "SOLVER_TIMEOUT": "120",
        "SOLVER_CPU_BUDGET": "24",
        "OPT_PROMPTS_DIR": "/tmp/prompts",
        "OPT_ALGORITHM_MANIFESTS_DIR": "/tmp/algorithms/manifests",
        "OPT_WORKSPACE_MAX_OUTPUT_CHARS": "12000",
        "OPT_WORKSPACE_MAX_LIST_ENTRIES": "1000",
        "OPT_WORKSPACE_DEFAULT_LIST_ENTRIES": "200",
        "OPT_WORKSPACE_MAX_READ_BYTES": "1000000",
        "OPT_AGENT_RESUME_POLL_INTERVAL_SECONDS": "0.1",
        "HOST": "127.0.0.1",
        "PORT": "8008",
        "DBN_DOCTOR_COMMAND_TIMEOUT_SECONDS": "5",
    }
    values.update(overrides)
    return values


def _write_dotenv(path: Path, **overrides: str) -> None:
    values = _settings_env(**overrides)
    path.write_text(
        "\n".join(f"{name}={value}" for name, value in values.items()),
        encoding="utf-8",
    )


def _clear_config_env(monkeypatch) -> None:
    for name in (
        "DBN_RUNS_DIR",
        "DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES",
        "DBN_MAX_ACTIVE_RUNS",
        "DBN_DEBUG",
        "DBN_CONFIG_FILE",
        "DBN_VIEW",
        "DBN_MAX_INPUT_FILE_SIZE",
        "LLM_MODEL_URL",
        "LLM_API_KEY",
        "LLM_CHAT_MODEL",
        "OPT_REASONING_EFFORT",
        "OPT_LLM_TIMEOUT",
        "OPT_LLM_TEMPERATURE",
        "OPT_LLM_MAX_ATTEMPTS",
        "OPT_LLM_RETRY_BASE_DELAY_SECONDS",
        "SOLVER_TIMEOUT",
        "SOLVER_CPU_BUDGET",
        "OPT_PROMPTS_DIR",
        "OPT_ALGORITHM_MANIFESTS_DIR",
        "OPT_WORKSPACE_MAX_OUTPUT_CHARS",
        "OPT_WORKSPACE_MAX_LIST_ENTRIES",
        "OPT_WORKSPACE_DEFAULT_LIST_ENTRIES",
        "OPT_WORKSPACE_MAX_READ_BYTES",
        "OPT_AGENT_RESUME_POLL_INTERVAL_SECONDS",
        "HOST",
        "PORT",
        "DBN_DOCTOR_COMMAND_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_requires_explicit_configuration(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)

    with pytest.raises(ValidationError):
        Settings()


def test_settings_has_no_implicit_defaults():
    optional_fields = [
        name for name, field in Settings.model_fields.items() if not field.is_required()
    ]

    assert optional_fields == ["ui_language"]


def test_settings_loads_dotenv_and_environment(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    _write_dotenv(
        tmp_path / ".env",
        DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES="1048576",
        DBN_MAX_ACTIVE_RUNS="7",
        SOLVER_TIMEOUT="9",
        OPT_PROMPTS_DIR="/tmp/custom-prompts",
        PORT="9000",
    )
    monkeypatch.setenv("DBN_DEBUG", "true")

    settings = Settings()

    assert settings.runs_dir == Path("dotenv-runs")
    assert settings.snapshot_max_file_size_bytes == 1048576
    assert settings.max_active_runs == 7
    assert settings.debug is True
    assert settings.llm_model_url == "http://example.invalid/chat/completions"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_chat_model == "test-model"
    assert settings.opt_reasoning_effort == ""
    assert settings.opt_llm_timeout == 300
    assert settings.opt_llm_temperature == 0.2
    assert settings.opt_llm_max_attempts == 4
    assert settings.opt_llm_retry_base_delay_seconds == 2
    assert settings.solver_timeout == 9
    assert settings.solver_cpu_budget == 24
    assert settings.opt_prompts_dir == Path("/tmp/custom-prompts")
    assert settings.opt_algorithm_manifests_dir == Path("/tmp/algorithms/manifests")
    assert settings.opt_workspace_max_output_chars == 12000
    assert settings.opt_workspace_max_list_entries == 1000
    assert settings.opt_workspace_default_list_entries == 200
    assert settings.opt_workspace_max_read_bytes == 1000000
    assert settings.opt_agent_resume_poll_interval_seconds == 0.1
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.doctor_command_timeout_seconds == 5


def test_legacy_config_names_are_ignored(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    _write_dotenv(tmp_path / ".env")
    monkeypatch.setenv("DBN_CONFIG_FILE", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("DBN_VIEW", "trace")
    monkeypatch.setenv("DBN_MAX_INPUT_FILE_SIZE", "1")

    settings = Settings()

    assert settings.snapshot_max_file_size_bytes == 50 * 1024 * 1024
    assert not hasattr(settings, "config_file")


def test_runtime_config_excludes_secrets(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _clear_config_env(monkeypatch)
    _write_dotenv(tmp_path / ".env", LLM_API_KEY="secret-key")

    config = Settings().to_runtime_config()

    assert "llm_api_key" not in config
    assert "api_key" not in config
    assert config["llm_model_url_configured"] is True
    assert config["max_active_runs"] == 5
    assert config["solver_timeout"] == 120
    assert config["solver_cpu_budget"] == 24
    assert config["opt_prompts_dir"] == "/tmp/prompts"
    assert config["opt_llm_temperature"] == 0.2
    assert config["opt_workspace_max_read_bytes"] == 1000000
