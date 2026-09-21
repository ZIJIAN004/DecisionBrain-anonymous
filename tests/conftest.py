from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from decisionbrain.core.models import (
    AgentInput,
    ClarificationQuestion,
    NeedsClarification,
    Succeeded,
)


_INTEGRATION_TEST_MODULES = {
    "test_api_flow.py",
    "test_benchmark_web.py",
    "test_cli_chat.py",
    "test_cli_resume.py",
    "test_cli_run.py",
    "test_runs.py",
    "test_runtime_session.py",
}


def pytest_collection_modifyitems(items):
    """Classify every test while keeping external-resource tests opt-in."""

    external_markers = {"solver", "network"}
    for item in items:
        marker_names = {marker.name for marker in item.iter_markers()}
        if marker_names & external_markers:
            continue
        if item.path.name in _INTEGRATION_TEST_MODULES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(autouse=True)
def required_settings_env(monkeypatch, request, tmp_path):
    if request.node.path.name == "test_settings.py":
        return
    values = {
        "DBN_RUNS_DIR": str(tmp_path / "runs"),
        "DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES": str(50 * 1024 * 1024),
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
        "OPT_PROMPTS_DIR": str(Path(__file__).resolve().parents[1] / "prompts"),
        "OPT_ALGORITHM_MANIFESTS_DIR": str(
            Path(__file__).resolve().parents[1] / "algorithms" / "manifests"
        ),
        "OPT_WORKSPACE_MAX_OUTPUT_CHARS": "12000",
        "OPT_WORKSPACE_MAX_LIST_ENTRIES": "1000",
        "OPT_WORKSPACE_DEFAULT_LIST_ENTRIES": "200",
        "OPT_WORKSPACE_MAX_READ_BYTES": "1000000",
        "OPT_AGENT_RESUME_POLL_INTERVAL_SECONDS": "0.1",
        "HOST": "127.0.0.1",
        "PORT": "8008",
        "DBN_DOCTOR_COMMAND_TIMEOUT_SECONDS": "5",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class CliTestAgent:
    version = "cli-test-1"

    def __init__(self, services=None, **workflow_options):
        self.state = {
            "clarify": False,
            "answered": False,
        }

    def initialize(self, agent_input: AgentInput):
        self.state = {
            "clarify": "NEED_CLARIFICATION" in agent_input.problem_description,
            "answered": False,
        }

    def resume(self, user_message):
        self.state["answered"] = True

    def outcome(self):
        if self.state["clarify"] and not self.state["answered"]:
            return NeedsClarification(
                message="需要澄清",
                questions=(ClarificationQuestion(id="q1", text="请补充目标"),),
            )
        return Succeeded(message="完成", summary={"status": "ok"})

    async def outputs(self, control):
        decision = self.outcome()
        if isinstance(decision, NeedsClarification):
            yield decision
            while not self.state["answered"]:
                if await control.is_cancelled():
                    raise asyncio.CancelledError
                await asyncio.sleep(0.01)
            yield self.outcome()
            return
        yield decision

    def snapshot_state(self):
        return dict(self.state)

    def restore_state(self, state):
        self.state = dict(state)


@pytest.fixture
def fake_cli_agent(monkeypatch):
    monkeypatch.setattr("decisionbrain.runtime.agent_runtime.OptimizationAgent", CliTestAgent)
