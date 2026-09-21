from pathlib import Path

from decisionbrain.events import RunEvent
from decisionbrain.core.contracts import CoreServices
from decisionbrain.core.models import CoreConfig
from decisionbrain.core.prompts import PromptBundle, load_prompt_bundle
from decisionbrain.core.workspace_tools import WorkspaceToolset
from decisionbrain.paths import PROMPTS_DIR


class MemoryEventSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def handle(self, event: RunEvent) -> None:
        self.events.append(event)


def fake_core_config(
    *,
    workspace_root: str | Path = "/tmp",
) -> CoreConfig:
    return CoreConfig(
        solver_timeout=120,
        prompts_dir=str(PROMPTS_DIR),
        workspace_root=str(workspace_root),
        resume_poll_interval_seconds=0.1,
    )


def fake_workspace_toolset(
    root: Path,
    *,
    timeout_s: float = 10,
    max_output_chars: int = 12_000,
    max_read_bytes: int = 1_000_000,
    process_timing_handler=None,
    components_enabled: bool = True,
) -> WorkspaceToolset:
    return WorkspaceToolset(
        root,
        timeout_s=timeout_s,
        max_output_chars=max_output_chars,
        max_list_entries=1_000,
        default_list_entries=200,
        max_read_bytes=max_read_bytes,
        process_timing_handler=process_timing_handler,
        components_enabled=components_enabled,
    )


def fake_core_services(
    *,
    llm=None,
    config: CoreConfig | None = None,
    prompts: PromptBundle | None = None,
    workspace_toolset=None,
) -> CoreServices:
    return CoreServices(
        llm=llm or object(),
        config=config or fake_core_config(),
        prompts=prompts or load_prompt_bundle(PROMPTS_DIR),
        workspace_toolset=workspace_toolset,
    )
