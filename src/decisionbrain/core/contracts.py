"""Port protocols between Core and runtime/infrastructure."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .package_policy import DEFAULT_PACKAGE_POLICY, PackagePolicy

if TYPE_CHECKING:
    from .models import (
        AgentInput,
        AgentOutcome,
        AgentReport,
        AgentStage,
        CoreConfig,
    )
    from .prompts import PromptBundle


@runtime_checkable
class AgentControl(Protocol):
    async def is_cancelled(self) -> bool: ...


@dataclass(frozen=True)
class CoreServices:
    llm: Any  # LLMClient (concrete, imported at runtime to avoid circular deps)
    config: CoreConfig
    prompts: PromptBundle
    workspace_toolset: Any = None
    algorithm_toolset: Any = None
    package_policy: PackagePolicy = DEFAULT_PACKAGE_POLICY
    stage_event_handler: Any = None  # set by Runtime for non-terminal stage events
    debug_event_handler: Any = None  # set by Runtime for debug event streaming
    agent_turn_handler: Any = None  # set by Runtime for complete post-run transcripts
    timing_handler: Any = None  # set by Runtime for structured execution timing

    def workspace_root(self) -> Path | None:
        """Resolve the workspace root, preferring the toolset over config."""
        toolset_root = getattr(self.workspace_toolset, "root", None)
        source = toolset_root or getattr(self.config, "workspace_root", None)
        if not source:
            return None
        return Path(str(source)).expanduser().resolve()

    def workspace_tools_for(
        self,
        stage: Any,
        *,
        solving_feasibility_self_check: bool = False,
        solving_outcome_submission: bool = False,
    ) -> tuple[Any, ...]:
        if self.workspace_toolset is None:
            return ()
        return tuple(
            self.workspace_toolset.tools(
                stage=stage,
                solving_feasibility_self_check=solving_feasibility_self_check,
                solving_outcome_submission=solving_outcome_submission,
            )
        )

    def algorithm_tools_for(self, stage: Any) -> tuple[Any, ...]:
        if self.algorithm_toolset is None:
            return ()
        return tuple(self.algorithm_toolset.tools(stage=stage))

    def algorithm_catalog_summary(self) -> str:
        if self.algorithm_toolset is None:
            return '{"count": 0, "algorithms": []}'
        return str(self.algorithm_toolset.catalog_summary())


@runtime_checkable
class CoreAgent(Protocol):
    version: str

    def initialize(self, agent_input: AgentInput) -> None: ...

    def resume(self, user_message: str) -> None: ...

    def outputs(
        self,
        control: AgentControl,
    ) -> AsyncIterator[AgentReport | AgentOutcome]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...

    def restore_state_for_stage(
        self, state: Mapping[str, Any], stage: AgentStage | str
    ) -> None: ...
