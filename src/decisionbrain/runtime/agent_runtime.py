"""Agent Runtime lifecycle orchestrator.

Creates Runs, persists configuration, snapshots inputs, initializes Core, consumes its output,
publishes events, updates records, stores results, and handles failures and cancellation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ..algorithm_library import AlgorithmToolset, LocalAlgorithmCatalog
from ..core.package_policy import DEFAULT_PACKAGE_POLICY, PackagePolicy
from ..config import Settings
from ..core import OptimizationAgent
from ..core.contracts import AgentControl, CoreAgent, CoreServices
from ..core.models import (
    AgentInput,
    AgentOutcome,
    AgentReport,
    AgentStage,
    CoreConfig,
)
from ..core.prompts import load_prompt_bundle
from ..core.workspace_tools import (
    WorkspaceToolset,
    clear_scratch_directory,
    ensure_scratch_directory,
    reset_workspace_for_stage,
)
from ..events import (
    RunEventDraft,
    CompositeEventSink,
    EVENT_DIAGNOSTIC,
    EVENT_PROGRESS_REPORTED,
    EVENT_RUN_CREATED,
    EVENT_RUN_FINISHED,
    EVENT_RUN_RESUMED,
    EVENT_RUN_STARTED,
    EVENT_USER_MESSAGE,
    EventLevel,
    EventPublisher,
    EventSink,
)
from ..exceptions import (
    AgentRuntimeError,
    ArtifactConflictError,
    ConfigurationError,
    LLMResponseError,
    UserInputError,
)
from ..infrastructure.llm_client import LLMClient
from ..run_storage import AgentHtmlReport, RunRepository, RunStatus
from ..run_storage.io import atomic_write_json
from .package_usage import PackageUsageObserver
from .timing import RunTiming
from .workspace_state import WorkspaceStateError, reconstruct_agent_state

logger = logging.getLogger(__name__)
TIMEZONE = timezone(timedelta(hours=8))


def generate_run_id() -> str:
    """Generate a ``YYYYMMDD-HHMMSSZ-xxxxxxxx`` Run ID in UTC+8."""
    timestamp = datetime.now(TIMEZONE)
    local_timestamp = timestamp.astimezone(TIMEZONE).strftime("%Y%m%d-%H%M%SZ")
    token = secrets.token_hex(4).lower()
    return f"{local_timestamp}-{token}"


# Exception type to RunStatus and error description.
_FAILURE_MAP: dict[type[Exception], tuple[RunStatus, str]] = {
    UserInputError: (RunStatus.FAILED, "User input error"),
    ConfigurationError: (RunStatus.FAILED, "Configuration error"),
    AgentRuntimeError: (RunStatus.FAILED, "Agent Runtime error"),
    LLMResponseError: (RunStatus.FAILED, "LLM request error"),
}
_NO_UPDATE = object()
EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
AgentFactory = Callable[[CoreServices], CoreAgent]


def build_core_services(
    settings: Settings,
    workspace_root: Path,
    *,
    solver_cpu_limit: int | None = None,
    input_schema_enabled: bool = True,
    algorithm_library_enabled: bool = True,
    feasibility_review_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    problem_contract_enabled: bool = True,
    components_enabled: bool = True,
    package_policy: PackagePolicy = DEFAULT_PACKAGE_POLICY,
    stage_event_handler: EventHandler | None = None,
    debug_event_handler: EventHandler | None = None,
    retry_event_handler: EventHandler | None = None,
    agent_turn_handler: EventHandler | None = None,
    timing_handler: EventHandler | None = None,
) -> CoreServices:
    """Assemble the concrete Core dependencies for one Runtime."""

    if (
        not algorithm_design_enabled
        and not components_enabled
        and algorithm_library_enabled
        and package_policy.pool != "gurobi-only"
    ):
        raise ConfigurationError(
            "Gurobi Formulator workflow requires package_pool=gurobi-only"
        )

    root = workspace_root.expanduser().resolve()
    config = CoreConfig(
        solver_timeout=settings.solver_timeout,
        prompts_dir=str(settings.opt_prompts_dir),
        workspace_root=str(root),
        resume_poll_interval_seconds=settings.opt_agent_resume_poll_interval_seconds,
    )
    llm = LLMClient(settings)
    llm.set_debug_stream_handler(debug_event_handler)
    llm.set_retry_event_handler(retry_event_handler)
    return CoreServices(
        llm=llm,
        config=config,
        prompts=load_prompt_bundle(
            config.prompts_dir,
            input_schema_enabled=input_schema_enabled,
            algorithm_library_enabled=algorithm_library_enabled,
            feasibility_review_enabled=feasibility_review_enabled,
            components_enabled=components_enabled,
            algorithm_design_enabled=algorithm_design_enabled,
            problem_contract_enabled=problem_contract_enabled,
        ),
        workspace_toolset=WorkspaceToolset(
            root,
            timeout_s=config.solver_timeout,
            solver_cpu_limit=solver_cpu_limit or settings.solver_cpu_budget,
            max_output_chars=settings.opt_workspace_max_output_chars,
            max_list_entries=settings.opt_workspace_max_list_entries,
            default_list_entries=settings.opt_workspace_default_list_entries,
            max_read_bytes=settings.opt_workspace_max_read_bytes,
            process_timing_handler=timing_handler,
            problem_contract_enabled=problem_contract_enabled,
            components_enabled=components_enabled,
            package_policy=package_policy,
        ),
        # Ablation 2 omits the toolset. CoreServices already maps None to no tools and an empty
        # summary, preserving LocalAlgorithmCatalog's useful fail-fast behavior for default arms.
        algorithm_toolset=(
            AlgorithmToolset(
                LocalAlgorithmCatalog(
                    settings.opt_algorithm_manifests_dir,
                    allowed_package_ids=package_policy.allowed_package_ids,
                )
            )
            if algorithm_library_enabled
            else None
        ),
        package_policy=package_policy,
        stage_event_handler=stage_event_handler,
        debug_event_handler=debug_event_handler,
        agent_turn_handler=agent_turn_handler,
        timing_handler=timing_handler,
    )


class AgentRuntime:
    """Orchestrate an Agent Run lifecycle.

    Each instance executes run() once and maps one-to-one to a Run record. Instances cannot
    be reused concurrently.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        repository: RunRepository | None = None,
        agent_factory: AgentFactory | None = None,
        solver_cpu_limit: int | None = None,
        input_schema_enabled: bool = True,
        algorithm_library_enabled: bool = True,
        feasibility_review_enabled: bool = True,
        algorithm_design_enabled: bool = True,
        problem_contract_enabled: bool = True,
        components_enabled: bool = True,
        package_policy: PackagePolicy = DEFAULT_PACKAGE_POLICY,
    ) -> None:
        self._settings = settings or (repository.settings if repository is not None else Settings())
        self._solver_cpu_limit = solver_cpu_limit
        self._input_schema_enabled = input_schema_enabled
        self._algorithm_library_enabled = algorithm_library_enabled
        self._feasibility_review_enabled = feasibility_review_enabled
        self._algorithm_design_enabled = algorithm_design_enabled
        self._problem_contract_enabled = problem_contract_enabled
        self._components_enabled = components_enabled
        self._package_policy = package_policy
        self._repository = repository or RunRepository(self._settings)
        self._agent_factory = agent_factory or (
            lambda services: OptimizationAgent(
                services=services,
                algorithm_design_enabled=algorithm_design_enabled,
                problem_contract_enabled=problem_contract_enabled,
            )
        )
        self._run_id: str | None = None
        self._status = RunStatus.CREATED
        self._cancelled = False
        self._active_agent: CoreAgent | None = None
        self._html_report: AgentHtmlReport | None = None
        self._timing: RunTiming | None = None
        self._package_usage: PackageUsageObserver | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def repository(self) -> RunRepository:
        return self._repository

    @property
    def llm_client(self) -> Any | None:
        """Expose the run's configured client to post-solver adapters."""
        agent = self._active_agent
        services = getattr(agent, "_services", None)
        return getattr(services, "llm", None)

    def _require_run_id(self) -> str:
        if self._run_id is None:
            raise AgentRuntimeError("Runtime has not created a Run")
        return self._run_id

    def _run_path(self) -> Path:
        return self._repository.run_path(self._require_run_id())

    def _create_agent(self, workspace: Path, publisher: EventPublisher) -> CoreAgent:
        report = self._html_report
        package_usage = self._package_usage

        async def record_turn(turn: dict[str, Any]) -> None:
            if report is not None:
                await report.record_turn(turn)

        async def record_timing(event: dict[str, Any]) -> None:
            if self._timing is not None:
                await self._timing.record(event)
            if package_usage is not None:
                await package_usage.record_event(event)

        services = build_core_services(
            self._settings,
            workspace,
            solver_cpu_limit=self._solver_cpu_limit,
            input_schema_enabled=self._input_schema_enabled,
            algorithm_library_enabled=self._algorithm_library_enabled,
            feasibility_review_enabled=self._feasibility_review_enabled,
            algorithm_design_enabled=self._algorithm_design_enabled,
            problem_contract_enabled=self._problem_contract_enabled,
            components_enabled=self._components_enabled,
            package_policy=self._package_policy,
            stage_event_handler=self._make_stage_event_handler(publisher),
            debug_event_handler=self._make_llm_debug_handler(publisher),
            retry_event_handler=self._make_llm_retry_handler(publisher),
            agent_turn_handler=record_turn,
            timing_handler=record_timing,
        )
        return self._agent_factory(services)

    def cancel(self) -> None:
        """Request cancellation of the current Runtime loop."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        *,
        sink: EventSink,
        workspace: Path,
        command: str,
        argv: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
        initial_input: str,
        max_snapshot_file_size_bytes: int | None = None,
        start_stage: AgentStage | None = None,
    ) -> None:
        """Start and maintain a Run runtime loop.

        Entering this method starts the Run. Returning means the Run reached a
        terminal state or was cancelled. The loop waits for ``submit_user_message``
        whenever user input is required.

        Args:
            sink: Event destination. The caller composes persistence, terminal, or SSE sinks.
            workspace: Initial workspace copied before execution.
            command: Command that started the Run.
            argv: Command arguments.
            config: Parsed configuration.
            initial_input: Initial user input.
            max_snapshot_file_size_bytes: Per-file snapshot size limit.
            start_stage: Initial stage. ``None`` or intake starts normally. Starting later
                requires all upstream artifacts to exist in the workspace.
        """

        resolved_config = dict(config or {})
        resolved_config["_runtime_execute_from_snapshot"] = True
        run_id = generate_run_id()
        self._run_id = run_id
        self._repository.create_run_record(
            run_id=run_id,
            workspace=workspace,
            command=command,
            argv=list(argv),
            config=resolved_config,
            max_snapshot_file_size_bytes=max_snapshot_file_size_bytes,
        )

        execution_workspace = self._run_path() / "workspace"
        ensure_scratch_directory(execution_workspace)
        self._html_report = AgentHtmlReport()
        self._timing = RunTiming() if resolved_config.get("timing_enabled", True) else None
        self._package_usage = self._build_package_usage_observer(
            execution_workspace,
            enabled=bool(resolved_config.get("algorithm_package_stats_enabled", True)),
        )
        self._event_publisher = EventPublisher(
            CompositeEventSink(sink, self._html_report),
            run_id=run_id,
        )
        publisher = self._event_publisher

        try:
            agent = self._create_agent(execution_workspace, publisher)
            await publisher.publish(RunEventDraft(type=EVENT_RUN_CREATED, message="Run created"))
            self._update_status(RunStatus.RUNNING)
            await publisher.publish(RunEventDraft(type=EVENT_RUN_STARTED, message="Agent started"))

            snapshot_payload: dict[str, Any] = {}
            snapshot_report = self._run_path() / "input.snapshot.json"
            if snapshot_report.is_file():
                snapshot_payload = json.loads(snapshot_report.read_text(encoding="utf-8"))

            await publisher.publish(
                RunEventDraft(
                    type=EVENT_DIAGNOSTIC,
                    level=EventLevel.DEBUG,
                    message="Runtime input and configuration diagnostics",
                    payload={
                        "agent_version": agent.version,
                        "source_workspace": str(workspace.expanduser().resolve()),
                        "execution_workspace": str(execution_workspace),
                        "run_directory": str(self._run_path()),
                        "snapshot": snapshot_payload,
                    },
                )
            )

            if start_stage is None or start_stage is AgentStage.INTAKE:
                agent.initialize(AgentInput(problem_description=initial_input))
            else:
                # Later-stage starts reconstruct state from existing workspace files without a parent Run.
                restored = reconstruct_agent_state(
                    execution_workspace,
                    start_stage,
                    problem_contract_enabled=self._problem_contract_enabled,
                    algorithm_design_enabled=self._algorithm_design_enabled,
                )
                agent.restore_state_for_stage(restored, start_stage)
                await publisher.publish(
                    RunEventDraft(
                        type=EVENT_DIAGNOSTIC,
                        level=EventLevel.DEBUG,
                        message=f"Skipped stages before {start_stage.value}",
                        payload={"start_stage": start_stage.value},
                    )
                )
            await self._drive_agent(agent)
        except asyncio.CancelledError:
            if self._status is RunStatus.NEEDS_CLARIFICATION and not self._cancelled:
                return
            await self._finalize_cancelled()
            # Cancellation is a normal terminal path; return CANCELLED without re-raising.
        except Exception as exc:
            await self._finalize_failed(exc)
        finally:
            try:
                clear_scratch_directory(execution_workspace)
            except OSError:
                logger.exception("Run %s scratch cleanup failed", self._run_id)
            self._write_html_report()
            self._write_timing_report()
            self._write_package_usage_report()
            if self._status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
                self._active_agent = None

    async def submit_user_message(
        self,
        *,
        user_message: str,
    ) -> None:
        """Submit user feedback and let the Runtime decide whether to resume.

        Args:
            user_message: User feedback submitted by the presentation layer.
        Raises:
            RunStateError: The current Run does not accept user feedback.
        """
        from ..exceptions import RunStateError

        if self._status is not RunStatus.NEEDS_CLARIFICATION:
            raise RunStateError(
                f"Runtime status is {self._status.value}; only "
                f"{RunStatus.NEEDS_CLARIFICATION.value} accepts user input"
            )
        agent = self._active_agent
        if agent is None:
            raise RunStateError("Runtime has no associated active Core Agent")
        self._update_status(RunStatus.RUNNING)
        await self._event_publisher.publish(
            RunEventDraft(
                type=EVENT_USER_MESSAGE,
                stage=AgentStage.INTAKE,
                message=user_message,
            )
        )
        agent.resume(user_message)

    async def resume_run(
        self,
        *,
        sink: EventSink,
        parent_run_id: str,
        resume_stage: AgentStage | str,
        command: str,
        argv: Sequence[str] = (),
        config: Mapping[str, Any] | None = None,
        max_snapshot_file_size_bytes: int | None = None,
    ) -> None:
        """Derive a new Run from an existing workspace and resume at a stage.

        Args:
            sink: Event destination.
            parent_run_id: Source Run to reuse.
            resume_stage: Stage at which execution resumes.
            command: Command that triggered the resume.
            argv: Command arguments.
            config: Parsed configuration.
            max_snapshot_file_size_bytes: Per-file snapshot size limit.
        """
        from ..exceptions import UserInputError

        # -- validate resume_stage --------------------------------------------
        stage: AgentStage
        if isinstance(resume_stage, AgentStage):
            stage = resume_stage
        else:
            try:
                stage = AgentStage(str(resume_stage))
            except ValueError as exc:
                raise UserInputError(f"Invalid stage: {resume_stage!r}") from exc

        # -- read parent data -------------------------------------------------
        try:
            parent_record = self._repository.read_record(parent_run_id)
        except Exception as exc:
            raise UserInputError(f"Cannot read parent Run {parent_run_id}") from exc

        parent_workspace = self._repository.run_path(parent_run_id) / "workspace"
        if not parent_workspace.is_dir():
            raise UserInputError(f"Parent Run {parent_run_id} has no workspace/ directory")
        try:
            restored_state = reconstruct_agent_state(
                parent_workspace,
                stage,
                problem_contract_enabled=self._problem_contract_enabled,
                algorithm_design_enabled=self._algorithm_design_enabled,
            )
        except WorkspaceStateError as exc:
            raise UserInputError(f"Cannot restore state from parent Run artifacts: {exc}") from exc

        # -- create derived run -----------------------------------------------
        resolved_config = dict(config or {})
        resolved_config["_runtime_execute_from_snapshot"] = True
        resolved_config["_resume_parent_run_id"] = parent_run_id
        resolved_config["_resume_stage"] = stage.value
        run_id = generate_run_id()
        self._run_id = run_id

        # Create a new run record with an empty temp workspace so
        # snapshot_workspace produces a minimal workspace/ stub. We then replace
        # it wholesale with the parent's workspace/ via copytree.
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_ws:
            self._repository.create_run_record(
                run_id=run_id,
                workspace=Path(tmp_ws),
                command=command,
                argv=list(argv),
                config=resolved_config,
                parent_run_id=parent_run_id,
                max_snapshot_file_size_bytes=max_snapshot_file_size_bytes,
            )

        # Replace the stub workspace/ with a full copy of the parent's workspace/.
        new_workspace = self._run_path() / "workspace"
        shutil.rmtree(new_workspace)
        shutil.copytree(parent_workspace, new_workspace)

        self._html_report = AgentHtmlReport()
        self._timing = RunTiming() if resolved_config.get("timing_enabled", True) else None
        self._package_usage = self._build_package_usage_observer(
            new_workspace,
            enabled=bool(resolved_config.get("algorithm_package_stats_enabled", True)),
        )
        self._event_publisher = EventPublisher(
            CompositeEventSink(sink, self._html_report),
            run_id=run_id,
        )
        publisher = self._event_publisher

        execution_workspace = self._run_path() / "workspace"
        ensure_scratch_directory(execution_workspace)
        try:
            # -- cleanup downstream workspace artifacts -----------------------
            reset_workspace_for_stage(execution_workspace, stage)
            agent = self._create_agent(execution_workspace, publisher)

            await publisher.publish(RunEventDraft(type=EVENT_RUN_CREATED, message="Run created"))
            await publisher.publish(
                RunEventDraft(
                    type=EVENT_RUN_RESUMED,
                    message=f"Resumed from {stage.value} of {parent_run_id}",
                    payload={
                        "parent_run_id": parent_run_id,
                        "resume_stage": stage.value,
                        "source_status": parent_record.status.value,
                        "execution_workspace": str(execution_workspace),
                    },
                )
            )
            self._update_status(RunStatus.RUNNING)
            await publisher.publish(
                RunEventDraft(type=EVENT_RUN_STARTED, message="Agent started (resume)")
            )

            # -- restore agent state for the target stage ---------------------
            agent.restore_state_for_stage(restored_state, stage)

            await self._drive_agent(agent)
        except asyncio.CancelledError:
            if self._status is RunStatus.NEEDS_CLARIFICATION and not self._cancelled:
                return
            await self._finalize_cancelled()
        except Exception as exc:
            await self._finalize_failed(exc)
        finally:
            try:
                clear_scratch_directory(execution_workspace)
            except OSError:
                logger.exception("Resumed run %s scratch cleanup failed", self._run_id)
            self._write_html_report()
            self._write_timing_report()
            self._write_package_usage_report()
            if self._status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}:
                self._active_agent = None

    async def _drive_agent(self, agent: CoreAgent) -> None:
        runtime = self

        class RuntimeControl(AgentControl):
            async def is_cancelled(self) -> bool:
                return runtime._cancelled

        self._active_agent = agent
        async for item in agent.outputs(RuntimeControl()):
            if isinstance(item, AgentReport):
                await self._handle_report(item)
                continue
            if not isinstance(item, AgentOutcome):
                raise AgentRuntimeError(f"Core Agent produced unknown output: {type(item).__name__}")
            should_continue = await self._handle_outcome(agent, item)
            if not should_continue:
                return

    def _update_status(
        self,
        status: RunStatus,
        *,
        input_summary: Mapping[str, Any] | object = _NO_UPDATE,
        result_summary: Mapping[str, Any] | None | object = _NO_UPDATE,
        error_summary: str | None | object = _NO_UPDATE,
    ) -> None:
        """Update in-memory Runtime state and persist the RunRecord."""

        self._status = status
        updates: dict[str, Any] = {"status": status}
        if input_summary is not _NO_UPDATE:
            updates["input_summary"] = input_summary
        if result_summary is not _NO_UPDATE:
            updates["result_summary"] = result_summary
        if error_summary is not _NO_UPDATE:
            updates["error_summary"] = error_summary
        self._repository.update_record(self._require_run_id(), **updates)

    def _write_html_report(self) -> None:
        report = self._html_report
        if report is None or self._run_id is None:
            return
        try:
            record = self._repository.read_record(self._run_id)
            report.write(self._run_path() / "agent-run.html", record)
        except Exception:
            logger.exception("Failed to generate HTML report for Run %s", self._run_id)

    def _write_timing_report(self) -> None:
        timing = self._timing
        if timing is None or self._run_id is None:
            return
        try:
            atomic_write_json(self._run_path() / "timing.json", timing.to_dict())
        except Exception:
            logger.exception("Run %s timing report generation failed", self._run_id)

    def _build_package_usage_observer(
        self, workspace: Path, *, enabled: bool
    ) -> PackageUsageObserver | None:
        if not enabled:
            return None
        try:
            catalog = LocalAlgorithmCatalog(self._settings.opt_algorithm_manifests_dir)
            return PackageUsageObserver(workspace, catalog)
        except Exception:
            logger.exception("Package usage observer initialization failed")
            return None

    def _write_package_usage_report(self) -> None:
        observer = self._package_usage
        if observer is None or self._run_id is None:
            return
        try:
            atomic_write_json(self._run_path() / "package-usage.json", observer.to_dict())
        except Exception:
            logger.exception("Run %s package usage report generation failed", self._run_id)

    async def _handle_report(
        self,
        report: AgentReport,
    ) -> None:
        """Write each Core report as one event or artifact."""
        from ..core.models import (
            ArtifactProduced,
            AssistantMessage,
            StageFinished,
            StageStarted,
            Warning,
        )

        publisher = self._event_publisher
        if isinstance(report, StageStarted):
            if self._timing is not None:
                self._timing.stage_started(report.stage.value)
            await publisher.publish(
                RunEventDraft(
                    type="stage_started",
                    stage=report.stage,
                    message=report.message,
                )
            )
        elif isinstance(report, StageFinished):
            if self._timing is not None:
                self._timing.stage_finished(report.stage.value)
            await publisher.publish(
                RunEventDraft(
                    type="stage_finished",
                    stage=report.stage,
                    message=report.message,
                )
            )
        elif isinstance(report, AssistantMessage):
            payload = dict(report.metadata)
            if report.stream is not None:
                payload["stream"] = report.stream.value
            await publisher.publish(
                RunEventDraft(
                    type="assistant_message",
                    stage=report.stage,
                    message=report.message,
                    payload=payload,
                )
            )
        elif isinstance(report, Warning):
            await publisher.publish(
                RunEventDraft(
                    type="warning",
                    stage=report.stage,
                    level=EventLevel.WARNING,
                    message=report.message,
                    payload=report.metadata,
                )
            )
        elif isinstance(report, ArtifactProduced):
            for draft in report.artifacts:
                relative_path = draft.relative_path
                revision = 1
                while True:
                    try:
                        metadata = self._repository.write_artifact_text(
                            self._require_run_id(),
                            relative_path,
                            draft.content,
                            media_type=draft.media_type,
                            category=draft.category,
                            producer="agent",
                            description=draft.description,
                        )
                        break
                    except ArtifactConflictError:
                        revision += 1
                        relative_path = self._artifact_revision_path(draft.relative_path, revision)
                await publisher.publish(
                    RunEventDraft(
                        type="artifact_created",
                        stage=report.stage,
                        message=f"Artifact created: {relative_path}",
                        payload=metadata.model_dump(mode="json"),
                    )
                )
        else:
            raise AgentRuntimeError(f"Core Agent produced unknown report: {type(report).__name__}")

    @staticmethod
    def _artifact_revision_path(relative_path: str, revision: int) -> str:
        """Preserve immutable remediation artifacts without overwriting prior rounds."""
        path = PurePosixPath(relative_path)
        suffix = "".join(path.suffixes)
        basename = path.name[: -len(suffix)] if suffix else path.name
        return (path.parent / f"{basename}-{revision}{suffix}").as_posix()

    async def _handle_outcome(
        self,
        agent: CoreAgent,
        outcome: AgentOutcome,
    ) -> bool:
        from ..core.models import Failed, NeedsClarification, Succeeded

        publisher = self._event_publisher
        if isinstance(outcome, NeedsClarification):
            questions = [item.model_dump(mode="json") for item in outcome.questions]
            self._update_status(RunStatus.NEEDS_CLARIFICATION)
            await publisher.publish(
                RunEventDraft(
                    type="clarification_requested",
                    stage=AgentStage.INTAKE,
                    message=outcome.message,
                    payload={"questions": questions},
                )
            )
            return True
        elif isinstance(outcome, Failed):
            await publisher.publish(
                RunEventDraft(
                    type="error",
                    stage=outcome.stage,
                    level=EventLevel.ERROR,
                    message=outcome.message,
                    payload=outcome.metadata,
                )
            )
            self._update_status(RunStatus.FAILED, error_summary=outcome.message)
            await publisher.publish(
                RunEventDraft(
                    type=EVENT_RUN_FINISHED,
                    level=EventLevel.ERROR,
                    message="Run execution failed",
                    payload={"status": "failed", "error": outcome.message},
                )
            )
            return False
        elif isinstance(outcome, Succeeded):
            self._update_status(RunStatus.SUCCEEDED, result_summary=outcome.summary)
            await publisher.publish(
                RunEventDraft(
                    type=EVENT_RUN_FINISHED,
                    message=outcome.message,
                    payload={**outcome.summary, "status": "succeeded"},
                )
            )
            return False
        else:
            raise AgentRuntimeError(f"Core Agent produced unknown outcome: {type(outcome).__name__}")

    def _make_stage_event_handler(self, publisher: EventPublisher) -> EventHandler:
        async def publish_stage_event(event: dict[str, Any]) -> None:
            kind = str(event.get("kind") or "")
            stage = self._stage_for_llm_role(str(event.get("agent_stage") or ""))
            payload = event.get("payload") or {}
            if not isinstance(payload, dict):
                payload = {"value": payload}
            if kind == EVENT_PROGRESS_REPORTED:
                await publisher.publish(
                    RunEventDraft(
                        type=EVENT_PROGRESS_REPORTED,
                        stage=stage,
                        message=str(payload.get("summary") or "Progress updated"),
                        payload=dict(payload),
                    )
                )
                return
            if kind == "agent_error":
                await publisher.publish(
                    RunEventDraft(
                        type="agent_error",
                        stage=stage,
                        level=EventLevel.WARNING,
                        message=str(payload.get("message") or "Recoverable Agent error"),
                        payload=dict(payload),
                    )
                )
                return

        return publish_stage_event

    @staticmethod
    def _make_llm_retry_handler(publisher: EventPublisher) -> EventHandler:
        async def publish_llm_retry(event: dict[str, Any]) -> None:
            await publisher.publish(
                RunEventDraft(
                    type="agent_error",
                    level=EventLevel.WARNING,
                    message=f"LLM network call failed; preparing attempt {event.get('next_attempt')}",
                    payload={
                        "flow_section": "llm_transport",
                        "error_kind": "llm_network_retry",
                        **event,
                    },
                )
            )

        return publish_llm_retry

    # ------------------------------------------------------------------
    # Internal: LLM debug stream
    # ------------------------------------------------------------------

    def _make_llm_debug_handler(self, publisher: EventPublisher) -> EventHandler | None:
        if not self._settings.debug:
            return None

        async def publish_llm_debug_event(event: dict[str, Any]) -> None:
            role = str(event.get("role") or "llm")
            stage = self._stage_for_llm_role(role)
            kind = str(event.get("kind") or "")
            payload = {key: value for key, value in event.items() if key != "text"}
            if kind == "started":
                await publisher.publish(
                    RunEventDraft(
                        type="llm_stream_started",
                        stage=stage,
                        level=EventLevel.DEBUG,
                        message=f"LLM {role} stream started",
                        payload=payload,
                    )
                )
                return
            if kind == "chunk":
                await publisher.publish(
                    RunEventDraft(
                        type="llm_stream_chunk",
                        stage=stage,
                        level=EventLevel.DEBUG,
                        message=str(event.get("text") or ""),
                        payload=payload,
                    )
                )
                return
            if kind == "tool_call_delta":
                # StageAgent publishes tool calls from execution; streamed deltas are merged only.
                return
            if kind == "tool_call_started":
                arguments = self._preview_tool_arguments(event.get("arguments") or {})
                await publisher.publish(
                    RunEventDraft(
                        type="tool_call",
                        stage=stage,
                        level=EventLevel.DEBUG,
                        message=f"tool call: {event.get('tool_name') or ''}",
                        payload={
                            "tool_call_id": str(event.get("tool_call_id") or ""),
                            "tool_name": str(event.get("tool_name") or ""),
                            "arguments": arguments,
                        },
                    )
                )
                return
            if kind == "tool_result":
                result = str(event.get("result") or "")
                metadata = self._tool_result_metadata(result)
                await publisher.publish(
                    RunEventDraft(
                        type="tool_result",
                        stage=stage,
                        level=EventLevel.DEBUG,
                        message=self._preview_text(result, 500),
                        payload={
                            "tool_call_id": str(event.get("tool_call_id") or ""),
                            "tool_name": str(event.get("tool_name") or ""),
                            **metadata,
                        },
                    )
                )
                return
            if kind == "finished":
                status = str(event.get("status") or "ok")
                await publisher.publish(
                    RunEventDraft(
                        type="llm_stream_finished",
                        stage=stage,
                        level=EventLevel.DEBUG,
                        message=f"LLM {role} stream completed ({status})",
                        payload=payload,
                    )
                )

        return publish_llm_debug_event

    @staticmethod
    def _stage_for_llm_role(role: str) -> AgentStage | None:
        return {
            "intake": AgentStage.INTAKE,
            "problem_contract": AgentStage.PROBLEM_CONTRACT,
            "algorithm_design": AgentStage.ALGORITHM_DESIGN,
            "code": AgentStage.SOLVING,
            "verify": AgentStage.SOLVING,
            "debug": AgentStage.SOLVING,
            "explain": AgentStage.EXPLANATION,
            "solving": AgentStage.SOLVING,
            "feasibility_review": AgentStage.FEASIBILITY_REVIEW,
            "explanation": AgentStage.EXPLANATION,
        }.get(role)

    @staticmethod
    def _preview_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "...[truncated]"

    @classmethod
    def _preview_tool_arguments(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"type": "string", "chars": len(value), "preview": cls._preview_text(value, 200)}
        if isinstance(value, dict):
            return {str(key): cls._preview_tool_arguments(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._preview_tool_arguments(item) for item in value[:20]]
        return value

    @staticmethod
    def _tool_result_metadata(result: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "result_chars": len(result),
            "truncated": "[truncated]" in result,
        }
        lines = result.splitlines()
        for line in lines[:3]:
            if line.startswith("Exit code:"):
                try:
                    metadata["exit_code"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Wall time:"):
                text = line.split(":", 1)[1].strip().split(" ", 1)[0]
                try:
                    metadata["wall_time_seconds"] = float(text)
                except ValueError:
                    pass
        return metadata

    # ------------------------------------------------------------------
    # Internal: Finalization
    # ------------------------------------------------------------------

    async def _finalize_cancelled(self) -> None:
        """Handle cancellation, persist state, and publish RUN_FINISHED."""
        publisher = self._event_publisher
        self._cancelled = True
        try:
            self._update_status(RunStatus.CANCELLED)
        except Exception:
            logger.exception("Failed to persist cancellation for Run %s", publisher.run_id)
        try:
            await publisher.publish(
                RunEventDraft(
                    type=EVENT_RUN_FINISHED,
                    message="Run cancelled by user",
                    payload={"status": "cancelled"},
                )
            )
        except Exception:
            logger.exception("Failed to publish cancellation event for Run %s", publisher.run_id)

    async def _finalize_failed(
        self,
        exc: Exception,
    ) -> None:
        """Handle failure, persist state, and publish an ERROR event."""
        publisher = self._event_publisher
        matched_status, matched_desc = _FAILURE_MAP.get(
            type(exc), (RunStatus.FAILED, "Unknown internal error")
        )
        error_msg = f"{matched_desc}：{exc}"

        try:
            self._update_status(matched_status, error_summary=error_msg)
            await publisher.publish(
                RunEventDraft(
                    type="error",
                    level=EventLevel.ERROR,
                    message=error_msg,
                )
            )
            await publisher.publish(
                RunEventDraft(
                    type=EVENT_RUN_FINISHED,
                    message=f"Run failed: {matched_desc}",
                    payload={"status": "failed", "error": str(exc)},
                )
            )
        except Exception:
            logger.exception("Failed to finalize failed Run %s", publisher.run_id)

        # Propagate exceptions by default so the CLI can map exit codes.
        raise
