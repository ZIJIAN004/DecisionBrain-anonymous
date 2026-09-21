"""Core-owned optimization agent state machine."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from .contracts import AgentControl, CoreServices
from .feasibility_routing import has_confirmed_instance_reviews
from .models import (
    AgentInput,
    AgentOutcome,
    AgentReport,
    AgentStage,
    AgentState,
    Failed,
    NeedsClarification,
    StageCompleted,
    StageFailed,
    StageNeedsClarification,
    StageResult,
    StateUpdated,
)
from .stage_flow import (
    STAGE_SEQUENCE,
    StageTransitionError,
    apply_stage_result,
    clarification_outcome,
    run_stage,
    success_outcome,
)
from .workspace_tools import clear_scratch_directory


class AgentProtocolError(RuntimeError):
    """Core received output that does not match its current stage."""


_CLEARED_FEASIBILITY: dict[str, Any] = {
    "feasibility_review": None,
    "feasibility_rounds": 0,
    "feasibility_directives": (),
    "feasibility_audits": (),
    "instance_infeasibility_reviews": (),
    "pending_feasibility_handoff": 0,
}


class OptimizationAgent:
    """The single optimization agent used by every presentation adapter."""

    version = "6.2.0"

    def __init__(
        self,
        *,
        services: CoreServices,
        feasibility_review_enabled: bool = True,
        input_schema_enabled: bool = True,
        algorithm_design_enabled: bool = True,
        problem_contract_enabled: bool = True,
    ) -> None:
        self._services = services
        self._feasibility_review_enabled = feasibility_review_enabled
        self._input_schema_enabled = input_schema_enabled
        self._algorithm_design_enabled = algorithm_design_enabled
        self._problem_contract_enabled = problem_contract_enabled
        self._gurobi_formulator_enabled = bool(services.prompts.gurobi_formulator_system)
        self._state: AgentState | None = None
        self._resume_event = asyncio.Event()

    def initialize(self, agent_input: AgentInput) -> None:
        self._resume_event.clear()
        self._state = AgentState(problem_description=agent_input.problem_description)

    def resume(self, user_message: str) -> None:
        state = self._require_state()
        if state.current_stage is not None or not state.pending_clarification_questions:
            raise AgentProtocolError("Only a needs_clarification state can resume")
        self._state = state.append_to(
            "conversation",
            {"role": "user", "content": user_message},
        ).with_updates(
            problem_contract=None,
            algorithm_design=None,
            generated_code="",
            solver_result=None,
            explanation=None,
            failure=None,
            current_stage=STAGE_SEQUENCE[0],
            pending_clarification_questions=(),
            pending_workspace_reset=True,
            **_CLEARED_FEASIBILITY,
        )
        self._resume_event.set()

    async def outputs(
        self,
        control: AgentControl,
    ) -> AsyncIterator[AgentReport | AgentOutcome]:
        """Run stages and stream reports until the agent pauses or terminates."""

        while True:
            state = self._require_state()
            if state.current_stage is None:
                outcome = self._outcome(state)
                if isinstance(outcome, NeedsClarification):
                    self._resume_event.clear()
                    yield outcome
                    await self._wait_for_resume(control)
                    continue
                yield outcome
                return

            stage = state.current_stage
            if await control.is_cancelled():
                raise asyncio.CancelledError
            result: StageResult | None = None
            try:
                async for output in run_stage(
                    state,
                    self._services,
                    control,
                    feasibility_review_enabled=self._feasibility_review_enabled,
                    input_schema_enabled=self._input_schema_enabled,
                    algorithm_design_enabled=self._algorithm_design_enabled,
                    problem_contract_enabled=self._problem_contract_enabled,
                    gurobi_formulator_enabled=self._gurobi_formulator_enabled,
                ):
                    if isinstance(output, StateUpdated):
                        self._state = output.state
                    elif isinstance(output, (StageCompleted, StageNeedsClarification, StageFailed)):
                        if result is not None:
                            raise AgentProtocolError("Core handler produced multiple StageResult values")
                        result = output
                    elif isinstance(output, AgentReport):
                        yield output
                    else:
                        raise AgentProtocolError(
                            f"Core handler produced unknown output: {type(output).__name__}"
                        )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = StageFailed(stage=stage, message=str(exc))

            if result is None:
                raise AgentProtocolError("Core handler did not produce a StageResult")
            try:
                self._state = apply_stage_result(
                    self._require_state(),
                    result,
                    feasibility_review_enabled=self._feasibility_review_enabled,
                    algorithm_design_enabled=self._algorithm_design_enabled,
                    problem_contract_enabled=self._problem_contract_enabled,
                    gurobi_formulator_enabled=self._gurobi_formulator_enabled,
                )
            except StageTransitionError as exc:
                raise AgentProtocolError(str(exc)) from exc
            finally:
                workspace_root = self._services.workspace_root()
                if workspace_root is not None:
                    try:
                        clear_scratch_directory(workspace_root)
                    except OSError:
                        # Scratch is an optimization aid; never mask the stage result.
                        pass

    def restore_state(self, state: Mapping[str, Any]) -> None:
        self._resume_event.clear()
        self._state = AgentState.from_dict(dict(state))

    def restore_state_for_stage(
        self,
        state: Mapping[str, Any],
        stage: AgentStage | str,
    ) -> None:
        """Restore reconstructed workspace state and clear target/downstream data."""

        self.restore_state(state)
        resolved = stage if isinstance(stage, AgentStage) else AgentStage(str(stage))
        current = self._require_state()

        if resolved is not AgentStage.INTAKE and not isinstance(current.problem_definition, dict):
            raise AgentProtocolError(f"Cannot resume from {resolved.value!r}: missing problem_definition")
        if resolved in {
            AgentStage.ALGORITHM_DESIGN,
            AgentStage.SOLVING,
            AgentStage.GUROBI_FORMULATOR,
            AgentStage.FEASIBILITY_REVIEW,
            AgentStage.EXPLANATION,
        }:
            if self._problem_contract_enabled and not isinstance(current.problem_contract, dict):
                raise AgentProtocolError(f"Cannot resume from {resolved.value!r}: missing problem_contract")
        if resolved in {
            AgentStage.SOLVING,
            AgentStage.FEASIBILITY_REVIEW,
            AgentStage.EXPLANATION,
        }:
            if self._algorithm_design_enabled and not isinstance(current.algorithm_design, dict):
                raise AgentProtocolError(f"Cannot resume from {resolved.value!r}: missing algorithm_design")
        if resolved in {AgentStage.SOLVING, AgentStage.FEASIBILITY_REVIEW, AgentStage.EXPLANATION} and self._gurobi_formulator_enabled:
            if not isinstance(current.gurobi_formulation, dict):
                raise AgentProtocolError(f"Cannot resume from {resolved.value!r}: missing gurobi_formulation")
        if resolved in {
            AgentStage.FEASIBILITY_REVIEW,
            AgentStage.EXPLANATION,
        } and not isinstance(
            current.solver_result, dict
        ):
            raise AgentProtocolError(f"Cannot resume from {resolved.value!r}: missing solver_result")
        if self._feasibility_review_enabled and resolved is AgentStage.EXPLANATION:
            feasibility_review = current.feasibility_review
            instance_confirmed = (
                isinstance(feasibility_review, dict)
                and feasibility_review.get("decision") == "reject"
                and feasibility_review.get("responsibility") == "instance"
                and has_confirmed_instance_reviews(current.instance_infeasibility_reviews)
            )
            if not isinstance(feasibility_review, dict) or (
                feasibility_review.get("decision") != "accept" and not instance_confirmed
            ):
                raise AgentProtocolError(
                    f"Cannot resume from {resolved.value!r}: missing accepted review or confirmed infeasibility"
                )

        if resolved is AgentStage.INTAKE:
            current = current.with_updates(
                problem_definition=None,
                problem_contract=None,
                algorithm_design=None,
                generated_code="",
                solver_result=None,
                explanation=None,
                **_CLEARED_FEASIBILITY,
            )
        elif resolved is AgentStage.PROBLEM_CONTRACT:
            current = current.with_updates(
                problem_contract=None,
                algorithm_design=None,
                generated_code="",
                solver_result=None,
                explanation=None,
                **_CLEARED_FEASIBILITY,
            )
        elif resolved is AgentStage.ALGORITHM_DESIGN:
            current = current.with_updates(
                algorithm_design=None,
                generated_code="",
                solver_result=None,
                explanation=None,
                **_CLEARED_FEASIBILITY,
            )
        elif resolved is AgentStage.SOLVING:
            current = current.with_updates(
                generated_code="",
                solver_result=None,
                explanation=None,
                **_CLEARED_FEASIBILITY,
            )
        elif resolved is AgentStage.FEASIBILITY_REVIEW:
            current = current.with_updates(
                explanation=None,
                **_CLEARED_FEASIBILITY,
            )
        else:
            current = current.with_updates(explanation=None)

        self._state = current.with_updates(
            current_stage=resolved,
            failure=None,
            pending_clarification_questions=(),
        )
        self._resume_event.clear()

    async def _wait_for_resume(self, control: AgentControl) -> None:
        while not self._resume_event.is_set():
            if await control.is_cancelled():
                raise asyncio.CancelledError
            try:
                await asyncio.wait_for(
                    self._resume_event.wait(),
                    timeout=self._services.config.resume_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue
        self._resume_event.clear()

    def _outcome(self, state: AgentState) -> AgentOutcome:
        if state.pending_clarification_questions:
            return clarification_outcome(state)
        if state.failure is not None:
            failure = state.failure
            return Failed(
                message=str(failure.get("reason") or "Agent execution failed"),
                metadata=failure,
            )
        if state.explanation is not None:
            return success_outcome(state)
        raise AgentProtocolError("Core has no current stage or publishable paused/terminal outcome")

    def _require_state(self) -> AgentState:
        if self._state is None:
            raise AgentProtocolError("OptimizationAgent is not initialized")
        return self._state
