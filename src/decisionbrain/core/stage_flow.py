"""Fixed stage sequence and transitions for the Core agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .feasibility_routing import (
    INSTANCE_AUDIT_KIND,
    REPAIR_AUDIT_KIND,
    annotate_feasibility_outcome,
    build_feasibility_directive,
    has_confirmed_instance_reviews,
    visible_review_history,
)
from .models import (
    AgentReport,
    AgentStage,
    AgentState,
    ClarificationQuestion,
    NeedsClarification,
    StageCompleted,
    StageFailed,
    StageNeedsClarification,
    StageResult,
    StateUpdated,
    Succeeded,
)
from .stage_agents import (
    AlgorithmDesignAgent,
    GurobiFormulatorAgent,
    ExplanationAgent,
    FeasibilityReviewAgent,
    IntakeAgent,
    ProblemContractAgent,
    SolvingAgent,
)
from .workspace_tools import (
    prepare_feasibility_handoff,
    reset_workspace_for_stage,
    write_feasibility_review_history,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .contracts import AgentControl, CoreServices
    from .stage_agent import StageAgent


STAGE_SEQUENCE = (
    AgentStage.INTAKE,
    AgentStage.PROBLEM_CONTRACT,
    AgentStage.ALGORITHM_DESIGN,
    AgentStage.GUROBI_FORMULATOR,
    AgentStage.SOLVING,
    AgentStage.FEASIBILITY_REVIEW,
    AgentStage.EXPLANATION,
)

REWIND_TARGETS: dict[str, AgentStage] = {
    "solving": AgentStage.SOLVING,
    "algorithm_design": AgentStage.ALGORITHM_DESIGN,
    "gurobi_formulator": AgentStage.GUROBI_FORMULATOR,
}
StageOutput = AgentReport | StageResult | StateUpdated


class StageTransitionError(RuntimeError):
    """The stage output does not match the current Core state."""


def next_stage(stage: AgentStage) -> AgentStage | None:
    try:
        index = STAGE_SEQUENCE.index(stage)
    except ValueError as exc:  # pragma: no cover - AgentStage limits the input set
        raise StageTransitionError(f"Unknown Core stage: {stage.value!r}") from exc
    if index + 1 == len(STAGE_SEQUENCE):
        return None
    return STAGE_SEQUENCE[index + 1]


def _instance_review_record(review: dict[str, Any]) -> dict[str, Any]:
    """Persist only the conclusion needed to confirm a repeated instance diagnosis."""

    return {
        "decision": "reject",
        "responsibility": "instance",
        "confidence": str(review.get("confidence") or ""),
        "summary": str(review.get("summary") or ""),
        "infeasibility_proof": str(review.get("infeasibility_proof") or ""),
    }


def _instance_audit_record(review: dict[str, Any], *, sequence: int) -> dict[str, Any]:
    """Record one instance verdict in the cumulative reviewer history.

    Instance rejections neither advance the repair attempt counter nor produce a
    handoff, so they carry no attempt number. They are kept so the next reviewer can
    see that this route was already taken and what it concluded, and can extend an
    incomplete infeasibility argument instead of restating it from scratch.
    """

    return {
        "kind": INSTANCE_AUDIT_KIND,
        "consecutive_instance_verdict": sequence,
        **_instance_review_record(review),
        "evidence": review.get("evidence") or [],
    }


def _prepare_pending_handoff(
    root: Path,
    attempt: int,
    directives: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Write the attempt's repair baselines and record the ones that exist.

    The prepared set is not fixed: a rejected solve that produced no candidate has
    no previous solution to copy, so the responsible stage must be told which
    baselines it actually has. Only the file names are kept; the directive already
    carries the handoff directory they live in.
    """

    index = next(
        (
            position
            for position in reversed(range(len(directives)))
            if directives[position].get("attempt") == attempt
        ),
        None,
    )
    if index is None:
        raise StageTransitionError("Missing feasibility-remediation directive for handoff")
    prepared = prepare_feasibility_handoff(root, attempt, directives[index])
    names = [item.rsplit("/", 1)[-1] for item in prepared]
    updated = {**directives[index], "baseline_files": names}
    return (*directives[:index], updated, *directives[index + 1 :])


async def run_stage(
    state: AgentState,
    services: CoreServices,
    control: AgentControl,
    *,
    feasibility_review_enabled: bool = True,
    input_schema_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    problem_contract_enabled: bool = True,
    gurobi_formulator_enabled: bool = False,
) -> AsyncIterator[StageOutput]:
    """Run the StageAgent selected by ``state.current_stage``."""

    stage = state.current_stage
    if stage is None:
        raise StageTransitionError("Current Core state has no executable stage")
    if await control.is_cancelled():
        return

    root = services.workspace_root()
    prepared_state: AgentState | None = None
    if state.pending_feasibility_handoff or state.pending_workspace_reset:
        directives = state.feasibility_directives
        if state.pending_feasibility_handoff:
            if root is None:
                # The remediation baseline exists only in the workspace; never skip it silently.
                raise StageTransitionError("A workspace is required to create a remediation handoff")
            directives = _prepare_pending_handoff(
                root,
                state.pending_feasibility_handoff,
                directives,
            )
        if root is not None and state.pending_workspace_reset:
            reset_workspace_for_stage(root, stage)
        prepared_state = state.with_updates(
            feasibility_directives=directives,
            pending_workspace_reset=False,
            pending_feasibility_handoff=0,
        )

    if root is not None:
        # Reviewer-owned history; the ACL keeps it out of both responsible stages.
        # Reproject before every stage so files cannot lag state on instance returns or terminal
        # confirmation. Do this after reset and before yielding to preserve cleanup and cancellation.
        write_feasibility_review_history(
            root,
            visible_review_history(
                state.feasibility_audits,
                instance_run=state.instance_infeasibility_reviews,
            ),
        )

    if prepared_state is not None:
        state = prepared_state
        yield StateUpdated(state=state)

    agent: StageAgent
    if stage is AgentStage.INTAKE:
        agent = IntakeAgent(services=services)
    elif stage is AgentStage.PROBLEM_CONTRACT:
        agent = ProblemContractAgent(
            services=services,
            input_schema_enabled=input_schema_enabled,
            feasibility_review_enabled=feasibility_review_enabled,
        )
    elif stage is AgentStage.ALGORITHM_DESIGN:
        agent = AlgorithmDesignAgent(
            services=services,
            problem_contract_enabled=problem_contract_enabled,
        )
    elif stage is AgentStage.GUROBI_FORMULATOR:
        if not isinstance(state.problem_contract, dict):
            yield StageFailed(stage=stage, message="Problem contract has not been generated")
            return
        if services.workspace_toolset is None:
            yield StageFailed(stage=stage, message="Workspace tools are not configured")
            return
        agent = GurobiFormulatorAgent(services=services)
    elif stage is AgentStage.SOLVING:
        if problem_contract_enabled and not isinstance(state.problem_contract, dict):
            yield StageFailed(stage=stage, message="Problem contract has not been generated")
            return
        if algorithm_design_enabled and not isinstance(state.algorithm_design, dict):
            yield StageFailed(stage=stage, message="Algorithm design has not been generated")
            return
        if gurobi_formulator_enabled and not isinstance(state.gurobi_formulation, dict):
            yield StageFailed(stage=stage, message="Gurobi formulation has not been generated")
            return
        if services.workspace_toolset is None:
            yield StageFailed(stage=stage, message="Workspace tools are not configured")
            return
        agent = SolvingAgent(
            services=services,
            feasibility_review_enabled=feasibility_review_enabled,
            input_schema_enabled=input_schema_enabled,
            algorithm_design_enabled=algorithm_design_enabled,
            problem_contract_enabled=problem_contract_enabled,
        )
    elif stage is AgentStage.FEASIBILITY_REVIEW:
        if not isinstance(state.solver_result, dict):
            yield StageFailed(stage=stage, message="Solver result has not been generated")
            return
        agent = FeasibilityReviewAgent(
            services=services,
            input_schema_enabled=input_schema_enabled,
            algorithm_design_enabled=algorithm_design_enabled,
        )
    elif stage is AgentStage.EXPLANATION:
        agent = ExplanationAgent(
            services=services,
            algorithm_design_enabled=algorithm_design_enabled,
            problem_contract_enabled=problem_contract_enabled,
        )
    else:  # pragma: no cover - exhaustive AgentStage guard
        raise StageTransitionError(f"Unknown Core stage: {stage!r}")

    async for output in agent.run(state, control):
        yield output

    if stage is AgentStage.INTAKE:
        transcript = agent.get_transcript(start_idx=1)
        if transcript:
            yield StateUpdated(state=state.with_updates(conversation=transcript))


def apply_stage_result(
    state: AgentState,
    result: StageResult,
    *,
    feasibility_review_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    problem_contract_enabled: bool = True,
    gurobi_formulator_enabled: bool = False,
) -> AgentState:
    """Apply one validated StageResult to immutable Core state."""

    stage = state.current_stage
    if stage is None or result.stage is not stage:
        waiting = stage.value if stage is not None else ""
        raise StageTransitionError(f"Stage {waiting!r} returned mismatched result {type(result).__name__}")
    if isinstance(result, StageFailed):
        return state.with_updates(
            failure={"reason": result.message, **result.metadata},
            current_stage=None,
        )
    if isinstance(result, StageNeedsClarification):
        if stage is not AgentStage.SOLVING:
            raise StageTransitionError(f"Stage {stage.value!r} does not support clarification")
        return pause_for_clarification(
            state,
            result.message,
            state.problem_definition,
            result.questions,
        )
    if not isinstance(result, StageCompleted):
        raise StageTransitionError(f"Unknown StageResult: {type(result).__name__}")

    if stage is AgentStage.INTAKE:
        return _apply_intake_result(
            state,
            result,
            problem_contract_enabled=problem_contract_enabled,
            algorithm_design_enabled=algorithm_design_enabled,
            gurobi_formulator_enabled=gurobi_formulator_enabled,
        )
    if stage is AgentStage.PROBLEM_CONTRACT:
        return state.with_updates(
            problem_contract=result.data["problem_contract"],
            current_stage=(
                AgentStage.ALGORITHM_DESIGN
                if algorithm_design_enabled
                else (
                    AgentStage.GUROBI_FORMULATOR
                    if gurobi_formulator_enabled
                    else AgentStage.SOLVING
                )
            ),
        )
    if stage is AgentStage.GUROBI_FORMULATOR:
        return state.with_updates(
            gurobi_formulation=result.data["gurobi_formulation"],
            current_stage=AgentStage.SOLVING,
        )
    if stage is AgentStage.ALGORITHM_DESIGN:
        return state.with_updates(
            algorithm_design=result.data["algorithm_design"],
            # Formulator is an explicit A-arm stage, not part of Full/D.
            current_stage=(
                AgentStage.GUROBI_FORMULATOR
                if gurobi_formulator_enabled
                else AgentStage.SOLVING
            ),
        )
    if stage is AgentStage.SOLVING:
        updates = {
            "solver_result": result.data["result"],
            "feasibility_review": None,
            "instance_infeasibility_reviews": (),
            "current_stage": (
                AgentStage.FEASIBILITY_REVIEW
                if feasibility_review_enabled
                else AgentStage.EXPLANATION
            ),
        }
        generated_code = result.data.get("generated_code")
        if isinstance(generated_code, str):
            updates["generated_code"] = generated_code
        return state.with_updates(**updates)
    if stage is AgentStage.FEASIBILITY_REVIEW:
        feasibility_review = result.data["feasibility_review"]
        feasibility_check = result.data["feasibility_check"]
        solver_result = dict(state.solver_result or {})
        if isinstance(feasibility_check, dict):
            solver_result["feasibility_result_file"] = "feasibility_result.json"
            solver_result["feasibility_check"] = feasibility_check
        else:
            solver_result.pop("feasibility_result_file", None)
            solver_result.pop("feasibility_check", None)
        directives = annotate_feasibility_outcome(
            state.feasibility_directives,
            review=feasibility_review,
        )
        if feasibility_review.get("decision") == "accept":
            return state.with_updates(
                feasibility_review=feasibility_review,
                solver_result=solver_result,
                feasibility_directives=directives,
                instance_infeasibility_reviews=(),
                current_stage=next_stage(stage),
            )
        reported_responsibility = str(feasibility_review.get("responsibility") or "")
        responsibility = reported_responsibility
        routed_review = feasibility_review
        if responsibility == "algorithm_design" and not algorithm_design_enabled:
            # There is no Algorithm Design stage in either no-design arm. Route the
            # legacy reviewer label to the arm's actual responsible stage.
            responsibility = (
                AgentStage.GUROBI_FORMULATOR.value
                if gurobi_formulator_enabled
                else AgentStage.SOLVING.value
            )
            routed_review = {**feasibility_review, "responsibility": responsibility}
        target = REWIND_TARGETS.get(responsibility)
        if target is not None:
            attempt = state.feasibility_rounds + 1
            directive = build_feasibility_directive(
                attempt_index=attempt,
                review=routed_review,
                solver_result=solver_result,
            )
            audit = {
                "kind": REPAIR_AUDIT_KIND,
                "attempt": attempt,
                "responsibility": responsibility,
                "feasibility_review": feasibility_review,
                "feasibility_check": feasibility_check,
            }
            if reported_responsibility != responsibility:
                audit["reported_responsibility"] = reported_responsibility
            updates: dict[str, object] = {
                "feasibility_review": None,
                "feasibility_rounds": attempt,
                "feasibility_directives": (*directives, directive),
                "feasibility_audits": (*state.feasibility_audits, audit),
                "instance_infeasibility_reviews": (),
                "pending_feasibility_handoff": attempt,
                "pending_workspace_reset": True,
                "current_stage": target,
                "failure": None,
                "explanation": None,
                "solver_result": None,
                "generated_code": "",
            }
            if target is AgentStage.ALGORITHM_DESIGN:
                updates["algorithm_design"] = None
            if target is AgentStage.GUROBI_FORMULATOR:
                updates["gurobi_formulation"] = None
            return state.with_updates(**updates)
        if responsibility != "instance":
            raise StageTransitionError(
                f"Feasibility Review returned unknown responsibility: {responsibility!r}"
            )
        instance_reviews = (
            *state.instance_infeasibility_reviews,
            _instance_review_record(feasibility_review),
        )
        # Keep only the current consecutive instance-review segment; audits remain cumulative.
        audits = (
            *state.feasibility_audits,
            _instance_audit_record(feasibility_review, sequence=len(instance_reviews)),
        )
        if has_confirmed_instance_reviews(instance_reviews):
            return state.with_updates(
                feasibility_review=feasibility_review,
                solver_result=solver_result,
                feasibility_directives=directives,
                feasibility_audits=audits,
                instance_infeasibility_reviews=instance_reviews,
                failure=None,
                pending_workspace_reset=False,
                current_stage=AgentStage.EXPLANATION,
            )
        return state.with_updates(
            feasibility_review=None,
            feasibility_directives=directives,
            feasibility_audits=audits,
            instance_infeasibility_reviews=instance_reviews,
            failure=None,
            pending_workspace_reset=True,
            current_stage=AgentStage.FEASIBILITY_REVIEW,
        )
    if stage is AgentStage.EXPLANATION:
        return state.with_updates(
            explanation=result.data["explanation"],
            current_stage=None,
        )
    raise StageTransitionError(f"Unknown Core stage: {stage.value!r}")  # pragma: no cover


def _apply_intake_result(
    state: AgentState,
    result: StageCompleted,
    *,
    problem_contract_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    gurobi_formulator_enabled: bool = False,
) -> AgentState:
    data = result.data
    if data.get("ready"):
        definition = data.get("problem_definition")
        if not definition:
            raise StageTransitionError("proceed is missing problem_definition")
        confirmed = dict(definition)
        confirmed["confirmed"] = True
        return state.with_updates(
            problem_definition=confirmed,
            pending_clarification_questions=(),
            current_stage=(
                AgentStage.PROBLEM_CONTRACT
                if problem_contract_enabled
                else (
                    AgentStage.ALGORITHM_DESIGN
                    if algorithm_design_enabled
                    else (
                        AgentStage.GUROBI_FORMULATOR
                        if gurobi_formulator_enabled
                        else AgentStage.SOLVING
                    )
                )
            ),
        )
    return pause_for_clarification(
        state,
        data.get("reply", ""),
        data.get("problem_definition"),
        _typed_questions(data.get("questions", [])),
    )


def pause_for_clarification(
    state: AgentState,
    message: str,
    problem_definition: dict | None,
    questions: tuple[ClarificationQuestion, ...],
) -> AgentState:
    updates: dict[str, object] = {}
    if problem_definition:
        definition = dict(problem_definition)
        definition["confirmed"] = False
        updates["problem_definition"] = definition
    serialized = [question.model_dump(mode="json") for question in questions]
    return state.with_updates(
        **updates,
        pending_clarification_questions=tuple(serialized),
        clarification_rounds=state.clarification_rounds + 1,
        current_stage=None,
    ).append_to("conversation", {"role": "assistant", "content": message})


def clarification_outcome(state: AgentState) -> NeedsClarification:
    questions = _typed_questions(state.pending_clarification_questions)
    last_message = "Additional information is required"
    for entry in reversed(state.conversation):
        if entry.get("role") == "assistant":
            last_message = entry.get("content", last_message)
            break
    return NeedsClarification(message=last_message, questions=questions)


def success_outcome(state: AgentState) -> Succeeded:
    review = state.feasibility_review or {}
    if review.get("decision") == "reject" and review.get("responsibility") == "instance":
        return Succeeded(
            message="Three independent feasibility reviews confirmed instance infeasibility",
            summary={
                "status": "instance_infeasible",
                "confidence": review.get("confidence"),
                "infeasibility_proof": review.get("infeasibility_proof"),
            },
        )
    parsed = state.solver_result or {}
    objective = parsed.get("objective") or parsed.get("objective_value")
    solver_status = parsed.get("status", "")
    message = f"Status: {solver_status}"
    if objective is not None:
        message = f"Objective: {objective}, status: {solver_status}"
    return Succeeded(
        message=message,
        summary={"status": "succeeded", "objective": objective},
    )


def _typed_questions(
    questions: list[dict] | tuple[ClarificationQuestion, ...],
) -> tuple[ClarificationQuestion, ...]:
    return tuple(
        question
        if isinstance(question, ClarificationQuestion)
        else ClarificationQuestion(
            id=str(question.get("id") or f"clarify_{index}"),
            text=str(question.get("text") or "").strip(),
            answer_type=str(question.get("answer_type") or "text"),
            options=tuple(str(item) for item in (question.get("options") or [])),
            context=str(question.get("context") or ""),
        )
        for index, question in enumerate(questions, 1)
    )
