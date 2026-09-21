"""Feasibility rejection directives shared by Core and rerun stages."""

from __future__ import annotations

import json
from typing import Any

from .stage_outputs import FEASIBILITY_REVIEW_HISTORY_FILE, feasibility_handoff_dir


INSTANCE_CONFIRMED_REVIEW_COUNT = 3
INSTANCE_AUDIT_KIND = "instance_infeasibility"
REPAIR_AUDIT_KIND = "repair_attempt"


def has_confirmed_instance_reviews(reviews: tuple[dict[str, Any], ...]) -> bool:
    """Three instance verdicts in a row confirm the diagnosis.

    ``instance_infeasibility_reviews`` is cleared whenever a verdict is not
    ``instance``, so its length already is the current consecutive run. Confidence is
    deliberately not part of the gate: requiring a confident streak let a single
    low-confidence verdict reset it forever, which had no upper bound.
    """

    return len(reviews) >= INSTANCE_CONFIRMED_REVIEW_COUNT


def visible_review_history(
    audits: tuple[dict[str, Any], ...],
    *,
    instance_run: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    """Select the audits that may be shown while ``instance_run`` is in progress.

    An open consecutive instance run withholds only its own verdicts, so each
    confirmation is reached without reading the conclusion it is about to repeat.
    Instance verdicts from runs that already ended stay visible: they say the route
    was taken and overturned, which is context rather than an answer to copy.

    A confirmed run is no longer open, so nothing is withheld and the stage that
    explains the outcome sees the verdicts that produced it.
    """

    withheld = 0 if has_confirmed_instance_reviews(instance_run) else len(instance_run)
    kept = list(audits)
    while withheld and kept and kept[-1].get("kind") == INSTANCE_AUDIT_KIND:
        kept.pop()
        withheld -= 1
    return [dict(item) for item in kept]


_ACCEPTANCE_CRITERIA = (
    "当前 solution 必须满足 problem.md 与 intake 中的全部业务硬约束",
    "必须逐项落实本 handoff 中的 required_changes",
    "不得修改上游问题定义和固定验收契约",
)


def build_feasibility_directive(
    *,
    attempt_index: int,
    review: dict[str, Any],
    solver_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a rejection verdict into a deterministic, replayable repair directive."""

    responsibility = str(review.get("responsibility") or "")
    handoff = review.get("remediation_handoff")
    if not isinstance(handoff, dict):
        raise ValueError("routed feasibility rejection requires remediation_handoff")
    return {
        "attempt": attempt_index,
        "responsibility": responsibility,
        "handoff": dict(handoff),
        "solver_summary": _compact_solver_summary(solver_result),
        "acceptance_criteria": list(_ACCEPTANCE_CRITERIA),
        "handoff_dir": feasibility_handoff_dir(responsibility, attempt_index),
    }


def annotate_feasibility_outcome(
    directives: tuple[dict[str, Any], ...],
    *,
    review: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Attach the next feasibility verdict to the pending repair directive."""

    if not directives or directives[-1].get("outcome"):
        return directives
    latest = dict(directives[-1])
    latest["outcome"] = str(review.get("decision") or "")
    if review.get("decision") == "reject":
        latest["outcome_responsibility"] = str(review.get("responsibility") or "")
    return (*directives[:-1], latest)


def render_for_executor(
    directives: tuple[dict[str, Any], ...],
    *,
    stage: str,
    components_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    gurobi_formulator_enabled: bool = False,
) -> str:
    """Render the active feasibility repair instruction for a rerun stage."""

    if not directives or directives[-1].get("outcome"):
        return ""
    latest = directives[-1]
    responsibility = str(latest.get("responsibility") or "")
    if stage not in {"algorithm_design", "gurobi_formulator", "solving"} or responsibility != stage:
        return ""

    handoff_dir = str(latest.get("handoff_dir") or "")
    baselines = [str(name) for name in (latest.get("baseline_files") or [])]
    payload = json.dumps(_active_directive(latest), ensure_ascii=False, indent=2)
    lines = [
        "【本阶段是被 Feasibility Reviewer 拒绝后的责任修复，必须处理以下指令】",
        payload,
        _baseline_instruction(handoff_dir, baselines),
        "- Core 私有审计历史、checker 结果和原始 evidence 不属于本阶段上下文。",
        "- 不得修改 problem contract、schema 或 feasibility checker。",
        "- 必须逐项落实 required_changes；无法落实时必须在本阶段正常输出中如实说明，",
        "  不得原样重跑或伪造 checker 通过结果。",
    ]
    if stage == "gurobi_formulator":
        lines.extend(
            [
                "- 重新建立覆盖全部业务变量、硬约束、目标和结果映射的全量完整 Gurobi 模型。",
                "- 不得退化为局部模型、启发式或只修补单个约束；若完整模型正确且没有其他建模方法上的改进空间，"
                "不得继续归责本阶段。",
            ]
        )
    elif stage == "algorithm_design":
        design_scope = (
            "算法选择、formulation、硬约束覆盖、fallback 或策略设计"
            if components_enabled
            else "算法选择、formulation、硬约束覆盖或策略设计"
        )
        lines.extend(
            [
                f"- 重新完成{design_scope}中被归责的部分。",
                "- handoff 中的旧设计和旧 solution 是失败反例，不是必须保留的模板。",
            ]
        )
    elif algorithm_design_enabled:
        lines.extend(
            [
                "- Algorithm Design 未变；应以 handoff 中的 previous_solver.py 为基线做定点修复，",
                "  避免无关重写。",
                "- 不得更换算法选择或建模范式；若证据表明必须改设计，应保留事实供下一次",
                "  Feasibility Reviewer 重新归责。",
            ]
        )
    elif gurobi_formulator_enabled:
        # Solving is translation-only in this arm; the model itself belongs to the
        # separate Gurobi Formulator stage and must not be redesigned here.
        lines.extend(
            [
                "- 本阶段只做 translation-only 的定点修复：把 gurobi_formulation.json 中已声明的模型",
                "  更忠实地翻译为 solver.py，应以 handoff 中的 previous_solver.py 为失败基线。",
                "- 不得修改变量、约束或目标语义，不得新增启发式、分解、候选生成、warm start 或 fallback；",
                "  若证据表明模型本身有误，应保留事实供下一次 Feasibility Reviewer 归责 gurobi_formulator。",
            ]
        )
    else:
        lines.extend(
            [
                "- 由同一个 Gurobi Formulator 修复完整模型。",
                "- Formulator 的修复范围包括变量、全部硬约束、完整目标、数据映射、求解调用和业务结果映射；",
                "  应以 handoff 中的 previous_solver.py 为失败基线。",
            ]
        )

    completed = [item for item in directives[:-1] if item.get("outcome")]
    if completed:
        history = json.dumps(
            [_history_ledger_entry(item) for item in completed],
            ensure_ascii=False,
            indent=2,
        )
        lines.append(
            "- 更早的可行性修复尝试及结果如下。不得重复已被证明无效的处理：\n" + history
        )
    return "\n".join(lines) + "\n"


def _baseline_instruction(handoff_dir: str, baselines: list[str]) -> str:
    """Point the responsible stage at the baselines this attempt actually produced."""

    if not baselines:
        return f"- 本轮修复基线位于 `{handoff_dir}/`，必须先 list_files 查看并读取其中全部文件再修改。"
    names = "、".join(f"`{name}`" for name in baselines)
    return f"- 本轮修复基线位于 `{handoff_dir}/`，包含 {names}，必须先全部读取再修改。"


def _active_directive(directive: dict[str, Any]) -> dict[str, Any]:
    """Project the active repair into the checker-independent stage view.

    This projection is the only copy of the repair instruction, so it carries the
    compact solver summary the ledger already exposes for earlier attempts. It stays
    an explicit whitelist to keep Core bookkeeping such as prepared baseline paths
    out of the stage prompt.
    """

    public = {
        "attempt": directive.get("attempt"),
        "responsibility": directive.get("responsibility"),
        "handoff": directive.get("handoff") or {},
        "solver_summary": directive.get("solver_summary") or {},
        "acceptance_criteria": directive.get("acceptance_criteria") or [],
        "handoff_dir": directive.get("handoff_dir"),
    }
    return public


def _history_ledger_entry(directive: dict[str, Any]) -> dict[str, Any]:
    """Compress one completed repair without carrying prose or checker evidence."""

    handoff = directive.get("handoff")
    if not isinstance(handoff, dict):
        handoff = {}
    return {
        "attempt": directive.get("attempt"),
        "responsibility": directive.get("responsibility"),
        "requirement_ids": list(handoff.get("requirement_ids") or []),
        "change_ids": list(handoff.get("change_ids") or []),
        "solver_summary": directive.get("solver_summary") or {},
        "outcome": directive.get("outcome"),
        "next_responsibility": directive.get("outcome_responsibility"),
    }


def _compact_solver_summary(solver_result: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only the prior solve status, diagnosis, and methods in the ledger."""

    if not isinstance(solver_result, dict):
        return {}
    executions = solver_result.get("executions")
    if not isinstance(executions, (list, tuple)):
        execution = solver_result.get("execution")
        executions = (execution,) if isinstance(execution, dict) else ()
    methods = [
        str(execution.get("method") or "unknown")[:80]
        for execution in executions
        if isinstance(execution, dict)
    ]
    return {
        "status": str(solver_result.get("status") or "")[:80],
        "diagnosis": " ".join(str(solver_result.get("diagnosis") or "").split())[:500],
        "methods": methods[:20],
    }


def render_for_reviewer(
    directives: tuple[dict[str, Any], ...],
) -> str:
    """Tell the reviewer which rejection directive produced the current candidate."""

    if not directives or directives[-1].get("outcome"):
        return ""
    latest = directives[-1]
    completed = [item for item in directives[:-1] if item.get("outcome")]
    history = [_history_ledger_entry(item) for item in completed]
    return (
        "【当前候选是一次可行性拒绝后的修复结果】\n"
        f"{json.dumps(_active_directive(latest), ensure_ascii=False, indent=2)}\n"
        "【更早尝试的压缩账本】\n"
        f"{json.dumps(history, ensure_ascii=False, indent=2)}\n"
        f"- 账本不含原始证据；历次裁决全文与 checker 结果在 `{FEASIBILITY_REVIEW_HISTORY_FILE}`，"
        "只有本阶段可读，需要核对时直接读取该文件。\n"
        "- 必须检查 required_changes 是否真实落实，并基于当前候选重新独立裁决。\n"
        "- 不得因为已经发生过修复就降低 accept 标准；仍不通过时重新判断本次责任。\n"
    )
