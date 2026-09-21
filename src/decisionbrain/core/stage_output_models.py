"""Strict output contracts for Core stages."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ProblemFamily = Literal[
    "linear_programming",
    "resource_scheduling",
    "routing",
    "assignment",
    "packing",
    "network_flow",
    "inventory",
    "production_planning",
    "facility_location",
    "cutting_stock",
    "robust_optimization",
    "distributionally_robust_optimization",
    "nonlinear_programming",
    "mixed_combinatorial",
    "other",
]
Method = Literal[
    "mixed_integer_linear_programming",
    "mixed_integer_nonlinear_programming",
    "linear_programming",
    "robust_optimization",
    "distributionally_robust_optimization",
    "constraint_programming",
    "constraint_programming_sat",
    "network_flow",
    "dynamic_programming",
    "iterated_local_search",
    "large_neighborhood_search",
    "simulated_annealing",
    "dispatching_rule",
    "vehicle_routing_search",
    "greedy",
    "custom_search",
    "other",
]
MethodClass = Literal[
    "exact",
    "constructive_heuristic",
    "local_search",
    "metaheuristic",
    "other",
]
Optimality = Literal["proven_optimal", "exact_or_incumbent", "heuristic_or_incumbent"]
FORMULATION_METHODS = {
    "mixed_integer_linear_programming",
    "mixed_integer_nonlinear_programming",
    "linear_programming",
    "robust_optimization",
    "distributionally_robust_optimization",
    "constraint_programming",
    "constraint_programming_sat",
    "network_flow",
    "dynamic_programming",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GurobiFormulationOutput(_StrictModel):
    """A complete, single Gurobi LP/MILP model definition without solving code."""

    formulation_type: Literal["lp", "milp"]
    variables: list[dict[str, Any]] = Field(min_length=1)
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(min_length=1)
    objective: dict[str, Any]
    data_mapping: list[dict[str, Any]] = Field(min_length=1)
    solution_mapping: list[dict[str, Any]] = Field(min_length=1)
    completeness_checklist: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_search_features(self) -> "GurobiFormulationOutput":
        text = str(self.model_dump(mode="json")).lower()
        # Prompts and artifacts are Chinese, so detect Chinese terms for heuristic and warm-start bypasses.
        forbidden = (
            "heuristic",
            "local_search",
            "local search",
            "lns",
            "warm_start",
            "warm start",
            "mipstart",
            "mip start",
            "fallback",
            "greedy",
            "tabu",
            "annealing",
            "decomposition",
            "column generation",
            "启发式",
            "局部搜索",
            "邻域搜索",
            "大邻域",
            "热启动",
            "暖启动",
            "模拟退火",
            "禁忌搜索",
            "列生成",
            "贪心",
            "候选生成",
            "分解",
            "回退",
        )
        hits = [term for term in forbidden if term in text]
        if hits:
            raise ValueError(
                "formulation 不能包含启发式、warm start、分解或 fallback 求解策略；"
                f"命中禁用词：{'、'.join(hits)}"
            )
        return self


class ProblemDiagnosis(_StrictModel):
    summary: str = Field(min_length=1)
    decision_type: Literal[
        "assign",
        "select",
        "sequence",
        "schedule",
        "route",
        "allocate",
        "batch",
        "partition",
        "activate",
        "quantity",
        "mixed",
    ]
    key_difficulty: str = Field(min_length=1)


class ScaleAssessment(_StrictModel):
    primary_items: int | None = Field(default=None, ge=0)
    resources: int | None = Field(default=None, ge=0)
    estimated_variables: int | None = Field(default=None, ge=0)
    estimated_constraints: int | None = Field(default=None, ge=0)
    risk: Literal["low", "medium", "high"]
    confidence: Literal["low", "medium", "high"]
    reason: str = Field(min_length=1)


class CandidateRoute(_StrictModel):
    method: Method
    method_class: MethodClass
    source: Literal["package", "generated"]
    package_id: str | None = None
    solver_id: str | None = None
    fit_reason: str = Field(min_length=1)
    main_risk: str = Field(min_length=1)
    recommendation: Literal["primary", "backup", "reject"]

    @model_validator(mode="after")
    def validate_source(self) -> "CandidateRoute":
        if self.source == "package" and (not self.package_id or not self.solver_id):
            raise ValueError("package candidate requires package_id and solver_id")
        if self.source == "generated" and (
            self.package_id is not None or self.solver_id is not None
        ):
            raise ValueError("generated candidate must not set package_id or solver_id")
        return self


class PackageReference(_StrictModel):
    id: str = Field(min_length=1)
    version: str = Field(min_length=1)


class CapabilityMapping(_StrictModel):
    business_requirement: str = Field(min_length=1)
    manifest_capability: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class PackageIntegration(_StrictModel):
    primary_api: str = Field(min_length=1)
    input_mapping: str = Field(min_length=1)
    result_mapping: str = Field(min_length=1)
    runtime_control: str = Field(min_length=1)
    random_seed: int | None = None
    availability_check: str = Field(min_length=1)


class StrategySpec(_StrictModel):
    representation: str = Field(min_length=1)
    feasibility: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    construction: str = Field(min_length=1)
    improvement: str = Field(min_length=1)
    stopping: str = Field(min_length=1)
    output: str = Field(min_length=1)


class FallbackPlan(_StrictModel):
    trigger: str = Field(min_length=1)
    description: str = Field(min_length=1)
    components: tuple["AlgorithmComponent", ...] = ()


class FormulationSpec(_StrictModel):
    method: Literal[
        "mixed_integer_linear_programming",
        "mixed_integer_nonlinear_programming",
        "linear_programming",
        "robust_optimization",
        "distributionally_robust_optimization",
        "constraint_programming",
        "constraint_programming_sat",
        "network_flow",
        "dynamic_programming",
    ]
    scope: Literal["full_problem", "subproblem", "relaxation", "repair_model"]
    summary: str = Field(min_length=1)
    variables: tuple[str, ...] = Field(min_length=1)
    objective: str = Field(min_length=1)
    constraints: tuple[str, ...] = Field(min_length=1)
    big_m_notes: str = ""


class SingleFormulationSpec(FormulationSpec):
    scope: Literal["full_problem"]


class AlgorithmComponent(_StrictModel):
    component_id: str = Field(pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
    role: str = Field(min_length=1)
    source: Literal["package", "generated"]
    package: PackageReference | None = None
    solver_id: str | None = None
    method: Method
    method_class: MethodClass
    optimality: Optimality
    reason: str = Field(min_length=1)
    capability_mapping: tuple[CapabilityMapping, ...] = ()
    integration: PackageIntegration | None = None
    formulation: FormulationSpec | None = None

    @model_validator(mode="after")
    def validate_component(self) -> "AlgorithmComponent":
        if self.source == "package":
            if self.package is None or not self.solver_id or self.integration is None:
                raise ValueError("package component requires package, solver_id, and integration")
            if not self.capability_mapping:
                raise ValueError("package component requires capability_mapping")
        elif any(value is not None for value in (self.package, self.solver_id, self.integration)):
            raise ValueError("generated component must not set package, solver_id, or integration")

        requires_formulation = self.method in FORMULATION_METHODS
        if requires_formulation and self.formulation is None:
            raise ValueError(f"{self.method} component requires formulation")
        if not requires_formulation and self.formulation is not None:
            raise ValueError(f"{self.method} component must not include formulation")
        if self.formulation is not None and self.formulation.method != self.method:
            raise ValueError("component method must match formulation method")
        return self


class AlgorithmSelection(_StrictModel):
    kind: Literal["single", "hybrid"]
    reason: str = Field(min_length=1)
    components: tuple[AlgorithmComponent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_component_count(self) -> "AlgorithmSelection":
        if self.kind == "single" and len(self.components) != 1:
            raise ValueError("single selection requires exactly one component")
        if self.kind == "hybrid" and len(self.components) < 2:
            raise ValueError("hybrid selection requires at least two components")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("selection component_id values must be unique")
        return self


class AlgorithmDesignOutput(_StrictModel):
    schema_version: Literal["1.1"]
    problem_family: ProblemFamily
    diagnosis: ProblemDiagnosis
    scale: ScaleAssessment
    candidates: tuple[CandidateRoute, ...] = Field(min_length=1)
    selection: AlgorithmSelection
    strategy: StrategySpec
    fallback: FallbackPlan
    risks: tuple[str, ...] = ()


class AlgorithmSpec(_StrictModel):
    source: Literal["package", "generated"]
    package: PackageReference | None = None
    solver_id: str | None = None
    method: Method
    method_class: MethodClass
    optimality: Optimality
    reason: str = Field(min_length=1)
    capability_mapping: tuple[CapabilityMapping, ...] = ()
    integration: PackageIntegration | None = None
    formulation: FormulationSpec | None = None

    @model_validator(mode="after")
    def validate_algorithm(self) -> "AlgorithmSpec":
        if self.source == "package":
            if self.package is None or not self.solver_id or self.integration is None:
                raise ValueError("package algorithm requires package, solver_id, and integration")
            if not self.capability_mapping:
                raise ValueError("package algorithm requires capability_mapping")
        elif any(value is not None for value in (self.package, self.solver_id, self.integration)):
            raise ValueError("generated algorithm must not set package, solver_id, or integration")
        requires_formulation = self.method in FORMULATION_METHODS
        if requires_formulation and self.formulation is None:
            raise ValueError(f"{self.method} algorithm requires formulation")
        if not requires_formulation and self.formulation is not None:
            raise ValueError(f"{self.method} algorithm must not include formulation")
        if self.formulation is not None and self.formulation.method != self.method:
            raise ValueError("algorithm method must match formulation method")
        return self


class SingleAlgorithmSpec(AlgorithmSpec):
    formulation: SingleFormulationSpec | None = None


class SingleAlgorithmDesignOutput(_StrictModel):
    schema_version: Literal["1.0"]
    problem_family: ProblemFamily
    diagnosis: ProblemDiagnosis
    scale: ScaleAssessment
    candidates: tuple[CandidateRoute, ...] = Field(min_length=1)
    algorithm: SingleAlgorithmSpec
    strategy: StrategySpec
    risks: tuple[str, ...] = ()


class AvailabilityResult(_StrictModel):
    ok: bool
    detail: str = Field(min_length=1)


class ComponentExecution(_StrictModel):
    component_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source: Literal["package", "generated"]
    planned_package_id: str | None = None
    planned_solver_id: str | None = None
    executed_package_id: str | None = None
    executed_solver_id: str | None = None
    method: Method
    method_class: MethodClass
    version: str | None = None
    guide_consulted: bool
    availability_check: AvailabilityResult
    seed: int | None = None
    runtime_limit_seconds: float = Field(ge=0)
    fallback_used: bool
    deviation_reason: str = ""

    @model_validator(mode="after")
    def validate_execution_source(self) -> "ComponentExecution":
        if self.source == "package":
            if not self.executed_package_id or not self.executed_solver_id or not self.version:
                raise ValueError(
                    "package execution requires executed_package_id, executed_solver_id, and version"
                )
            if not self.guide_consulted or not self.availability_check.ok:
                raise ValueError(
                    "package execution requires guide and successful availability check"
                )
        else:
            package_fields = (
                self.planned_package_id,
                self.planned_solver_id,
                self.executed_package_id,
                self.executed_solver_id,
                self.version,
            )
            if any(value is not None for value in package_fields):
                raise ValueError("generated execution must not claim package fields")
        if self.fallback_used and not self.deviation_reason:
            raise ValueError("fallback execution requires deviation_reason")
        if (
            self.planned_package_id
            and self.executed_package_id != self.planned_package_id
            and not self.fallback_used
        ):
            raise ValueError("executed package differs from plan without fallback_used")
        if (
            self.planned_solver_id
            and self.executed_solver_id != self.planned_solver_id
            and not self.fallback_used
        ):
            raise ValueError("executed solver differs from plan without fallback_used")
        return self


class CheckResult(_StrictModel):
    ok: bool
    issues: tuple[Any, ...] = ()


class AlgorithmMappingCheck(CheckResult):
    mapped_capabilities: tuple[str, ...] = ()


class FeasibilityResult(_StrictModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    schema_valid: bool
    feasible: bool
    objective_value: float | int | None
    violations: tuple[Any, ...] = ()
    warnings: tuple[Any, ...] = ()


class FeasibilityCheckReceipt(_StrictModel):
    """Runtime-owned proof that one checker execution produced feasibility_result.json.

    The receipt exists so acceptance can be bound to provenance, not only to the
    shape of the result file.  ``feasibility_result.json`` alone cannot say which
    rules ran, on which instance, against which candidate, or whether it was
    rewritten afterwards; the four digests below pin exactly that.  A verdict is
    only allowed to rely on a check whose recorded digests still match the files
    present when the verdict is written.
    """

    schema_version: Literal["1.0"]
    outcome_source: Literal["runtime_checker_execution"]
    checker_file: Literal["feasibility_checker.py"]
    result_file: Literal["feasibility_result.json"]
    # sha256 of the four files the check is defined by
    checker_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    solution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    # execution facts
    argv: tuple[str, ...] = ()
    exit_code: int
    elapsed_seconds: float = Field(ge=0)


class SolverExecutionReceipt(_StrictModel):
    """Runtime-owned proof of one solver execution and its optional candidate."""

    schema_version: Literal["1.0"]
    outcome_source: Literal["runtime_solver_execution"]
    solver_file: Literal["solver.py"]
    input_file: Literal["input.json"]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    solver_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    solution_file: Literal["solution.json"] | None = None
    solution_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    argv: tuple[str, ...] = ()
    exit_code: int
    elapsed_seconds: float = Field(ge=0)
    timed_out: bool = False

    @model_validator(mode="after")
    def validate_solution_binding(self) -> "SolverExecutionReceipt":
        if (self.solution_file is None) != (self.solution_sha256 is None):
            raise ValueError("solution_file and solution_sha256 must be set together")
        return self


class FeasibilityReviewEvidence(_StrictModel):
    source: str = Field(min_length=1)
    finding: str = Field(min_length=1)


class FeasibilityRemediationHandoff(_StrictModel):
    """Checker-independent repair context safe to expose to an upstream stage."""

    requirement_ids: tuple[
        Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")], ...
    ] = Field(min_length=1)
    change_ids: tuple[
        Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")], ...
    ] = Field(min_length=1)
    problem_requirement: str = Field(min_length=1)
    observed_solution_behavior: str = Field(min_length=1)
    responsibility_reason: str = Field(min_length=1)
    required_changes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_checker_details(self) -> "FeasibilityRemediationHandoff":
        text = " ".join(
            (
                *self.requirement_ids,
                *self.change_ids,
                self.problem_requirement,
                self.observed_solution_behavior,
                self.responsibility_reason,
                *self.required_changes,
            )
        ).casefold()
        forbidden = (
            "checker",
            "feasibility_result",
            "feasibility_checker",
            "检查器",
            ".py:",
        )
        leaked = [marker for marker in forbidden if marker in text]
        if leaked:
            raise ValueError(
                "remediation_handoff must not expose checker details: "
                + ", ".join(leaked)
            )
        if len(set(self.requirement_ids)) != len(self.requirement_ids):
            raise ValueError("remediation_handoff requirement_ids must be unique")
        if len(set(self.change_ids)) != len(self.change_ids):
            raise ValueError("remediation_handoff change_ids must be unique")
        return self


class FeasibilityReviewOutput(_StrictModel):
    """Independent acceptance verdict and responsibility attribution."""

    schema_version: Literal["1.0"]
    decision: Literal["accept", "reject"]
    responsibility: Literal["instance", "algorithm_design", "gurobi_formulator", "solving"] | None = None
    summary: str = Field(min_length=1)
    evidence: tuple[FeasibilityReviewEvidence, ...] = ()
    required_changes: tuple[str, ...] = ()
    confidence: Literal["low", "medium", "high"]
    infeasibility_proof: str = ""
    remediation_handoff: FeasibilityRemediationHandoff | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "FeasibilityReviewOutput":
        if self.decision == "accept":
            if self.responsibility is not None:
                raise ValueError("accept decision must not assign responsibility")
            if self.required_changes:
                raise ValueError("accept decision must not require changes")
            if self.infeasibility_proof:
                raise ValueError("accept decision must not include infeasibility_proof")
            if self.remediation_handoff is not None:
                raise ValueError("accept decision must not include remediation_handoff")
            return self
        if self.responsibility is None:
            raise ValueError("reject decision requires responsibility")
        if not self.evidence:
            raise ValueError("reject decision requires evidence")
        if self.responsibility == "instance":
            if not self.infeasibility_proof:
                raise ValueError("instance responsibility requires infeasibility_proof")
            if self.required_changes:
                raise ValueError("instance responsibility must not require implementation changes")
            if self.remediation_handoff is not None:
                raise ValueError("instance responsibility must not include remediation_handoff")
        else:
            if self.infeasibility_proof:
                raise ValueError(
                    "implementation responsibility must not claim infeasibility proof"
                )
            if not self.required_changes:
                raise ValueError(
                    "algorithm_design/solving responsibility requires concrete changes"
                )
            if self.remediation_handoff is None:
                raise ValueError(
                    "algorithm_design/solving responsibility requires remediation_handoff"
                )
        return self


class SolverValidation(_StrictModel):
    schema_check: CheckResult
    feasibility_checker: FeasibilityResult | None = None
    business_semantic_check: CheckResult
    algorithm_mapping_check: AlgorithmMappingCheck
    implementation_check: CheckResult


class ConstraintCheck(_StrictModel):
    name: str = Field(min_length=1)
    ok: bool
    detail: str = ""


class InfeasibilityProof(_StrictModel):
    proof_type: Literal["exact_solver_status", "solver_certificate"]
    solver: str = Field(min_length=1)
    raw_status: str = Field(min_length=1)
    model_scope: Literal["complete"]
    hard_constraints_covered: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)


class SolverResultOutput(_StrictModel):
    status: Literal[
        "optimal",
        "feasible",
        "time_limit",
        "unknown",
        "no_solution_found",
        "infeasible_or_unbounded",
        "infeasible",
        "failed",
    ]
    objective: float | int | None
    mip_gap: float | None = Field(default=None, ge=0)
    optimality: Optimality | None
    input_file: Literal["input.json"]
    solution_file: Literal["solution.json"] | None
    code_file: Literal["solver.py"]
    feasibility_result_file: Literal["feasibility_result.json"] | None = None
    executions: tuple[ComponentExecution, ...] = Field(min_length=1)
    solution_summary: dict[str, Any]
    constraint_check: tuple[ConstraintCheck, ...] = ()
    validation: SolverValidation
    diagnosis: str = ""
    infeasibility_proof: InfeasibilityProof | None = None

    @model_validator(mode="after")
    def validate_result_semantics(self) -> "SolverResultOutput":
        component_ids = [execution.component_id for execution in self.executions]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("execution component_id values must be unique")
        if self.status == "infeasible":
            if self.objective is not None:
                raise ValueError("infeasible status requires objective=null")
            if self.mip_gap is not None:
                raise ValueError("infeasible status requires mip_gap=null")
            if self.optimality is not None:
                raise ValueError("infeasible status requires optimality=null")
            if self.solution_file is not None:
                raise ValueError("infeasible status requires solution_file=null")
            if self.feasibility_result_file is not None:
                raise ValueError("infeasible status requires feasibility_result_file=null")
            if self.infeasibility_proof is None:
                raise ValueError("infeasible status requires infeasibility_proof")
        elif self.solution_file == "solution.json":
            if self.status not in {"optimal", "feasible", "time_limit", "unknown"}:
                raise ValueError(f"{self.status} status cannot carry a candidate solution")
            if self.optimality is None:
                raise ValueError("candidate solution requires optimality")
            if self.infeasibility_proof is not None:
                raise ValueError("infeasibility_proof is only allowed for infeasible status")
        elif self.status in {
            "time_limit",
            "unknown",
            "no_solution_found",
            "infeasible_or_unbounded",
        }:
            if self.objective is not None:
                raise ValueError("no-candidate status requires objective=null")
            if self.mip_gap is not None:
                raise ValueError("no-candidate status requires mip_gap=null")
            if self.optimality is not None:
                raise ValueError("no-candidate status requires optimality=null")
            if self.feasibility_result_file is not None:
                raise ValueError("no-candidate status requires feasibility_result_file=null")
            if self.infeasibility_proof is not None:
                raise ValueError("no-candidate status must not claim infeasibility proof")
            if not self.diagnosis.strip():
                raise ValueError("no-candidate status requires diagnosis")
        else:
            raise ValueError(f"{self.status} status requires solution_file=solution.json")
        return self


class AlgorithmExecution(_StrictModel):
    source: Literal["package", "generated"]
    planned_package_id: str | None = None
    planned_solver_id: str | None = None
    executed_package_id: str | None = None
    executed_solver_id: str | None = None
    method: Method
    method_class: MethodClass
    version: str | None = None
    guide_consulted: bool
    availability_check: AvailabilityResult
    seed: int | None = None
    runtime_limit_seconds: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_execution(self) -> "AlgorithmExecution":
        if self.source == "package":
            if not self.executed_package_id or not self.executed_solver_id or not self.version:
                raise ValueError(
                    "package execution requires executed_package_id, executed_solver_id, and version"
                )
            if not self.guide_consulted or not self.availability_check.ok:
                raise ValueError(
                    "package execution requires guide and successful availability check"
                )
        elif any(
            value is not None
            for value in (
                self.planned_package_id,
                self.planned_solver_id,
                self.executed_package_id,
                self.executed_solver_id,
                self.version,
            )
        ):
            raise ValueError("generated execution must not set package fields")
        if (
            self.planned_package_id
            and self.executed_package_id != self.planned_package_id
        ):
            raise ValueError("executed package must match planned package")
        if (
            self.planned_solver_id
            and self.executed_solver_id != self.planned_solver_id
        ):
            raise ValueError("executed solver must match planned solver")
        return self


class SingleSolverResultOutput(_StrictModel):
    status: Literal[
        "optimal", "feasible", "time_limit", "unknown", "no_solution_found",
        "infeasible_or_unbounded", "infeasible", "failed",
    ]
    objective: float | int | None
    mip_gap: float | None = Field(default=None, ge=0)
    optimality: Optimality | None
    input_file: Literal["input.json"]
    solution_file: Literal["solution.json"] | None
    code_file: Literal["solver.py"]
    feasibility_result_file: Literal["feasibility_result.json"] | None = None
    execution: AlgorithmExecution
    solution_summary: dict[str, Any]
    constraint_check: tuple[ConstraintCheck, ...] = ()
    validation: SolverValidation
    diagnosis: str = ""
    infeasibility_proof: InfeasibilityProof | None = None

    @model_validator(mode="after")
    def validate_result_semantics(self) -> "SingleSolverResultOutput":
        if self.status == "infeasible":
            if self.objective is not None:
                raise ValueError("infeasible status requires objective=null")
            if self.mip_gap is not None:
                raise ValueError("infeasible status requires mip_gap=null")
            if self.optimality is not None:
                raise ValueError("infeasible status requires optimality=null")
            if self.solution_file is not None:
                raise ValueError("infeasible status requires solution_file=null")
            if self.feasibility_result_file is not None:
                raise ValueError("infeasible status requires feasibility_result_file=null")
            if self.infeasibility_proof is None:
                raise ValueError("infeasible status requires infeasibility_proof")
        elif self.solution_file == "solution.json":
            if self.status not in {"optimal", "feasible", "time_limit", "unknown"}:
                raise ValueError(f"{self.status} status cannot carry a candidate solution")
            if self.optimality is None:
                raise ValueError("candidate solution requires optimality")
            if self.infeasibility_proof is not None:
                raise ValueError("infeasibility_proof is only allowed for infeasible status")
        elif self.status in {
            "time_limit",
            "unknown",
            "no_solution_found",
            "infeasible_or_unbounded",
        }:
            if self.objective is not None:
                raise ValueError("no-candidate status requires objective=null")
            if self.mip_gap is not None:
                raise ValueError("no-candidate status requires mip_gap=null")
            if self.optimality is not None:
                raise ValueError("no-candidate status requires optimality=null")
            if self.feasibility_result_file is not None:
                raise ValueError("no-candidate status requires feasibility_result_file=null")
            if self.infeasibility_proof is not None:
                raise ValueError("no-candidate status must not claim infeasibility proof")
            if not self.diagnosis.strip():
                raise ValueError("no-candidate status requires diagnosis")
        else:
            raise ValueError(f"{self.status} status requires solution_file=solution.json")
        return self


class SolvingResultReference(_StrictModel):
    solver_result_file: Literal["solver_result.json"] | None = None
    runtime_solver_outcome_file: Literal["runtime_solver_outcome.json"] | None = None

    @model_validator(mode="after")
    def validate_single_reference(self) -> "SolvingResultReference":
        references = (
            self.solver_result_file is not None,
            self.runtime_solver_outcome_file is not None,
        )
        if sum(references) != 1:
            raise ValueError("result requires exactly one solving outcome file reference")
        return self


class RuntimeSolverTimeoutOutput(_StrictModel):
    schema_version: Literal["1.0"]
    outcome_source: Literal["runtime_external_timeout"]
    # ``time_limit`` is accepted only for reading pre-rename historical Runs;
    # Runtime always writes the disambiguated ``runtime_timeout`` value.
    status: Literal["runtime_timeout", "time_limit"]
    timeout_source: Literal["runtime_watchdog", "gnu_timeout"]
    timeout_seconds: float | None = Field(default=None, gt=0)
    elapsed_seconds: float = Field(ge=0)
    exit_code: int
    termination_signal: Literal["SIGTERM", "SIGKILL"] | None = None
    solver_file: Literal["solver.py"]
    solver_execution_observed: Literal[True]
    solution_detected: bool
    solution_schema_valid: bool | None
    solution_file: Literal["solution.json"] | None = None
    stdout_tail: str = ""

    @model_validator(mode="after")
    def validate_candidate(self) -> "RuntimeSolverTimeoutOutput":
        if self.solution_file is not None:
            if not self.solution_detected or self.solution_schema_valid is not True:
                raise ValueError("solution_file requires a detected, schema-valid candidate")
        elif self.solution_schema_valid is True:
            raise ValueError("schema-valid candidate requires solution_file=solution.json")
        return self


class ClarificationQuestionOutput(_StrictModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    answer_type: str = "text"
    options: tuple[str, ...] = ()
    context: str = ""


class SolvingStageOutput(_StrictModel):
    decision: Literal["solved", "needs_clarification", "failed"]
    message: str = Field(min_length=1)
    result: SolvingResultReference | None = None
    questions: tuple[ClarificationQuestionOutput, ...] = ()

    @model_validator(mode="after")
    def validate_decision_payload(self) -> "SolvingStageOutput":
        if self.decision == "solved" and self.result is None:
            raise ValueError("solved decision requires result.solver_result_file")
        if self.decision == "needs_clarification" and not self.questions:
            raise ValueError("needs_clarification requires questions")
        if self.decision != "needs_clarification" and self.questions:
            raise ValueError(f"{self.decision} decision must not include questions")
        if self.decision != "solved" and self.result is not None:
            raise ValueError(f"{self.decision} decision must not include result")
        return self
