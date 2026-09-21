"""Strict models for algorithm capability manifests."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.stage_output_models import ProblemFamily


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlgorithmSpec(_StrictModel):
    # Keep this vocabulary aligned with ProblemFamily output by algorithm_design.
    # Catalog problem_families are shortlist keys, and prompts prohibit catalog traversal
    # for discovery; a package using a different vocabulary would be unreachable.
    problem_families: tuple[ProblemFamily, ...] = Field(min_length=1)
    method: str = Field(min_length=1)
    method_class: Literal[
        "exact",
        "constructive_heuristic",
        "local_search",
        "metaheuristic",
        "mixed",
        "other",
    ]
    optimality: Literal[
        "proven_optimal",
        "exact_or_incumbent",
        "heuristic_or_incumbent",
        "mixed",
    ]
    supports_warm_start: bool
    supports_random_seed: bool
    supports_time_limit: bool
    capabilities: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = ()


class ScaleProfile(_StrictModel):
    confidence: Literal["high", "medium", "low", "unknown"]
    hard_limit: int | None = Field(default=None, ge=1)
    note: str = Field(min_length=1)


class SelectionSpec(_StrictModel):
    required_features: tuple[str, ...] = ()
    good_fit_when: tuple[str, ...] = ()
    reject_when: tuple[str, ...] = ()
    scale_profile: ScaleProfile


class InstalledArtifact(_StrictModel):
    origin: str = Field(min_length=1)
    self_compiled: bool
    wheel_tag: str | None = Field(default=None, min_length=1)
    modules: tuple[str, ...] = ()


class VerifiedEnvironment(_StrictModel):
    conda_environment: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)


class DistributionSpec(_StrictModel):
    package: str = Field(min_length=1)
    verified_version: str = Field(min_length=1)
    compatible_version_spec: str = Field(min_length=1)
    import_name: str = Field(min_length=1)
    requires_python: str = Field(min_length=1)
    license_spdx: str = Field(min_length=1)
    homepage: str = Field(min_length=1)
    documentation: str = Field(min_length=1)
    source_repository: str = Field(min_length=1)
    native_extensions: bool
    installed_artifact: InstalledArtifact | None = None
    verified_environment: VerifiedEnvironment


class ModelBuilderInterface(_StrictModel):
    constructor: str = Field(min_length=1)
    methods: tuple[str, ...] = Field(min_length=1)
    alternative_inputs: tuple[str, ...] = ()


class StoppingCriterionFact(_StrictModel):
    name: str = Field(min_length=1)
    budget_unit: Literal[
        "wall_clock_seconds",
        "iterations",
        "non_improving_iterations",
        "first_feasible_solution",
        "composite",
    ]
    workload_stable: bool
    environment_sensitive: bool
    recommended_uses: tuple[str, ...] = Field(min_length=1)
    caveat: str = Field(min_length=1)


class SolveInterface(_StrictModel):
    call: str = Field(min_length=1)
    stopping_criteria: tuple[StoppingCriterionFact, ...] = Field(min_length=1)
    important_parameters: dict[str, str] = Field(default_factory=dict)


class ResultInterface(_StrictModel):
    type: str = Field(min_length=1)
    fields: dict[str, str] = Field(min_length=1)
    warning: str = Field(min_length=1)


class PyVRPWarmStartFact(_StrictModel):
    supported: Literal[True]
    parameter: str = Field(min_length=1)
    value_type: str = Field(min_length=1)
    purpose: str = Field(min_length=1)


class PyVRPReproducibilityFact(_StrictModel):
    fixed_seed_scope: Literal["random_stream_only"]
    time_based_stop_strictly_reproducible: Literal[False]
    workload_stable_criteria: tuple[str, ...] = Field(min_length=1)
    strict_reproducibility_requirements: tuple[str, ...] = Field(min_length=1)


class PyVRPFeasibilityFact(_StrictModel):
    true_meaning: str = Field(min_length=1)
    false_meaning: str = Field(min_length=1)
    false_proves_problem_infeasible: Literal[False]
    proof_requirement: str = Field(min_length=1)


class PyVRPNumericInputFact(_StrictModel):
    distance_python_annotation: str = Field(min_length=1)
    duration_python_annotation: str = Field(min_length=1)
    internal_storage: Literal["integer"]
    positive_float_behavior: Literal["truncated_toward_zero"]
    required_conversion: str = Field(min_length=1)


class PyVRPRouteVisitsFact(_StrictModel):
    returns: Literal["client_location_indices"]
    includes_depot: Literal[False]
    business_id_mapping: str = Field(min_length=1)
    complete_route_mapping: str = Field(min_length=1)


class PyVRPObjectiveFact(_StrictModel):
    general_objective: str = Field(min_length=1)
    pure_distance: str = Field(min_length=1)
    distance_unit_matches_edge_distance: Literal[True]
    general_objective_equals_distance_when: tuple[str, ...] = Field(min_length=1)


class PyVRPInterfaceFacts(_StrictModel):
    warm_start: PyVRPWarmStartFact
    reproducibility: PyVRPReproducibilityFact
    feasibility: PyVRPFeasibilityFact
    numeric_inputs: PyVRPNumericInputFact
    route_visits: PyVRPRouteVisitsFact
    objective: PyVRPObjectiveFact


class PyVRPIteratedLocalSearchInterface(_StrictModel):
    profile: Literal["pyvrp.ils"]
    preferred_api: str = Field(min_length=1)
    imports: tuple[str, ...] = Field(min_length=1)
    model_builder: ModelBuilderInterface
    solve: SolveInterface
    result: ResultInterface
    facts: PyVRPInterfaceFacts


class PyJobShopBackend(_StrictModel):
    id: Literal["ortools", "cpoptimizer"]
    method: Literal["constraint_programming"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]
    installed: bool
    dependency: str = Field(min_length=1)
    call: str = Field(min_length=1)
    guarantee_boundary: str = Field(min_length=1)


class PyJobShopWarmStartFact(_StrictModel):
    supported: Literal[True]
    parameter: Literal["initial_solution"]
    value_type: Literal["pyjobshop.Solution"]


class PyJobShopProofStatusFact(_StrictModel):
    optimal: Literal["SolveStatus.OPTIMAL"]
    infeasible: Literal["SolveStatus.INFEASIBLE"]
    incumbent: Literal["SolveStatus.FEASIBLE"]
    budget_exhausted: Literal["SolveStatus.TIME_LIMIT"]


class PyJobShopNumericInputFact(_StrictModel):
    representation: Literal["integer"]
    guidance: str = Field(min_length=1)


class PyJobShopSolutionMappingFact(_StrictModel):
    source: Literal["result.best.tasks"]
    guidance: str = Field(min_length=1)


class PyJobShopInterfaceFacts(_StrictModel):
    warm_start: PyJobShopWarmStartFact
    proof_status: PyJobShopProofStatusFact
    numeric_inputs: PyJobShopNumericInputFact
    solution_mapping: PyJobShopSolutionMappingFact


class PyJobShopInterface(_StrictModel):
    profile: Literal["pyjobshop"]
    preferred_api: Literal["model_builder"]
    imports: tuple[str, ...] = Field(min_length=1)
    model_builder: ModelBuilderInterface
    solve: SolveInterface
    result: ResultInterface
    backends: tuple[PyJobShopBackend, ...] = Field(min_length=2, max_length=2)
    facts: PyJobShopInterfaceFacts

    @model_validator(mode="after")
    def validate_backends(self) -> "PyJobShopInterface":
        if {backend.id for backend in self.backends} != {"ortools", "cpoptimizer"}:
            raise ValueError("pyjobshop interface requires ortools and cpoptimizer backends")
        return self


class JobShopLibSolverBase(_StrictModel):
    imports: tuple[str, ...] = Field(min_length=1)
    constructor: str = Field(min_length=1)
    call: str = Field(min_length=1)
    method: str = Field(min_length=1)
    method_class: Literal["exact", "constructive_heuristic", "metaheuristic"]
    supports_warm_start: bool
    supports_random_seed: bool
    supports_time_limit: bool
    stopping_criteria: tuple[StoppingCriterionFact, ...] = ()
    important_parameters: dict[str, str] = Field(default_factory=dict)
    result: ResultInterface


class JobShopLibConstraintProgrammingSolver(JobShopLibSolverBase):
    id: Literal["constraint_programming"]
    method: Literal["constraint_programming_sat"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]


class JobShopLibDispatchingRulesSolver(JobShopLibSolverBase):
    id: Literal["dispatching_rules"]
    method: Literal["dispatching_rule"]
    method_class: Literal["constructive_heuristic"]
    optimality: Literal["heuristic_or_incumbent"]


class JobShopLibSimulatedAnnealingSolver(JobShopLibSolverBase):
    id: Literal["simulated_annealing"]
    method: Literal["simulated_annealing"]
    method_class: Literal["metaheuristic"]
    optimality: Literal["heuristic_or_incumbent"]


JobShopLibSolver = Annotated[
    JobShopLibConstraintProgrammingSolver
    | JobShopLibDispatchingRulesSolver
    | JobShopLibSimulatedAnnealingSolver,
    Field(discriminator="id"),
]


class JobShopLibInstanceMappingFact(_StrictModel):
    operation_type: Literal["job_shop_lib.Operation"]
    instance_type: Literal["job_shop_lib.JobShopInstance"]
    guidance: str = Field(min_length=1)


class JobShopLibScheduleFact(_StrictModel):
    result_type: Literal["job_shop_lib.Schedule"]
    machine_sequences: Literal["schedule.schedule"]
    makespan: Literal["schedule.makespan()"]
    completeness: Literal["schedule.is_complete()"]


class JobShopLibInterfaceFacts(_StrictModel):
    instance_mapping: JobShopLibInstanceMappingFact
    schedule: JobShopLibScheduleFact
    feasibility_boundary: str = Field(min_length=1)


class JobShopLibInterface(_StrictModel):
    profile: Literal["job_shop_lib"]
    preferred_api: Literal["solver_suite"]
    imports: tuple[str, ...] = Field(min_length=1)
    model_builder: ModelBuilderInterface
    solvers: tuple[JobShopLibSolver, ...] = Field(min_length=3, max_length=3)
    facts: JobShopLibInterfaceFacts

    @model_validator(mode="after")
    def validate_solver_suite(self) -> "JobShopLibInterface":
        required = {"constraint_programming", "dispatching_rules", "simulated_annealing"}
        if {solver.id for solver in self.solvers} != required:
            raise ValueError(f"job_shop_lib interface requires solver suite {sorted(required)}")
        return self


class ModelingSolverBase(_StrictModel):
    imports: tuple[str, ...] = Field(min_length=1)
    constructor: str = Field(min_length=1)
    call: str = Field(min_length=1)
    method: str = Field(min_length=1)
    method_class: Literal["exact", "metaheuristic"]
    optimality: Literal["exact_or_incumbent", "heuristic_or_incumbent"]
    supports_warm_start: bool
    supports_random_seed: bool
    supports_time_limit: bool
    stopping_criteria: tuple[StoppingCriterionFact, ...] = ()
    important_parameters: dict[str, str] = Field(default_factory=dict)
    result: ResultInterface


class GurobiMilpSolver(ModelingSolverBase):
    id: Literal["milp"]
    method: Literal["mixed_integer_linear_programming"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]


class GurobiLpSolver(ModelingSolverBase):
    id: Literal["lp"]
    method: Literal["linear_programming"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]


GurobiSolver = Annotated[
    GurobiMilpSolver | GurobiLpSolver,
    Field(discriminator="id"),
]


class GurobiInterfaceFacts(_StrictModel):
    license_required: Literal[True]
    proof_boundary: str = Field(min_length=1)
    formulation_guidance: str = Field(min_length=1)


class GurobiInterface(_StrictModel):
    profile: Literal["gurobipy"]
    preferred_api: Literal["solver_suite"]
    imports: tuple[str, ...] = Field(min_length=1)
    model_builder: ModelBuilderInterface
    solvers: tuple[GurobiSolver, ...] = Field(min_length=2, max_length=2)
    facts: GurobiInterfaceFacts

    @model_validator(mode="after")
    def validate_solver_suite(self) -> "GurobiInterface":
        if {solver.id for solver in self.solvers} != {"milp", "lp"}:
            raise ValueError("gurobipy interface requires milp and lp solvers")
        return self


class ORToolsCpSatSolver(ModelingSolverBase):
    id: Literal["cp_sat"]
    method: Literal["constraint_programming_sat"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]


class ORToolsLinearSolver(ModelingSolverBase):
    id: Literal["linear_solver"]
    method: Literal["mixed_integer_linear_programming"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]


class ORToolsMinCostFlowSolver(ModelingSolverBase):
    id: Literal["min_cost_flow"]
    method: Literal["network_flow"]
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]


class ORToolsRoutingSolver(ModelingSolverBase):
    id: Literal["routing"]
    method: Literal["vehicle_routing_search"]
    method_class: Literal["metaheuristic"]
    optimality: Literal["heuristic_or_incumbent"]


ORToolsSolver = Annotated[
    ORToolsCpSatSolver | ORToolsLinearSolver | ORToolsMinCostFlowSolver | ORToolsRoutingSolver,
    Field(discriminator="id"),
]


class ORToolsInterfaceFacts(_StrictModel):
    cp_sat_integer_only: Literal[True]
    proof_boundary: str = Field(min_length=1)
    formulation_guidance: str = Field(min_length=1)


class ORToolsInterface(_StrictModel):
    profile: Literal["ortools"]
    preferred_api: Literal["solver_suite"]
    imports: tuple[str, ...] = Field(min_length=1)
    model_builder: ModelBuilderInterface
    solvers: tuple[ORToolsSolver, ...] = Field(min_length=4, max_length=4)
    facts: ORToolsInterfaceFacts

    @model_validator(mode="after")
    def validate_solver_suite(self) -> "ORToolsInterface":
        required = {"cp_sat", "linear_solver", "min_cost_flow", "routing"}
        if {solver.id for solver in self.solvers} != required:
            raise ValueError(f"ortools interface requires solver suite {sorted(required)}")
        return self


class RSOMEModelBase(_StrictModel):
    imports: tuple[str, ...] = Field(min_length=1)
    constructor: str = Field(min_length=1)
    model_builder: ModelBuilderInterface
    call: str = Field(min_length=1)
    method: str = Field(min_length=1)
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]
    supports_warm_start: Literal[False]
    supports_random_seed: bool
    supports_time_limit: bool
    stopping_criteria: tuple[StoppingCriterionFact, ...] = ()
    important_parameters: dict[str, str] = Field(default_factory=dict)
    result: ResultInterface


class RSOMERobustModel(RSOMEModelBase):
    id: Literal["robust_optimization"]
    constructor: Literal["rsome.ro.Model(name=None)"]
    method: Literal["robust_optimization"]


class RSOMEDistributionallyRobustModel(RSOMEModelBase):
    id: Literal["distributionally_robust_optimization"]
    constructor: Literal["rsome.dro.Model(scens=1, name=None)"]
    method: Literal["distributionally_robust_optimization"]


RSOMEModel = Annotated[
    RSOMERobustModel | RSOMEDistributionallyRobustModel,
    Field(discriminator="id"),
]


class RSOMESolverAdapter(_StrictModel):
    id: Literal["clp", "copt", "cplex", "ecos", "gurobi", "mosek", "ortools", "scipy"]
    module: str = Field(min_length=1)
    dependency: str = Field(min_length=1)
    installed: bool
    guarantee_boundary: str = Field(min_length=1)


class RSOMEInterfaceFacts(_StrictModel):
    default_solver: Literal["scipy"]
    reformulation_boundary: str = Field(min_length=1)
    optimal_flag_boundary: str = Field(min_length=1)
    ambiguity_set_boundary: str = Field(min_length=1)


class RSOMEInterface(_StrictModel):
    profile: Literal["rsome"]
    preferred_api: Literal["modeling_suite"]
    imports: tuple[str, ...] = Field(min_length=2)
    solvers: tuple[RSOMEModel, ...] = Field(min_length=2, max_length=2)
    solver_adapters: tuple[RSOMESolverAdapter, ...] = Field(min_length=8, max_length=8)
    facts: RSOMEInterfaceFacts

    @model_validator(mode="after")
    def validate_interface_suite(self) -> "RSOMEInterface":
        required_models = {"robust_optimization", "distributionally_robust_optimization"}
        if {solver.id for solver in self.solvers} != required_models:
            raise ValueError(f"rsome interface requires model suite {sorted(required_models)}")
        required_adapters = {
            "clp",
            "copt",
            "cplex",
            "ecos",
            "gurobi",
            "mosek",
            "ortools",
            "scipy",
        }
        if {adapter.id for adapter in self.solver_adapters} != required_adapters:
            raise ValueError(
                f"rsome interface requires solver adapters {sorted(required_adapters)}"
            )
        return self


class ALNSStateFact(_StrictModel):
    protocol: Literal["alns.State"]
    required_method: Literal["objective() -> float"]
    optimization_direction: Literal["minimization"]


class ALNSOperatorFact(_StrictModel):
    destroy_signature: Literal["destroy(state, rng, **kwargs) -> state"]
    repair_signature: Literal["repair(state, rng, **kwargs) -> state"]
    copy_requirement: str = Field(min_length=1)
    feasibility_responsibility: str = Field(min_length=1)


class ALNSReproducibilityFact(_StrictModel):
    rng_constructor: Literal["numpy.random.default_rng(seed)"]
    fixed_seed_scope: Literal["random_stream_only"]
    runtime_stop_strictly_reproducible: Literal[False]


class ALNSInterfaceFacts(_StrictModel):
    state: ALNSStateFact
    operators: ALNSOperatorFact
    reproducibility: ALNSReproducibilityFact
    guarantee_boundary: str = Field(min_length=1)


class ALNSSolver(_StrictModel):
    id: Literal["adaptive_large_neighborhood_search"]
    method: Literal["large_neighborhood_search"]
    method_class: Literal["metaheuristic"]
    optimality: Literal["heuristic_or_incumbent"]
    constructor: Literal["alns.ALNS(rng)"]
    registration_calls: tuple[str, ...] = Field(min_length=2, max_length=2)
    call: Literal["alns.iterate(initial_solution, op_select, accept, stop, **kwargs)"]
    supports_warm_start: Literal[True]
    supports_random_seed: Literal[True]
    supports_time_limit: Literal[True]
    stopping_criteria: tuple[StoppingCriterionFact, ...] = Field(min_length=3, max_length=3)
    result: ResultInterface


ALNSSelectionScheme = Literal[
    "AlphaUCB",
    "MABSelector",
    "RandomSelect",
    "RouletteWheel",
    "SegmentedRouletteWheel",
]
ALNSAcceptanceCriterion = Literal[
    "AlwaysAccept",
    "GreatDeluge",
    "HillClimbing",
    "LateAcceptanceHillClimbing",
    "MovingAverageThreshold",
    "NonLinearGreatDeluge",
    "RandomAccept",
    "RecordToRecordTravel",
    "SimulatedAnnealing",
]
ALNSStoppingCriterion = Literal["MaxIterations", "MaxRuntime", "NoImprovement"]


class ALNSInterface(_StrictModel):
    profile: Literal["alns"]
    preferred_api: Literal["search_framework"]
    imports: tuple[str, ...] = Field(min_length=1)
    solvers: tuple[ALNSSolver, ...] = Field(min_length=1, max_length=1)
    selection_schemes: tuple[ALNSSelectionScheme, ...] = Field(min_length=5, max_length=5)
    acceptance_criteria: tuple[ALNSAcceptanceCriterion, ...] = Field(min_length=9, max_length=9)
    stopping_criteria: tuple[ALNSStoppingCriterion, ...] = Field(min_length=3, max_length=3)
    facts: ALNSInterfaceFacts

    @model_validator(mode="after")
    def validate_framework_suite(self) -> "ALNSInterface":
        if self.solvers[0].id != "adaptive_large_neighborhood_search":
            raise ValueError("alns interface requires its adaptive search solver")
        if len(set(self.selection_schemes)) != 5:
            raise ValueError("alns interface requires every operator selection scheme once")
        if len(set(self.acceptance_criteria)) != 9:
            raise ValueError("alns interface requires every acceptance criterion once")
        if set(self.stopping_criteria) != {"MaxIterations", "MaxRuntime", "NoImprovement"}:
            raise ValueError("alns interface requires every stopping criterion once")
        return self


class PySCIPOptSolverBase(_StrictModel):
    imports: tuple[str, ...] = Field(min_length=1)
    constructor: Literal["pyscipopt.Model(problemName='model')"]
    call: Literal["model.optimize()"]
    method: str = Field(min_length=1)
    method_class: Literal["exact"]
    optimality: Literal["exact_or_incumbent"]
    supports_warm_start: Literal[True]
    supports_random_seed: Literal[True]
    supports_time_limit: Literal[True]
    stopping_criteria: tuple[StoppingCriterionFact, ...] = Field(min_length=1)
    important_parameters: dict[str, str] = Field(default_factory=dict)
    result: ResultInterface


class PySCIPOptMILPSolver(PySCIPOptSolverBase):
    id: Literal["milp"]
    method: Literal["mixed_integer_linear_programming"]


class PySCIPOptMINLPSolver(PySCIPOptSolverBase):
    id: Literal["minlp"]
    method: Literal["mixed_integer_nonlinear_programming"]


PySCIPOptSolver = Annotated[
    PySCIPOptMILPSolver | PySCIPOptMINLPSolver,
    Field(discriminator="id"),
]


class PySCIPOptPluginFacts(_StrictModel):
    base_classes: tuple[Literal["Pricer", "Heur", "Conshdlr"], ...] = Field(
        min_length=3, max_length=3
    )
    include_calls: tuple[
        Literal["Model.includePricer", "Model.includeHeur", "Model.includeConshdlr"], ...
    ] = Field(min_length=3, max_length=3)
    implementation_boundary: str = Field(min_length=1)
    branch_and_price_boundary: str = Field(min_length=1)


class PySCIPOptInterfaceFacts(_StrictModel):
    plugins: PySCIPOptPluginFacts
    nonlinear_objective_boundary: str = Field(min_length=1)
    proof_boundary: str = Field(min_length=1)
    solution_access_boundary: str = Field(min_length=1)


class PySCIPOptInterface(_StrictModel):
    profile: Literal["pyscipopt"]
    preferred_api: Literal["solver_suite"]
    imports: tuple[str, ...] = Field(min_length=1)
    model_builder: ModelBuilderInterface
    solvers: tuple[PySCIPOptSolver, ...] = Field(min_length=2, max_length=2)
    facts: PySCIPOptInterfaceFacts

    @model_validator(mode="after")
    def validate_solver_suite(self) -> "PySCIPOptInterface":
        if {solver.id for solver in self.solvers} != {"milp", "minlp"}:
            raise ValueError("pyscipopt interface requires milp and minlp solver modes")
        if set(self.facts.plugins.base_classes) != {"Pricer", "Heur", "Conshdlr"}:
            raise ValueError("pyscipopt interface requires Pricer, Heur, Conshdlr plugin facts")
        if set(self.facts.plugins.include_calls) != {
            "Model.includePricer",
            "Model.includeHeur",
            "Model.includeConshdlr",
        }:
            raise ValueError("pyscipopt interface requires all plugin registration calls")
        return self


AlgorithmInterface = Annotated[
    PyVRPIteratedLocalSearchInterface
    | PyJobShopInterface
    | JobShopLibInterface
    | GurobiInterface
    | ORToolsInterface
    | RSOMEInterface
    | ALNSInterface
    | PySCIPOptInterface,
    Field(discriminator="profile"),
]


class AgentGuidance(_StrictModel):
    algorithm_design: tuple[str, ...] = ()
    solving: tuple[str, ...] = ()
    minimal_call_example: str = Field(min_length=1)


class CommonSolverError(_StrictModel):
    error: str = Field(min_length=1)
    wrong: str = Field(min_length=1)
    correct: str = Field(min_length=1)


class SolverDocumentation(_StrictModel):
    id: str = Field(pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
    summary: str = Field(min_length=1)
    minimal_call_example: str = Field(min_length=1)
    common_errors: tuple[CommonSolverError, ...] = ()


class AvailabilityCheck(_StrictModel):
    command: str = Field(min_length=1)
    expected_version: str = Field(min_length=1)


class ValidationSpec(_StrictModel):
    availability_check: AvailabilityCheck
    acceptance_requirements: tuple[str, ...] = Field(min_length=1)
    benchmark_status: str = Field(min_length=1)


class AlgorithmManifest(_StrictModel):
    schema_version: Literal["1.3"]
    id: str = Field(pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
    display_name: str = Field(min_length=1)
    status: Literal["experimental", "available", "deprecated", "disabled"]
    kind: Literal["python_library", "executable", "service"]
    summary: str = Field(min_length=1)
    algorithm: AlgorithmSpec
    selection: SelectionSpec
    distribution: DistributionSpec
    interface: AlgorithmInterface
    solver_documentation: tuple[SolverDocumentation, ...] = Field(min_length=1)
    agent_guidance: AgentGuidance
    validation: ValidationSpec

    @model_validator(mode="after")
    def validate_solver_documentation(self) -> "AlgorithmManifest":
        documented = [item.id for item in self.solver_documentation]
        if len(documented) != len(set(documented)):
            raise ValueError("solver_documentation contains duplicate solver ids")
        interface_solvers = getattr(self.interface, "solvers", None)
        if interface_solvers is not None:
            expected = {str(item.id) for item in interface_solvers}
        elif self.interface.profile == "pyvrp.ils":
            expected = {"iterated_local_search"}
        elif self.interface.profile == "pyjobshop":
            expected = {"constraint_programming"}
        else:
            expected = set()
        if set(documented) != expected:
            raise ValueError(
                "solver_documentation ids must match the package solver ids: "
                f"expected {sorted(expected)}, got {sorted(documented)}"
            )
        return self


class AlgorithmGuideRequest(_StrictModel):
    package_id: str = Field(min_length=1)
    solver_id: str | None = Field(default=None, min_length=1)
