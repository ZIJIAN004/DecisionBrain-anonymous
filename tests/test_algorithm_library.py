import asyncio
import json

import pytest
import yaml
from pydantic import ValidationError

from decisionbrain.algorithm_library import (
    AlgorithmManifest,
    AlgorithmToolset,
    LocalAlgorithmCatalog,
)
from decisionbrain.core.models import AgentStage
from decisionbrain.paths import ALGORITHM_MANIFESTS_DIR, PROMPTS_DIR


def run(coro):
    return asyncio.run(coro)


def load_manifest(algorithm_id: str) -> dict:
    path = ALGORITHM_MANIFESTS_DIR / f"{algorithm_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_real_catalog_strictly_validates_all_package_manifests():
    catalog = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR)

    assert [item.id for item in catalog.list_available()] == [
        "alns",
        "gurobipy",
        "job_shop_lib",
        "ortools",
        "pyjobshop",
        "pyscipopt",
        "pyvrp.ils",
        "rsome",
    ]
    assert all(item.schema_version == "1.3" for item in catalog.list_available())


def test_real_pyvrp_manifest_preserves_numeric_and_feasibility_boundaries():
    manifest = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR).get("pyvrp.ils")

    assert manifest.distribution.verified_version == "0.13.4"
    assert manifest.algorithm.method_class == "metaheuristic"
    assert manifest.interface.profile == "pyvrp.ils"
    assert manifest.interface.facts.route_visits.includes_depot is False
    assert manifest.interface.facts.feasibility.false_proves_problem_infeasible is False
    assert (
        manifest.interface.facts.numeric_inputs.positive_float_behavior == "truncated_toward_zero"
    )


def test_real_scheduling_package_manifests_preserve_complete_suites():
    catalog = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR)
    pyjobshop = catalog.get("pyjobshop")
    job_shop_lib = catalog.get("job_shop_lib")

    assert {backend.id for backend in pyjobshop.interface.backends} == {
        "ortools",
        "cpoptimizer",
    }
    assert {solver.id for solver in job_shop_lib.interface.solvers} == {
        "constraint_programming",
        "dispatching_rules",
        "simulated_annealing",
    }


def test_real_rsome_manifest_covers_both_models_and_every_packaged_adapter():
    manifest = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR).get("rsome")

    assert manifest.distribution.verified_version == "1.3.1"
    assert {solver.id for solver in manifest.interface.solvers} == {
        "robust_optimization",
        "distributionally_robust_optimization",
    }
    assert {adapter.id for adapter in manifest.interface.solver_adapters} == {
        "clp",
        "copt",
        "cplex",
        "ecos",
        "gurobi",
        "mosek",
        "ortools",
        "scipy",
    }
    assert manifest.interface.facts.default_solver == "scipy"


def test_rsome_manifest_rejects_incomplete_adapter_suite():
    raw = load_manifest("rsome")
    raw["interface"]["solver_adapters"].pop()

    with pytest.raises(ValidationError, match="at least 8 items"):
        AlgorithmManifest.model_validate(raw)


def test_real_alns_manifest_exposes_framework_contract_not_domain_operators():
    manifest = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR).get("alns")

    assert manifest.distribution.verified_version == "7.0.0"
    assert manifest.interface.solvers[0].id == "adaptive_large_neighborhood_search"
    assert manifest.interface.facts.state.required_method == "objective() -> float"
    assert manifest.interface.facts.operators.destroy_signature.startswith("destroy(")
    assert set(manifest.interface.stopping_criteria) == {
        "MaxIterations",
        "MaxRuntime",
        "NoImprovement",
    }


def test_alns_manifest_rejects_unknown_package_fact():
    raw = load_manifest("alns")
    raw["interface"]["facts"]["domain_solver"] = True

    with pytest.raises(ValidationError, match="domain_solver"):
        AlgorithmManifest.model_validate(raw)


def test_real_pyscipopt_manifest_separates_milp_minlp_and_plugin_boundaries():
    manifest = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR).get("pyscipopt")

    assert manifest.distribution.verified_version == "6.2.1"
    assert {solver.id for solver in manifest.interface.solvers} == {"milp", "minlp"}
    assert set(manifest.interface.facts.plugins.base_classes) == {
        "Pricer",
        "Heur",
        "Conshdlr",
    }
    assert "only one building block" in manifest.interface.facts.plugins.branch_and_price_boundary


def test_pyscipopt_manifest_rejects_incomplete_solver_suite():
    raw = load_manifest("pyscipopt")
    raw["interface"]["solvers"].pop()

    with pytest.raises(ValidationError, match="at least 2 items"):
        AlgorithmManifest.model_validate(raw)


def test_manifest_rejects_unknown_common_fields():
    raw = load_manifest("pyvrp.ils")
    raw["unexpected"] = True

    with pytest.raises(ValidationError, match="unexpected"):
        AlgorithmManifest.model_validate(raw)


def test_catalog_omits_disabled_algorithms(tmp_path):
    active = load_manifest("alns")
    disabled = load_manifest("alns")
    disabled["id"] = "alns.disabled"
    disabled["status"] = "disabled"
    (tmp_path / "active.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    (tmp_path / "disabled.yaml").write_text(yaml.safe_dump(disabled), encoding="utf-8")

    catalog = LocalAlgorithmCatalog(tmp_path)

    assert [item.id for item in catalog.list_available()] == ["alns"]


def test_algorithm_toolset_summary_is_shallow_and_guides_are_complete():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    summary = json.loads(toolset.catalog_summary())
    rsome_guide = json.loads(run(toolset.get_algorithm_guide(package_id="rsome")))
    alns_guide = json.loads(run(toolset.get_algorithm_guide(package_id="alns")))
    pyscipopt_guide = json.loads(run(toolset.get_algorithm_guide(package_id="pyscipopt")))

    assert summary["count"] == 8
    entries = {item["id"]: item for item in summary["algorithms"]}
    assert set(entries["rsome"]) == {
        "id",
        "display_name",
        "status",
        "kind",
        "summary",
        "problem_families",
        "method",
        "method_class",
        "solvers",
    }
    assert "interface" not in entries["rsome"]
    assert len(entries["rsome"]["summary"].splitlines()) == 1
    assert len(entries["rsome"]["solvers"]) == 2
    assert all(set(item) == {"id", "summary"} for item in entries["rsome"]["solvers"])
    assert len(rsome_guide["solvers"]) == 2
    assert alns_guide["solvers"][0]["id"] == "adaptive_large_neighborhood_search"
    assert {solver["id"] for solver in pyscipopt_guide["solvers"]} == {
        "milp",
        "minlp",
    }


def test_solver_guide_returns_one_solver_with_its_minimal_example():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    guide = json.loads(
        run(toolset.get_algorithm_guide(package_id="gurobipy", solver_id="milp"))
    )

    assert guide["package_id"] == "gurobipy"
    assert guide["solver_id"] == "milp"
    assert guide["interface"]["method"] == "mixed_integer_linear_programming"
    assert "model.optimize()" in guide["minimal_call_example"]
    assert 'Model("lp")' not in guide["minimal_call_example"]
    assert all(
        set(item) == {"error", "wrong", "correct"}
        for item in guide["common_errors"]
    )
    assert any("MIPStarts" in item["wrong"] for item in guide["common_errors"])


def test_single_solver_packages_use_the_same_three_level_contract():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    summary = json.loads(toolset.catalog_summary())
    entries = {item["id"]: item for item in summary["algorithms"]}
    assert entries["pyvrp.ils"]["solvers"][0]["id"] == "iterated_local_search"
    assert entries["pyjobshop"]["solvers"][0]["id"] == "constraint_programming"

    detail = json.loads(
        run(
            toolset.get_algorithm_guide(
                package_id="pyjobshop", solver_id="constraint_programming"
            )
        )
    )
    assert detail["interface"]["solve"]["call"].startswith("model.solve(")
    assert "model.solve" in detail["minimal_call_example"]


def test_solver_examples_and_common_error_snippets_are_separate():
    catalog = LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR)

    for manifest in catalog.list_available():
        for documentation in manifest.solver_documentation:
            example = documentation.minimal_call_example.strip()
            for common_error in documentation.common_errors:
                assert common_error.wrong.strip() not in example
                assert common_error.correct.strip() not in example


def test_gurobi_package_guide_states_the_official_size_limited_license_bounds():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    guide = json.loads(run(toolset.get_algorithm_guide(package_id="gurobipy")))
    limitations = " ".join(guide["package"]["algorithm"]["limitations"])

    assert "2,000 variables" in limitations
    assert "2,000 linear constraints" in limitations
    assert "200 variables when quadratic terms are present" in limitations


def test_algorithm_tools_have_stage_specific_visibility():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    for stage in (AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING):
        assert [tool.name for tool in toolset.tools(stage=stage)] == ["get_algorithm_guide"]
    assert toolset.tools(stage=AgentStage.INTAKE) == ()
    assert toolset.tools(stage=AgentStage.PROBLEM_CONTRACT) == ()
    assert toolset.tools(stage=AgentStage.EXPLANATION) == ()
    assert toolset.tools(stage=None) == ()


def test_get_algorithm_guide_returns_readable_unknown_id_error():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    result = run(toolset.get_algorithm_guide(package_id="missing.algorithm"))

    assert result.startswith("get_algorithm_guide error: unknown package_id")
    assert "rsome" in result
    assert "pyscipopt" in result


def test_get_algorithm_guide_returns_readable_unknown_solver_error():
    toolset = AlgorithmToolset(LocalAlgorithmCatalog(ALGORITHM_MANIFESTS_DIR))

    result = run(toolset.get_algorithm_guide(package_id="gurobipy", solver_id="cp_sat"))

    assert result.startswith("get_algorithm_guide error: unknown solver_id")
    assert "milp" in result
    assert "lp" in result


def test_stage_prompts_require_shortlisting_guides_and_formulations():
    common = (PROMPTS_DIR / "system" / "stage_common_system.txt").read_text(encoding="utf-8")
    library = (PROMPTS_DIR / "system" / "algorithm_library_system.txt").read_text(encoding="utf-8")
    contract = (PROMPTS_DIR / "developer" / "algorithm_design_contract.txt").read_text(
        encoding="utf-8"
    )
    solving = (PROMPTS_DIR / "system" / "solving_system.txt").read_text(encoding="utf-8")

    assert "get_algorithm_guide" not in common
    assert "不得为了发现候选而逐个调用" in library
    assert "系统不提供 recipe" in library
    assert "mixed_integer_nonlinear_programming" in contract
    assert "distributionally_robust_optimization" in contract
    assert '"capability_mapping"' in contract
    assert '"integration"' in contract
    assert "package 组件必须真实调用 manifest 中所选 solver_id 的 API" in solving
