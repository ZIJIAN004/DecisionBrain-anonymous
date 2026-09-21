"""包池（臂 A）与单包（臂 B）两个单因子消融开关的契约测试。

实验定义里 numpy + scipy 合起来算作一个包（unit ``generated``），因此：
- 臂 A（池 = {gurobipy}）必须同时排除其他求解器**和** numpy/scipy；
- 臂 B（完整池、一次只能落一个包）声明了 gurobipy 就拿不到 numpy/scipy，
  反之不声明任何 package 就只剩 numpy/scipy，求解器全被拦。
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from decisionbrain.algorithm_library import LocalAlgorithmCatalog
from decisionbrain.algorithm_library.catalog import AlgorithmCatalogError
from decisionbrain.benchmark import execution as runner_module
from decisionbrain.config import Settings
from decisionbrain.core.models import AgentStage
from decisionbrain.core.package_policy import (
    GENERATED_UNIT,
    PackagePolicy,
    blocked_modules_for,
    declared_units,
    design_addendum,
    import_roots,
    render_import_guard,
    solving_addendum,
    solving_units,
    validate_design,
    validate_executions,
    validate_source_imports,
)
from decisionbrain.core.workspace_tools import WorkspaceToolset

MANIFESTS_DIR = runner_module.PROJECT_ROOT / "algorithms" / "manifests"

ARM_A = PackagePolicy(pool="gurobi-only")
ARM_B = PackagePolicy(cross_package_enabled=False)
DEFAULT = PackagePolicy()


def _design(*package_ids: str | None) -> dict:
    components = []
    for index, package_id in enumerate(package_ids):
        if package_id is None:
            components.append({"component_id": f"c{index}", "source": "generated"})
        else:
            components.append(
                {
                    "component_id": f"c{index}",
                    "source": "package",
                    "package": {"id": package_id, "version": "1.0"},
                }
            )
    return {"selection": {"kind": "hybrid", "components": components}}


def _executions(*package_ids: str) -> dict:
    return {
        "executions": [
            {"source": "package", "executed_package_id": package_id} for package_id in package_ids
        ]
    }


def _toolset(tmp_path, policy: PackagePolicy) -> WorkspaceToolset:
    return WorkspaceToolset(
        tmp_path,
        timeout_s=60,
        solver_cpu_limit=1,
        max_output_chars=20000,
        max_list_entries=200,
        default_list_entries=50,
        max_read_bytes=200000,
        package_policy=policy,
    )


def _run_shell(toolset: WorkspaceToolset, command: str, stage=AgentStage.SOLVING) -> str:
    return asyncio.run(toolset.shell(command=command, stage=stage))


def _probe(tmp_path, module: str) -> str:
    path = tmp_path / f"probe_{module}.py"
    path.write_text(
        f"try:\n"
        f"    import {module}\n"
        f"    print('IMPORTED')\n"
        f"except ImportError as exc:\n"
        f"    print('BLOCKED', exc)\n",
        encoding="utf-8",
    )
    return path.name


def test_default_policy_is_inert():
    assert DEFAULT.is_default
    assert solving_units(DEFAULT, _design("gurobipy")) is None
    assert render_import_guard(None) is None
    assert design_addendum(DEFAULT) == ""
    assert solving_addendum(DEFAULT, _design("gurobipy")) == ""
    assert validate_design(_design("gurobipy", "ortools"), DEFAULT) == ()
    assert validate_executions(_executions("gurobipy", "ortools"), DEFAULT) == ()
    assert validate_source_imports({"solver.py": "import ortools"}, DEFAULT, None) == ()


def test_unknown_pool_is_rejected():
    with pytest.raises(ValueError, match="unknown package pool"):
        PackagePolicy(pool="scipy-only")


# Unit model: NumPy and SciPy count as one package.


def test_generated_unit_is_numpy_plus_scipy():
    blocked = set(blocked_modules_for({GENERATED_UNIT}))
    # A generated implementation permits NumPy/SciPy and blocks solver packages.
    assert {"gurobipy", "ortools", "pyscipopt", "alns", "pyvrp", "rsome", "pyomo"} <= blocked
    assert not blocked & {"numpy", "scipy"}


def test_choosing_a_solver_package_costs_numpy_and_scipy():
    blocked = set(blocked_modules_for({"gurobipy"}))
    assert {"numpy", "scipy"} <= blocked
    assert "gurobipy" not in blocked


def test_undeclarable_solvers_are_blocked_whenever_the_pool_narrows():
    # Packages without manifests cannot be declared and must not bypass policy through Pyomo/PuLP.
    for unit in ("gurobipy", GENERATED_UNIT):
        blocked = set(blocked_modules_for({unit}))
        assert {"pyomo", "pulp", "cvxpy", "mip", "highspy", "swiglpk"} <= blocked


def test_declared_units_treats_a_package_free_design_as_generated():
    assert declared_units(_design("gurobipy", None)) == frozenset({"gurobipy"})
    assert declared_units(_design(None, None)) == frozenset({GENERATED_UNIT})
    assert declared_units(None) == frozenset()


# Arm A: package pool


def test_arm_a_catalog_exposes_only_the_pool():
    full = LocalAlgorithmCatalog(MANIFESTS_DIR)
    pooled = LocalAlgorithmCatalog(MANIFESTS_DIR, allowed_package_ids=ARM_A.allowed_package_ids)
    assert len(full.list_available()) > 1
    assert [manifest.id for manifest in pooled.list_available()] == ["gurobipy"]
    # Catalog summaries and guides must expose the same pool.
    assert pooled.get("gurobipy").id == "gurobipy"
    with pytest.raises(AlgorithmCatalogError, match="unknown package_id"):
        pooled.get("ortools")


def test_arm_a_execution_environment_is_gurobipy_only_regardless_of_design():
    # Arm A is design-independent; generated components do not unlock NumPy/SciPy.
    for design in (None, _design(None, None), _design("gurobipy")):
        assert solving_units(ARM_A, design) == frozenset({"gurobipy"})
    assert {"numpy", "scipy", "ortools"} <= set(blocked_modules_for(solving_units(ARM_A, None)))


def test_arm_a_rejects_out_of_pool_design_and_execution():
    (violation,) = validate_design(_design("ortools"), ARM_A)
    assert violation.actual == "ortools"
    assert violation.path == "$.selection.components[0].package.id"
    assert validate_design(_design("gurobipy"), ARM_A) == ()

    (violation,) = validate_executions(_executions("pyscipopt"), ARM_A)
    assert violation.actual == "pyscipopt"
    assert validate_executions(_executions("gurobipy"), ARM_A) == ()


def test_arm_a_rejects_generated_components():
    """only 取字面意义：拦掉 numpy/scipy 还不够，自写组件本身也不允许。

    否则可行解容易构造的问题（枢纽选址、集合覆盖）可以靠一个纯标准库贪心通过
    hidden checker，臂 A 与主实验就无法分开。
    """

    (violation,) = validate_design(_design(None), ARM_A)
    assert violation.actual == "generated"
    assert violation.path == "$.selection.components[0]"

    # Mixing with a valid package component is also prohibited.
    violations = validate_design(_design("gurobipy", None), ARM_A)
    assert [v.actual for v in violations] == ["generated"]

    generated_execution = {"executions": [{"source": "generated"}]}
    (violation,) = validate_executions(generated_execution, ARM_A)
    assert violation.actual == "generated"

    # The main experiment remains unaffected.
    assert validate_design(_design("gurobipy", None), DEFAULT) == ()
    assert validate_executions(generated_execution, DEFAULT) == ()


def test_arm_a_prompt_states_that_numpy_and_scipy_are_gone():
    text = design_addendum(ARM_A)
    assert "numpy" in text and "scipy" in text
    # State positive requirements only; do not reveal generated alternatives.
    assert "每个组件都必须是 source=package" in text
    for word in ("generated", "自写"):
        assert word not in text


# Arm B: single package


def test_arm_b_keeps_the_full_catalog():
    """臂 B 只砍组合，不砍池：目录必须与主实验完全一致。"""

    assert ARM_B.allowed_package_ids is None
    assert solving_units(ARM_B, None) is None


def test_arm_b_narrows_the_runtime_to_the_declared_unit():
    assert solving_units(ARM_B, _design("gurobipy", "gurobipy")) == frozenset({"gurobipy"})
    # With a declared package, generated components use only the standard library.
    assert {"numpy", "scipy"} <= set(
        blocked_modules_for(solving_units(ARM_B, _design("gurobipy", None)))
    )
    # Declaring no package selects the NumPy/SciPy unit.
    assert solving_units(ARM_B, _design(None, None)) == frozenset({GENERATED_UNIT})


def test_arm_b_allows_multiple_components_from_one_package():
    # Components, decomposition, and warm starts remain; all share one package_id.
    assert validate_design(_design("gurobipy", "gurobipy", None), ARM_B) == ()
    assert validate_executions(_executions("gurobipy", "gurobipy"), ARM_B) == ()


def test_arm_b_rejects_cross_package_design_and_execution():
    (violation,) = validate_design(_design("gurobipy", "ortools"), ARM_B)
    assert violation.actual == "gurobipy, ortools"
    assert "禁止跨包组合" in violation.message

    (violation,) = validate_executions(_executions("gurobipy", "ortools"), ARM_B)
    assert violation.actual == "gurobipy, ortools"


def test_arm_b_also_covers_fallback_and_single_algorithm_shapes():
    design = {
        "selection": {
            "components": [{"source": "package", "package": {"id": "gurobipy", "version": "1"}}]
        },
        "fallback": {
            "components": [{"source": "package", "package": {"id": "ortools", "version": "1"}}]
        },
    }
    assert validate_design(design, ARM_B)
    single = {"algorithm": {"source": "package", "package": {"id": "ortools"}}}
    assert validate_design(single, ARM_A)


# Contract layer: generated code


def test_import_roots_sees_plain_and_dynamic_imports():
    source = (
        "import numpy as np\n"
        "from scipy.optimize import linprog\n"
        "import importlib\n"
        "m = importlib.import_module('ortools.sat.python.cp_model')\n"
    )
    assert {"numpy", "scipy", "ortools"} <= set(import_roots(source))


def test_source_import_check_catches_receipt_and_code_disagreeing():
    """receipt 声明 gurobipy、代码里却 import ortools —— 只看 receipt 抓不到。"""

    sources = {"solver.py": "import gurobipy\nimport ortools\n"}
    (violation,) = validate_source_imports(sources, ARM_B, _design("gurobipy"))
    assert violation.actual == "ortools"
    assert violation.path == "$.files.solver.py"


def test_source_import_check_enforces_numpy_as_a_package_in_arm_b():
    sources = {"solver.py": "import gurobipy\nimport numpy as np\n"}
    (violation,) = validate_source_imports(sources, ARM_B, _design("gurobipy"))
    assert violation.actual == "numpy"
    # A generated selection permits NumPy but not solver packages.
    generated = _design(None)
    assert validate_source_imports({"solver.py": "import numpy"}, ARM_B, generated) == ()
    assert validate_source_imports({"solver.py": "import gurobipy"}, ARM_B, generated)


def test_source_import_check_allows_the_standard_library():
    sources = {"solver.py": "import json\nimport itertools\nimport gurobipy\n"}
    assert validate_source_imports(sources, ARM_A, None) == ()


# Execution layer: subprocess guard


@pytest.mark.solver
def test_arm_a_import_guard_blocks_numpy_and_other_solvers_in_a_subprocess(tmp_path):
    """守卫必须真的让子进程 import 失败——提示词层面的禁止一定会有漏网。"""

    default_tools = _toolset(tmp_path, DEFAULT)
    arm_a_tools = _toolset(tmp_path, ARM_A)
    probe = _probe(tmp_path, "numpy")

    assert "IMPORTED" in _run_shell(default_tools, f'"{sys.executable}" {probe}')
    blocked = _run_shell(arm_a_tools, f'"{sys.executable}" {probe}')
    assert "BLOCKED" in blocked
    assert "package ablation" in blocked
    # Packages inside the pool remain available.
    assert "IMPORTED" in _run_shell(
        arm_a_tools, f'"{sys.executable}" {_probe(tmp_path, "gurobipy")}'
    )


def test_import_guard_survives_python_dash_s(tmp_path):
    """``python -S`` 不加载 sitecustomize；同名遮蔽模块负责堵住这个口。

    shell 工具会在命令层直接拒绝 ``-S``，这里绕过那一层，单独验证遮蔽模块本身，
    这样两道防线各自都有独立证据。
    """

    import os

    toolset = _toolset(tmp_path, ARM_A)
    probe = _probe(tmp_path, "numpy")
    env = toolset._guarded_env()
    assert env is not None

    async def run() -> str:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-S",
            probe,
            cwd=os.fspath(toolset.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        stdout, _ = await process.communicate()
        return stdout.decode("utf-8", "replace")

    assert "BLOCKED" in asyncio.run(run())


def test_shell_rejects_commands_that_would_bypass_the_guard(tmp_path):
    arm_a_tools = _toolset(tmp_path, ARM_A)
    for command in (
        f'"{sys.executable}" -S solver.py',
        f'"{sys.executable}" -E solver.py',
        f'"{sys.executable}" -I solver.py',
        f'PYTHONPATH=/tmp "{sys.executable}" solver.py',
        f'env -i "{sys.executable}" solver.py',
    ):
        assert "绕过" in _run_shell(arm_a_tools, command), command
    # Ordinary calls remain unaffected.
    assert "绕过" not in _run_shell(arm_a_tools, f'"{sys.executable}" -c "print(1)"')


@pytest.mark.solver
def test_guard_applies_to_solving_only(tmp_path):
    """包约束只针对求解本身；理解数据的阶段不该被牵连，否则不再是单因子。"""

    arm_a_tools = _toolset(tmp_path, ARM_A)
    probe = _probe(tmp_path, "numpy")
    assert "IMPORTED" in _run_shell(
        arm_a_tools, f'"{sys.executable}" {probe}', stage=AgentStage.INTAKE
    )
    assert "BLOCKED" in _run_shell(arm_a_tools, f'"{sys.executable}" {probe}')


@pytest.mark.solver
def test_arm_b_guard_follows_the_bound_design(tmp_path):
    tools = _toolset(tmp_path, ARM_B)
    numpy_probe = _probe(tmp_path, "numpy")
    gurobi_probe = _probe(tmp_path, "gurobipy")

    # Do not narrow before design is known.
    assert "IMPORTED" in _run_shell(tools, f'"{sys.executable}" {numpy_probe}')

    tools.bind_solving_design(_design("gurobipy"))
    assert "BLOCKED" in _run_shell(tools, f'"{sys.executable}" {numpy_probe}')
    assert "IMPORTED" in _run_shell(tools, f'"{sys.executable}" {gurobi_probe}')

    # A redesign to generated code must update the guard after review rollback.
    tools.bind_solving_design(_design(None))
    assert "IMPORTED" in _run_shell(tools, f'"{sys.executable}" {numpy_probe}')
    assert "BLOCKED" in _run_shell(tools, f'"{sys.executable}" {gurobi_probe}')


# Controlled-composition protection


@pytest.mark.parametrize(
    "policy", [ARM_A, ARM_B, PackagePolicy(pool="gurobi-only", cross_package_enabled=False)]
)
@pytest.mark.parametrize(
    "disabled",
    [
        {"algorithm_library_enabled": False},
        {"input_schema_enabled": False},
        {"components_enabled": False},
    ],
)
def test_package_arms_refuse_to_combine_with_other_ablations(policy, disabled):
    with pytest.raises(ValueError, match="package pool and cross-package ablations"):
        asyncio.run(
            runner_module.run_benchmark(
                [],
                Settings(),
                jobs=1,
                package_policy=policy,
                **disabled,
            )
        )


@pytest.mark.parametrize(
    "policy", [ARM_B, PackagePolicy(pool="gurobi-only", cross_package_enabled=False)]
)
def test_non_gurobi_only_package_arms_still_refuse_no_review(policy):
    with pytest.raises(ValueError, match="package pool and cross-package ablations"):
        asyncio.run(
            runner_module.run_benchmark(
                [],
                Settings(),
                jobs=1,
                package_policy=policy,
                feasibility_review_enabled=False,
            )
        )


def test_cli_exposes_package_pool_and_keeps_cross_package_enabled():
    parser = runner_module._parser()
    parser_args = parser.parse_args(["--package-pool", "gurobi-only"])
    assert parser_args.package_pool == "gurobi-only"
    assert parser_args.cross_package is True

    defaults = parser.parse_args([])
    assert defaults.package_pool == "full"
    assert defaults.cross_package is True
