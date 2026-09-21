"""Single-factor ablation policy for package pools and cross-package composition.

The core abstraction is a **package unit**, the smallest unit counted as one package.

- Each manifest ``package_id`` is one unit.
- ``generated`` is a virtual unit representing NumPy plus SciPy. This pairing is
  part of the experiment definition and is not a free foundation.
- The Python standard library belongs to no unit and is always available.

Two switches control the policy:

- ``pool`` (arm A) narrows the available units. ``gurobi-only`` permits only gurobipy.
- ``cross_package_enabled`` (arm B) keeps the complete catalog and composition
  features, but constrains one run to the unit selected by the algorithm design.

The restriction is enforced at three layers: catalog visibility, contract and
static-import validation, and subprocess import enforcement.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# A handwritten numerical implementation counts as one unit: NumPy and SciPy together.
GENERATED_UNIT = "generated"
GENERATED_UNIT_MODULES: tuple[str, ...] = ("numpy", "scipy")

# Unit to top-level import module names.
UNIT_IMPORT_MODULES: dict[str, tuple[str, ...]] = {
    "alns": ("alns",),
    "gurobipy": ("gurobipy",),
    "job_shop_lib": ("job_shop_lib",),
    "ortools": ("ortools",),
    "pyjobshop": ("pyjobshop",),
    "pyscipopt": ("pyscipopt",),
    "pyvrp.ils": ("pyvrp",),
    "rsome": ("rsome",),
    GENERATED_UNIT: GENERATED_UNIT_MODULES,
}

# Manifest package IDs exposed by the catalog, excluding virtual units.
MANIFEST_UNITS: frozenset[str] = frozenset(UNIT_IMPORT_MODULES) - {GENERATED_UNIT}

# Block installed solvers/modeling layers without manifests in restricted environments so a
# gurobipy-only arm cannot reach CBC, GLPK, or HiGHS through Pyomo or PuLP.
UNDECLARABLE_SOLVER_MODULES: tuple[str, ...] = (
    "cvxpy",
    "highspy",
    "mip",
    "pulp",
    "pyomo",
    "swiglpk",
    # pyvroom is installed as vroom but has no manifest, so every arm must block it. Otherwise a
    # restricted arm could bypass the catalog and the ablation would no longer be single-factor.
    "vroom",
)

# Pool name to allowed manifest package IDs; None leaves the catalog unfiltered.
PACKAGE_POOLS: dict[str, frozenset[str] | None] = {
    "full": None,
    "gurobi-only": frozenset({"gurobipy"}),
}

_MODULE_TO_UNIT: dict[str, str] = {
    module: unit for unit, modules in UNIT_IMPORT_MODULES.items() for module in modules
}


@dataclass(frozen=True)
class PackagePolicy:
    """Package-pool and cross-package policy for one run."""

    pool: str = "full"
    cross_package_enabled: bool = True

    def __post_init__(self) -> None:
        if self.pool not in PACKAGE_POOLS:
            raise ValueError(
                f"unknown package pool {self.pool!r}; available: "
                + ", ".join(sorted(PACKAGE_POOLS))
            )

    @property
    def allowed_package_ids(self) -> frozenset[str] | None:
        """Return catalog-visible package IDs; ``None`` means the full catalog."""

        return PACKAGE_POOLS[self.pool]

    @property
    def is_default(self) -> bool:
        return self.pool == "full" and self.cross_package_enabled

DEFAULT_PACKAGE_POLICY = PackagePolicy()


# Unit resolution


def module_unit(module_root: str) -> str | None:
    """Map a top-level module to its unit, or return ``None`` when unrestricted."""

    return _MODULE_TO_UNIT.get(module_root)


def blocked_modules_for(allowed_units: Collection[str]) -> tuple[str, ...]:
    """Return modules whose imports must fail at execution time."""

    allowed = frozenset(allowed_units)
    blocked = {
        module
        for unit, modules in UNIT_IMPORT_MODULES.items()
        if unit not in allowed
        for module in modules
    }
    blocked.update(UNDECLARABLE_SOLVER_MODULES)
    for unit in allowed:
        blocked.difference_update(UNIT_IMPORT_MODULES.get(unit, ()))
    return tuple(sorted(blocked))


def declared_units(design: Mapping[str, Any] | None) -> frozenset[str]:
    """Return units declared by a design, using ``generated`` when appropriate."""

    if not isinstance(design, Mapping):
        return frozenset()
    packages = {package_id for _, package_id in _walk_design_packages(design)}
    if packages:
        return frozenset(packages)
    if any(_walk_design_sources(design)):
        return frozenset({GENERATED_UNIT})
    return frozenset()


def solving_units(policy: PackagePolicy, design: Mapping[str, Any] | None) -> frozenset[str] | None:
    """Return units allowed in solving subprocesses, or ``None`` if unrestricted.

    Arm A is fixed by the pool. Arm B follows the algorithm design and remains
    unrestricted until that design is known.
    """

    pool = policy.allowed_package_ids
    if pool is not None:
        return pool
    if policy.cross_package_enabled:
        return None
    declared = declared_units(design)
    if not declared:
        return None
    # A declared package takes precedence; generated components then use only the standard library.
    manifest = declared & MANIFEST_UNITS
    return manifest or frozenset({GENERATED_UNIT})


# Additional prompt constraints


def _unit_description(units: Collection[str]) -> str:
    return "、".join(
        "自写实现（numpy + scipy）" if unit == GENERATED_UNIT else unit for unit in sorted(units)
    )


def design_addendum(policy: PackagePolicy) -> str:
    """Build additional hard constraints for the algorithm-design stage."""

    lines: list[str] = []
    pool = policy.allowed_package_ids
    if pool is not None:
        blocked = "、".join(blocked_modules_for(pool))
        lines.append(
            "【算法包池限制（系统注入，只读）】\n"
            f"- 本次运行只允许使用 {_unit_description(pool)}，目录里也只有它。\n"
            f"- solving 子进程会拦截以下模块的导入并直接抛 ImportError：{blocked}。\n"
            "- 其中 numpy 与 scipy 同样被拦截。\n"
            f"- 每个组件都必须是 source=package，且 package.id 属于 {_unit_description(pool)}。\n"
        )
    if not policy.cross_package_enabled:
        lines.append(
            "【单包限制（系统注入，只读）】\n"
            "- 完整算法目录仍然可用，hybrid、多组件、问题分解、warm start 都不受限制。\n"
            "- 唯一的限制是本次运行只能落到一个包上：所有 source=package 的组件必须\n"
            "  共用同一个 package_id；同一个包内的不同 solver_id 可以自由组合。\n"
            "- numpy + scipy 合起来算作一个包（记作 generated）。一旦你声明了任何\n"
            "  source=package 组件，solving 子进程就会拦截 numpy 与 scipy 的导入，\n"
            "  此时 source=generated 的组件只能使用 Python 标准库。\n"
            "- 反过来，如果整套方案不声明任何 package，就等于选择了 numpy + scipy，\n"
            "  此时所有求解器包都会被拦截。请在两者之间明确择一。\n"
        )
    return "".join(lines)


def solving_addendum(policy: PackagePolicy, design: Mapping[str, Any] | None = None) -> str:
    """Build solving-stage constraints specialized to the selected units."""

    units = solving_units(policy, design)
    if units is None:
        if policy.cross_package_enabled:
            return ""
        return (
            "【单包限制（系统注入，只读）】\n"
            "- 本次运行只能落到一个包上；numpy + scipy 合起来算作一个包。\n"
            "- 所有 source=package 的执行记录必须共用同一个 executed_package_id。\n"
        )
    blocked = "、".join(blocked_modules_for(units))
    return (
        "【本次运行的可用算法包（系统注入，只读）】\n"
        f"- 只有 {_unit_description(units)} 可以使用。\n"
        f"- solver.py 与 solving 阶段的 shell 中导入以下模块会被运行时拦截并抛\n"
        f"  ImportError：{blocked}。\n"
        "- 该拦截由子进程环境强制执行，不是建议；请不要尝试用 python -S / -E / -I\n"
        "  或改写 PYTHONPATH 绕过，这类命令会被 shell 工具直接拒绝。\n"
        "- Python 标准库不受限制。\n"
        "- 每个组件都必须是 source=package。\n"
        + (
            "- 所有 source=package 的执行记录必须共用同一个 executed_package_id。\n"
            if not policy.cross_package_enabled
            else ""
        )
    )


# Contract validation


def _package_id(node: Mapping[str, Any]) -> str | None:
    if node.get("source") != "package":
        return None
    package = node.get("package")
    if isinstance(package, Mapping):
        value = package.get("id")
        return str(value) if value else None
    return None


def _iter_component_nodes(design: Mapping[str, Any]) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for holder in ("selection", "fallback"):
        node = design.get(holder)
        if not isinstance(node, Mapping):
            continue
        components = node.get("components")
        if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
            continue
        for index, component in enumerate(components):
            if isinstance(component, Mapping):
                yield f"$.{holder}.components[{index}]", component
    algorithm = design.get("algorithm")
    if isinstance(algorithm, Mapping):
        yield "$.algorithm", algorithm


def _walk_design_packages(design: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield ``(json_path, package_id)`` for component and single-algorithm outputs."""

    for path, node in _iter_component_nodes(design):
        package_id = _package_id(node)
        if package_id:
            yield f"{path}.package.id", package_id


def _walk_design_sources(design: Mapping[str, Any]) -> Iterator[str]:
    for _, node in _iter_component_nodes(design):
        source = node.get("source")
        if isinstance(source, str):
            yield source


@dataclass(frozen=True)
class PolicyViolation:
    path: str
    message: str
    expected: str
    actual: str | None = None


def _pool_violations(
    entries: Sequence[tuple[str, str]], policy: PackagePolicy, subject: str
) -> list[PolicyViolation]:
    allowed = policy.allowed_package_ids
    if allowed is None:
        return []
    return [
        PolicyViolation(
            path=path,
            message=f"{subject} package 不在本次运行的算法包池内",
            expected=", ".join(sorted(allowed)),
            actual=package_id,
        )
        for path, package_id in entries
        if package_id not in allowed
    ]


def _generated_violations(
    policy: PackagePolicy,
    nodes: Sequence[tuple[str, Mapping[str, Any]]],
    subject: str,
) -> list[PolicyViolation]:
    """Disallow ``source=generated`` when the package pool is restricted.

    Blocking NumPy and SciPy alone is insufficient because standard-library heuristics
    could pass easy feasibility checks. ``only`` therefore requires every component to
    use a package in the selected pool.
    """

    if policy.allowed_package_ids is None:
        return []
    return [
        PolicyViolation(
            path=path,
            message=f"{subject}不允许 source=generated：本次运行只能使用算法包池内的包",
            expected=", ".join(sorted(policy.allowed_package_ids)),
            actual="generated",
        )
        for path, node in nodes
        if node.get("source") == "generated"
    ]


def _cross_package_violation(
    entries: Sequence[tuple[str, str]], policy: PackagePolicy, path: str
) -> list[PolicyViolation]:
    if policy.cross_package_enabled:
        return []
    distinct = sorted({package_id for _, package_id in entries})
    if len(distinct) <= 1:
        return []
    return [
        PolicyViolation(
            path=path,
            message="本次运行禁止跨包组合，所有 source=package 必须共用同一个 package_id",
            expected="exactly one distinct package_id",
            actual=", ".join(distinct),
        )
    ]


def validate_design(
    design: Mapping[str, Any], policy: PackagePolicy
) -> tuple[PolicyViolation, ...]:
    """Check an algorithm design against package-pool and single-package rules."""

    if policy.is_default:
        return ()
    entries = list(_walk_design_packages(design))
    violations = _pool_violations(entries, policy, "设计中的")
    violations += _generated_violations(policy, list(_iter_component_nodes(design)), "设计中的组件")
    violations += _cross_package_violation(entries, policy, "$.selection.components[].package.id")
    return tuple(violations)


def validate_executions(
    solver_result: Mapping[str, Any], policy: PackagePolicy
) -> tuple[PolicyViolation, ...]:
    """Check execution records against package-pool and single-package rules."""

    if policy.is_default:
        return ()
    executions: list[tuple[str, Mapping[str, Any]]] = []
    raw = solver_result.get("executions")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        executions.extend(
            (f"$.solver_result.executions[{index}].executed_package_id", item)
            for index, item in enumerate(raw)
            if isinstance(item, Mapping)
        )
    single = solver_result.get("execution")
    if isinstance(single, Mapping):
        executions.append(("$.solver_result.execution.executed_package_id", single))

    entries: list[tuple[str, str]] = []
    for path, execution in executions:
        if execution.get("source") != "package":
            continue
        package_id = execution.get("executed_package_id")
        if package_id:
            entries.append((path, str(package_id)))

    violations = _pool_violations(entries, policy, "实际执行的")
    violations += _generated_violations(policy, executions, "实际执行的组件")
    violations += _cross_package_violation(
        entries, policy, "$.solver_result.executions[].executed_package_id"
    )
    return tuple(violations)


def import_roots(source: str) -> tuple[str, ...]:
    """Parse top-level imports statically, returning an empty set on syntax errors."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            # Track dynamic imports such as importlib.import_module here and in runtime guards.
            name = node.func
            attribute = getattr(name, "attr", None) or getattr(name, "id", None)
            if attribute in {"import_module", "__import__"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    roots.add(first.value.split(".", 1)[0])
    return tuple(sorted(roots))


def validate_source_imports(
    sources: Mapping[str, str],
    policy: PackagePolicy,
    design: Mapping[str, Any] | None,
) -> tuple[PolicyViolation, ...]:
    """Validate written code statically instead of trusting receipt package IDs.

    The execution guard remains authoritative, while this catches declaration and
    implementation mismatches before code is run.
    """

    units = solving_units(policy, design)
    if units is None:
        return ()
    blocked = frozenset(blocked_modules_for(units))
    violations: list[PolicyViolation] = []
    for filename, source in sources.items():
        for root in import_roots(source or ""):
            if root not in blocked:
                continue
            unit = module_unit(root)
            violations.append(
                PolicyViolation(
                    path=f"$.files.{filename}",
                    message=(
                        f"`{filename}` 导入了本次运行不允许的模块 `{root}`"
                        + (f"（属于 {unit}）" if unit else "")
                    ),
                    expected=f"只使用 {_unit_description(units)} 与 Python 标准库",
                    actual=root,
                )
            )
    return tuple(violations)


# Execution guard

_GUARD_TEMPLATE = '''"""Runtime import guard injected by the DecisionBrain package ablation."""

import sys

_BLOCKED = frozenset(__BLOCKED_JSON__)
_MESSAGE = " is disabled by the DecisionBrain package ablation for this run"


class _BlockedPackageFinder:
    """Raise ImportError for modules outside this run's package unit."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in _BLOCKED:
            raise ImportError("package " + root + _MESSAGE)
        return None


if not any(isinstance(item, _BlockedPackageFinder) for item in sys.meta_path):
    sys.meta_path.insert(0, _BlockedPackageFinder())
'''

_STUB_TEMPLATE = (
    '"""Shadow module injected by the DecisionBrain package ablation."""\n\n'
    "raise ImportError(\n"
    '    "package {module} is disabled by the DecisionBrain package ablation for this run"\n'
    ")\n"
)


def render_import_guard(allowed_units: Collection[str] | None) -> str | None:
    """Generate ``sitecustomize`` source, or ``None`` for an unrestricted environment."""

    if allowed_units is None:
        return None
    blocked = blocked_modules_for(allowed_units)
    if not blocked:
        return None
    return _GUARD_TEMPLATE.replace("__BLOCKED_JSON__", json.dumps(list(blocked)))


def render_module_stub(module: str) -> str:
    """Generate shadow-module source that also blocks imports under ``python -S``."""

    return _STUB_TEMPLATE.format(module=module)
