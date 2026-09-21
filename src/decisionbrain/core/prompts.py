"""Core prompt resource loader.

Core declares required prompts; Runtime resolves and injects their directory from Settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class PromptBundle(BaseModel):
    """All prompt text required by one Core execution."""

    stage_common_system: str
    algorithm_library_system: str = ""
    intake_system: str
    intake_contract: str
    problem_contract_system: str
    problem_contract_contract: str
    algorithm_design_system: str
    algorithm_design_contract: str
    gurobi_formulator_system: str = ""
    gurobi_formulator_contract: str = ""
    solving_system: str
    solving_contract: str
    solving_translation_system: str = ""
    solving_translation_contract: str = ""
    solving_self_check_system: str = ""
    solving_self_check_contract: str = ""
    feasibility_review_system: str
    feasibility_review_contract: str
    explain_system: str
    explain_contract: str
    components_enabled: bool = True

    model_config = ConfigDict(extra="forbid", frozen=True)


def _read_prompt(directory: Path, name: str) -> str:
    return (directory / name).read_text(encoding="utf-8").strip()


def _read_optional_prompt(directory: Path, name: str, *, fallback: str) -> str:
    path = directory / name
    return path.read_text(encoding="utf-8").strip() if path.is_file() else fallback


# Use one ordered suffix per ablation dimension for deterministic combined-arm names.
ABLATION_SUFFIXES: dict[str, str] = {
    "input_schema": "_no_input_schema",
    "algorithm_library": "_no_algorithm_library",
    "feasible_review": "_no_feasible_review",
}

# Each prompt is sensitive only to dimensions it mentions. Maintain this map explicitly so
# variants neither require irrelevant files nor receive contradictory instructions.
_PROMPT_SENSITIVITY: dict[str, tuple[str, ...]] = {
    "problem_contract_system.txt": ("input_schema", "feasible_review"),
    "problem_contract_contract.txt": ("input_schema", "feasible_review"),
    "algorithm_library_system.txt": ("algorithm_library",),
    "algorithm_design_system.txt": ("algorithm_library", "feasible_review"),
    "algorithm_design_contract.txt": ("algorithm_library", "feasible_review"),
    "solving_system.txt": ("input_schema", "algorithm_library"),
    "solving_contract.txt": ("input_schema", "algorithm_library"),
    "solving_self_check_system.txt": ("input_schema", "algorithm_library", "feasible_review"),
    "solving_self_check_contract.txt": ("input_schema", "algorithm_library", "feasible_review"),
}


class MissingPromptVariantError(FileNotFoundError):
    """An ablation arm has no rewritten prompt for a file that names what it removed."""


def _suffix_for(name: str, disabled: frozenset[str]) -> str:
    """Compose the variant suffix from the disabled dimensions this file cares about."""

    relevant = _PROMPT_SENSITIVITY.get(name, ())
    return "".join(
        ABLATION_SUFFIXES[dim] for dim in ABLATION_SUFFIXES if dim in relevant and dim in disabled
    )


def _read_variant_prompt(directory: Path, name: str, disabled: frozenset[str]) -> str:
    """Load the arm's prompt, or raise if it removes something this file still describes.

    Silently falling back can give an ablation arm contradictory instructions.
    A missing variant is a repository error and must fail at load time.
    """

    suffix = _suffix_for(name, disabled)
    if not suffix:
        return _read_prompt(directory, name)
    stem, ext = Path(name).stem, Path(name).suffix
    path = directory / f"{stem}{suffix}{ext}"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    raise MissingPromptVariantError(
        f"Ablation arm is missing prompt variant {path}; `{name}` references content removed "
        "by that arm, so an explicit variant is required."
    )


def load_prompt_bundle(
    prompts_dir: str | Path,
    *,
    input_schema_enabled: bool = True,
    algorithm_library_enabled: bool = True,
    feasibility_review_enabled: bool = True,
    components_enabled: bool = True,
    algorithm_design_enabled: bool = True,
    problem_contract_enabled: bool = True,
) -> PromptBundle:
    """Load prompts from the specified directory.

    Directory layout::
        {dir}/system/*.txt     system-role prompts
        {dir}/developer/*.txt developer contracts and format prompts

    Prompts affected by a disabled ablation dimension use suffixed variants.
    Missing variants raise an error rather than silently falling back. The
    components ablation is an independent single-factor arm.
    """
    if not components_enabled and not all(
        (input_schema_enabled, algorithm_library_enabled, feasibility_review_enabled, problem_contract_enabled)
    ) and algorithm_design_enabled:
        raise ValueError(
            "The components ablation is a single-factor experiment and cannot be combined "
            "with other workflow ablations"
        )

    root = Path(prompts_dir).expanduser().resolve()
    sys_dir = root / "system"
    dev_dir = root / "developer"
    disabled = frozenset(
        dim
        for dim, on in (
            ("input_schema", input_schema_enabled),
            ("algorithm_library", algorithm_library_enabled),
            ("feasible_review", feasibility_review_enabled),
        )
        if not on
    )
    if not components_enabled:
        algorithm_design_system = _read_prompt(
            sys_dir, "algorithm_design_system_no_components.txt"
        )
        algorithm_design_contract = _read_prompt(
            dev_dir, "algorithm_design_contract_no_components.txt"
        )
        solving_system = _read_prompt(
            sys_dir,
            "solving_system_gurobi_formulator.txt"
            if not algorithm_design_enabled
            else "solving_system_no_components.txt",
        )
        solving_contract = _read_prompt(
            dev_dir,
            "solving_contract_gurobi_formulator.txt"
            if not algorithm_design_enabled
            else "solving_contract_no_components.txt",
        )
        if not algorithm_design_enabled and algorithm_library_enabled:
            gurobi_formulator_system = _read_prompt(sys_dir, "gurobi_formulator_system.txt")
            gurobi_formulator_contract = _read_prompt(dev_dir, "gurobi_formulator_contract.txt")
            solving_translation_system = _read_prompt(sys_dir, "solving_system_gurobi_translation_only.txt")
            solving_translation_contract = _read_prompt(dev_dir, "solving_contract_gurobi_translation_only.txt")
        else:
            gurobi_formulator_system = ""
            gurobi_formulator_contract = ""
            solving_translation_system = ""
            solving_translation_contract = ""
        solving_self_check_system = solving_system
        solving_self_check_contract = solving_contract
    else:
        gurobi_formulator_system = ""
        gurobi_formulator_contract = ""
        solving_translation_system = ""
        solving_translation_contract = ""
        solving_system = _read_variant_prompt(sys_dir, "solving_system.txt", disabled)
        solving_contract = _read_variant_prompt(dev_dir, "solving_contract.txt", disabled)
        algorithm_design_system = (
            _read_variant_prompt(sys_dir, "algorithm_design_system.txt", disabled)
            if algorithm_design_enabled
            else _read_prompt(sys_dir, "algorithm_design_system.txt")
        )
        algorithm_design_contract = (
            _read_variant_prompt(dev_dir, "algorithm_design_contract.txt", disabled)
            if algorithm_design_enabled
            else _read_prompt(dev_dir, "algorithm_design_contract.txt")
        )
        if feasibility_review_enabled:
            solving_self_check_system = _read_optional_prompt(
                sys_dir, "solving_self_check_system.txt", fallback=solving_system
            )
            solving_self_check_contract = _read_optional_prompt(
                dev_dir, "solving_self_check_contract.txt", fallback=solving_contract
            )
        else:
            solving_self_check_system = _read_variant_prompt(
                sys_dir, "solving_self_check_system.txt", disabled
            )
            solving_self_check_contract = _read_variant_prompt(
                dev_dir, "solving_self_check_contract.txt", disabled
            )
    feasibility_review_contract = _read_prompt(
        dev_dir,
        (
            "feasibility_review_contract_gurobi_formulator.txt"
            if not algorithm_design_enabled and not components_enabled
            else "feasibility_review_contract_no_components.txt"
            if not components_enabled
            else "feasibility_review_contract.txt"
        ),
    )
    if not components_enabled:
        algorithm_library_system = _read_prompt(
            sys_dir, "algorithm_library_system_no_components.txt"
        )
    else:
        algorithm_library_system = _read_variant_prompt(
            sys_dir, "algorithm_library_system.txt", disabled
        )
    problem_contract_system = (
        _read_variant_prompt(sys_dir, "problem_contract_system.txt", disabled)
        if problem_contract_enabled
        else _read_prompt(sys_dir, "problem_contract_system.txt")
    )
    problem_contract_contract = (
        _read_variant_prompt(dev_dir, "problem_contract_contract.txt", disabled)
        if problem_contract_enabled
        else _read_prompt(dev_dir, "problem_contract_contract.txt")
    )
    bundle = PromptBundle(
        stage_common_system=_read_prompt(sys_dir, "stage_common_system.txt"),
        algorithm_library_system=algorithm_library_system,
        intake_system=_read_prompt(sys_dir, "intake_system.txt"),
        intake_contract=_read_prompt(dev_dir, "intake_contract.txt"),
        problem_contract_system=problem_contract_system,
        problem_contract_contract=problem_contract_contract,
        algorithm_design_system=algorithm_design_system,
        algorithm_design_contract=algorithm_design_contract,
        gurobi_formulator_system=gurobi_formulator_system,
        gurobi_formulator_contract=gurobi_formulator_contract,
        solving_translation_system=solving_translation_system,
        solving_translation_contract=solving_translation_contract,
        solving_system=solving_system,
        solving_contract=solving_contract,
        # Parse self-check prompts strictly because review-off arms actually use them; falling back to
        # solving_* would reintroduce Feasibility Reviewer instructions.
        solving_self_check_system=solving_self_check_system,
        solving_self_check_contract=solving_self_check_contract,
        feasibility_review_system=_read_prompt(
            sys_dir,
            (
                "feasibility_review_system_gurobi_formulator.txt"
                if not algorithm_design_enabled and not components_enabled
                else "feasibility_review_system_no_components.txt"
                if not components_enabled
                else "feasibility_review_system.txt"
            ),
        ),
        feasibility_review_contract=feasibility_review_contract,
        explain_system=_read_prompt(sys_dir, "explain_system.txt"),
        explain_contract=_read_prompt(dev_dir, "explain_contract.txt"),
        components_enabled=components_enabled,
    )
    return bundle
