"""Agent-facing tools for the validated algorithm package catalog."""

from __future__ import annotations

import json
from typing import Any

from ..core.models import AgentStage
from ..core.stage_agent import Tool

from .catalog import AlgorithmCatalog, AlgorithmCatalogError
from .models import AlgorithmGuideRequest, AlgorithmManifest


_GUIDE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "package_id": {
            "type": "string",
            "description": "Exact package id injected in the algorithm catalog.",
        },
        "solver_id": {
            "type": "string",
            "description": "Optional solver id from that package's solver list.",
        },
    },
    "required": ["package_id"],
    "additionalProperties": False,
}


class AlgorithmToolset:
    """Expose package summaries and complete guides to relevant stages."""

    def __init__(self, catalog: AlgorithmCatalog) -> None:
        self.catalog = catalog

    def tools(self, *, stage: AgentStage | None = None) -> tuple[Tool, ...]:
        if stage not in {AgentStage.ALGORITHM_DESIGN, AgentStage.SOLVING}:
            return ()
        return (
            Tool(
                name="get_algorithm_guide",
                description=(
                    "Return a package guide when only package_id is provided. Add solver_id to "
                    "return that solver's detailed interface and minimal usage example."
                ),
                parameters=_GUIDE_PARAMETERS,
                handler=self.get_algorithm_guide,
            ),
        )

    def catalog_summary(self) -> str:
        manifests = self.catalog.list_available()
        payload = {
            "count": len(manifests),
            "algorithms": [self._catalog_entry(manifest) for manifest in manifests],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    async def get_algorithm_guide(self, **kwargs: Any) -> str:
        try:
            request = AlgorithmGuideRequest.model_validate(kwargs)
            manifest = self.catalog.get(request.package_id)
            payload = (
                self._solver_guide(manifest, request.solver_id)
                if request.solver_id is not None
                else self._package_guide(manifest)
            )
        except (ValueError, AlgorithmCatalogError) as exc:
            return f"get_algorithm_guide error: {exc}"
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def _catalog_entry(manifest: AlgorithmManifest) -> dict[str, Any]:
        return {
            "id": manifest.id,
            "display_name": manifest.display_name,
            "status": manifest.status,
            "kind": manifest.kind,
            "summary": manifest.summary,
            "problem_families": list(manifest.algorithm.problem_families),
            "method": manifest.algorithm.method,
            "method_class": manifest.algorithm.method_class,
            "solvers": [
                {"id": item.id, "summary": item.summary}
                for item in manifest.solver_documentation
            ],
        }

    @staticmethod
    def _solver_entries(manifest: AlgorithmManifest) -> list[dict[str, Any]]:
        interface = manifest.interface.model_dump(mode="json")
        solvers = interface.get("solvers")
        if isinstance(solvers, list):
            return solvers
        if manifest.interface.profile == "pyvrp.ils":
            return [{
                "id": "iterated_local_search",
                "method": manifest.algorithm.method,
                "method_class": manifest.algorithm.method_class,
                "optimality": manifest.algorithm.optimality,
                "supports_warm_start": manifest.algorithm.supports_warm_start,
                "supports_random_seed": manifest.algorithm.supports_random_seed,
                "supports_time_limit": manifest.algorithm.supports_time_limit,
                "solve": interface["solve"],
                "result": interface["result"],
            }]
        if manifest.interface.profile == "pyjobshop":
            return [{
                "id": "constraint_programming",
                "method": manifest.algorithm.method,
                "method_class": manifest.algorithm.method_class,
                "optimality": manifest.algorithm.optimality,
                "supports_warm_start": manifest.algorithm.supports_warm_start,
                "supports_random_seed": manifest.algorithm.supports_random_seed,
                "supports_time_limit": manifest.algorithm.supports_time_limit,
                "solve": interface["solve"],
                "result": interface["result"],
            }]
        return []

    @classmethod
    def _package_guide(cls, manifest: AlgorithmManifest) -> dict[str, Any]:
        details = {item.id: item.summary for item in manifest.solver_documentation}
        solvers = [dict(item, summary=details[item["id"]]) for item in cls._solver_entries(manifest)]
        return {
            "package": {
                "id": manifest.id,
                "status": manifest.status,
                "kind": manifest.kind,
                "summary": manifest.summary,
                "algorithm": manifest.algorithm.model_dump(mode="json"),
                "selection": manifest.selection.model_dump(mode="json"),
                "distribution": manifest.distribution.model_dump(mode="json"),
                "validation": manifest.validation.model_dump(mode="json"),
            },
            "solvers": solvers,
        }

    @classmethod
    def _solver_guide(cls, manifest: AlgorithmManifest, solver_id: str) -> dict[str, Any]:
        documentation = {item.id: item for item in manifest.solver_documentation}
        solver = next(
            (item for item in cls._solver_entries(manifest) if item["id"] == solver_id),
            None,
        )
        if solver is None or solver_id not in documentation:
            available = ", ".join(sorted(documentation)) or "none"
            raise AlgorithmCatalogError(
                f"unknown solver_id {solver_id!r} for package {manifest.id!r}; "
                f"available: {available}"
            )
        doc = documentation[solver_id]
        return {
            "package_id": manifest.id,
            "solver_id": solver_id,
            "summary": doc.summary,
            "interface": solver,
            "package_facts": manifest.interface.model_dump(mode="json").get("facts", {}),
            "minimal_call_example": doc.minimal_call_example,
            "common_errors": [
                item.model_dump(mode="json") for item in doc.common_errors
            ],
            "solving_guidance": list(manifest.agent_guidance.solving),
        }
