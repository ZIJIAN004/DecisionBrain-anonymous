import ast
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LLM_MODEL_URL", "http://example.invalid/chat/completions")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_CHAT_MODEL", "test-model")


class CoreModuleTests(unittest.TestCase):
    def test_core_does_not_import_web_frameworks(self):
        forbidden = {"fastapi", "starlette", "typer", "rich"}
        for path in (ROOT / "src" / "decisionbrain" / "core").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".", 1)[0])
            self.assertFalse(imported & forbidden, f"{path.name}: {imported & forbidden}")

    def test_core_uses_ports_not_runtime_or_concrete_infrastructure(self):
        for path in (ROOT / "src" / "decisionbrain" / "core").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("from ..runtime", source, path.name)
            if path.name == "agent.py":
                # agent.py 是 Core ↔ infra 的装配点，允许导入 concrete adapters
                continue
            self.assertNotIn("from ..infrastructure", source, path.name)

    def test_runtime_driver_does_not_import_optimization_stages(self):
        source = (ROOT / "src" / "decisionbrain" / "runtime" / "agent_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("from ..core.workflow", source)
        self.assertNotIn("from ..core.stage_flow", source)

    def test_retired_validation_and_debugging_modules_are_removed(self):
        core_dir = ROOT / "src" / "decisionbrain" / "core"

        self.assertFalse((core_dir / "validation.py").exists())
        self.assertFalse((core_dir / "debugging.py").exists())
        self.assertFalse((core_dir / "workflow.py").exists())


if __name__ == "__main__":
    unittest.main()
