"""Central definitions for project resource paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
ALGORITHM_MANIFESTS_DIR = PROJECT_ROOT / "algorithms" / "manifests"
