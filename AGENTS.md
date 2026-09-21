# Repository Guidelines

## Project Structure & Module Organization

DecisionBrain is a Python 3.10+ optimization-agent toolkit with a FastAPI backend and Typer CLI. Core package code lives in `src/decisionbrain/`: `api/` contains HTTP/SSE routes, `core/` owns the fixed six-stage workflow and validation, `infrastructure/` contains the LLM client, `runtime/` owns Run execution, and `run_storage/` stores Run records, artifacts, events, and workspaces. `events/` publishes and renders structured events, while `cli/` implements the `dbn` command. Tests live in `tests/`; prompt contracts in `prompts/`; docs in `docs/`; the simple web UI in `frontend/index.html`.

## Build, Test, and Development Commands

DecisionBrain is distributed only as a source repository. Do not add PyPI,
wheel, or sdist publication workflows. Install from a retained checkout for
local development:

```bash
conda activate decisionbrain
python -m pip install -e ".[dev]"
```

Useful commands:

- `dbn --help` / `python -m decisionbrain --help`: inspect CLI commands.
- `dbn doctor --json`: check environment, config, runs directory, and optional tools.
- `dbn dev demo-events --view normal`: create a sample Run and render event output.
- `conda run -n decisionbrain pytest`: run the test suite.
- `conda run -n decisionbrain ruff check .`: lint Python files.
- `conda run -n decisionbrain python -m decisionbrain.api.app`: start the local API/UI after loading `.env`.

## Coding Style & Naming Conventions

Use 4-space indentation, type hints for public interfaces, and clear module boundaries. Prefer snake_case for functions, variables, files, and tests; use PascalCase for classes and Pydantic models. Ruff targets Python 3.10 with a 100-character line length and rules `E4,E7,E9,F`. Runtime code should emit `RunEventDraft` through `EventPublisher`; do not print directly or write `events.jsonl` by hand.

## Testing Guidelines

Pytest discovers tests under `tests/`; name files `test_*.py` and tests `test_...`. Use deterministic LLM and solver fakes rather than real model calls or Gurobi licenses. Add focused tests near the behavior changed, e.g. CLI behavior in `tests/test_cli.py`, run storage in `tests/test_runs.py`, and events in `tests/test_events.py`.

## Commit & Pull Request Guidelines

Use concise Conventional Commit-style messages, for example `feat(cli): add event stream and demo command` or `refactor: separate core Agent and API layers`. Include a type, an optional scope, and a concrete English summary. Change descriptions should explain user-visible behavior, list verification commands, mention configuration or migration impacts, and include screenshots only for UI changes.

## Security & Configuration Tips

Keep secrets in `.env`; update `.env.example` when adding variables. CLI settings use the `DBN_` prefix, including `DBN_RUNS_DIR`, `DBN_LOG_LEVEL`, `DBN_DEFAULT_VIEW`, and `DBN_SNAPSHOT_MAX_FILE_SIZE_BYTES`. Avoid committing generated caches, local sessions, licenses, or large run artifacts unless explicitly needed.
