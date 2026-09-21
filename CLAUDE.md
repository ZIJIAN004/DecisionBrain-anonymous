# CLAUDE.md

Repository guidance for coding agents working on DecisionBrain.

## Project Overview

DecisionBrain is a Python 3.10+ conversational optimization toolkit. A single Core
`OptimizationAgent` serves both the FastAPI Run API and the Typer `dbn` CLI. The agent uses an
OpenAI-compatible LLM and workspace-scoped tools to clarify a problem, define contracts, design an
algorithm, solve it, review the solution quality, and explain the result.

## Commands

DecisionBrain is source-repository-only: do not add PyPI, wheel, or sdist
publication. Keep the checkout and use an editable installation.

```bash
conda activate decisionbrain
python -m pip install -e ".[dev]"

dbn --help
dbn init examples/demo
dbn run examples/demo
dbn run . --debug
dbn run . --json
dbn chat --file data/orders.csv
dbn resume RUN_ID solving
dbn doctor --json
dbn dev demo-events

set -a
source .env
set +a
conda run -n decisionbrain python -m decisionbrain.api.app

conda run -n decisionbrain ruff format --check .
conda run -n decisionbrain ruff check .
conda run -n decisionbrain pytest
```

Tests use deterministic LLM and scripted CoreAgent fakes. Do not use real model calls or require a
Gurobi license in tests.

## Architecture

### Core (`src/decisionbrain/core/`)

`OptimizationAgent.outputs()` is the only execution protocol. It owns immutable `AgentState`, emits
typed `AgentReport` values, and finishes with a typed `AgentOutcome`.

`stage_flow.py` defines the fixed sequence and all state transitions:

```text
intake -> problem_contract -> algorithm_design -> solving -> review -> explanation
```

Each stage is a `StageAgent` subclass under `core/stage_agents/`. A StageAgent builds messages,
registers stage-aware workspace tools, runs the LLM tool-calling loop, reads one fixed output file,
and validates it before returning a `StageResult`. `report_progress` is the only shared non-terminal
control tool.

Do not add a dynamic stage registry, a second status enum, string pseudo-stages, or another driver
protocol. Stage order belongs in `stage_flow.py`; fixed filenames belong in `stage_outputs.py`.

`review` is the only stage that may move the pipeline backwards. It rewinds to `solving` or
`algorithm_design` at most `OPT_REVIEW_MAX_ROUNDS` times, archives each round under
`review/round{N}/` before the workspace is reset, and always improves from the `best_round`
baseline rather than the most recent round.

### Runtime (`src/decisionbrain/runtime/`)

`AgentRuntime` owns the Run lifecycle. It creates a Run, assembles `CoreServices`, consumes
`outputs()`, maps reports to events or artifacts, persists AgentState, handles clarification and
cancellation, and writes the unique terminal event.

Runtime depends on the generic `CoreAgent` protocol and must not import concrete stage handlers.
CoreServices are assembled once for each Agent instance. Tests may inject an `agent_factory`.

### Run Storage (`src/decisionbrain/run_storage/`)

`RunRepository` is stateless: every operation receives an explicit `run_id`. It must never retain a
current Run or bind itself to one Run.

Each `DBN_RUNS_DIR/<run_id>/` contains:

- `run.json`: lifecycle and result summary
- `config.resolved.yaml`: non-secret resolved configuration
- `events.jsonl`: append-only structured events
- `input.snapshot.json`: initial copy report
- `workspace/`: writable execution copy
- `state/agent.json`: resumable Core state
- `artifacts/`: user-facing outputs and index

The initial input directory is copied into `workspace/`; stages then write their outputs into the
same directory. Do not describe `workspace/` as immutable. New external input cannot be attached to
an active Run, but internal stage output is expected to change.

### Events (`src/decisionbrain/events/`)

Runtime publishes `RunEventDraft` through `EventPublisher`. The publisher assigns event IDs,
timestamps, and monotonic sequence numbers before dispatching to sinks. Event types are plain
strings. Never print from Runtime or write `events.jsonl` directly.

### API (`src/decisionbrain/api/`)

`decisionbrain.api.app` is the process entry point. `RunExecutionManager` owns active background
tasks and SSE subscriptions:

- `POST /api/runs`
- `POST /api/runs/{run_id}/continue`
- `POST /api/runs/{run_id}/cancel`
- `GET /api/runs` and `GET /api/runs/{run_id}`
- `GET /api/runs/{run_id}/events`
- `GET /api/runs/{run_id}/artifacts`

Disconnecting SSE does not cancel the Run. Only the cancel endpoint changes execution state.

### CLI (`src/decisionbrain/cli/`)

The `dbn` entry point is declared in `pyproject.toml`. `dbn run` accepts a folder containing
`problem.md` and optional `data/`; `dbn chat` stages a prompt and attached files; `dbn resume`
creates a derived Run from a parent checkpoint and workspace. CLI code renders events but does not
implement optimization behavior.

## Configuration

`Settings` in `src/decisionbrain/config/settings.py` is the only configuration model. Core receives
an immutable `CoreConfig` and never reads environment variables.

Relevant variables are documented in `.env.example`: LLM endpoint/key/model, reasoning effort and
timeout, solver timeout, prompt directory, API host/port, Run directory, snapshot limit, active Run
limit, and debug mode. Keep secrets out of persisted configuration and committed files.

## Change Rules

- Preserve Core's independence from FastAPI, Typer, Rich, and concrete storage.
- Emit generated user artifacts through `ArtifactProduced`; do not mutate the artifact index.
- Keep AgentState schema and agent version changes explicit when persistence compatibility changes.
- Add focused tests beside the behavior changed and run the full suite for shared contracts.
- Prefer deleting obsolete abstractions and docs over leaving compatibility shims with no consumer.
