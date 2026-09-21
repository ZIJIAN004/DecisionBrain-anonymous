# Optimization Agent Development Guide

> The HTTP API and CLI share `AgentRuntime`, `OptimizationAgent`, and Run storage.

## Layers and Boundaries

```text
HTTP API / CLI
       |
       v
AgentRuntime --- RunRepository / EventPublisher / CoreServices
       |
       v
OptimizationAgent.outputs()
       |
       v
stage_flow --- StageAgent --- LLM / workspace tools
```

- `core/` defines the fixed stage flow, state, reports, terminal results, and workspace rules. It does not depend on FastAPI, Typer, or Rich.
- `runtime/` creates Runs, assembles `CoreServices`, consumes the single `outputs()` protocol, and owns cancellation, resume, and terminal Run state.
- `run_storage/` is a stateless repository. Every read or write method receives an explicit `run_id`; there is no implicit current Run.
- `api/` and `cli/` stage inputs, call Runtime, and present events. They do not implement a second workflow.

## Run Layout

Each Run is stored under `DBN_RUNS_DIR/<run_id>/`:

- `run.json`: lifecycle, command, configuration, and result summary.
- `agent-run.html`: complete LLM context, reasoning output, responses, and tool-call report.
- `input.snapshot.json`: initial workspace copy statistics and exclusions.
- `workspace/`: isolated writable directory used by the stages.
- `artifacts/`: user-facing outputs and `.index.json`.

`workspace/` is an execution copy of the user's input directory, not an immutable input directory. A Run does not accept new external input files after it starts, but StageAgents continue to write stage outputs there. `input.snapshot.json` describes only the initial copy.

## Lifecycle

Runtime calls `OptimizationAgent.outputs()` and handles:

- `AgentReport`: map it to an event or artifact, then persist `AgentState`.
- `NeedsClarification`: set the Run to `needs_clarification` and wait for the same Runtime to receive the answer.
- `Succeeded` or `Failed`: update `run.json` and publish the single `run_finished` event.
- Cancellation: set the Run to `cancelled` and publish `run_finished`.

`dbn resume` reconstructs upstream state from `problem.md` and `stage_outputs/*.json`. Do not introduce a second checkpoint format. Use only `report_progress` for semantic progress.

## Change Rules

- Change stage order or transitions only in `core/stage_flow.py`; do not add a dynamic registry.
- Runtime depends only on `CoreAgent`, generic reports, and terminal results. It does not import concrete StageAgents.
- Publish every live execution event through `EventPublisher`; Runtime assembles the complete HTML report.
- Write stage artifacts to fixed workspace paths before a StageAgent parses and validates them.
- Register user-facing outputs through `ArtifactProduced`; do not write the artifact index directly.
- Tests use deterministic LLM fakes or a scripted `CoreAgent`, never a real model or licensed solver.

## Verification

```bash
conda run -n decisionbrain ruff format --check .
conda run -n decisionbrain ruff check .
conda run -n decisionbrain pytest
```
