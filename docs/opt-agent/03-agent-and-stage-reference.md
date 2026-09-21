# Agent and Stage Reference

## Fixed Pipeline

`core/stage_flow.py` defines the only stage sequence:

```text
intake -> problem_contract -> algorithm_design -> solving
       -> feasibility_review -> explanation
```

| Stage | Responsibility | Fixed outputs |
| --- | --- | --- |
| `intake` | Inspect the problem and data; request clarification when needed | `stage_outputs/intake.json` |
| `problem_contract` | Define input and solution schemas plus an independent feasibility checker | `stage_outputs/problem_contract.json`, `input_schema.json`, `solution_schema.json`, `feasibility_checker.py` |
| `algorithm_design` | Design the solution strategy independently | `stage_outputs/algorithm_design.json` |
| `solving` | Prepare input, execute the solver, and submit a complete candidate without checker access | `input.json`, `solver.py`, `solution.json`, `solver_result.json`, `stage_outputs/solving.json` |
| `feasibility_review` | Run the checker through fixed tools, inspect the candidate read-only, and assign instance, design, or solving responsibility after rejection | `feasibility_result.json`, `stage_outputs/feasibility_review.json` |
| `explanation` | Convert the result into a user-facing explanation | `stage_outputs/explanation.json` |

Stages exchange fixed files through the Run `workspace/`. Later stages read them with workspace tools instead of copying complete upstream JSON into prompts.

## Core State

`AgentState` is an immutable dataclass with these principal fields:

| Field | Purpose |
| --- | --- |
| `current_stage` | Current stage; `None` means paused or terminated |
| `problem_definition` | Problem definition confirmed by Intake |
| `problem_contract` | Problem-level input, output, and feasibility contract |
| `algorithm_design` | Independent algorithm design |
| `generated_code` | Final `solver.py` content |
| `solver_result` | Compact solver result |
| `feasibility_review` | Independent feasibility decision and responsibility assignment |
| `feasibility_rounds` | Number of feasibility remediation attempts |
| `feasibility_directives` | Remediation directives and subsequent review outcomes |
| `feasibility_audits` | Raw rejection evidence visible only to Core and Feasibility Review |
| `explanation` | Final explanation |
| `failure` | Failure reason and metadata |
| `conversation` | Clarification conversation |
| `pending_clarification_questions` | Questions awaiting answers |

`stage_flow.apply_stage_result()` is the only place where a stage result updates state. `StageFailed` and clarification set `current_stage` to `None`; a completed result advances to the next `AgentStage`.

After `feasibility_review` passes, execution proceeds directly to Explanation. Review must invoke a fixed tool with no path arguments to run the audited checker against the current `input.json` and `solution.json`; Solving cannot read the checker or its results. A rejection assigned to `solving` preserves the design and retries Solving. A rejection assigned to `algorithm_design` clears the old design and downstream artifacts before rerunning Algorithm Design. Raw evidence is stored only in `AgentState.feasibility_audits`, never in the execution workspace. Core creates `feasibility_handoffs/<target>/attempt{N}/` for the responsible stage: Design receives the prior design, solution, and semantic remediation directives; Solving also receives the prior solver. The latest handoff retains complete safe semantics, while earlier attempts expose only the attempt, target, requirement IDs, change IDs, outcome, and next responsibility. Feasibility Review can retrieve a private audit for a specific attempt when required. A handoff never contains checker output, raw evidence, or checker implementation details. An `instance` responsibility records the conclusion and terminates the Run.

## StageAgent Protocol

Each StageAgent has four responsibilities:

1. Build messages from `AgentState` and the stage prompt.
2. Register stage-specific workspace tools and the shared `report_progress` tool.
3. Drive the LLM tool-calling loop.
4. Read fixed output files, validate JSON and domain rules, and return `StageResult`.

A stage handler may produce `AgentReport`, internal `StateUpdated` events, and one final `StageResult`. Runtime sees only generic reports or terminal results aggregated by `OptimizationAgent.outputs()` and does not depend on stage implementations.

## Workspace Rules

`WorkspaceAccessPolicy` enforces stage-aware permissions for fixed artifacts:

- Each fixed output can be written only by its owning stage.
- `algorithm_design` cannot read problem-contract files, preserving design independence; `solving` cannot read the checker or checker results.
- `list_files` hides protected files that the current stage cannot read.
- `write_file`, `replace_in_file`, and `shell` all enforce protected-write rules.
- Every stage can read `problem.md` and `data/**`.

During Solving, `input.json` and `solution.json` must satisfy their corresponding schemas. `stage_outputs/solving.json` stores only stage decisions and a compact summary rather than copying a large solution. Only the fixed Feasibility Review tool executes `feasibility_checker.py`.

## Changing a Stage

When adding or changing a stage, inspect all of the following:

1. `AgentStage`, `STAGE_SEQUENCE`, and `run_stage()` dispatch.
2. Fixed files in `stage_outputs.py`.
3. Messages, parsing, and result mapping in the corresponding StageAgent.
4. `WorkspaceAccessPolicy` and resume cleanup scope.
5. Prompt contracts and focused tests.

The stages are fixed and few; do not reintroduce a dynamic registry or string-based pseudo-state.
