# Prompt Resources

DecisionBrain keeps its versioned system prompts and developer contracts in this directory.
Each prompt file places the Chinese runtime prompt first and its English translation immediately
below it. The two sections are separated by this fixed marker:

```text
--- BEGIN ENGLISH TRANSLATION (DOCUMENTATION ONLY; NOT LOADED AT RUNTIME) ---
```

`src/decisionbrain/core/prompts.py` loads only the text above the marker. The English section is
for peer review and does not alter the prompt sent to the model in the reported configuration.

## Layout

```text
prompts/
|-- system/       Agent roles, operating principles, and stage-specific policies
`-- developer/    Artifact contracts, output schemas, and runtime instructions
```

The standard workflow loads prompt pairs for Intake, Problem Contract, Algorithm Design,
Solving, Feasibility Review, and Explanation. Shared workspace rules live in
`system/stage_common_system.txt`, while algorithm-library guidance is injected only into stages
that can use the catalog.

Additional prompt pairs support reported ablations and the Gurobi-formulator workflow:

- `*_no_input_schema.txt`, `*_no_algorithm_library.txt`, and
  `*_no_feasible_review.txt` remove one corresponding workflow capability.
- `*_no_components.txt` supports the components ablation.
- `gurobi_formulator_*` and `solving_*_gurobi_*` support the no-design Gurobi workflow.
- `solving_self_check_*` is the Solving path used when independent Feasibility Review is
  disabled; Solving runs the checker itself in that configuration.

## Loading and Validation

The loader selects the exact files required by a runtime configuration. For prompt-sensitive
ablations, a missing explicit variant is an error rather than an implicit fallback. Prompt files
are loaded when the runtime starts, so a running service must be restarted after a prompt change.

System prompts should describe the Agent's role and decision principles. Developer contracts
should contain artifact names, structured output requirements, runtime limits, and parser-facing
details. Keep experimental variants narrowly scoped so that disabling one capability does not
silently change unrelated behavior.
