# FAMOU Benchmark Pipeline

This pipeline evaluates FAMOU on `FrontierOR65-Fea` as a blind benchmark. The suite contains 65 feasible large instances selected by `problem_class x formulation_type`. It evaluates FAMOU's end-to-end solutions, not a human reconstruction based on hidden reference implementations.

## Public Inputs

Export the public evaluation package:

```powershell
cd "<DecisionBrain-repository>"
python -m decisionbrain.benchmark.famou export `
  --large-root "<FrontierOR-large-root>" `
  --output "<output-root>\famou-FrontierOR65-Fea-input"
```

`--large-root` points to the original FrontierOR large dataset with layout `<large_root>/<paper_id>/instance/large_instance_<index>.json`. Each exported task contains:

- `problem.md`: natural-language problem statement.
- `data/instance.json`: selected large instance.
- `submission_contract.json`: required fields and types for FAMOU's `solution.json`; this is a public interface contract, not a reference solution.
- `manifest.json`: fixed case, direction, class, and instance index.

The export root contains one shared `FAMOU_PROMPT.md`. For each task, use that prompt and attach only the three public task materials. Do not expose feasibility checkers, reference solutions, Gurobi code, mathematical formulations, or reference logs from `hidden/`.

## Collecting FAMOU Results

For every `tasks/<task_id>/`, submit `problem.md`, `data/instance.json`, and `submission_contract.json` with the shared `FAMOU_PROMPT.md`. Normalize platform results as:

```text
famou-submissions/
  <task_id>/
    solution.json   # required when a feasible solution is returned
    result.json     # only for an infeasibility claim: {"status":"infeasible"}
```

`solution.json` must match the case's `submission_contract.json`, including an `objective_value` computed by FAMOU. Models, logs, screenshots, and variable tables may be retained as audit evidence, but do not replace the solution file. For `frey2017` and `levin2017`, an infeasibility claim passes only when the hidden reference also marks the task infeasible.

### Importing Natural-Language Replies

If FAMOU returns JSON inside a conversation, save each complete reply as UTF-8 text under the exported task ID:

```text
famou-replies/
  laporte2003-large-2/
    laporte2003-large-2.txt
```

Replies may contain prose and Markdown JSON blocks. An existing `solution.json` is preserved. Otherwise, the importer extracts the unique JSON object containing every top-level field required by `submission_contract.json`. Ambiguous or invalid replies produce `fail_to_match.md`; unanswered tasks produce no result file.

```powershell
python -m decisionbrain.benchmark.famou import `
  --input "<output-root>\famou-FrontierOR65-Fea-input" `
  --replies "<replies-root>"
```

The replies directory is also the submissions root. Point `--submissions` at it during private evaluation. Keep `solution.json` as pure JSON rather than copying explanatory prose into it.

## Private Evaluation

Run evaluation only in an environment containing the complete dataset:

```powershell
python -m decisionbrain.benchmark.famou evaluate `
  --large-root "<FrontierOR-large-root>" `
  --submissions "<submissions-root>" `
  --output "<output-root>\famou-FrontierOR65-Fea-report.json"
```

For each submission, the evaluator runs the hidden checker and computes the objective gap against the hidden Gurobi reference. A task passes when its answer exists, is valid JSON, and the checker finds it feasible; there is no hard gap threshold. `gap > 0` is worse than the reference, `gap < 0` is better, and a zero reference objective uses absolute difference. The report includes execution and feasibility rates, median/best/worst gaps, constraint violations, and errors.

## Automation Boundary

No public FAMOU batch submission and download API has been verified. The module therefore implements only the reproducible cross-platform boundary. Once authentication, endpoints, attachment formats, and exports are confirmed, a thin API adapter may be placed between `export_suite()` and `evaluate_submissions()`. It may read only the exported public package and must write the `solution.json` layout above; it must never read benchmark `hidden/` directories.
