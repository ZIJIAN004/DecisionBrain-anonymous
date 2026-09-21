# Hard32-Fea

Hard32-Fea contains 32 discriminating cases across three tasks. Each task adds a hard constraint chosen to separate an Agent with access to the algorithm library from a Gurobi-only arm under a 30-second solver budget. Discrimination is determined entirely by the checker: `passed` means `executed and feasible is True`.

| Task | Hard constraint | Effective package | Cases | Weak-arm evidence |
| --- | --- | --- | ---: | --- |
| `jssp_deadline` | Global deadline D; makespan <= D | CP-SAT | 6 | Real two-arm runs; three cases disproved |
| `vrptw_minfleet` | Fleet cap K; number of routes <= K | PyVRP | 6 | Real two-arm runs separated 5/6 cases |
| `pdptw_minfleet` | Fleet cap K, same-vehicle pickup and delivery, pickup before delivery | VROOM | 20 | Deterministic proxy with warm start |

## Evidence Status

The three tasks have different evidence strengths:

- `vrptw_minfleet` has the strongest result: the real full arm passed 5/6 and the Gurobi-only arm passed 0/6.
- Half of `jssp_deadline` was disproved. The three theta=1.1 cases (instances 1, 3, and 5) separated the arms, while all three theta=1.2 cases (instances 2, 4, and 6) passed in the Gurobi-only arm. The latter remain as an explicit record of falsification and must not count toward discriminating power.
- `pdptw_minfleet` has not received a real two-arm run. Its deterministic proxy is stronger than those used for the other tasks: a reinforced MIP still found no first feasible solution in 20/20 cases when warm-started from handwritten heuristic partial solutions covering 76%-89% of requests.

In both real two-arm runs, every Gurobi-only failure was a 7,200-second task timeout with no solution, not a submitted solution rejected by the checker. Reports must distinguish these outcomes.

## Arithmetic Conventions

| Task | Distance/time convention | Time scale |
| --- | --- | --- |
| `jssp_deadline` | No distance | Original values |
| `vrptw_minfleet` | `floor(10 * Euclidean)` | x10 |
| `pdptw_minfleet` | `round(Euclidean) = int(sqrt(.) + 0.5)` | Original values |

All values are integral and feasibility checks use no tolerance. Objective values are not comparable across tasks or directly against externally published best-known solutions.

## Interpretation

Use `passed`, not `gap`, as the primary result. Gap is meaningful for the total-distance objectives in `vrptw_minfleet` and `pdptw_minfleet`. `jssp_deadline` is a pure feasibility task whose CP-SAT reference can be far below the deadline; a valid schedule at the deadline may therefore show a large positive gap without being a worse feasibility result.

## Layout

```text
benchmarks/Hard32-Fea/
  jssp_deadline/          task.json, input/problem.md, hidden checker and schemas
  vrptw_minfleet/         same, plus hidden/mathematical_formulation.md
  pdptw_minfleet/         same
  selection.json          fixed selection of 32 cases
  screening_record_*.md   screening evidence for each task
data/Hard32-Fea/instances/<task>/instance/
data/Hard32-Fea/solutions/<task>/gurobi_solution/
```

Task-local indices are contiguous from 1 and match the benchmark runner layout.

## Validation

```powershell
python benchmarks\Hard32-Fea\<task>\hidden\feasibility_check.py `
  --instance_path data\Hard32-Fea\instances\<task>\instance\large_instance_<n>.json `
  --solution_path data\Hard32-Fea\solutions\<task>\gurobi_solution\large_solution_<n>.json `
  --result_path r.json
```

All 32 reference solutions were verified as `feasible: true`. Checkers depend only on the Python standard library and return `{feasible, violated_constraints, violations, violation_magnitudes}`.

## Remaining Work

- Validate all 20 `pdptw_minfleet` cases with real two-arm runs.
- Replace the three disproved theta=1.2 `jssp_deadline` cases with cases using theta <= 1.1 if six discriminating JSSP cases are required.
