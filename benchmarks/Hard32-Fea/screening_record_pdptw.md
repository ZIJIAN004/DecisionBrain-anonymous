# Capped-Fleet PDPTW Screening Record

This document records evidence for 20 `pdptw_minfleet` cases. The intended property is that VROOM serves every request with K vehicles within 30 seconds while a Gurobi-side implementation fails even with a handwritten heuristic and warm start. Screening was completed on 2026-09-04 in the paper's evaluation environment.

## Screening Rule

K is the route count used by the reference solver with an unrestricted fleet (`freeK`). A candidate required:

| Arm | Criterion |
| --- | --- |
| Package | VROOM serves every request with K vehicles; `summary.unassigned == 0` |
| Weak heuristic | Handwritten pickup-and-delivery insertion fails to place every request within 30 seconds |
| Weak MIP | Reinforced three-index MIP finds no first feasible solution within 30 seconds |
| Combined | The same MIP still finds no first feasible solution when warm-started from the heuristic partial solution |

## Candidate Pool

The pool contained all 296 Li & Lim PDPTW instances from Li100 through Li800. VROOM was feasible at K for 230; 189 survived the heuristic; 182 survived the cold MIP. Of those, 100 large cases were skipped because `K*N^2 > 10^6` and cannot be counted as failures. The evidence-backed pool therefore contained 82 cases. The final 20 have `placed_gap >= 12`, where `placed_gap` measures the remaining request-placement deficit. The largest observed gap was 17.

## Implementations

- VROOM 1.15.2 used native `Shipment` objects, integral duration and cost matrices, exactly K vehicles, `exploration_level=5`, and a 30-second timeout. Feasibility was determined by `summary.unassigned == 0` because VROOM returns unassigned jobs rather than raising on infeasibility.
- The weak heuristic used paired pickup-delivery insertion and repair for 30 seconds.
- The MIP used three-index `X[k,i,j]`, time-window arc pruning, tight arc-specific big-M values, vehicle symmetry breaking, `MIPFocus=1`, and `Heuristics=0.3` for 30 seconds.
- The combined arm generated a partial solution for 10 seconds and assigned path arcs and served requests as a partial MIP start before another 30-second MIP solve.

## Warm-Start Results

| Metric | Result |
| --- | --- |
| Requests covered by heuristic | 76%-89% |
| Warm-start arcs rejected by model | 0 in every case |
| MIP duration | 30.0-32.0 seconds; all reached the limit |
| MIP `SolCount` | 0 in every case |
| Cases eliminated | 0/20 |

The rebuilt model also reproduced opposite control cases: it found a verified feasible solution for `lc101, K=10` and found no solution within 30 seconds for `lr108, K=9`.

## Limitations

- All weak-arm evidence is deterministic; no real LLM two-arm validation has been completed. A JSSP validation showed that the real LLM arm can outperform its deterministic proxy, so these 20 cases remain provisional.
- Only Li100 and Li200 are represented. Li400, Li600, and Li800 were skipped by the scale guard rather than solved and failed.
- Cases are correlated: seven are `lc1_2_*`, six are `lc2_2_*`, and two each are `lrc1_2_*` and `lrc2_2_*`. Report results clustered by family and size.
- Column generation, branch-and-price, set partitioning, separate subproblems for unserved requests, and multi-start metaheuristics were not tested.
- The maximum `placed_gap` is only 17; a stronger ejection-based pickup-and-delivery heuristic may eliminate cases near the threshold of 12.
- A real bug in the original heuristic chained arrival time instead of departure time, omitting about 90 seconds of service per node. It was fixed and validated on 700 real-data trials with zero false acceptances and zero false rejections. All reported values are post-fix.

## Data and Arithmetic

Instances come from `PyVRP/Instances/PDPTW` and originate from Li & Lim data converted to SINTEF format. Distance and travel time use `int(sqrt(dx^2 + dy^2) + 0.5)` with original integral time values and no feasibility tolerance. This differs from `vrptw_minfleet`, so objective values cannot be compared across the two tasks. Vehicles leave the depot at time zero, wait when early, reject late arrivals, and must return within the depot time window.
