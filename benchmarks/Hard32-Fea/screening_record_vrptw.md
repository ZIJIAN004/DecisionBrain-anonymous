# Capped-Fleet VRPTW Screening Record

This document records evidence for six `vrptw_minfleet` cases. The intended property is that PyVRP finds a feasible solution using at most K vehicles within 30 seconds while the Gurobi-only arm does not. Screening was completed on 2026-09-04 in the paper's evaluation environment.

## Screening Rule

K is the SINTEF best-known fleet size. PyVRP had to return a feasible solution with `num_available=K`; all Gurobi-only approaches had to remain above K; and the cap had to exclude trivial distance-driven solutions. The free-fleet PyVRP route count, `freeK`, measured how restrictive the cap was.

The candidate pool contained all 180 GH instances with 400, 600, or 800 customers. C1, R1, RC1, and C2 typically had `freeK-K` of only 0-1 and were excluded. R-family separation also disappeared at larger sizes. Only RC2 retained a growing fleet gap, and all six selected cases are RC2.

## Implementations

Every arm received 30 seconds:

- PyVRP: `solve(data, MaxRuntime(30), seed=42)` with `num_available=K`, checked by `Solution.is_feasible()`.
- Cold-start Gurobi: two-index arc MIP with time-window big-M constraints and depot out-degree at most K.
- Construction heuristic: route-minimizing sequential insertion with an ejection pool, penalty counts, window ejections, and feasible perturbations.
- Route pool plus set cover: collect feasible routes, solve a minimum-cardinality cover with Gurobi, and remove duplicate visits. A warm MIP variant minimized depot out-degree from the constructed start.

## Results

The cold-start MIP found no first feasible solution in all 37 tested cells. The decisive weak baseline was the construction heuristic:

| Instance | K | Best at 30 s | Best at 120 s | Gap |
| --- | ---: | ---: | ---: | ---: |
| RC2_6_6 | 11 | 14 | 13 | 2 |
| RC2_8_3 | 15 | 17 | not run | 2 |
| RC2_4_6 | 8 | 9 | 9 | 1 |
| RC2_6_7 | 11 | 12 | 12 | 1 |
| RC2_8_6 | 15 | 16 | not run | 1 |
| RC2_8_7 | 15 | 16 | not run | 1 |

Two independent route-pool generators produced pools ranging from 67 to 2,526 routes. Set cover and warm MIP did not eliminate any selected case and generally proved the best route-pool fleet size equal to the construction result.

PyVRP failed to reach K on 5 of 33 candidates, including three 800-customer cases. This confirms that both sides of the separation window are restrictive.

## Limitations

- Strengthening the proxy eliminated 10 of 13 initially separated cases. It has not been proved converged; squeeze moves and inter-route local search may eliminate cases with a one-vehicle margin.
- All six cases are RC2, and three 800-customer cases share the same family, size, and generator. Report results clustered by family and size rather than as independent samples.
- PyVRP is compiled C++ while the proxy is Python, so implementation speed contributes to separation, especially at 800 customers.
- The original screening evidence used a deterministic proxy rather than an LLM. A subsequent real two-arm run passed 5/6 cases in the full arm and 0/6 in the Gurobi-only arm.
- An early run incorrectly mixed truncated and floating-point Euclidean distances. It was discarded. Final screening consistently uses DIMACS arithmetic: `floor(10 * Euclidean)`, times scaled by 10, and integral values.

## Data Sources

Instances come from `PyVRP/Instances` under `VRPTW/GH400`, `GH600`, and `GH800`. Fleet caps come from SINTEF's Gehring and Homberger best-known-solution tables. SINTEF reports floating-point distance, so those objective values are provenance only and are not directly comparable with this suite's integral convention.
