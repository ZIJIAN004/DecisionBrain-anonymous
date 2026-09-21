# Deadline JSSP Screening Record

This document records the evidence used to select six `jssp_deadline` cases. The intended property is that a CP-SAT-class solver finds a schedule with `makespan <= D` within budget while a Gurobi-only arm does not. Screening was completed on 2026-09-04 in the paper's evaluation environment. Values below are copied from the experiment JSON.

## Screening Rule

`D = round(theta * ub)`, where `ub` is `job_shop_lib`'s `metadata['upper_bound']`. A candidate required:

| Arm | Criterion |
| --- | --- |
| CP | CP-SAT finds `makespan <= D` within budget |
| Gurobi-only | Discrete MIP with minimization and warm start does not find `makespan <= D` within budget |
| Heuristic | Best priority-list dispatch schedule has makespan greater than D |

Theta=1.0 was used only for calibration. Screening used 30 seconds; reference generation used 60 seconds.

CP-SAT minimized makespan and recorded the first incumbent at or below D. OR-Tools 9.12 required explicit interval end variables for `AddNoOverlap`; using `end=None` caused a segmentation fault and those earlier results were discarded. The stronger Gurobi variant minimized makespan and used the best MWKR dispatch schedule as a warm start. Dispatch candidates were SPT, LPT, MWKR, LWKR, LMFT, and random priority.

## Archived Results

### CP-SAT and References

| Case | D | Result |
| --- | ---: | --- |
| ta41, theta=1.0 | 2005 | best 2153 at 30 s; excluded |
| ta41, theta=1.1 | 2206 | best 2152; reached D at 18.11 s |
| ta41, theta=1.2 | 2406 | best 2227; reached D at 2.2 s |
| ta51, theta=1.1 | 3036 | reference makespan 2823 within 60 s |
| ta51, theta=1.2 | 3312 | reference makespan 2818 within 60 s |
| ta61, theta=1.1 | 3155 | reference makespan 3007 within 60 s |
| ta61, theta=1.2 | 3442 | reference makespan 2994 within 60 s |

### Gurobi-Only Proxy at 30 Seconds

| Case | D | Warm-start makespan | Final incumbent | At or below D |
| --- | ---: | ---: | ---: | --- |
| ta41, theta=1.1 | 2206 | 3169 | 2522 | No |
| ta41, theta=1.2 | 2406 | 3169 | 2528 | No |
| ta51, theta=1.1 | 3036 | 4468 | 3955 | No |
| ta51, theta=1.2 | 3312 | 4468 | 3910 | No |
| ta61, theta=1.1 | 3155 | 4566 | 4148 | No |
| ta61, theta=1.2 | 3442 | 4566 | 4148 | No |

The cold-start Gurobi model also reached its time limit with no first feasible solution. Best dispatch makespans were approximately 3166-3169 for ta41, 4468 for ta51, and 4566 for ta61, all above their deadlines.

## Evidence Limitations and Corrections

- All six CP-SAT references exist and satisfy D. The reference makespans are 2088, 2131, 2823, 2818, 3007, and 2994. Selected files were independently checked as feasible by the suite checker.
- The ta51 records establish a reference within 60 seconds but do not archive the first-at-D time under a symmetric 30-second budget.
- Earlier results with missing CP-SAT fields came from the invalid `end=None` interval construction and are not evidence.
- The generator previously duplicated `th` in directory names. It was corrected to emit names such as `ta41_th1.1`, after which ta61 references and the six-case manifest were regenerated.
- A later real two-arm run disproved the intended property for all three theta=1.2 cases: the Gurobi-only Agent arm passed them. Only the theta=1.1 cases should count as empirically discriminating.

## Upper-Bound Provenance

The metadata upper bounds from JobShopLib's thomasWeise source are ta41=2005, ta51=2760, and ta61=2868. A published ta41 best-known upper bound of 2006 differs by one; this does not change the theta>=1.1 screening regime, but reports should state which source was used.

## Checker Decisions

The solution schema intentionally contains only `objective_value` and `schedule`; descriptive `status`, `deadline`, and `makespan` keys in reference files are ignored. The checker treats a non-numeric `objective_value` as a C7 violation. Other malformed structures follow the FrontierOR checker convention: a checker crash is recorded as `executed=True, feasible=None` and cannot become a false pass.
