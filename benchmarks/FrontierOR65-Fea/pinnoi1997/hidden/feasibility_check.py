"""
Feasibility Checker for the Workload Smoothing Problem (WSP)
From: "A Branch and Cut Approach for Workload Smoothing on Assembly Lines"
by Anulark Pinnoi and Wilbert F. Wilhelm (1997).

Checks constraints (2)-(7) of the WSP formulation against a candidate solution
plus a new constraint (8) for objective-value consistency (Tier-C defense
against LLM score-gaming exploits that fabricate `objective_value`).

Constraints (numbered as in the paper, except (8) which is checker-internal):
  (2) Assignment: each task assigned to exactly one station
  (3) Precedence: if (i,j) in Theta, station(i) <= station(j)
  (4) Capacity: sum of processing times at each station <= cycle_time
  (5) Binary/integrality: x_{si} in {0,1} (task assigned to integer station in valid range)
  (6) Idle time: workload at each station + z_max >= cycle_time
  (7) Non-negativity: z_max >= 0
  (8) Objective consistency: reported objective_value must equal
      max(0, max_s (cycle_time - workload(s))) within tol = 0.5
      (z_max is integer-valued per the paper).
"""

import argparse
import json
import math


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # --- Extract instance data ---
    tasks = sorted(int(t) for t in instance["task_processing_times"].keys())
    p = {int(t): v for t, v in instance["task_processing_times"].items()}
    c = instance["cycle_time"]
    S_star = instance["num_stations"]
    arcs = [(a[0], a[1]) for a in instance["precedence_arcs"]]
    E = {int(t): v for t, v in instance["earliest_station"].items()}
    L = {int(t): v for t, v in instance["latest_station"].items()}

    # --- Extract solution data ---
    task_assignments = {int(t): int(s) for t, s in solution["task_assignments"].items()}
    z_max = solution.get("z_max", solution.get("objective_value"))
    if z_max is None:
        z_max = 0.0
    z_max = float(z_max)

    # Helper to record a violation
    def record(constraint_idx, msg, lhs, rhs, operator):
        """Record a constraint violation.
        operator: 'eq', 'leq', 'geq'
        """
        if operator == "leq":
            violation_amount = lhs - rhs
        elif operator == "geq":
            violation_amount = rhs - lhs
        elif operator == "eq":
            violation_amount = abs(lhs - rhs)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            violated_set.add(constraint_idx)
            violations.append(msg)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # =====================================================================
    # Constraint (2): Assignment — each task assigned to exactly one station
    #   sum_{s=E_i}^{L_i} x_{si} = 1  for all i in T
    #
    # In the solution representation, each task maps to one station.
    # We check: (a) every task is assigned, (b) no task assigned multiple
    # times (impossible with dict), (c) assignment count == 1.
    # =====================================================================
    for i in tasks:
        if i not in task_assignments:
            # Task not assigned at all => LHS=0, RHS=1
            record(2, f"Constraint (2): Task {i} is not assigned to any station.",
                   0.0, 1.0, "eq")

    # Check for tasks in solution that are not in instance
    for i in task_assignments:
        if i not in p:
            record(2, f"Constraint (2): Task {i} in solution is not a valid task.",
                   0.0, 1.0, "eq")

    # =====================================================================
    # Constraint (3): Precedence
    #   sum s*x_{si} - sum s*x_{sj} <= 0  for all (i,j) in Theta
    #   i.e., station(i) <= station(j)
    # =====================================================================
    for (i, j) in arcs:
        if i in task_assignments and j in task_assignments:
            s_i = task_assignments[i]
            s_j = task_assignments[j]
            lhs = float(s_i - s_j)
            rhs = 0.0
            if lhs > tol:
                record(3,
                       f"Constraint (3): Precedence violated for arc ({i},{j}): "
                       f"task {i} at station {s_i}, task {j} at station {s_j} "
                       f"(requires station({i}) <= station({j})).",
                       lhs, rhs, "leq")

    # =====================================================================
    # Constraint (4): Capacity
    #   sum_{i in T(s)} p_i * x_{si} <= c  for all s = 1,...,S*
    # =====================================================================
    from collections import defaultdict
    station_workloads = defaultdict(int)
    for i, s in task_assignments.items():
        if i in p:
            station_workloads[s] += p[i]

    for s in range(1, S_star + 1):
        workload = station_workloads[s]
        lhs = float(workload)
        rhs = float(c)
        if lhs - rhs > tol:
            record(4,
                   f"Constraint (4): Capacity exceeded at station {s}: "
                   f"workload {workload} > cycle time {c}.",
                   lhs, rhs, "leq")

    # =====================================================================
    # Constraint (5): Binary/integrality domain
    #   x_{si} in {0,1} for s = E_i,...,L_i and i in T
    #
    # In the solution, task i is assigned to station s. We verify:
    #   (a) s is an integer
    #   (b) E_i <= s <= L_i  (valid station range)
    #   (c) 1 <= s <= S*
    # =====================================================================
    for i in tasks:
        if i not in task_assignments:
            continue  # already caught by constraint (2)
        s = task_assignments[i]
        # Check station is in valid range [1, S*]
        if s < 1 or s > S_star:
            lhs = float(s)
            if s < 1:
                rhs_val = 1.0
                record(5,
                       f"Constraint (5): Task {i} assigned to station {s}, "
                       f"which is below the minimum station 1.",
                       lhs, rhs_val, "geq")
            else:
                rhs_val = float(S_star)
                record(5,
                       f"Constraint (5): Task {i} assigned to station {s}, "
                       f"which exceeds the number of stations S*={S_star}.",
                       lhs, rhs_val, "leq")
        # Check station is within task-specific bounds [E_i, L_i]
        e_i = E.get(i, 1)
        l_i = L.get(i, S_star)
        if s < e_i:
            record(5,
                   f"Constraint (5): Task {i} assigned to station {s}, "
                   f"which is below its earliest station E_{i}={e_i}.",
                   float(s), float(e_i), "geq")
        elif s > l_i:
            record(5,
                   f"Constraint (5): Task {i} assigned to station {s}, "
                   f"which exceeds its latest station L_{i}={l_i}.",
                   float(s), float(l_i), "leq")

    # =====================================================================
    # Constraint (6): Idle time
    #   sum_i p_i * x_{si} + z_max >= c  for all s = 1,...,S*
    # =====================================================================
    for s in range(1, S_star + 1):
        workload = station_workloads[s]
        lhs = float(workload) + z_max
        rhs = float(c)
        if rhs - lhs > tol:
            record(6,
                   f"Constraint (6): Idle time constraint violated at station {s}: "
                   f"workload ({workload}) + z_max ({z_max}) = {lhs} < cycle time {c}.",
                   lhs, rhs, "geq")

    # =====================================================================
    # Constraint (7): Non-negativity
    #   z_max >= 0
    # =====================================================================
    lhs_z = z_max
    rhs_z = 0.0
    if rhs_z - lhs_z > tol:
        record(7,
               f"Constraint (7): z_max = {z_max} is negative.",
               lhs_z, rhs_z, "geq")

    # =====================================================================
    # Constraint (8): Objective consistency (Tier-C anti-exploit defense)
    #   reported objective_value must equal the true z_max computed from the
    #   solution: true_z_max = max(0, max_{s=1..S*} (c - workload(s))).
    #   All obj-determining variables (task_assignments) are present in the
    #   solution, so this is a full recompute. Tolerance 0.5 because c and
    #   p_t are integers and the paper notes z_max attains integer values.
    # =====================================================================
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and not math.isnan(reported_obj):
            # Compute true z_max from task assignments and instance data.
            # Only iterate s = 1..S* to match the formulation's index range
            # for constraints (4) and (6); idle is clamped at 0 to match
            # constraint (7) (z_max >= 0).
            max_idle = 0.0
            for s in range(1, S_star + 1):
                idle = float(c) - float(station_workloads[s])
                if idle > max_idle:
                    max_idle = idle
            true_z_max = max_idle  # already >= 0
            abs_diff = abs(reported_obj - true_z_max)
            # Integer-valued objective: tighten to 0.5 so a mismatch of >=1
            # always fires while floating-point noise around an integer
            # is tolerated.
            obj_tol = 0.5
            if abs_diff > obj_tol:
                violated_set.add(8)
                msg = (
                    f"Constraint (8): Objective consistency violated: "
                    f"reported objective_value={reported_obj} differs from "
                    f"recomputed z_max=max(0, max_s(c - workload(s)))={true_z_max} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol})."
                )
                violations.append(msg)
                normalizer = max(abs(true_z_max), eps)
                ratio = abs_diff / normalizer
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": float(reported_obj),
                    "rhs": float(true_z_max),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(ratio),
                })

    # --- Build result ---
    feasible = len(violated_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Workload Smoothing Problem (WSP)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON candidate solution file.")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE — no constraint violations found.")
    else:
        print(f"INFEASIBLE — {len(result['violated_constraints'])} constraint(s) violated: "
              f"{result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
