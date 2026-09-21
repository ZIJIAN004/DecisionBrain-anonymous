#!/usr/bin/env python3
"""
Feasibility checker for the Nurse Rostering Problem (Rahimian et al., 2017).
Checks hard constraints HC1-HC10 from the mathematical formulation.
Soft constraints (SC1, SC2) are ignored.

Tier C addition (constraint 11): Objective consistency / lower-bound check.
The reported objective_value must be at least the obj implied by the
submitted schedule, where the implied obj uses the OPTIMAL choice of
y_{dt}, z_{dt} given x_{idt} (per SC1 and SC2 these are determined by x:
v_{idt} = q*(1-x) + p*x, y_{dt} = max(0, u-sum_x), z_{dt} = max(0, sum_x-u)).
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for NRP (Rahimian et al., 2017)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
        sol_data = json.load(f)

    result = check_feasibility(instance, sol_data)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if not result['feasible']:
        print(f"Violated constraints: {result['violated_constraints']}")


def check_feasibility(instance, sol_data):
    # Extract schedule - handle both efficient and gurobi solution formats
    if ("solution" in sol_data and isinstance(sol_data["solution"], dict)
            and "schedule" in sol_data["solution"]):
        schedule = sol_data["solution"]["schedule"]
        cov_root = sol_data["solution"]
    else:
        schedule = sol_data["schedule"]
        cov_root = sol_data

    # Domain check: coverage_under (y_{dt}) and coverage_over (z_{dt}) are
    # integer ≥ 0 (math_model L116). They are submitted in solution_schema
    # but were previously never validated. Verify each entry's domain.
    coverage_violations = []
    for cov_field in ("coverage_under", "coverage_over"):
        cov = cov_root.get(cov_field)
        if cov is None:
            continue
        if not isinstance(cov, dict):
            coverage_violations.append(
                f"{cov_field} must be a dict; got {type(cov).__name__}")
            continue
        for key, val in cov.items():
            if not isinstance(val, (int, float)):
                coverage_violations.append(
                    f"{cov_field}['{key}']={val!r} is not numeric "
                    f"(integer ≥ 0 required)")
                continue
            if val < -1e-9:
                coverage_violations.append(
                    f"{cov_field}['{key}']={val} is negative "
                    f"(non-negativity violated)")
            if isinstance(val, float) and abs(val - round(val)) > 1e-6:
                coverage_violations.append(
                    f"{cov_field}['{key}']={val} is not integer "
                    f"(integrality violated)")

    # Parse instance data
    num_days = instance["num_days"]
    num_nurses = instance["num_nurses"]
    num_shift_types = instance["num_shift_types"]
    num_weekends = instance["num_weekends"]

    D = list(range(num_days))
    I = list(range(num_nurses))
    T = list(range(num_shift_types))
    W = list(range(1, num_weekends + 1))

    shift_lengths = {}
    for st in instance["shift_types"]:
        shift_lengths[st["id"]] = st["length_minutes"]

    # Forbidden rotations: R[t] = set of shift types that cannot follow t
    R = {t: set() for t in T}
    for rot in instance["forbidden_shift_rotations"]:
        R[rot["shift_type"]].add(rot["cannot_be_followed_by"])

    nurse_data = {}
    for nurse in instance["nurses"]:
        nurse_data[nurse["id"]] = nurse

    # Build x[i][d] = assigned shift type (int) or None
    x = {}
    for i in I:
        x[i] = {}
        nurse_sched = schedule[str(i)]
        for d in D:
            x[i][d] = nurse_sched[str(d)]

    def works(i, d):
        return x[i][d] is not None

    tol = 1e-5
    eps = 1e-5

    all_violations = {}

    def check_violation(constraint_idx, msg, lhs, rhs, constraint_type):
        if constraint_type == 'le':
            va = lhs - rhs
        elif constraint_type == 'ge':
            va = rhs - lhs
        else:
            va = abs(lhs - rhs)

        if va > tol:
            normalizer = max(abs(rhs), eps)
            ratio = va / normalizer
            record = {
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(va),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            }
            if constraint_idx not in all_violations:
                all_violations[constraint_idx] = []
            all_violations[constraint_idx].append((msg, record))

    # =========================================================================
    # Constraint 1 (HC1): At most one shift per day per nurse
    # =========================================================================
    for i in I:
        for d in D:
            lhs = 1 if works(i, d) else 0
            check_violation(1, f"Nurse {i} assigned more than one shift on day {d}",
                            lhs, 1, 'le')

    # =========================================================================
    # Constraint 2 (HC2): Forbidden shift rotations
    # =========================================================================
    for i in I:
        for d in range(num_days - 1):
            if x[i][d] is not None:
                t = x[i][d]
                if x[i][d + 1] is not None:
                    u = x[i][d + 1]
                    if u in R[t]:
                        check_violation(
                            2,
                            f"Nurse {i}: forbidden rotation shift {t} on day {d} "
                            f"followed by shift {u} on day {d+1}",
                            2, 1, 'le')

    # =========================================================================
    # Constraint 3 (HC3): Maximum number of shifts per type
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        for t in T:
            t_str = str(t)
            if t_str in nd["max_shifts_per_type"]:
                max_s = nd["max_shifts_per_type"][t_str]
                count = sum(1 for d in D if x[i][d] == t)
                check_violation(
                    3,
                    f"Nurse {i}: {count} shifts of type {t} exceeds max {max_s}",
                    count, max_s, 'le')

    # =========================================================================
    # Constraint 4 (HC4): Maximum total minutes
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        total_min = sum(shift_lengths[x[i][d]] for d in D if x[i][d] is not None)
        check_violation(
            4,
            f"Nurse {i}: total minutes {total_min} exceeds maximum {nd['max_total_minutes']}",
            total_min, nd['max_total_minutes'], 'le')

    # =========================================================================
    # Constraint 5 (HC5): Minimum total minutes
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        total_min = sum(shift_lengths[x[i][d]] for d in D if x[i][d] is not None)
        check_violation(
            5,
            f"Nurse {i}: total minutes {total_min} below minimum {nd['min_total_minutes']}",
            total_min, nd['min_total_minutes'], 'ge')

    # =========================================================================
    # Constraint 6 (HC6): Maximum consecutive shifts
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        c_max = nd["max_consecutive_shifts"]
        for d in range(num_days - c_max):
            count = sum(1 for j in range(d, d + c_max + 1) if works(i, j))
            check_violation(
                6,
                f"Nurse {i}: {count} shifts in window days {d}-{d+c_max} "
                f"exceeds max consecutive {c_max}",
                count, c_max, 'le')

    # =========================================================================
    # Constraint 7 (HC7): Minimum consecutive shifts
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        c_min = nd["min_consecutive_shifts"]
        for c in range(1, c_min):
            for d in range(num_days - c - 1):
                w_d = 1 if works(i, d) else 0
                mid_sum = sum(1 for j in range(d + 1, d + c + 1) if works(i, j))
                w_end = 1 if works(i, d + c + 1) else 0
                lhs_val = w_d + (c - 1 - mid_sum) + w_end
                check_violation(
                    7,
                    f"Nurse {i}: isolated block of {mid_sum} shift(s) at days "
                    f"{d+1}-{d+c} violates min consecutive shifts {c_min}",
                    lhs_val, 0, 'ge')

    # =========================================================================
    # Constraint 8 (HC8): Minimum consecutive days off
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        o_min = nd["min_consecutive_days_off"]
        for b in range(1, o_min):
            for d in range(num_days - b - 1):
                off_d = 1 if not works(i, d) else 0
                mid_sum = sum(1 for j in range(d + 1, d + b + 1) if works(i, j))
                w_end = 1 if works(i, d + b + 1) else 0
                lhs_val = off_d + mid_sum + w_end
                check_violation(
                    8,
                    f"Nurse {i}: isolated block of {b} day(s) off at days "
                    f"{d+1}-{d+b} violates min consecutive days off {o_min}",
                    lhs_val, 0, 'ge')

    # =========================================================================
    # Constraint 9 (HC9): Maximum number of weekends worked
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        weekends_worked = 0
        for w in W:
            sat = 7 * w - 2
            sun = 7 * w - 1
            if sat < num_days and sun < num_days:
                if works(i, sat) or works(i, sun):
                    weekends_worked += 1
        check_violation(
            9,
            f"Nurse {i}: works {weekends_worked} weekends, "
            f"exceeds max {nd['max_weekends']}",
            weekends_worked, nd['max_weekends'], 'le')

    # =========================================================================
    # Constraint 10 (HC10): Requested days off
    # =========================================================================
    for i in I:
        nd = nurse_data[i]
        for d_off in nd["day_off_requests"]:
            if d_off < num_days and x[i][d_off] is not None:
                check_violation(
                    10,
                    f"Nurse {i}: assigned shift on requested day off {d_off}",
                    1, 0, 'eq')

    # =========================================================================
    # Domain violations on coverage_under/coverage_over (computed earlier)
    # =========================================================================
    if coverage_violations:
        for msg in coverage_violations:
            check_violation(
                "coverage_domain",
                msg,
                1.0, 0.0, "geq",
            )

    # =========================================================================
    # Constraint 11: Objective consistency (Tier C anti-exploit).
    # Two-sided check on reported objective_value vs schedule-implied obj.
    # Per SC1 (equality): v_{idt} = q_{idt}*(1-x_{idt}) + p_{idt}*x_{idt}
    #   (deterministic in x).
    # Per SC2 (equality): y_{dt} - z_{dt} = u_{dt} - sum_i x_{idt}, y,z>=0.
    # The cost-optimal (y,z) given x is y=max(0,u-sx), z=max(0,sx-u), and
    # since w_min,w_max >= 0 this gives a true LOWER BOUND on the obj.
    #   (a) reported < obj_lb - tol  → catches under-report (e.g. obj=0)
    #   (b) when submitted coverage_under/coverage_over are present (per
    #       solution_schema), compute obj_from_submitted = sum_v +
    #       sum_dt(w_min*y_sub + w_max*z_sub) and require
    #       |reported - obj_from_submitted| <= tol — catches over-report
    #       exploits (e.g. obj=sys.float_info.max) and any deviation from
    #       the program's own data.
    # =========================================================================
    reported_obj = sol_data.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # sum_v (deterministic from x)
            sum_v = 0.0
            for r in instance.get("shift_on_requests", []):
                nid = r["nurse_id"]
                d = r["day"]
                t = r["shift_type"]
                pen = r["penalty"]
                if 0 <= nid < num_nurses and 0 <= d < num_days:
                    if x[nid][d] != t:
                        sum_v += pen
            for r in instance.get("shift_off_requests", []):
                nid = r["nurse_id"]
                d = r["day"]
                t = r["shift_type"]
                pen = r["penalty"]
                if 0 <= nid < num_nurses and 0 <= d < num_days:
                    if x[nid][d] == t:
                        sum_v += pen

            # nurse counts per (d, t)
            counts = {}
            for i in I:
                for d in D:
                    t = x[i][d]
                    if t is not None:
                        counts[(d, t)] = counts.get((d, t), 0) + 1

            # (a) lower bound from optimal (y, z) given x
            opt_cov_cost = 0.0
            for cr in instance.get("coverage_requirements", []):
                d = cr["day"]
                t = cr["shift_type"]
                pref = cr["preferred"]
                uw = cr["under_weight"]
                ow = cr["over_weight"]
                sx = counts.get((d, t), 0)
                opt_cov_cost += uw * max(0, pref - sx) + ow * max(0, sx - pref)
            obj_lb = sum_v + opt_cov_cost

            # (b) full recompute using submitted (y, z), if both provided as dicts
            sub_y = cov_root.get("coverage_under")
            sub_z = cov_root.get("coverage_over")
            has_submitted_yz = isinstance(sub_y, dict) and isinstance(sub_z, dict)
            if has_submitted_yz:
                sub_cov_cost = 0.0
                for cr in instance.get("coverage_requirements", []):
                    d = cr["day"]
                    t = cr["shift_type"]
                    uw = cr["under_weight"]
                    ow = cr["over_weight"]
                    key = f"{d}_{t}"
                    yv = sub_y.get(key, 0)
                    zv = sub_z.get(key, 0)
                    try:
                        yv = float(yv) if yv is not None else 0.0
                    except (TypeError, ValueError):
                        yv = 0.0
                    try:
                        zv = float(zv) if zv is not None else 0.0
                    except (TypeError, ValueError):
                        zv = 0.0
                    if yv < 0:
                        yv = 0.0
                    if zv < 0:
                        zv = 0.0
                    sub_cov_cost += uw * yv + ow * zv
                obj_from_submitted = sum_v + sub_cov_cost
            else:
                obj_from_submitted = None

            # 0.1% relative tolerance with 1.0 absolute floor (integer obj)
            obj_tol = max(1.0, 1e-3 * abs(obj_lb))

            if reported < obj_lb - obj_tol:
                shortfall = obj_lb - reported
                check_violation(
                    11,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} is below recomputed lower bound "
                    f"sum_v + sum_dt(w_min*y_opt + w_max*z_opt) = {obj_lb} "
                    f"(shortfall={shortfall:.3g}, tol={obj_tol:.3g})",
                    reported, obj_lb, 'ge')
            elif obj_from_submitted is not None:
                eq_tol = max(1.0, 1e-3 * abs(obj_from_submitted))
                abs_diff = abs(reported - obj_from_submitted)
                if abs_diff > eq_tol:
                    check_violation(
                        11,
                        f"Objective consistency violated: reported objective_value="
                        f"{reported} differs from recompute using submitted "
                        f"coverage_under/coverage_over = {obj_from_submitted} "
                        f"(|diff|={abs_diff:.3g}, tol={eq_tol:.3g})",
                        reported, obj_from_submitted, 'eq')

    # =========================================================================
    # Build output
    # =========================================================================
    sorted_violated = sorted(all_violations.keys(), key=lambda x: (isinstance(x, str), x))
    feasible = len(sorted_violated) == 0

    violations_msgs = []
    violation_magnitudes = []
    for c_idx in sorted_violated:
        records = all_violations[c_idx]
        count = len(records)
        first_msg = records[0][0]
        if count > 1:
            violations_msgs.append(f"{first_msg} (and {count - 1} more)")
        else:
            violations_msgs.append(first_msg)
        for _, rec in records:
            violation_magnitudes.append(rec)

    return {
        "feasible": feasible,
        "violated_constraints": sorted_violated,
        "violations": violations_msgs,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    }


if __name__ == "__main__":
    main()
