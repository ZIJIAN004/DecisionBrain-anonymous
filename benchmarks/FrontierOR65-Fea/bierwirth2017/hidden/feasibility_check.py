#!/usr/bin/env python3
"""
Feasibility checker for JSPTWT (Job Shop Scheduling Problem with Total Weighted Tardiness).
Checks constraints C1-C6 from the mathematical formulation in math_model.txt, plus:

Constraint 7: Objective consistency (Tier C anti-exploit defense). The reported
objective_value must agree (within tolerance) with the recomputed total weighted
tardiness: sum_j w_j * max(0, c_j - d_j), where c_j = s_last + p_last is computed
from the schedule's actual start times and the instance's processing times. This
catches LLM exploits that fabricate objective_value while the schedule itself is
constraint-feasible.
"""

import argparse
import json
import sys


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def load_solution(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_schedule(instance, solution):
    """
    Extract a unified schedule representation from either efficient or gurobi solution format.
    Returns:
        start_times: dict (job, op_index) -> start_time
        completion_times_op: dict (job, op_index) -> completion_time
        job_completion: dict job -> completion_time (of last op)
        job_tardiness: dict job -> tardiness reported in solution
    """
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]
    jobs = instance["jobs"]

    start_times = {}
    completion_times_op = {}
    job_completion = {}
    job_tardiness = {}

    if "machine_schedules" in solution:
        # Efficient algorithm format: machine_schedules + job_completions
        for ms in solution["machine_schedules"]:
            for op in ms["operations"]:
                key = (op["job"], op["operation_index"])
                start_times[key] = float(op["start_time"])
                completion_times_op[key] = float(op["completion_time"])
        for jc in solution["job_completions"]:
            job_completion[jc["job"]] = float(jc["completion_time"])
            job_tardiness[jc["job"]] = float(jc["tardiness"])
    elif "schedule" in solution:
        # Gurobi format: schedule is per-job with operations in technological order
        for job_sched in solution["schedule"]:
            j = job_sched["job_id"]
            job_completion[j] = float(job_sched["completion_time"])
            job_tardiness[j] = float(job_sched["tardiness"])
            for op_idx, op in enumerate(job_sched["operations"]):
                key = (j, op_idx)
                start_times[key] = float(op["start_time"])
                # The solution contract may omit end_time and processing_time.
                # In that format processing time comes from the instance; avoid
                # an eagerly evaluated dict.get default causing a KeyError.
                if "end_time" in op:
                    end = float(op["end_time"])
                else:
                    end = float(op["start_time"]) + float(
                        jobs[j]["operations"][op_idx]["processing_time"]
                    )
                completion_times_op[key] = end
    else:
        raise ValueError("Unrecognized solution format")

    return start_times, completion_times_op, job_completion, job_tardiness


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    jobs = instance["jobs"]
    num_jobs = instance["num_jobs"]
    num_machines = instance["num_machines"]

    start_times, completion_times_op, job_completion, job_tardiness = extract_schedule(
        instance, solution
    )

    violations = []
    violation_magnitudes = []

    def record(constraint_idx, msg, lhs, rhs, op_type):
        """op_type: 'ge' for >=, 'eq' for =, 'le' for <="""
        if op_type == "ge":
            violation_amount = max(0.0, rhs - lhs)
        elif op_type == "le":
            violation_amount = max(0.0, lhs - rhs)
        elif op_type == "eq":
            violation_amount = abs(lhs - rhs)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violations.append((constraint_idx, msg))
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # Build lookup: for each job, the technological sequence of (machine, processing_time)
    job_ops = {}
    for job in jobs:
        j = job["job_id"]
        job_ops[j] = []
        for op in job["operations"]:
            job_ops[j].append((op["machine"], op["processing_time"]))

    # =====================================================================
    # Constraint 1: Tardiness definition
    #   T_j >= c_j - d_j   and   T_j >= 0
    # =====================================================================
    for job in jobs:
        j = job["job_id"]
        d_j = float(job["due_date"])
        c_j = job_completion.get(j)
        T_j = job_tardiness.get(j)
        if c_j is None or T_j is None:
            violations.append((1, f"Job {j}: missing completion or tardiness data"))
            continue

        # T_j >= c_j - d_j
        lhs = T_j
        rhs = c_j - d_j
        record(1, f"Job {j}: T_j ({T_j}) < c_j - d_j ({rhs})", lhs, rhs, "ge")

        # T_j >= 0
        record(1, f"Job {j}: T_j ({T_j}) < 0", T_j, 0.0, "ge")

    # =====================================================================
    # Constraint 2: Job completion time
    #   c_j = s_{last_op} + p_{last_op}
    # =====================================================================
    for job in jobs:
        j = job["job_id"]
        last_op_idx = len(job_ops[j]) - 1
        last_machine, last_pt = job_ops[j][last_op_idx]
        key = (j, last_op_idx)
        if key not in start_times:
            violations.append((2, f"Job {j}: last operation not found in solution"))
            continue
        s_last = start_times[key]
        expected_c = s_last + last_pt
        c_j = job_completion.get(j, 0.0)
        record(2, f"Job {j}: c_j ({c_j}) != s_last + p_last ({expected_c})",
               c_j, expected_c, "eq")

    # =====================================================================
    # Constraint 3: Technological precedence within each job
    #   s_{op(k+1)} >= s_{op(k)} + p_{op(k)}   for k = 0..m-2
    # =====================================================================
    for job in jobs:
        j = job["job_id"]
        for k in range(len(job_ops[j]) - 1):
            key_curr = (j, k)
            key_next = (j, k + 1)
            if key_curr not in start_times or key_next not in start_times:
                continue
            s_curr = start_times[key_curr]
            p_curr = job_ops[j][k][1]
            s_next = start_times[key_next]
            rhs_val = s_curr + p_curr
            record(3, f"Job {j}, op {k}->{k+1}: s[{k+1}] ({s_next}) < s[{k}]+p[{k}] ({rhs_val})",
                   s_next, rhs_val, "ge")

    # =====================================================================
    # Constraint 4: Release date
    #   s_{first_op} >= r_j
    # =====================================================================
    for job in jobs:
        j = job["job_id"]
        r_j = float(job.get("release_date", 0))
        key = (j, 0)
        if key not in start_times:
            continue
        s_first = start_times[key]
        record(4, f"Job {j}: s_first ({s_first}) < release_date ({r_j})",
               s_first, r_j, "ge")

    # =====================================================================
    # Constraint 5: Machine capacity (disjunctive) — no overlap on same machine
    #   For each pair of ops on the same machine, one must finish before the other starts.
    # =====================================================================
    # Group operations by machine
    machine_ops = {}  # machine -> list of (job, op_index, start, end, processing_time)
    for job in jobs:
        j = job["job_id"]
        for k, (m, pt) in enumerate(job_ops[j]):
            key = (j, k)
            if key in start_times:
                s = start_times[key]
                e = completion_times_op.get(key, s + pt)
                machine_ops.setdefault(m, []).append((j, k, s, e, pt))

    for m, ops_list in machine_ops.items():
        # Sort by start time for efficient pairwise checking
        ops_list.sort(key=lambda x: x[2])
        for i in range(len(ops_list)):
            for ii in range(i + 1, len(ops_list)):
                j1, k1, s1, e1, p1 = ops_list[i]
                j2, k2, s2, e2, p2 = ops_list[ii]
                # Either j1 finishes before j2 starts, or j2 finishes before j1 starts
                # Since sorted by start, check if e1 <= s2 (with tolerance)
                overlap = min(e1, e2) - max(s1, s2)
                if overlap > tol:
                    # Violation: overlap
                    violation_amount = overlap
                    # For the magnitude, treat as: LHS = overlap, RHS = 0 (should be <= 0)
                    normalizer = max(abs(max(p1, p2)), eps)
                    ratio = violation_amount / normalizer
                    violations.append(
                        (5, f"Machine {m}: ops (job {j1}, op {k1}) [{s1},{e1}] and "
                            f"(job {j2}, op {k2}) [{s2},{e2}] overlap by {overlap:.4f}")
                    )
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": overlap,
                        "rhs": 0.0,
                        "raw_excess": violation_amount,
                        "normalizer": normalizer,
                        "ratio": ratio,
                    })

    # =====================================================================
    # Constraint 6: Non-negativity
    #   s_{ij} >= 0, T_j >= 0, c_j >= 0
    # =====================================================================
    for key, s_val in start_times.items():
        if s_val < -tol:
            record(6, f"Op (job {key[0]}, op {key[1]}): start_time ({s_val}) < 0",
                   s_val, 0.0, "ge")

    for job in jobs:
        j = job["job_id"]
        T_j = job_tardiness.get(j, 0.0)
        if T_j < -tol:
            record(6, f"Job {j}: T_j ({T_j}) < 0", T_j, 0.0, "ge")
        c_j = job_completion.get(j, 0.0)
        if c_j < -tol:
            record(6, f"Job {j}: c_j ({c_j}) < 0", c_j, 0.0, "ge")

    # =====================================================================
    # Constraint 7: Objective consistency (Tier C anti-exploit defense)
    #   Reported objective_value must agree with the recomputed total weighted
    #   tardiness from the schedule's actual start times:
    #     true_obj = sum_j w_j * max(0, (s_last_j + p_last_j) - d_j)
    #   Tolerance: max(0.5, 1e-3 * |true_obj|). Inputs (w_j, p_j, d_j) are
    #   integers, so an integer mismatch (>= 1) always fires.
    # =====================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            obj_computable = True
            for job in jobs:
                j = job["job_id"]
                w_j = float(job["weight"])
                d_j = float(job["due_date"])
                last_idx = len(job_ops[j]) - 1
                key = (j, last_idx)
                if key not in start_times:
                    obj_computable = False
                    break
                s_last = start_times[key]
                p_last = float(job_ops[j][last_idx][1])
                c_j = s_last + p_last
                T_j = max(0.0, c_j - d_j)
                true_obj += w_j * T_j
            if obj_computable:
                abs_diff = abs(reported - true_obj)
                obj_tol = max(0.5, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    normalizer = max(abs(true_obj), eps)
                    ratio = abs_diff / normalizer
                    violations.append((
                        7,
                        f"Objective consistency violated: reported objective_value="
                        f"{reported} differs from recomputed sum_j w_j*max(0,c_j-d_j)="
                        f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    ))
                    violation_magnitudes.append({
                        "constraint": 7,
                        "lhs": reported,
                        "rhs": true_obj,
                        "raw_excess": abs_diff,
                        "normalizer": normalizer,
                        "ratio": ratio,
                    })

    # Build result
    violated_indices = sorted(set(v[0] for v in violations))
    violation_messages = [v[1] for v in violations]
    feasible = len(violated_indices) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for JSPTWT solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"{status}: {len(result['violated_constraints'])} constraint(s) violated")


if __name__ == "__main__":
    main()
