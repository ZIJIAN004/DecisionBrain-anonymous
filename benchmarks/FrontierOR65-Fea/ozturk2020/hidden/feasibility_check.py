"""
Feasibility checker for Parallel Batch Scheduling (Ozturk 2020).
Checks all hard constraints from the MILP formulation in math_model.txt.

Constraints (numbered top-to-bottom from the formulation):
  (1) Each job assigned to exactly one batch on one machine
  (2) Capacity constraint for each batch
  (3) Batch processing time >= processing time of any assigned job
  (4) Batch start time >= release date of any assigned job
  (5) Consecutive batches on the same machine do not overlap
  (6) Completion time of each job >= batch end time for its assigned batch
  (7) Variable domains: x in {0,1}, S >= 0, p >= 0 (C >= 0 implied)
  (8) Objective consistency: reported objective_value must equal sum_j C_j
      (full recompute; falls back to a lower bound via batch_end for any
      missing completion times).
"""

import json
import argparse


def load_instance(instance_path):
    with open(instance_path, 'r') as f:
        return json.load(f)


def load_solution(solution_path):
    with open(solution_path, 'r') as f:
        return json.load(f)


def parse_solution(sol, instance):
    """
    Normalize both efficient and gurobi solution formats into a common structure.
    Returns:
      batches: list of dicts with keys: jobs, machine, start_time, processing_time
      job_completions: dict mapping job_id (int) -> completion_time (float)
    """
    batches = []
    job_completions = {}

    for b in sol.get("batches", []):
        batch = {
            "jobs": b["jobs"],
            "machine": b["machine"],
            "start_time": b["start_time"],
            "processing_time": b["processing_time"],
        }
        # Some formats store batch_id
        if "batch_id" in b:
            batch["batch_id"] = b["batch_id"]
        batches.append(batch)

    # Efficient solution format: job_completions dict
    if "job_completions" in sol:
        for jid_str, ct in sol["job_completions"].items():
            job_completions[int(jid_str)] = float(ct)

    # Gurobi solution format: job_assignments dict
    if "job_assignments" in sol:
        for jid_str, info in sol["job_assignments"].items():
            job_completions[int(jid_str)] = float(info["completion_time"])

    return batches, job_completions


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    N = instance["num_jobs"]
    M = instance["num_machines"]
    Cap = instance["batch_capacity"]
    jobs = instance["jobs"]

    # Build job lookup by job_id
    job_by_id = {}
    for j in jobs:
        job_by_id[j["job_id"]] = j

    batches, job_completions = parse_solution(solution, instance)

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    def record_violation(constraint_idx, message, lhs, rhs, operator=">="):
        """Record a violation with normalized magnitude."""
        if operator == ">=":
            violation_amount = rhs - lhs
        elif operator == "<=":
            violation_amount = lhs - rhs
        elif operator == "=":
            violation_amount = abs(lhs - rhs)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            violated_constraints_set.add(constraint_idx)
            violations.append((constraint_idx, message))
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

    # =========================================================================
    # Constraint (1): Each job assigned to exactly one batch on one machine
    # sum_k sum_m x_{jkm} = 1 for all j
    # =========================================================================
    job_assignment_count = {j["job_id"]: 0 for j in jobs}
    for b in batches:
        for jid in b["jobs"]:
            if jid in job_assignment_count:
                job_assignment_count[jid] += 1

    for j in jobs:
        jid = j["job_id"]
        count = job_assignment_count[jid]
        if abs(count - 1) > tol:
            record_violation(
                1,
                f"Job {jid} is assigned to {count} batch(es) instead of exactly 1",
                float(count), 1.0, "="
            )

    # =========================================================================
    # Constraint (2): Capacity constraint
    # sum_j x_{jkm} * v_j <= Cap for each batch
    # =========================================================================
    for i, b in enumerate(batches):
        total_size = sum(job_by_id[jid]["size"] for jid in b["jobs"] if jid in job_by_id)
        if total_size - Cap > tol:
            record_violation(
                2,
                f"Batch {i+1} on machine {b['machine']}: total size {total_size} exceeds capacity {Cap}",
                float(total_size), float(Cap), "<="
            )

    # =========================================================================
    # Constraint (3): Batch processing time >= processing time of any assigned job
    # p_{km} >= x_{jkm} * p_j
    # =========================================================================
    for i, b in enumerate(batches):
        p_km = b["processing_time"]
        for jid in b["jobs"]:
            if jid in job_by_id:
                p_j = job_by_id[jid]["processing_time"]
                if p_j - p_km > tol:
                    record_violation(
                        3,
                        f"Batch {i+1} on machine {b['machine']}: processing time {p_km} < job {jid} processing time {p_j}",
                        float(p_km), float(p_j), ">="
                    )

    # =========================================================================
    # Constraint (4): Batch start time >= release date of any assigned job
    # S_{km} >= x_{jkm} * r_j
    # =========================================================================
    for i, b in enumerate(batches):
        s_km = b["start_time"]
        for jid in b["jobs"]:
            if jid in job_by_id:
                r_j = job_by_id[jid]["release_date"]
                if r_j - s_km > tol:
                    record_violation(
                        4,
                        f"Batch {i+1} on machine {b['machine']}: start time {s_km} < release date {r_j} of job {jid}",
                        float(s_km), float(r_j), ">="
                    )

    # =========================================================================
    # Constraint (5): Consecutive batches on same machine do not overlap
    # S_{km} >= S_{k-1,m} + p_{k-1,m}
    # Sort batches on each machine by start time and check non-overlapping.
    # =========================================================================
    machine_batches = {}
    for i, b in enumerate(batches):
        m = b["machine"]
        if m not in machine_batches:
            machine_batches[m] = []
        machine_batches[m].append((i, b))

    for m, mb_list in machine_batches.items():
        # Sort by start time
        mb_list.sort(key=lambda x: x[1]["start_time"])
        for idx in range(1, len(mb_list)):
            prev_i, prev_b = mb_list[idx - 1]
            curr_i, curr_b = mb_list[idx]
            prev_end = prev_b["start_time"] + prev_b["processing_time"]
            curr_start = curr_b["start_time"]
            if prev_end - curr_start > tol:
                record_violation(
                    5,
                    f"Machine {m}: batch {prev_i+1} ends at {prev_end} but batch {curr_i+1} starts at {curr_start} (overlap)",
                    float(curr_start), float(prev_end), ">="
                )

    # =========================================================================
    # Constraint (6): Completion time of each job
    # C_j >= (S_{km} + p_{km}) - Q(1 - x_{jkm})
    # For assigned jobs (x=1): C_j >= S_{km} + p_{km}
    # =========================================================================
    # Build mapping: job_id -> (batch start_time, batch processing_time)
    job_batch_info = {}
    for i, b in enumerate(batches):
        batch_end = b["start_time"] + b["processing_time"]
        for jid in b["jobs"]:
            job_batch_info[jid] = (b["start_time"], b["processing_time"], batch_end, i)

    for j in jobs:
        jid = j["job_id"]
        if jid in job_completions and jid in job_batch_info:
            c_j = job_completions[jid]
            s_km, p_km, batch_end, batch_idx = job_batch_info[jid]
            if batch_end - c_j > tol:
                record_violation(
                    6,
                    f"Job {jid}: completion time {c_j} < batch end time {batch_end} (batch {batch_idx+1})",
                    float(c_j), float(batch_end), ">="
                )

    # =========================================================================
    # Constraint (7): Variable domains
    # S_{km} >= 0, p_{km} >= 0, C_j >= 0
    # (x binary is structural and checked implicitly via constraint 1)
    # =========================================================================
    for i, b in enumerate(batches):
        if b["start_time"] < -tol:
            record_violation(
                7,
                f"Batch {i+1} on machine {b['machine']}: start time {b['start_time']} is negative",
                float(b["start_time"]), 0.0, ">="
            )
        if b["processing_time"] < -tol:
            record_violation(
                7,
                f"Batch {i+1} on machine {b['machine']}: processing time {b['processing_time']} is negative",
                float(b["processing_time"]), 0.0, ">="
            )

    for j in jobs:
        jid = j["job_id"]
        if jid in job_completions and job_completions[jid] < -tol:
            record_violation(
                7,
                f"Job {jid}: completion time {job_completions[jid]} is negative",
                float(job_completions[jid]), 0.0, ">="
            )

    # =========================================================================
    # Constraint (8): Objective consistency
    # objective_value must equal sum_j C_j (total flow time).
    # Full recompute when every job has a completion time in the solution;
    # otherwise fall back to a per-job lower bound using its batch_end
    # (since constraint 6 forces C_j >= batch_end).
    # =========================================================================
    reported_obj = solution.get("objective_value")
    try:
        reported_obj = float(reported_obj) if reported_obj is not None else None
    except (TypeError, ValueError):
        reported_obj = None

    if reported_obj is not None:
        parts = []
        have_all = True
        for j in jobs:
            jid = j["job_id"]
            if jid in job_completions:
                parts.append(float(job_completions[jid]))
            elif jid in job_batch_info:
                _, _, batch_end, _ = job_batch_info[jid]
                parts.append(float(batch_end))
                have_all = False
            else:
                # Job is unassigned (constraint 1 will already flag); contribute 0
                # so we don't over-constrain the lower bound here.
                have_all = False

        recomputed = float(sum(parts))
        # 0.1% relative tolerance with 1e-3 absolute floor.
        obj_tol = max(1e-3, 1e-3 * abs(recomputed))

        if have_all:
            # Equality check (full recompute).
            abs_diff = abs(reported_obj - recomputed)
            if abs_diff > obj_tol:
                record_violation(
                    8,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} differs from recomputed sum_j(C_j)="
                    f"{recomputed} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    float(reported_obj), float(recomputed), "="
                )
        else:
            # Lower-bound check: reported_obj must be at least sum of
            # known C_j (or batch_end fallbacks). Only catches under-reporting.
            shortfall = recomputed - reported_obj
            if shortfall > obj_tol:
                record_violation(
                    8,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} is below lower bound sum_j(C_j or batch_end)="
                    f"{recomputed} (shortfall={shortfall:.6g}, tol={obj_tol:.6g})",
                    float(reported_obj), float(recomputed), ">="
                )

    # Build output
    violated_constraints = sorted(violated_constraints_set)
    # Aggregate violation messages by constraint index
    violation_messages = []
    seen_constraints_msg = set()
    for cidx, msg in violations:
        violation_messages.append(msg)

    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violation_messages,
        "violation_magnitudes": violation_magnitudes,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Parallel Batch Scheduling (Ozturk 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)
    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
