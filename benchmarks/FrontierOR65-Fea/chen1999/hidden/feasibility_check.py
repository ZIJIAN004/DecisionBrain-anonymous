"""
Feasibility checker for parallel machine scheduling solutions from
Chen & Powell (1999) "Solving Parallel Machine Scheduling Problems by Column Generation".

Checks constraints from the mathematical formulations in the paper:

For Weighted Completion Time problems (IP1: Eqs 2-6; IP2: Eqs 8-12):
  General (non-identical) IP1:
    Constraint 1 (Eq 2): Each job assigned exactly once
    Constraint 2 (Eq 3): At most one first job per machine
    Constraint 3 (Eq 4): Flow conservation (each job has exactly one predecessor and one successor)
    Constraint 4 (Eq 5): Completion time consistency
    Constraint 5 (Eq 6): Binary/integrality of assignment variables

  Identical machines IP2:
    Constraint 1 (Eq 8): Each job assigned exactly once
    Constraint 2 (Eq 9): Number of machines used <= m
    Constraint 3 (Eq 10): Flow conservation
    Constraint 4 (Eq 11): Completion time consistency
    Constraint 5 (Eq 12): Binary/integrality of assignment variables

For Weighted Tardy Jobs problems (IP1': Eqs 33-39):
    Constraint 1 (Eq 33): Each job is either on-time on some machine or tardy (z_j + sum x = 1)
    Constraint 2 (Eq 34): At most one first job per machine
    Constraint 3 (Eq 35): Flow conservation for on-time jobs
    Constraint 4 (Eq 36): Completion time consistency for on-time jobs
    Constraint 5 (Eq 37): On-time jobs finish by due date (0 <= C_j <= d_j)
    Constraint 6 (Eq 38): Binary/integrality of x variables
    Constraint 7 (Eq 39): Binary/integrality of z variables
    Constraint 8 (Eq 32, obj consistency): reported objective_value must equal
        sum_{j} w_j * z_j (full recompute from the solution's tardy set).

Since the candidate solutions represent schedules (lists of job indices per machine),
we verify the constraints by reconstructing the implied assignment and computing
completion times from the schedule.
"""

import argparse
import json
import sys


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def get_processing_time(instance, job, machine):
    """Get processing time of job on machine."""
    pt_2d = instance["jobs"]["processing_times"]
    return pt_2d[job][machine]


def check_weighted_completion_time(instance, solution):
    """
    Check feasibility for the total weighted completion time problem.

    For identical machines, checks constraints from IP2 (Eqs 8-12):
      Constraint 1 (Eq 8): Each job assigned exactly once
      Constraint 2 (Eq 9): At most m machines used
      Constraint 3 (Eq 10): Flow conservation
      Constraint 4 (Eq 11): Completion time consistency
      Constraint 5 (Eq 12): Binary/integrality

    For non-identical machines (uniform/unrelated), checks constraints from IP1 (Eqs 2-6):
      Constraint 1 (Eq 2): Each job assigned exactly once
      Constraint 2 (Eq 3): At most one first job per machine
      Constraint 3 (Eq 4): Flow conservation
      Constraint 4 (Eq 5): Completion time consistency
      Constraint 5 (Eq 6): Binary/integrality
    """
    tol = 1e-5
    eps = 1e-5

    n = instance["num_jobs"]
    m = instance["num_machines"]
    machine_type = instance.get("machine_type", "identical")
    weights = instance["jobs"]["weights"]

    schedule = solution.get("schedule", {})
    reported_obj = solution.get("objective_value")

    violations = []
    violation_magnitudes = []

    # Reconstruct assignment from schedule
    job_assignment = {}  # job -> machine
    job_count = {}  # job -> count of appearances
    for mk, job_list in schedule.items():
        k = int(mk)
        for job in job_list:
            job_count[job] = job_count.get(job, 0) + 1
            job_assignment[job] = k

    # --- Constraint 1: Each job assigned exactly once ---
    # IP2 Eq(8): sum_{i in B_j union {0}} x_{ij} = 1, for all j
    # IP1 Eq(2): sum_{k} sum_{i in B_j^k union {0}} x_{ij}^k = 1, for all j
    # This means every job must appear exactly once in the schedule.
    for j in range(n):
        count = job_count.get(j, 0)
        if count != 1:
            rhs = 1.0
            lhs = float(count)
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                normalizer = max(abs(rhs), eps)
                if count == 0:
                    violations.append(f"Job {j} is not assigned to any machine")
                else:
                    violations.append(f"Job {j} is assigned {count} times (expected exactly 1)")
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # --- Constraint 2: Machine capacity ---
    # IP2 Eq(9): sum_j x_{0j} <= m  (number of machines used <= m)
    # IP1 Eq(3): sum_j x_{0j}^k <= 1 for all k (at most one first job per machine)
    machines_used = len([k for k, jobs in schedule.items() if len(jobs) > 0])

    if machine_type == "identical":
        # Eq(9): number of machines used <= m
        lhs = float(machines_used)
        rhs = float(m)
        violation_amount = max(0.0, lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Number of machines used ({machines_used}) exceeds available machines ({m})")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })
    else:
        # Eq(3): For each machine k, at most 1 first job
        # Since the schedule is given as a list per machine, each machine has at most
        # one first job by construction. But we also need to check that machine indices
        # are valid (0..m-1).
        for mk in schedule.keys():
            k = int(mk)
            if k < 0 or k >= m:
                lhs = float(k)
                rhs = float(m - 1)
                violation_amount = max(0.0, lhs - rhs)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Machine index {k} is out of range [0, {m-1}]")
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })
        # Also check that number of machines used does not exceed m
        if machines_used > m:
            lhs = float(machines_used)
            rhs = float(m)
            violation_amount = lhs - rhs
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Number of machines used ({machines_used}) exceeds available machines ({m})")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # --- Constraint 3: Flow conservation ---
    # IP2 Eq(10): sum_{i in B_j union {0}} x_{ij} = sum_{i in A_j union {n+1}} x_{ji}, for all j
    # IP1 Eq(4): same but per machine k
    # In the schedule representation, each job on a machine has exactly one predecessor
    # (the previous job or the start) and one successor (the next job or the end).
    # This is satisfied by construction of the list representation. We verify that
    # job indices are valid (in range [0, n-1]).
    for mk, job_list in schedule.items():
        k = int(mk)
        for idx, job in enumerate(job_list):
            if job < 0 or job >= n:
                violations.append(
                    f"Invalid job index {job} on machine {k} (must be in [0, {n-1}])")
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(job),
                    "rhs": float(n - 1),
                    "raw_excess": max(0.0, float(job) - float(n - 1)),
                    "normalizer": max(abs(float(n - 1)), eps),
                    "ratio": max(0.0, float(job) - float(n - 1)) / max(abs(float(n - 1)), eps)
                })

    # --- Constraint 4: Completion time consistency ---
    # IP2 Eq(11): C_j = p_j * x_{0j} + sum_{i in B_j} (C_i + p_j) * x_{ij}, for all j
    # IP1 Eq(5): C_j = sum_k (p_{jk} * x_{0j}^k + sum_{i in B_j^k} (C_i + p_{jk}) * x_{ij}^k)
    # We compute completion times from the schedule and verify against reported values
    # (if available). The completion times must be non-negative.
    computed_completion_times = {}
    computed_obj = 0.0

    for mk, job_list in schedule.items():
        k = int(mk)
        cumulative_time = 0.0
        for job in job_list:
            p_jk = get_processing_time(instance, job, k)
            cumulative_time += p_jk
            computed_completion_times[job] = cumulative_time
            computed_obj += weights[job] * cumulative_time

    # If the solution provides completion times, check consistency
    reported_completion_times = solution.get("completion_times")
    if reported_completion_times is not None:
        for j_str, reported_cj in reported_completion_times.items():
            j = int(j_str)
            if j in computed_completion_times:
                computed_cj = computed_completion_times[j]
                diff = abs(computed_cj - reported_cj)
                rhs = reported_cj
                if diff > tol:
                    normalizer = max(abs(rhs), eps)
                    violations.append(
                        f"Completion time mismatch for job {j}: "
                        f"computed={computed_cj:.4f}, reported={reported_cj:.4f}")
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": computed_cj,
                        "rhs": rhs,
                        "raw_excess": diff,
                        "normalizer": normalizer,
                        "ratio": diff / normalizer
                    })

    # Check that all completion times are non-negative
    for job, cj in computed_completion_times.items():
        if cj < -tol:
            rhs = 0.0
            violation_amount = abs(cj)
            normalizer = max(abs(rhs), eps)
            violations.append(f"Completion time of job {job} is negative: {cj:.4f}")
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": cj,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # Check objective value consistency
    # This is not a mathematical constraint from the formulation, but a
    # consistency check. Use relative tolerance for large objective values
    # to avoid false positives from solver floating-point rounding.
    if reported_obj is not None:
        obj_diff = abs(computed_obj - reported_obj)
        normalizer = max(abs(reported_obj), eps)
        relative_diff = obj_diff / normalizer
        if obj_diff > tol and relative_diff > tol:
            rhs = float(reported_obj)
            violations.append(
                f"Objective value mismatch: computed={computed_obj:.4f}, "
                f"reported={reported_obj}")
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": computed_obj,
                "rhs": rhs,
                "raw_excess": obj_diff,
                "normalizer": normalizer,
                "ratio": relative_diff
            })

    # --- Constraint 5: Binary/integrality ---
    # IP2 Eq(12) / IP1 Eq(6): x_{ij} in {0,1}
    # In the schedule representation, assignments are inherently binary (a job is either
    # in a machine's list or not). We verify that all job indices are integers.
    for mk, job_list in schedule.items():
        for job in job_list:
            if not isinstance(job, int):
                violations.append(
                    f"Job index {job} on machine {mk} is not an integer")
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": float(job),
                    "rhs": round(float(job)),
                    "raw_excess": abs(float(job) - round(float(job))),
                    "normalizer": max(abs(round(float(job))), eps),
                    "ratio": abs(float(job) - round(float(job))) / max(abs(round(float(job))), eps)
                })

    return violations, violation_magnitudes


def check_weighted_tardy_jobs(instance, solution):
    """
    Check feasibility for the weighted number of tardy jobs problem.

    Checks constraints from IP1' (Eqs 33-39):
      Constraint 1 (Eq 33): Each job is either on-time on some machine or tardy
      Constraint 2 (Eq 34): At most one first job per machine
      Constraint 3 (Eq 35): Flow conservation for on-time jobs
      Constraint 4 (Eq 36): Completion time consistency for on-time jobs
      Constraint 5 (Eq 37): On-time jobs finish by due date (0 <= C_j <= d_j)
      Constraint 6 (Eq 38): Binary/integrality of x variables
      Constraint 7 (Eq 39): Binary/integrality of z variables
      Constraint 8 (Eq 32, obj consistency): reported objective_value must equal
        sum_{j} w_j * z_j computed from the solution's tardy set.
    """
    tol = 1e-5
    eps = 1e-5

    n = instance["num_jobs"]
    m = instance["num_machines"]
    machine_type = instance.get("machine_type", "identical")
    weights = instance["jobs"]["weights"]
    due_dates = instance["jobs"]["due_dates"]

    schedule = solution.get("schedule", {})
    reported_obj = solution.get("objective_value")
    reported_tardy = solution.get("tardy_jobs")

    violations = []
    violation_magnitudes = []

    # Determine on-time and tardy jobs from the solution
    on_time_jobs_in_schedule = set()
    job_assignment = {}
    job_count = {}

    for mk, job_list in schedule.items():
        k = int(mk)
        for job in job_list:
            on_time_jobs_in_schedule.add(job)
            job_count[job] = job_count.get(job, 0) + 1
            job_assignment[job] = k

    # Tardy jobs: either explicitly listed or inferred as not in any schedule
    if reported_tardy is not None:
        tardy_jobs = set(reported_tardy)
    else:
        tardy_jobs = set(range(n)) - on_time_jobs_in_schedule

    # On-time jobs: from schedule or from explicit list
    reported_on_time = solution.get("on_time_jobs")
    if reported_on_time is not None:
        on_time_jobs = set(reported_on_time)
    else:
        on_time_jobs = on_time_jobs_in_schedule

    # For solutions without schedules (e.g., gurobi tardy solutions that only
    # report tardy_jobs/on_time_jobs), use the on_time/tardy lists for constraint 1
    # and update job_count accordingly.
    has_schedule = len(schedule) > 0
    if not has_schedule:
        for j in on_time_jobs:
            job_count[j] = job_count.get(j, 0) + 1

    # --- Constraint 1 (Eq 33): sum_k sum_i x_{ij}^k + z_j = 1, for all j ---
    # Each job must be either on-time (in schedule) or tardy, but not both and not missing.
    for j in range(n):
        in_schedule = job_count.get(j, 0)
        is_tardy = 1 if j in tardy_jobs else 0
        lhs = float(in_schedule + is_tardy)
        rhs = 1.0
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            if in_schedule == 0 and is_tardy == 0:
                violations.append(
                    f"Job {j} is neither on-time nor tardy")
            elif in_schedule > 0 and is_tardy > 0:
                violations.append(
                    f"Job {j} is both on-time (in schedule) and marked tardy")
            elif in_schedule > 1:
                violations.append(
                    f"Job {j} appears {in_schedule} times in schedule (expected at most 1)")
            else:
                violations.append(
                    f"Job {j}: on-time count ({in_schedule}) + tardy ({is_tardy}) != 1")
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # --- Constraint 2 (Eq 34): sum_j x_{0j}^k <= 1, for all k ---
    # At most one first job per machine (satisfied by list structure).
    # Also check machine indices are valid and number of machines used <= m.
    machines_used = len([k for k, jobs in schedule.items() if len(jobs) > 0])
    for mk in schedule.keys():
        k = int(mk)
        if k < 0 or k >= m:
            lhs = float(k)
            rhs = float(m - 1)
            violation_amount = max(0.0, lhs - rhs)
            normalizer = max(abs(rhs), eps)
            violations.append(f"Machine index {k} is out of range [0, {m-1}]")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })
    if machines_used > m:
        lhs = float(machines_used)
        rhs = float(m)
        violation_amount = lhs - rhs
        normalizer = max(abs(rhs), eps)
        violations.append(
            f"Number of machines used ({machines_used}) exceeds available machines ({m})")
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": violation_amount / normalizer
        })

    # --- Constraint 3 (Eq 35): Flow conservation ---
    # Verified by list structure. Check valid job indices.
    for mk, job_list in schedule.items():
        k = int(mk)
        for job in job_list:
            if job < 0 or job >= n:
                violations.append(
                    f"Invalid job index {job} on machine {k} (must be in [0, {n-1}])")
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(job),
                    "rhs": float(n - 1),
                    "raw_excess": max(0.0, float(job) - float(n - 1)),
                    "normalizer": max(abs(float(n - 1)), eps),
                    "ratio": max(0.0, float(job) - float(n - 1)) / max(abs(float(n - 1)), eps)
                })

    # --- Constraint 4 (Eq 36): Completion time consistency ---
    # C_j = sum_k (p_{jk} * x_{0j}^k + sum_{i in B_j} (C_i + p_{jk}) * x_{ij}^k)
    # Compute completion times from the schedule for on-time jobs.
    computed_completion_times = {}
    for mk, job_list in schedule.items():
        k = int(mk)
        cumulative_time = 0.0
        for job in job_list:
            p_jk = get_processing_time(instance, job, k)
            cumulative_time += p_jk
            computed_completion_times[job] = cumulative_time

    # Check against reported completion times if available
    reported_completion_times = solution.get("completion_times")
    if reported_completion_times is not None:
        for j_str, reported_cj in reported_completion_times.items():
            j = int(j_str)
            if j in computed_completion_times:
                computed_cj = computed_completion_times[j]
                diff = abs(computed_cj - reported_cj)
                if diff > tol:
                    rhs = reported_cj
                    normalizer = max(abs(rhs), eps)
                    violations.append(
                        f"Completion time mismatch for job {j}: "
                        f"computed={computed_cj:.4f}, reported={reported_cj:.4f}")
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": computed_cj,
                        "rhs": rhs,
                        "raw_excess": diff,
                        "normalizer": normalizer,
                        "ratio": diff / normalizer
                    })

    # --- Constraint 5 (Eq 37): 0 <= C_j <= d_j for on-time jobs ---
    for job in on_time_jobs:
        if job in computed_completion_times:
            cj = computed_completion_times[job]
            dj = due_dates[job]

            # Check C_j >= 0
            if cj < -tol:
                rhs = 0.0
                violation_amount = abs(cj)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Completion time of on-time job {job} is negative: {cj:.4f}")
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": cj,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

            # Check C_j <= d_j
            violation_amount = max(0.0, cj - dj)
            if violation_amount > tol:
                rhs = float(dj)
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"On-time job {job} finishes at {cj:.4f} but due date is {dj} "
                    f"(exceeds by {violation_amount:.4f})")
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": cj,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # --- Constraint 6 (Eq 38): Binary x variables ---
    # Satisfied by construction of list-based schedule.
    for mk, job_list in schedule.items():
        for job in job_list:
            if not isinstance(job, int):
                violations.append(
                    f"Job index {job} on machine {mk} is not an integer")
                violation_magnitudes.append({
                    "constraint": 6,
                    "lhs": float(job),
                    "rhs": round(float(job)),
                    "raw_excess": abs(float(job) - round(float(job))),
                    "normalizer": max(abs(round(float(job))), eps),
                    "ratio": abs(float(job) - round(float(job))) / max(abs(round(float(job))), eps)
                })

    # --- Constraint 7 (Eq 39): Binary z variables ---
    # z_j in {0,1}: each job is either tardy or not. Verified by checking
    # no job is both on-time and tardy (already checked in constraint 1).
    # Nothing additional to check here beyond constraint 1.

    # --- Constraint 8 (Eq 32, obj consistency): reported objective_value must
    # equal sum_{j} w_j * z_j. The tardy set z is fully present in the solution
    # (or unambiguously inferable from on_time_jobs / schedule), so this is a
    # full recompute rather than a lower bound. Tier C defense against
    # candidates that lie about objective_value (e.g. obj=0 or obj=MAX_FLOAT)
    # while leaving the constraint-level structure feasible.
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(sum(weights[j] for j in tardy_jobs
                                 if 0 <= j < n))
            obj_diff = abs(reported - true_obj)
            # weights are integer and the objective is an integer sum;
            # tighten to 0.5 so any integer-magnitude mismatch fires, with
            # a relative floor for very large recomputed values.
            obj_tol = max(0.5, 1e-6 * abs(true_obj))
            if obj_diff > obj_tol:
                rhs = reported
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed sum_j w_j*z_j="
                    f"{true_obj} (|diff|={obj_diff:.4g}, tol={obj_tol:.4g})")
                violation_magnitudes.append({
                    "constraint": 8,
                    "lhs": true_obj,
                    "rhs": rhs,
                    "raw_excess": obj_diff,
                    "normalizer": normalizer,
                    "ratio": obj_diff / normalizer
                })

    return violations, violation_magnitudes


def check_feasibility(instance, solution):
    """Dispatch on problem_type and return a result dict matching main()'s output."""
    problem_type = instance.get(
        "problem_type",
        solution.get("problem_type", "weighted_completion_time"))

    if problem_type == "weighted_completion_time":
        violations, violation_magnitudes = check_weighted_completion_time(instance, solution)
    elif problem_type in ("weighted_tardy_jobs", "weighted_number_of_tardy_jobs"):
        violations, violation_magnitudes = check_weighted_tardy_jobs(instance, solution)
    else:
        violations = [f"Unknown problem type: {problem_type}"]
        violation_magnitudes = []

    violated_constraints = sorted(set(
        vm["constraint"] for vm in violation_magnitudes
    ))
    feasible = len(violations) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for parallel machine scheduling solutions "
                    "(Chen & Powell 1999)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"{status}: {len(result['violations'])} violation(s) found")
    for v in result["violations"]:
        print(f"  - {v}")


if __name__ == "__main__":
    main()
