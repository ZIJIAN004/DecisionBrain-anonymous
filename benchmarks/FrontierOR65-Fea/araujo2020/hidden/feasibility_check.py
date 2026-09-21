#!/usr/bin/env python3
"""
Feasibility checker for MMRCPSP / MMRCMPSP solutions.

Checks all hard constraints from the PDT formulation (Eqs. 8-16) of:
  Araujo et al. (2019): "Strong Bounds for Resource Constrained Project
  Scheduling: Preprocessing and Cutting Planes"

Constraint numbering (top to bottom in formulation):
  1: Eq (8)  - Assignment: each job to exactly one mode and start time
  2: Eq (9)  - Non-renewable resource capacity
  3: Eq (10) - Renewable resource capacity per time period
  4: Eq (11) - Precedence constraints
  5: Eq (12) - Linking z and x variables
  6: Eq (13) - Makespan >= project completion times
  7: Eq (14) - Binary domain of x
  8: Eq (15) - Binary domain of z
  9: Eq (16) - Non-negativity of h
 10: Eq (7)  - Objective consistency: reported objective_value must equal
              the recomputed Total Project Delay (TPD) from the schedule
"""

import argparse
import json
from collections import defaultdict, deque

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _topological_sort(job_ids, successors_map):
    pred = defaultdict(list)
    for jid in job_ids:
        for s in successors_map.get(jid, []):
            pred[s].append(jid)
    indeg = {jid: len(pred[jid]) for jid in job_ids}
    q = deque([jid for jid in job_ids if indeg[jid] == 0])
    order = []
    while q:
        jid = q.popleft()
        order.append(jid)
        for s in successors_map.get(jid, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    return order


def _compute_cpd(jobs_data, project):
    """CPM forward pass with min-mode durations: lambda_p = est[sink] - sigma_p."""
    pid = project["project_id"]
    sigma = project["release_date"]
    sink = project["artificial_sink_job_id"]
    job_ids = [j["job_id"] for j in jobs_data if j["project_id"] == pid]
    successors_map = {
        j["job_id"]: list(j["successors"])
        for j in jobs_data if j["project_id"] == pid
    }
    min_dur = {
        j["job_id"]: min(m["duration"] for m in j["modes"])
        for j in jobs_data if j["project_id"] == pid
    }
    order = _topological_sort(job_ids, successors_map)
    est = {jid: sigma for jid in job_ids}
    for jid in order:
        for s in successors_map.get(jid, []):
            est[s] = max(est[s], est[jid] + min_dur[jid])
    if sink not in est:
        return None
    return est[sink] - sigma


def check_feasibility(instance, solution):
    violations = []
    violation_magnitudes = []

    jobs_data = instance["jobs"]
    projects = instance["projects"]
    resources = instance["resources"]
    renewable_res = resources.get("renewable", [])
    nonrenewable_res = resources.get("nonrenewable", [])

    schedule = solution["schedule"]
    h_val = solution.get("makespan", 0)

    # Build lookup structures
    job_by_id = {j["job_id"]: j for j in jobs_data}
    mode_by_jm = {}
    for j in jobs_data:
        for m in j["modes"]:
            mode_by_jm[(j["job_id"], m["mode_id"])] = m

    # Build assignment from solution: job_id -> (mode_id, start_time)
    assignment = {}
    for entry in schedule:
        jid = entry["job_id"]
        mid = entry["mode_id"]
        st = entry["start_time"]
        if jid in assignment:
            assignment[jid] = None  # Mark as duplicate
        else:
            assignment[jid] = (mid, st)

    all_job_ids = set(j["job_id"] for j in jobs_data)
    assigned_job_ids = set(entry["job_id"] for entry in schedule)

    # Helper to record a violation
    def record(constraint_idx, msg, lhs, rhs, operator):
        if operator == "eq":
            violation_amount = abs(lhs - rhs)
        elif operator == "leq":
            violation_amount = max(0.0, lhs - rhs)
        elif operator == "geq":
            violation_amount = max(0.0, rhs - lhs)
        else:
            violation_amount = 0.0

        if violation_amount > TOL:
            normalizer = max(abs(rhs), EPS)
            ratio = violation_amount / normalizer
            violations.append((constraint_idx, msg))
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # =========================================================================
    # Constraint 1: Eq (8) - Each job assigned to exactly one mode/start time
    # sum_{m,t} x_{jmt} = 1  for all j
    # =========================================================================
    for j in jobs_data:
        jid = j["job_id"]
        count = sum(1 for entry in schedule if entry["job_id"] == jid)
        if count != 1:
            lhs = float(count)
            rhs = 1.0
            record(1, f"Job {jid}: assigned {count} times (expected exactly 1)", lhs, rhs, "eq")

    # Check that assigned mode is valid for the job
    for entry in schedule:
        jid = entry["job_id"]
        mid = entry["mode_id"]
        if jid not in job_by_id:
            record(1, f"Job {jid} in solution does not exist in instance", 0.0, 1.0, "eq")
            continue
        valid_modes = [m["mode_id"] for m in job_by_id[jid]["modes"]]
        if mid not in valid_modes:
            record(1, f"Job {jid}: mode {mid} not in valid modes {valid_modes}", 0.0, 1.0, "eq")

    # =========================================================================
    # Constraint 2: Eq (9) - Non-renewable resource capacity
    # sum_j sum_m sum_t q_{kjm} * x_{jmt} <= check_q_k  for all k
    # =========================================================================
    for k_data in nonrenewable_res:
        kid = k_data["resource_id"]
        cap = k_data["capacity"]
        total_usage = 0.0
        for entry in schedule:
            jid = entry["job_id"]
            mid = entry["mode_id"]
            key = (jid, mid)
            if key in mode_by_jm:
                total_usage += mode_by_jm[key]["nonrenewable_consumption"][kid]
        lhs = total_usage
        rhs = float(cap)
        record(2, f"Non-renewable resource {kid}: usage {lhs} exceeds capacity {rhs}", lhs, rhs, "leq")

    # =========================================================================
    # Constraint 3: Eq (10) - Renewable resource capacity per time period
    # sum_j sum_m q_{rjm} * z_{jmt} <= check_q_r  for all r, t
    # =========================================================================
    # Derive z from x: z_{j,m,t}=1 iff job j in mode m is processing at time t
    # i.e., start_time <= t < start_time + duration
    max_time = 0
    processing_intervals = {}  # job_id -> (mode_id, start, end_exclusive)
    for entry in schedule:
        jid = entry["job_id"]
        mid = entry["mode_id"]
        st = entry["start_time"]
        key = (jid, mid)
        if key in mode_by_jm:
            dur = mode_by_jm[key]["duration"]
            processing_intervals[jid] = (mid, st, st + dur)
            if st + dur > max_time:
                max_time = st + dur

    for r_data in renewable_res:
        rid = r_data["resource_id"]
        cap = r_data["capacity"]
        for t in range(max_time):
            usage = 0.0
            active_jobs = []
            for jid, (mid, start, end) in processing_intervals.items():
                if start <= t < end:
                    usage += mode_by_jm[(jid, mid)]["renewable_consumption"][rid]
                    active_jobs.append(jid)
            lhs = usage
            rhs = float(cap)
            if lhs - rhs > TOL:
                record(
                    3,
                    f"Renewable resource {rid} at time {t}: usage {lhs} exceeds capacity {rhs} (active jobs: {active_jobs})",
                    lhs, rhs, "leq",
                )

    # =========================================================================
    # Constraint 4: Eq (11) - Precedence constraints
    # sum_m sum_t (t + d_{jm}) * x_{jmt} - sum_z sum_i i * x_{szi} <= 0
    # i.e., finish_time(j) <= start_time(s) for each (j,s) in precedence
    # =========================================================================
    for j in jobs_data:
        jid = j["job_id"]
        if jid not in assigned_job_ids:
            continue
        assigned_j = next((e for e in schedule if e["job_id"] == jid), None)
        if assigned_j is None:
            continue
        mid_j = assigned_j["mode_id"]
        st_j = assigned_j["start_time"]
        dur_j = mode_by_jm.get((jid, mid_j), {}).get("duration", 0) if (jid, mid_j) in mode_by_jm else 0
        finish_j = st_j + dur_j

        for s_id in j.get("successors", []):
            assigned_s = next((e for e in schedule if e["job_id"] == s_id), None)
            if assigned_s is None:
                continue
            st_s = assigned_s["start_time"]
            # LHS = finish_j - start_s, must be <= 0
            lhs = float(finish_j - st_s)
            rhs = 0.0
            if lhs > TOL:
                record(
                    4,
                    f"Precedence: job {jid} finishes at {finish_j} but successor {s_id} starts at {st_s}",
                    lhs, rhs, "leq",
                )

    # =========================================================================
    # Constraint 5: Eq (12) - Linking z and x
    # z_{jmt} - sum_{t'=(t-d+1)}^{t} x_{jmt'} = 0
    # This is definitional: z is derived from x. We verify consistency.
    # Since z is derived from the schedule, this is satisfied by construction.
    # We still check: for each job, the derived z values are consistent with
    # being binary and matching exactly the processing interval.
    # =========================================================================
    for entry in schedule:
        jid = entry["job_id"]
        mid = entry["mode_id"]
        st = entry["start_time"]
        key = (jid, mid)
        if key not in mode_by_jm:
            continue
        dur = mode_by_jm[key]["duration"]
        if dur == 0:
            continue
        # z_{j,m,t} should be 1 for t in [st, st+dur-1] and 0 elsewhere
        # For each t in processing window, check:
        # z_{jmt} = sum_{t'=max(e_j, t-d+1)}^{min(l_jm, t)} x_{jmt'}
        # Since x_{j,mid,st}=1 and all other x_{j,*,*}=0, z_{j,mid,t}=1
        # iff st <= t and t-dur+1 <= st, i.e., st <= t <= st+dur-1.
        # This is exactly the processing interval, so by construction OK.
        # We verify start_time is non-negative.
        if st < 0:
            record(5, f"Job {jid}: start time {st} is negative", float(st), 0.0, "geq")

    # =========================================================================
    # Constraint 6: Eq (13) - Makespan computation
    # h - sum_m sum_t t * x_{a_p,m,t} >= 0  for all p
    # i.e., h >= completion_time(sink_job) for each project
    # =========================================================================
    for p in projects:
        pid = p["project_id"]
        a_p = p["artificial_sink_job_id"]
        assigned_sink = next((e for e in schedule if e["job_id"] == a_p), None)
        if assigned_sink is None:
            continue
        sink_start = assigned_sink["start_time"]
        # sink has duration 0, so completion = start
        # h >= t * x_{a_p,m,t} which equals sink_start
        lhs = float(h_val)
        rhs = float(sink_start)
        # Constraint: h - sink_start >= 0, i.e., lhs = h, rhs = sink_start, geq
        # Rewriting: lhs_of_constraint = h - sink_start >= 0
        # So we check h >= sink_start
        if rhs - lhs > TOL:
            record(
                6,
                f"Project {pid}: makespan h={h_val} < sink job {a_p} start time {sink_start}",
                lhs, rhs, "geq",
            )

    # =========================================================================
    # Constraint 7: Eq (14) - Binary domain of x
    # x_{jmt} in {0,1}
    # By construction from schedule, x values are 0 or 1. Verify start times
    # are integers.
    # =========================================================================
    for entry in schedule:
        st = entry["start_time"]
        if not isinstance(st, int) and (isinstance(st, float) and st != int(st)):
            record(7, f"Job {entry['job_id']}: start_time {st} is not integer (binary x violated)",
                   float(abs(st - round(st))), 0.0, "eq")

    # =========================================================================
    # Constraint 8: Eq (15) - Binary domain of z
    # z_{jmt} in {0,1}
    # Derived from binary x, so satisfied by construction. Check durations
    # are non-negative integers.
    # =========================================================================
    for entry in schedule:
        jid = entry["job_id"]
        mid = entry["mode_id"]
        key = (jid, mid)
        if key in mode_by_jm:
            dur = mode_by_jm[key]["duration"]
            if not isinstance(dur, int) or dur < 0:
                record(8, f"Job {jid} mode {mid}: duration {dur} invalid (z binary violated)",
                       float(abs(dur - round(dur))), 0.0, "eq")

    # =========================================================================
    # Constraint 9: Eq (16) - Non-negativity AND integrality of h
    # h ∈ Z_{>=0}
    # =========================================================================
    if h_val < -TOL:
        record(9, f"Makespan h={h_val} is negative", float(h_val), 0.0, "geq")
    if isinstance(h_val, float) and abs(h_val - round(h_val)) > 1e-6:
        record(9, f"Makespan h={h_val} is not integer (Eq 16)",
               float(abs(h_val - round(h_val))), 0.0, "eq")
    elif not isinstance(h_val, (int, float)):
        record(9, f"Makespan h={h_val!r} is not numeric (Eq 16)",
               0.0, 0.0, "eq")

    # =========================================================================
    # Constraint 10: Eq (7) - Objective consistency (full recompute)
    # TPD = sum_{p} (sink_start_p - sigma_p - lambda_p)
    #   sink_start_p : start_time of project p's artificial_sink_job in schedule
    #   sigma_p      : project release_date
    #   lambda_p     : Critical Path Duration via forward CPM on min-mode durations
    # The Gurobi reference reports TPD only (epsilon * h tiebreaker excluded
    # from objective_value), so the comparison is exact.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None

        # Skip the obj check only if any sink job is missing from the schedule
        # (already flagged by constraint 1). Otherwise the recompute is well-defined.
        sink_starts = {}
        sinks_present = True
        for p in projects:
            sink_id = p["artificial_sink_job_id"]
            entry = next((e for e in schedule if e["job_id"] == sink_id), None)
            if entry is None:
                sinks_present = False
                break
            sink_starts[p["project_id"]] = entry["start_time"]

        if reported is not None and sinks_present:
            true_tpd = 0.0
            cpd_ok = True
            for p in projects:
                lam = _compute_cpd(jobs_data, p)
                if lam is None:
                    cpd_ok = False
                    break
                sigma_p = p["release_date"]
                true_tpd += sink_starts[p["project_id"]] - sigma_p - lam

            if cpd_ok:
                abs_diff = abs(reported - true_tpd)
                # TPD is integer-valued; floor at 0.5 catches any integer mismatch.
                tol = max(0.5, 1e-3 * abs(true_tpd))
                if abs_diff > tol:
                    record(
                        10,
                        (
                            f"Objective consistency violated: reported objective_value="
                            f"{reported} differs from recomputed Total Project Delay="
                            f"{true_tpd} (|diff|={abs_diff:.3g}, tol={tol:.3g})"
                        ),
                        float(reported),
                        float(true_tpd),
                        "eq",
                    )

    # Build output
    violated_indices = sorted(set(c for c, _ in violations))
    violation_messages = []
    for idx in violated_indices:
        msgs = [msg for c, msg in violations if c == idx]
        violation_messages.extend(msgs)

    feasible = len(violated_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MMRCPSP solutions (Araujo et al. 2019)"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility result written to {args.result_path}")
    print(f"Feasible: {result['feasible']}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
