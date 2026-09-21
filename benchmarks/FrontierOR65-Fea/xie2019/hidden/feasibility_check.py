#!/usr/bin/env python3
"""
Feasibility checker for the Integrated Berth Allocation and Quay Crane
Assignment Problem (BACAP) from Xie, Wu, and Zhang (2019).

Checks:
1. Each vessel is assigned exactly one berth.
2. Each vessel starts within its time window [a_i, b_i].
3. Each vessel starts within its assigned berth's time window [a_k, b_k].
4. Vessel end time = start time + profile duration - 1, and end <= H_size.
5. No two vessels overlap on the same berth at the same time step.
6. QC capacity constraint: total QCs used at each time step <= total_qc.
7. Objective value matches recomputation.
"""

import argparse
import json
import random
from itertools import product


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def generate_qc_profiles(instance, rng):
    """Generate QC profiles (same logic as gurobi_code.py)."""
    S_size = instance["time_steps_per_shift"]
    big_cfg = instance["qc_profiles"]["big"]
    small_cfg = instance["qc_profiles"]["small"]
    num_big = instance["num_big_profiles"]
    num_small = instance["num_small_profiles"]

    def make_profiles(cfg, count):
        """Stratified sampling -- must match gurobi_code.py exactly so profile
        indices line up. One profile per duration bucket first, then random
        fill from the rest using the same RNG sequence."""
        min_qc, max_qc = cfg["min_qc"], cfg["max_qc"]
        min_s, max_s = cfg["min_handling_shifts"], cfg["max_handling_shifts"]
        qc_range = list(range(min_qc, max_qc + 1))
        by_dur = {}
        for dur in range(min_s, max_s + 1):
            by_dur[dur] = [
                {"duration_shifts": dur, "qc_per_shift": list(combo)}
                for combo in product(qc_range, repeat=dur)
            ]
        all_possible = [p for ps in by_dur.values() for p in ps]
        if len(all_possible) <= count:
            return all_possible
        n_buckets = len(by_dur)
        selected = []
        for dur in by_dur:
            selected.append(rng.choice(by_dur[dur]))
        remaining_count = count - n_buckets
        if remaining_count > 0:
            taken_ids = {id(p) for p in selected}
            rest = [p for p in all_possible if id(p) not in taken_ids]
            extra = rng.sample(rest, min(remaining_count, len(rest)))
            selected.extend(extra)
        return selected

    all_raw = make_profiles(big_cfg, num_big) + make_profiles(small_cfg, num_small)

    profiles_per_vessel = {}
    for v in instance["vessels"]:
        vid, a_i, b_i = v["id"], v["a"], v["b"]
        feasible = []
        for praw in all_raw:
            dur_steps = praw["duration_shifts"] * S_size
            if dur_steps <= (b_i - a_i + 1):
                qc_per_step = []
                for sq in praw["qc_per_shift"]:
                    qc_per_step.extend([sq] * S_size)
                feasible.append({
                    "pid": len(feasible),
                    "duration_steps": dur_steps,
                    "duration_shifts": praw["duration_shifts"],
                    "qc_per_shift": praw["qc_per_shift"],
                    "qc_per_step": qc_per_step,
                })
        profiles_per_vessel[vid] = feasible
    return profiles_per_vessel


def get_shift_index(h, S_size):
    return ((h - 1) % S_size) + 1


def compute_d_sp(profiles_per_vessel, S_size):
    d = {}
    for vid, profs in profiles_per_vessel.items():
        for prof in profs:
            pid = prof["pid"]
            D = prof["duration_shifts"]
            for s in range(1, S_size + 1):
                d[(vid, s, pid)] = (S_size - s + 1) + (D - 1) * S_size
    return d


def compute_q_spu(profiles_per_vessel, S_size):
    q = {}
    for vid, profs in profiles_per_vessel.items():
        for prof in profs:
            pid = prof["pid"]
            D = prof["duration_shifts"]
            qc_shifts = prof["qc_per_shift"]
            for s in range(1, S_size + 1):
                u = 1
                for _ in range(S_size - s + 1):
                    q[(vid, s, pid, u)] = qc_shifts[0]
                    u += 1
                for si in range(1, D):
                    for _ in range(S_size):
                        q[(vid, s, pid, u)] = qc_shifts[si]
                        u += 1
    return q


def check_feasibility(instance, solution):
    violations = []
    violated_constraints = set()

    # If no solution
    if solution.get("objective_value") is None or solution.get("status") in ("NO_SOLUTION", "INFEASIBLE"):
        return {
            "feasible": False,
            "violated_constraints": ["no_solution"],
            "violations": ["No solution found."],
            "objective_value": None,
            "recomputed_objective": None
        }

    rng = random.Random(instance.get("random_seed", 42))
    S_size = instance["time_steps_per_shift"]
    F_size = instance["num_shifts_per_step"]
    H_size = S_size * F_size
    total_qc = instance["total_qc"]
    c1, c2 = instance["c1"], instance["c2"]
    vessels_data = {v["id"]: v for v in instance["vessels"]}
    berths_data = {b["id"]: b for b in instance["berths"]}
    n = instance["num_vessels"]

    profiles_per_vessel = generate_qc_profiles(instance, rng)
    d_sp = compute_d_sp(profiles_per_vessel, S_size)
    q_spu = compute_q_spu(profiles_per_vessel, S_size)

    sol_vessels = solution.get("vessels", [])
    if len(sol_vessels) != n:
        violations.append(f"Expected {n} vessel assignments, got {len(sol_vessels)}.")
        violated_constraints.add("assignment_count")

    # Track berth occupancy and QC usage
    berth_usage = {}  # (k, h) -> list of vessel ids
    qc_usage = {}     # h -> total QC count
    recomputed_obj = 0.0

    for vs in sol_vessels:
        vid = vs["id"]
        v = vessels_data[vid]
        berth = vs["berth"]
        start = vs["start_time"]
        end = vs["end_time"]
        pid = vs.get("profile")

        # Check berth assignment
        if berth not in berths_data:
            violations.append(f"Vessel {vid}: invalid berth {berth}.")
            violated_constraints.add("berth_assignment")
            continue

        bk = berths_data[berth]

        # Check vessel time window
        if start < v["a"] or start > v["b"]:
            violations.append(f"Vessel {vid}: start={start} outside time window [{v['a']}, {v['b']}].")
            violated_constraints.add("vessel_time_window")

        # Check berth time window
        if start < bk["a_k"] or start > bk["b_k"]:
            violations.append(f"Vessel {vid}: start={start} outside berth {berth} window [{bk['a_k']}, {bk['b_k']}].")
            violated_constraints.add("berth_time_window")

        # Check profile and duration
        if pid is not None and pid < len(profiles_per_vessel.get(vid, [])):
            prof = profiles_per_vessel[vid][pid]
            s = get_shift_index(start, S_size)
            if (vid, s, pid) in d_sp:
                expected_end = start + d_sp[(vid, s, pid)] - 1
                if end != expected_end:
                    violations.append(f"Vessel {vid}: end={end} expected {expected_end}.")
                    violated_constraints.add("end_time")

        # Check end time within horizon
        if end > H_size:
            violations.append(f"Vessel {vid}: end={end} exceeds horizon {H_size}.")
            violated_constraints.add("horizon")

        # Track berth occupancy
        for h in range(start, end + 1):
            key = (berth, h)
            if key not in berth_usage:
                berth_usage[key] = []
            berth_usage[key].append(vid)

        # Track QC usage
        if pid is not None and pid < len(profiles_per_vessel.get(vid, [])):
            prof = profiles_per_vessel[vid][pid]
            s = get_shift_index(start, S_size)
            if (vid, s, pid) in d_sp:
                dur = d_sp[(vid, s, pid)]
                for u in range(1, dur + 1):
                    h = start + u - 1
                    qc_val = q_spu.get((vid, s, pid, u), 0)
                    qc_usage[h] = qc_usage.get(h, 0) + qc_val

        # Compute cost
        delta_k = abs(berth - v["k_bar"])
        delta_t = abs(start - v["t_bar"])
        recomputed_obj += c1 * delta_k + c2 * delta_t

    # Check no-overlap constraint
    for (k, h), vids in berth_usage.items():
        if len(vids) > 1:
            violations.append(f"Berth {k} at time {h}: overlapping vessels {vids}.")
            violated_constraints.add("no_overlap")

    # Check QC capacity
    for h, qc_total in qc_usage.items():
        if qc_total > total_qc:
            violations.append(f"Time step {h}: QC usage {qc_total} > capacity {total_qc}.")
            violated_constraints.add("qc_capacity")

    # Check objective
    reported_obj = solution.get("objective_value")
    obj_match = True
    if reported_obj is not None:
        # c1, c2 are integers and delta_k, delta_t are integer differences,
        # so the recomputed objective is always exactly integer. Use 1e-6 to
        # tolerate float-representation noise without permitting any real
        # integer-level objective drift.
        if abs(reported_obj - recomputed_obj) > 1e-6:
            violations.append(f"Objective mismatch: reported={reported_obj}, recomputed={recomputed_obj}.")
            violated_constraints.add("objective")
            obj_match = False

    feasible = len(violations) == 0

    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "objective_value": reported_obj,
        "recomputed_objective": recomputed_obj
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility check for BACAP (Xie et al. 2019)"
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {result['feasible']}")
    if result['violations']:
        for v in result['violations'][:10]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
