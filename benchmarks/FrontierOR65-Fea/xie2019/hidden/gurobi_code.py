#!/usr/bin/env python3
"""
Gurobi implementation of the Integrated Berth Allocation and Quay Crane
Assignment Problem (BACAP) from:

Xie, Wu, and Zhang (2019), "A Branch-and-Price Algorithm for the Integrated
Berth Allocation and Quay Crane Assignment Problem", Transportation Science.

This implements the Original Problem (OP) formulation (equations 1-35).
"""

import argparse
import json
import random
import sys
from itertools import product

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass


def load_instance(path):
    """Load instance from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_qc_profiles(instance, rng):
    """
    Generate feasible QC profiles for each vessel based on QC parameter ranges.

    A QC profile specifies, for each working shift the vessel occupies the berth,
    the number of QCs assigned during that shift. QCs cannot be moved until the
    end of a working shift (end-of-shift assumption from Giallombardo et al. 2010).

    The profile is parameterized by (s, p) where s is the starting shift-index
    and p encodes the duration and QC assignment per shift.

    **INFERRED ASSUMPTION**: The paper does not give the exact random generation
    procedure for QC profiles. We generate profiles by enumerating combinations
    of (handling_duration_in_shifts, qc_per_shift_tuple) within the min/max
    bounds from Table 4, then randomly sampling the requested number of big
    and small profiles. Each profile's QC assignment per shift is drawn uniformly
    from [min_qc, max_qc] for that category.
    """
    S_size = instance["time_steps_per_shift"]
    F_size = instance["num_shifts_per_step"]
    big_cfg = instance["qc_profiles"]["big"]
    small_cfg = instance["qc_profiles"]["small"]
    num_big = instance["num_big_profiles"]
    num_small = instance["num_small_profiles"]

    def make_profiles_for_category(cfg, count):
        """Generate 'count' profiles for a given QC category, stratified across
        duration buckets so every duration value has >= 1 profile when count
        permits. Without stratification, plain rng.sample can miss the
        minimum-duration bucket entirely (probability ~45%) and leave any
        vessel whose time window only fits min-duration profiles unable to
        select any plan -> sum_p Lambda_i^p = 1 becomes 0 = 1 infeasible."""
        min_qc = cfg["min_qc"]
        max_qc = cfg["max_qc"]
        min_shifts = cfg["min_handling_shifts"]
        max_shifts = cfg["max_handling_shifts"]
        qc_range = list(range(min_qc, max_qc + 1))

        # Group by duration bucket
        by_dur = {}
        for dur in range(min_shifts, max_shifts + 1):
            by_dur[dur] = [
                {"duration_shifts": dur, "qc_per_shift": list(combo)}
                for combo in product(qc_range, repeat=dur)
            ]

        all_possible = [p for ps in by_dur.values() for p in ps]
        if len(all_possible) <= count:
            return all_possible

        # Phase 1: take 1 profile per duration bucket (stratified guarantee)
        n_buckets = len(by_dur)
        selected = []
        for dur in by_dur:
            selected.append(rng.choice(by_dur[dur]))

        # Phase 2: fill remaining slots randomly from the rest
        remaining_count = count - n_buckets
        if remaining_count > 0:
            taken_ids = {id(p) for p in selected}
            rest = [p for p in all_possible if id(p) not in taken_ids]
            extra = rng.sample(rest, min(remaining_count, len(rest)))
            selected.extend(extra)
        return selected

    big_profiles = make_profiles_for_category(big_cfg, num_big)
    small_profiles = make_profiles_for_category(small_cfg, num_small)
    all_profiles_raw = big_profiles + small_profiles

    # For each vessel, determine which profiles are feasible given its duration
    # and time window. A profile is feasible if its total handling time (in time
    # steps) fits within the vessel's time window [a_i, b_i].
    vessels = instance["vessels"]
    profiles_per_vessel = {}
    profile_data = {}  # (vessel_id, profile_id) -> profile info

    for v in vessels:
        vid = v["id"]
        a_i = v["a"]
        b_i = v["b"]
        feasible = []
        pid = 0
        for praw in all_profiles_raw:
            dur_shifts = praw["duration_shifts"]
            dur_steps = dur_shifts * S_size
            # Check if this profile can fit within the vessel's time window
            if dur_steps <= (b_i - a_i + 1):
                # Build per-time-step QC usage: repeat each shift's QC count
                # for S_size time steps
                qc_per_step = []
                for shift_qc in praw["qc_per_shift"]:
                    qc_per_step.extend([shift_qc] * S_size)
                profile_info = {
                    "pid": pid,
                    "duration_steps": dur_steps,
                    "duration_shifts": dur_shifts,
                    "qc_per_shift": praw["qc_per_shift"],
                    "qc_per_step": qc_per_step
                }
                feasible.append(profile_info)
                profile_data[(vid, pid)] = profile_info
                pid += 1
        profiles_per_vessel[vid] = feasible

    return profiles_per_vessel, profile_data


def compute_profile_parameters(profiles_per_vessel, S_size, H_size):
    """
    Compute d_i^{sp} and q_i^{spu} for each vessel, profile, and shift-index s.

    d_i^{sp}: duration in time steps for profile p when vessel i starts at
              a time step with shift-index s.
    q_i^{spu}: number of QCs at relative time step u for profile p when
               vessel i starts at shift-index s.

    **INFERRED ASSUMPTION**: The paper states that d_i^{sp} depends on the
    starting shift-index s because the end-of-shift assumption means that if
    a vessel starts mid-shift, the remaining time in that shift still counts.
    We compute: if vessel starts at shift-index s within a shift, the first
    partial shift has (S_size - s + 1) time steps, then full shifts follow.
    The total duration d_i^{sp} = (S_size - s + 1) + (num_full_shifts - 1) * S_size
    if the profile has num_full_shifts working shifts. But the QC assignment
    in the first partial shift uses the profile's first-shift QC count.

    Actually, re-reading the paper more carefully (Figure 3): the duration
    d_i^{sp} is the total number of time steps from the starting time step to
    the end of the last working shift of the profile. For a profile covering
    D working shifts, if the vessel starts at shift-index s:
    d_i^{sp} = (S_size - s + 1) + (D - 1) * S_size

    The QC at relative step u follows the profile's per-shift assignment,
    where the first (S_size - s + 1) steps use the first shift's QC count,
    then each subsequent block of S_size steps uses the next shift's QC count.
    """
    F_size = H_size // S_size  # number of shifts

    d_sp = {}    # (vessel_id, s, pid) -> duration in time steps
    q_spu = {}   # (vessel_id, s, pid, u) -> QC count at relative step u

    for vid, profiles in profiles_per_vessel.items():
        for prof in profiles:
            pid = prof["pid"]
            D = prof["duration_shifts"]
            qc_shifts = prof["qc_per_shift"]

            for s in range(1, S_size + 1):
                # Time steps remaining in the first shift
                first_partial = S_size - s + 1
                # Total duration
                total_dur = first_partial + (D - 1) * S_size
                d_sp[(vid, s, pid)] = total_dur

                # Build QC per relative step u (1-indexed)
                u = 1
                # First partial shift
                for _ in range(first_partial):
                    q_spu[(vid, s, pid, u)] = qc_shifts[0]
                    u += 1
                # Remaining full shifts
                for shift_idx in range(1, D):
                    for _ in range(S_size):
                        q_spu[(vid, s, pid, u)] = qc_shifts[shift_idx]
                        u += 1

    return d_sp, q_spu


def get_shift_index(h, S_size):
    """Get the shift-index s for time step h. s = ((h-1) % S_size) + 1."""
    return ((h - 1) % S_size) + 1


def build_H_s_sets(S_size, H_size):
    """Build H^s sets: for each s in S, the set of h in H with that shift index."""
    H_s = {}
    for s in range(1, S_size + 1):
        H_s[s] = []
        for h in range(1, H_size + 1):
            if get_shift_index(h, S_size) == s:
                H_s[s].append(h)
    return H_s


def build_P_i_s(profiles_per_vessel, d_sp, S_size):
    """
    Build P_i^s: for each vessel i and shift-index s, the set of profiles
    that are dedicated to time steps with index s within a working shift.

    A profile p is in P_i^s if d_i^{sp} is defined (i.e., the profile is
    compatible with starting at shift-index s).
    """
    P_i_s = {}
    for vid, profiles in profiles_per_vessel.items():
        for s in range(1, S_size + 1):
            P_i_s[(vid, s)] = []
            for prof in profiles:
                pid = prof["pid"]
                if (vid, s, pid) in d_sp:
                    P_i_s[(vid, s)].append(pid)
    return P_i_s


def solve_bacap(instance, time_limit):
    """Build and solve the BACAP model using Gurobi."""
    rng = random.Random(instance.get("random_seed", 42))

    n = instance["num_vessels"]
    m = instance["num_berths"]
    S_size = instance["time_steps_per_shift"]
    F_size = instance["num_shifts_per_step"]
    H_size = S_size * F_size  # total time steps
    total_qc = instance["total_qc"]
    c1 = instance["c1"]
    c2 = instance["c2"]

    vessels = instance["vessels"]
    berths = instance["berths"]

    # Generate QC profiles
    profiles_per_vessel, profile_data = generate_qc_profiles(instance, rng)

    # Compute profile parameters
    d_sp, q_spu = compute_profile_parameters(profiles_per_vessel, S_size, H_size)

    # Build index sets
    N = [v["id"] for v in vessels]
    M = [b["id"] for b in berths]
    H = list(range(1, H_size + 1))
    S = list(range(1, S_size + 1))
    H_s = build_H_s_sets(S_size, H_size)
    P_i_s = build_P_i_s(profiles_per_vessel, d_sp, S_size)

    # Berth time windows
    a_k = {b["id"]: b["a_k"] for b in berths}
    b_k = {b["id"]: b["b_k"] for b in berths}

    # Vessel parameters
    a_i = {v["id"]: v["a"] for v in vessels}
    b_i = {v["id"]: v["b"] for v in vessels}
    k_bar = {v["id"]: v["k_bar"] for v in vessels}
    t_bar = {v["id"]: v["t_bar"] for v in vessels}

    # Big-M: **NOT SPECIFIED IN PAPER**. We use H_size + 1 as a safe upper bound.
    BIG_M = H_size + 1

    # QC capacity at each time step
    # **INFERRED ASSUMPTION**: q^h is constant = total_qc for all h.
    q_h = {h: total_qc for h in H}

    # All profile IDs per vessel
    P_i = {}
    for vid in N:
        P_i[vid] = [p["pid"] for p in profiles_per_vessel[vid]]

    # ====== Build Gurobi Model ======
    model = gp.Model("BACAP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # --- Decision Variables ---
    # Y_i^k: berth k assigned to vessel i
    Y = {}
    for i in N:
        for k in M:
            Y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"Y_{i}_{k}")

    # Lambda_i^p: QC profile p assigned to vessel i
    Lambda = {}
    for i in N:
        for p in P_i[i]:
            Lambda[i, p] = model.addVar(vtype=GRB.BINARY, name=f"Lam_{i}_{p}")

    # Gamma_i^h: time step h assigned as start to vessel i
    Gamma = {}
    for i in N:
        for h in H:
            Gamma[i, h] = model.addVar(vtype=GRB.BINARY, name=f"Gam_{i}_{h}")

    # T_i: start time, E_i: end time
    T = {}
    E = {}
    for i in N:
        T[i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"T_{i}")
        E[i] = model.addVar(vtype=GRB.INTEGER, lb=0, name=f"E_{i}")

    # Omega_i^{ph}: profile p and time step h both assigned to vessel i
    Omega = {}
    for i in N:
        for p in P_i[i]:
            for h in H:
                Omega[i, p, h] = model.addVar(vtype=GRB.BINARY,
                                               name=f"Om_{i}_{p}_{h}")

    # AS_i^h, BE_i^h, BT_i^h: auxiliary indicator variables
    AS = {}
    BE = {}
    BT = {}
    for i in N:
        for h in H:
            AS[i, h] = model.addVar(vtype=GRB.BINARY, name=f"AS_{i}_{h}")
            BE[i, h] = model.addVar(vtype=GRB.BINARY, name=f"BE_{i}_{h}")
            BT[i, h] = model.addVar(vtype=GRB.BINARY, name=f"BT_{i}_{h}")

    # X_i^{kh}: berth k occupied by vessel i at time step h
    X = {}
    for i in N:
        for k in M:
            for h in H:
                X[i, k, h] = model.addVar(vtype=GRB.BINARY,
                                           name=f"X_{i}_{k}_{h}")

    # Linearization variables for absolute values
    U1 = {}
    V1 = {}
    U2 = {}
    V2 = {}
    for i in N:
        U1[i] = model.addVar(lb=0, name=f"U1_{i}")
        V1[i] = model.addVar(lb=0, name=f"V1_{i}")
        U2[i] = model.addVar(lb=0, name=f"U2_{i}")
        V2[i] = model.addVar(lb=0, name=f"V2_{i}")

    model.update()

    # --- Objective (35): min sum_i (c1*(U1_i+V1_i) + c2*(U2_i+V2_i)) ---
    model.setObjective(
        gp.quicksum(c1 * (U1[i] + V1[i]) + c2 * (U2[i] + V2[i]) for i in N),
        GRB.MINIMIZE
    )

    # --- Linearization constraints (32)-(34) ---
    for i in N:
        # (32): sum_k k*Y_i^k - k_bar_i + U1_i - V1_i = 0
        model.addConstr(
            gp.quicksum(k * Y[i, k] for k in M) - k_bar[i] + U1[i] - V1[i] == 0,
            name=f"lin32_{i}"
        )
        # (33): T_i - t_bar_i + U2_i - V2_i = 0
        model.addConstr(
            T[i] - t_bar[i] + U2[i] - V2[i] == 0,
            name=f"lin33_{i}"
        )

    # --- Assignment constraints ---
    # (2): sum_k Y_i^k = 1
    for i in N:
        model.addConstr(
            gp.quicksum(Y[i, k] for k in M) == 1,
            name=f"assign_berth_{i}"
        )

    # (3): sum_p Lambda_i^p = 1
    for i in N:
        model.addConstr(
            gp.quicksum(Lambda[i, p] for p in P_i[i]) == 1,
            name=f"assign_profile_{i}"
        )

    # (4): sum_h Gamma_i^h = 1
    for i in N:
        model.addConstr(
            gp.quicksum(Gamma[i, h] for h in H) == 1,
            name=f"assign_time_{i}"
        )

    # (5): sum_{h in H^s} Gamma_i^h <= sum_{p in P_i^s} Lambda_i^p
    for i in N:
        for s in S:
            lhs = gp.quicksum(Gamma[i, h] for h in H_s[s])
            rhs_pids = P_i_s.get((i, s), [])
            if rhs_pids:
                rhs = gp.quicksum(Lambda[i, p] for p in rhs_pids)
            else:
                rhs = 0
            model.addConstr(lhs <= rhs, name=f"compat_{i}_{s}")

    # (6): T_i = sum_h h*Gamma_i^h
    for i in N:
        model.addConstr(
            T[i] == gp.quicksum(h * Gamma[i, h] for h in H),
            name=f"def_T_{i}"
        )

    # (7): E_i = sum_h h*Gamma_i^h + sum_{h in H^s} sum_{p in P_i^s} d_i^{sp} * Omega_i^{ph} - 1
    # This must hold for all s in S. Since only one Gamma_i^h=1 (and thus only one s active),
    # the constraint is effectively: E_i = T_i + d_i^{s*,p*} - 1 for the active (s*, p*).
    # We implement it as: for each s, add the contribution from profiles in P_i^s.
    for i in N:
        # E_i = T_i + sum_s sum_{h in H^s} sum_{p in P_i^s} d_i^{sp} * Omega_i^{ph} - 1
        dur_expr = gp.LinExpr()
        for s in S:
            for h in H_s[s]:
                for p in P_i_s.get((i, s), []):
                    if (i, s, p) in d_sp:
                        dur_expr.add(Omega[i, p, h], d_sp[(i, s, p)])
        model.addConstr(
            E[i] == gp.quicksum(h * Gamma[i, h] for h in H) + dur_expr - 1,
            name=f"def_E_{i}"
        )

    # (8)-(9): Time window on vessels
    for i in N:
        model.addConstr(a_i[i] <= T[i], name=f"tw_lo_{i}")
        model.addConstr(T[i] <= b_i[i], name=f"tw_hi_{i}")

    # (10)-(11): Time window on berths
    for i in N:
        model.addConstr(
            gp.quicksum(a_k[k] * Y[i, k] for k in M) <= T[i],
            name=f"berth_tw_lo_{i}"
        )
        model.addConstr(
            T[i] <= gp.quicksum(b_k[k] * Y[i, k] for k in M),
            name=f"berth_tw_hi_{i}"
        )

    # (12): h - T_i + 1 <= M * AS_i^h
    for i in N:
        for h in H:
            model.addConstr(
                h - T[i] + 1 <= BIG_M * AS[i, h],
                name=f"def_AS_{i}_{h}"
            )

    # (13): E_i - h + 1 <= M * BE_i^h
    for i in N:
        for h in H:
            model.addConstr(
                E[i] - h + 1 <= BIG_M * BE[i, h],
                name=f"def_BE_{i}_{h}"
            )

    # (14): BT_i^h >= AS_i^h + BE_i^h - 1
    for i in N:
        for h in H:
            model.addConstr(
                BT[i, h] >= AS[i, h] + BE[i, h] - 1,
                name=f"def_BT_lo_{i}_{h}"
            )

    # (15): T_i - h <= M*(1 - BT_i^h)
    for i in N:
        for h in H:
            model.addConstr(
                T[i] - h <= BIG_M * (1 - BT[i, h]),
                name=f"def_BT_hi1_{i}_{h}"
            )

    # (16): h - E_i <= M*(1 - BT_i^h)
    for i in N:
        for h in H:
            model.addConstr(
                h - E[i] <= BIG_M * (1 - BT[i, h]),
                name=f"def_BT_hi2_{i}_{h}"
            )

    # (17): 2 - Lambda_i^p - Gamma_i^h <= M*(1 - Omega_i^{ph})
    for i in N:
        for p in P_i[i]:
            for h in H:
                model.addConstr(
                    2 - Lambda[i, p] - Gamma[i, h] <= BIG_M * (1 - Omega[i, p, h]),
                    name=f"def_Om_hi_{i}_{p}_{h}"
                )

    # (18): Omega_i^{ph} >= Lambda_i^p + Gamma_i^h - 1
    for i in N:
        for p in P_i[i]:
            for h in H:
                model.addConstr(
                    Omega[i, p, h] >= Lambda[i, p] + Gamma[i, h] - 1,
                    name=f"def_Om_lo_{i}_{p}_{h}"
                )

    # (19): 2 - BT_i^h - Y_i^k <= M*(1 - X_i^{kh})
    for i in N:
        for k in M:
            for h in H:
                model.addConstr(
                    2 - BT[i, h] - Y[i, k] <= BIG_M * (1 - X[i, k, h]),
                    name=f"def_X_hi_{i}_{k}_{h}"
                )

    # (20): X_i^{kh} >= BT_i^h + Y_i^k - 1
    for i in N:
        for k in M:
            for h in H:
                model.addConstr(
                    X[i, k, h] >= BT[i, h] + Y[i, k] - 1,
                    name=f"def_X_lo_{i}_{k}_{h}"
                )

    # (21): QC capacity constraint
    # sum_i sum_p sum_s sum_{t in H^s, t<=h<=t+d_i^{sp}-1} q_i^{sp(h-t+1)} * Omega_i^{pt} <= q^h
    for h_cur in H:
        qc_expr = gp.LinExpr()
        for i in N:
            for p in P_i[i]:
                for s in S:
                    if (i, s) not in P_i_s or p not in P_i_s[(i, s)]:
                        continue
                    if (i, s, p) not in d_sp:
                        continue
                    dur = d_sp[(i, s, p)]
                    for t in H_s[s]:
                        if t <= h_cur <= t + dur - 1:
                            u = h_cur - t + 1
                            qc_val = q_spu.get((i, s, p, u), 0)
                            if qc_val > 0:
                                qc_expr.add(Omega[i, p, t], qc_val)
        model.addConstr(qc_expr <= q_h[h_cur], name=f"qc_cap_{h_cur}")

    # (22): No overlapping constraint
    for k in M:
        for h in H:
            model.addConstr(
                gp.quicksum(X[i, k, h] for i in N) <= 1,
                name=f"no_overlap_{k}_{h}"
            )

    # --- Solve ---
    model.optimize()

    # --- Extract solution ---
    result = {"objective_value": None, "status": model.Status}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["vessels"] = []
        for i in N:
            v_sol = {"id": i}
            # Berth assignment
            for k in M:
                if Y[i, k].X > 0.5:
                    v_sol["berth"] = k
                    break
            # Start time
            v_sol["start_time"] = round(T[i].X)
            v_sol["end_time"] = round(E[i].X)
            # QC profile
            for p in P_i[i]:
                if Lambda[i, p].X > 0.5:
                    v_sol["profile"] = p
                    break
            result["vessels"].append(v_sol)

        if model.Status == GRB.OPTIMAL:
            result["status_str"] = "OPTIMAL"
        elif model.Status == GRB.TIME_LIMIT:
            result["status_str"] = "TIME_LIMIT"
        else:
            result["status_str"] = "FEASIBLE"
    else:
        result["objective_value"] = None
        result["status_str"] = "NO_SOLUTION"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve BACAP using Gurobi (Xie et al. 2019 OP formulation)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    result = solve_bacap(instance, args.time_limit)

    # Write solution
    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status_str']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
