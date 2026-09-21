#!/usr/bin/env python3
"""
Feasibility checker for the IMSOAN two-stage stochastic integer program
from Wang & Jacquillat (2020).

Checks all hard constraints from the mathematical formulation against a
candidate solution. Constraints are numbered 1..19 counting from top to
bottom in the formulation:

  Constraint  1 : (1b) w_dep non-increasing
  Constraint  2 : (1c) w_arr non-increasing
  Constraint  3 : (1d) scheduled en-route time equality
  Constraint  4 : (1e) first-stage connection constraints
  Constraint  5 : (1f) w_dep, w_arr binary and in domain
  Constraint  6 : (2b) x_dep non-increasing
  Constraint  7 : (2c) x_arr non-increasing
  Constraint  8 : (2d) departure delay definition (equality)
  Constraint  9 : (2e) arrival delay definition (inequality)
  Constraint 10 : (2f) departure delay bound
  Constraint 11 : (2g) arrival delay bound
  Constraint 12 : (2h) second-stage aircraft connections
  Constraint 13 : (2i) minimum en-route time
  Constraint 14 : (2j) maximum en-route time
  Constraint 15 : (2k) capacity envelope
  Constraint 16 : (2l) valid inequality: no early departure
  Constraint 17 : (2m) valid inequality: en-route savings
  Constraint 18 : (2n) x_dep, x_arr binary
  Constraint 19 : (2o) v_dep, v_arr >= 0
  Constraint 20 : (Tier C, obj-consistency) reported objective_value must lie within
                  [min(disp_cost, delay_cost), max(disp_cost, delay_cost)] where
                  disp_cost = sum_i sum_{t in bar_T_dep_i} g_{it}*(w_dep[t-1]-w_dep[t])
                  and  delay_cost = sum_s p_s * sum_i (c^dep_i*v^dep_{is} + c^arr_i*v^arr_{is}).
                  Because obj = rho*disp + (1-rho)*delay with rho in [0,1] (rho is not
                  carried in the instance; the paper tests rho in {0.46,0.67,0.82,0.95}),
                  this rho-independent envelope holds for any honest objective value.
"""

import argparse
import json
import math

TOL = 1e-5
EPS = 1e-5


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def load_solution(path):
    with open(path) as f:
        return json.load(f)


def norm_ratio(raw_excess, rhs):
    normalizer = max(abs(rhs), EPS)
    return raw_excess / normalizer, normalizer


def record_violation(violations_list, magnitudes_list,
                     constraint_idx, msg, lhs, rhs, raw_excess):
    violations_list.append((constraint_idx, msg))
    ratio, normalizer = norm_ratio(raw_excess, rhs)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw_excess),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def reconstruct_w_dep(scheduled_dep, S_dep, delta):
    """Reconstruct the w_dep non-increasing binary vector from scheduled_dep.

    The w_dep variables are defined over bar_T_dep = {S_dep - delta + 1, ..., S_dep + delta}.
    By convention, w[t]=1 for t <= S_dep - delta, w[t]=0 for t > S_dep + delta.
    The scheduled departure time = S_dep - delta + sum of w variables over bar_T_dep.
    So sum_var = scheduled_dep - (S_dep - delta).
    The non-increasing encoding: first sum_var entries are 1, rest are 0.
    """
    bar_T = list(range(S_dep - delta + 1, S_dep + delta + 1))
    sum_var = scheduled_dep - (S_dep - delta)
    w = {}
    for idx, t in enumerate(bar_T):
        w[t] = 1 if idx < sum_var else 0
    return w, bar_T


def reconstruct_w_arr(scheduled_arr, S_arr, delta):
    bar_T = list(range(S_arr - delta + 1, S_arr + delta + 1))
    sum_var = scheduled_arr - (S_arr - delta)
    w = {}
    for idx, t in enumerate(bar_T):
        w[t] = 1 if idx < sum_var else 0
    return w, bar_T


def check_feasibility(instance, solution):
    violations = []
    magnitudes = []

    delta = instance["maximum_displacement_delta"]
    flights = instance["flights"]
    flight_by_id = {fl["id"]: fl for fl in flights}

    # Build E^1 (union of aircraft + passenger connections)
    e1 = {}
    for ac in instance["aircraft_connections"]:
        pair = (ac["flight_i"], ac["flight_j"])
        tau = ac["min_connection_time"]
        e1[pair] = max(e1.get(pair, 0), tau)
    for pc in instance["passenger_connections"]:
        pair = (pc["flight_i"], pc["flight_j"])
        tau = pc["min_connection_time"]
        e1[pair] = max(e1.get(pair, 0), tau)

    # Check if solution has schedule data
    schedule = solution.get("schedule")
    if not schedule:
        record_violation(violations, magnitudes, 0,
                         "Solution is missing 'schedule'; cannot verify first-stage feasibility.",
                         0.0, 0.0, 1.0)
        return violations, magnitudes

    # Parse schedule: keys may be strings
    sched = {}
    for key, val in schedule.items():
        fid = int(key) if isinstance(key, str) else key
        sched[fid] = val

    # =========================================================================
    # Constraint 1 (1b): w_dep non-increasing
    # =========================================================================
    for fl in flights:
        fid = fl["id"]
        if fid not in sched:
            continue
        S_dep = fl["dep_period"]
        sd = sched[fid]["scheduled_dep"]
        w_dep, bar_T = reconstruct_w_dep(sd, S_dep, delta)
        for idx in range(1, len(bar_T)):
            t = bar_T[idx]
            t_prev = bar_T[idx - 1]
            # Constraint: w[t] <= w[t-1], i.e., w[t] - w[t-1] <= 0
            lhs_val = w_dep[t]
            rhs_val = w_dep[t_prev]
            excess = lhs_val - rhs_val
            if excess > TOL:
                record_violation(violations, magnitudes, 1,
                                 f"Flight {fid}: w_dep non-increasing violated at t={t} "
                                 f"(w[{t}]={lhs_val} > w[{t_prev}]={rhs_val})",
                                 lhs_val, rhs_val, excess)

    # =========================================================================
    # Constraint 2 (1c): w_arr non-increasing
    # =========================================================================
    for fl in flights:
        fid = fl["id"]
        if fid not in sched:
            continue
        S_arr = fl["arr_period"]
        sa = sched[fid]["scheduled_arr"]
        w_arr, bar_T = reconstruct_w_arr(sa, S_arr, delta)
        for idx in range(1, len(bar_T)):
            t = bar_T[idx]
            t_prev = bar_T[idx - 1]
            lhs_val = w_arr[t]
            rhs_val = w_arr[t_prev]
            excess = lhs_val - rhs_val
            if excess > TOL:
                record_violation(violations, magnitudes, 2,
                                 f"Flight {fid}: w_arr non-increasing violated at t={t} "
                                 f"(w[{t}]={lhs_val} > w[{t_prev}]={rhs_val})",
                                 lhs_val, rhs_val, excess)

    # =========================================================================
    # Constraint 3 (1d): sum_T(w_arr - w_dep) = delta_sch
    #   Equivalently: scheduled_arr - scheduled_dep = delta_sch
    # =========================================================================
    for fl in flights:
        fid = fl["id"]
        if fid not in sched:
            continue
        delta_sch = fl["delta_sch"]
        sd = sched[fid]["scheduled_dep"]
        sa = sched[fid]["scheduled_arr"]
        lhs_val = sa - sd
        rhs_val = delta_sch
        excess = abs(lhs_val - rhs_val)
        if excess > TOL:
            record_violation(violations, magnitudes, 3,
                             f"Flight {fid}: en-route time {lhs_val} != delta_sch {rhs_val} "
                             f"(scheduled_arr={sa}, scheduled_dep={sd})",
                             lhs_val, rhs_val, excess)

    # =========================================================================
    # Constraint 4 (1e): sum_T(w_dep_j - w_arr_i) >= tau for (i,j) in E^1
    #   Equivalently: scheduled_dep(j) - scheduled_arr(i) >= tau
    # =========================================================================
    for (fi, fj), tau in e1.items():
        if fi not in sched or fj not in sched:
            continue
        sa_i = sched[fi]["scheduled_arr"]
        sd_j = sched[fj]["scheduled_dep"]
        lhs_val = sd_j - sa_i
        rhs_val = tau
        excess = rhs_val - lhs_val  # >= constraint: violation if RHS > LHS
        if excess > TOL:
            record_violation(violations, magnitudes, 4,
                             f"Connection ({fi},{fj}): scheduled_dep({fj})={sd_j} - "
                             f"scheduled_arr({fi})={sa_i} = {lhs_val} < tau={tau}",
                             lhs_val, rhs_val, excess)

    # =========================================================================
    # Constraint 5 (1f): w_dep, w_arr binary and in domain
    #   scheduled_dep in {S_dep - delta, ..., S_dep + delta} (integer)
    #   scheduled_arr in {S_arr - delta, ..., S_arr + delta} (integer)
    # =========================================================================
    for fl in flights:
        fid = fl["id"]
        if fid not in sched:
            continue
        S_dep = fl["dep_period"]
        S_arr = fl["arr_period"]
        sd = sched[fid]["scheduled_dep"]
        sa = sched[fid]["scheduled_arr"]

        # Check departure domain
        dep_lo = S_dep - delta
        dep_hi = S_dep + delta
        if not isinstance(sd, int) and not (isinstance(sd, float) and sd == int(sd)):
            lhs_val = sd
            rhs_val = round(sd)
            excess = abs(sd - round(sd))
            if excess > TOL:
                record_violation(violations, magnitudes, 5,
                                 f"Flight {fid}: scheduled_dep={sd} is not integer",
                                 lhs_val, rhs_val, excess)
        sd_int = int(round(sd))
        if sd_int < dep_lo:
            excess = dep_lo - sd_int
            record_violation(violations, magnitudes, 5,
                             f"Flight {fid}: scheduled_dep={sd_int} < lower bound {dep_lo}",
                             float(sd_int), float(dep_lo), float(excess))
        elif sd_int > dep_hi:
            excess = sd_int - dep_hi
            record_violation(violations, magnitudes, 5,
                             f"Flight {fid}: scheduled_dep={sd_int} > upper bound {dep_hi}",
                             float(sd_int), float(dep_hi), float(excess))

        # Check arrival domain
        arr_lo = S_arr - delta
        arr_hi = S_arr + delta
        if not isinstance(sa, int) and not (isinstance(sa, float) and sa == int(sa)):
            lhs_val = sa
            rhs_val = round(sa)
            excess = abs(sa - round(sa))
            if excess > TOL:
                record_violation(violations, magnitudes, 5,
                                 f"Flight {fid}: scheduled_arr={sa} is not integer",
                                 lhs_val, rhs_val, excess)
        sa_int = int(round(sa))
        if sa_int < arr_lo:
            excess = arr_lo - sa_int
            record_violation(violations, magnitudes, 5,
                             f"Flight {fid}: scheduled_arr={sa_int} < lower bound {arr_lo}",
                             float(sa_int), float(arr_lo), float(excess))
        elif sa_int > arr_hi:
            excess = sa_int - arr_hi
            record_violation(violations, magnitudes, 5,
                             f"Flight {fid}: scheduled_arr={sa_int} > upper bound {arr_hi}",
                             float(sa_int), float(arr_hi), float(excess))

    # =========================================================================
    # Second-stage constraints (6-19) require x, v variables per scenario.
    # These are only checked if the solution provides them.
    # =========================================================================
    scenarios_sol = solution.get("scenario_solutions")
    if not scenarios_sol:
        record_violation(violations, magnitudes, 0,
                         "Solution is missing 'scenario_solutions'; cannot verify second-stage feasibility.",
                         0.0, 0.0, 1.0)
    if scenarios_sol:
        T = instance["num_time_periods"]
        scenarios = instance["scenarios"]

        # Build E^2 (aircraft connections only)
        e2 = {}
        for ac in instance["aircraft_connections"]:
            pair = (ac["flight_i"], ac["flight_j"])
            tau = ac["min_connection_time"]
            e2[pair] = max(e2.get(pair, 0), tau)

        dep_at = {}
        arr_at = {}
        for fl in flights:
            dep_at.setdefault(fl["dep_airport"], []).append(fl["id"])
            arr_at.setdefault(fl["arr_airport"], []).append(fl["id"])

        for sc_sol in scenarios_sol:
            s_id = sc_sol["scenario_id"]
            # Find matching scenario data
            sc_data = None
            for sc in scenarios:
                if sc["id"] == s_id:
                    sc_data = sc
                    break
            if sc_data is None:
                continue

            oc = sc_data["operating_conditions"]

            # Extract x_dep, x_arr, v_dep, v_arr from solution
            x_dep_sol = sc_sol.get("x_dep", {})
            x_arr_sol = sc_sol.get("x_arr", {})
            v_dep_sol = sc_sol.get("v_dep", {})
            v_arr_sol = sc_sol.get("v_arr", {})

            def get_x_dep(fid, t):
                key = f"{fid}_{t}"
                if key in x_dep_sol:
                    return x_dep_sol[key]
                fl = flight_by_id[fid]
                S = fl["dep_period"]
                l = fl["max_dep_delay"]
                return 1.0 if t <= S - delta else 0.0

            def get_x_arr(fid, t):
                key = f"{fid}_{t}"
                if key in x_arr_sol:
                    return x_arr_sol[key]
                fl = flight_by_id[fid]
                S_dep = fl["dep_period"]
                dmin = fl["delta_min"]
                return 1.0 if t <= S_dep - delta + dmin else 0.0

            def full_sum_x_dep(fid):
                fl = flight_by_id[fid]
                S = fl["dep_period"]
                l = fl["max_dep_delay"]
                base = S - delta
                ts = range(S - delta + 1, S + delta + l + 1)
                return base + sum(get_x_dep(fid, t) for t in ts)

            def full_sum_x_arr(fid):
                fl = flight_by_id[fid]
                S_dep = fl["dep_period"]
                S_arr = fl["arr_period"]
                dmin = fl["delta_min"]
                l = fl["max_arr_delay"]
                base = S_dep - delta + dmin
                ts = range(S_dep - delta + dmin + 1, S_arr + delta + l + 1)
                return base + sum(get_x_arr(fid, t) for t in ts)

            # Constraint 6 (2b): x_dep non-increasing
            for fl in flights:
                fid = fl["id"]
                S = fl["dep_period"]
                l = fl["max_dep_delay"]
                ts = list(range(S - delta + 1, S + delta + l + 1))
                for idx in range(1, len(ts)):
                    t = ts[idx]
                    t_prev = ts[idx - 1]
                    xc = get_x_dep(fid, t)
                    xp = get_x_dep(fid, t_prev)
                    excess = xc - xp
                    if excess > TOL:
                        record_violation(violations, magnitudes, 6,
                                         f"Scenario {s_id}, flight {fid}: x_dep non-increasing "
                                         f"violated at t={t}",
                                         xc, xp, excess)

            # Constraint 7 (2c): x_arr non-increasing
            for fl in flights:
                fid = fl["id"]
                S_dep = fl["dep_period"]
                S_arr = fl["arr_period"]
                dmin = fl["delta_min"]
                l = fl["max_arr_delay"]
                ts = list(range(S_dep - delta + dmin + 1, S_arr + delta + l + 1))
                for idx in range(1, len(ts)):
                    t = ts[idx]
                    t_prev = ts[idx - 1]
                    xc = get_x_arr(fid, t)
                    xp = get_x_arr(fid, t_prev)
                    excess = xc - xp
                    if excess > TOL:
                        record_violation(violations, magnitudes, 7,
                                         f"Scenario {s_id}, flight {fid}: x_arr non-increasing "
                                         f"violated at t={t}",
                                         xc, xp, excess)

            # Constraint 8 (2d): full_sum_x_dep - full_sum_w_dep = v_dep (equality)
            for fl in flights:
                fid = fl["id"]
                if fid not in sched:
                    continue
                sd = sched[fid]["scheduled_dep"]
                sum_xd = full_sum_x_dep(fid)
                v_d = v_dep_sol.get(str(fid), 0.0)
                lhs_val = sum_xd - sd
                rhs_val = v_d
                excess = abs(lhs_val - rhs_val)
                if excess > TOL:
                    record_violation(violations, magnitudes, 8,
                                     f"Scenario {s_id}, flight {fid}: dep delay def violated "
                                     f"(sum_x_dep - sched_dep = {lhs_val} != v_dep = {v_d})",
                                     lhs_val, rhs_val, excess)

            # Constraint 9 (2e): full_sum_x_arr - full_sum_w_arr <= v_arr
            for fl in flights:
                fid = fl["id"]
                if fid not in sched:
                    continue
                sa = sched[fid]["scheduled_arr"]
                sum_xa = full_sum_x_arr(fid)
                v_a = v_arr_sol.get(str(fid), 0.0)
                lhs_val = sum_xa - sa
                rhs_val = v_a
                excess = lhs_val - rhs_val
                if excess > TOL:
                    record_violation(violations, magnitudes, 9,
                                     f"Scenario {s_id}, flight {fid}: arr delay violated "
                                     f"(sum_x_arr - sched_arr = {lhs_val} > v_arr = {v_a})",
                                     lhs_val, rhs_val, excess)

            # Constraint 10 (2f): v_dep <= l_dep
            for fl in flights:
                fid = fl["id"]
                v_d = v_dep_sol.get(str(fid), 0.0)
                l_dep = fl["max_dep_delay"]
                excess = v_d - l_dep
                if excess > TOL:
                    record_violation(violations, magnitudes, 10,
                                     f"Scenario {s_id}, flight {fid}: v_dep={v_d} > "
                                     f"l_dep={l_dep}",
                                     v_d, float(l_dep), excess)

            # Constraint 11 (2g): v_arr <= l_arr
            for fl in flights:
                fid = fl["id"]
                v_a = v_arr_sol.get(str(fid), 0.0)
                l_arr = fl["max_arr_delay"]
                excess = v_a - l_arr
                if excess > TOL:
                    record_violation(violations, magnitudes, 11,
                                     f"Scenario {s_id}, flight {fid}: v_arr={v_a} > "
                                     f"l_arr={l_arr}",
                                     v_a, float(l_arr), excess)

            # Constraint 12 (2h): full_sum_x_dep(j) - full_sum_x_arr(i) >= tau
            for (fi, fj), tau in e2.items():
                sum_xd_j = full_sum_x_dep(fj)
                sum_xa_i = full_sum_x_arr(fi)
                lhs_val = sum_xd_j - sum_xa_i
                rhs_val = tau
                excess = rhs_val - lhs_val
                if excess > TOL:
                    record_violation(violations, magnitudes, 12,
                                     f"Scenario {s_id}, connection ({fi},{fj}): "
                                     f"x_dep_sum({fj}) - x_arr_sum({fi}) = {lhs_val} < tau={tau}",
                                     lhs_val, float(rhs_val), excess)

            # Constraint 13 (2i): full_sum_x_arr - full_sum_x_dep >= delta_min
            for fl in flights:
                fid = fl["id"]
                dmin = fl["delta_min"]
                enroute = full_sum_x_arr(fid) - full_sum_x_dep(fid)
                excess = dmin - enroute
                if excess > TOL:
                    record_violation(violations, magnitudes, 13,
                                     f"Scenario {s_id}, flight {fid}: en-route {enroute} < "
                                     f"delta_min={dmin}",
                                     enroute, float(dmin), excess)

            # Constraint 14 (2j): full_sum_x_arr - full_sum_x_dep <= delta_max
            for fl in flights:
                fid = fl["id"]
                dmax = fl["delta_max"]
                enroute = full_sum_x_arr(fid) - full_sum_x_dep(fid)
                excess = enroute - dmax
                if excess > TOL:
                    record_violation(violations, magnitudes, 14,
                                     f"Scenario {s_id}, flight {fid}: en-route {enroute} > "
                                     f"delta_max={dmax}",
                                     enroute, float(dmax), excess)

            # Constraint 15 (2k): capacity envelope
            cap_env = instance["capacity_envelopes"]
            for k in instance["airports"]:
                cond_list = oc[k]
                ce = cap_env[k]
                dep_flights = dep_at.get(k, [])
                arr_flights = arr_at.get(k, [])
                for t in range(1, T + 1):
                    cond = cond_list[t - 1]
                    segments = ce[cond]
                    for q_idx, seg in enumerate(segments):
                        a_val = seg["a"]
                        b_val = seg["b"]
                        Q_val = seg["Q"]
                        lhs_val = 0.0
                        if a_val != 0:
                            for fid in dep_flights:
                                xp = get_x_dep(fid, t - 1)
                                xc = get_x_dep(fid, t)
                                lhs_val += a_val * (xp - xc)
                        if b_val != 0:
                            for fid in arr_flights:
                                xp = get_x_arr(fid, t - 1)
                                xc = get_x_arr(fid, t)
                                lhs_val += b_val * (xp - xc)
                        excess = lhs_val - Q_val
                        if excess > TOL:
                            record_violation(violations, magnitudes, 15,
                                             f"Scenario {s_id}, airport {k}, t={t}, "
                                             f"seg {q_idx}: capacity {lhs_val} > Q={Q_val}",
                                             lhs_val, float(Q_val), excess)

            # Constraint 18 (2n): x_dep, x_arr binary
            for fl in flights:
                fid = fl["id"]
                S = fl["dep_period"]
                l = fl["max_dep_delay"]
                for t in range(S - delta + 1, S + delta + l + 1):
                    xd = get_x_dep(fid, t)
                    frac = min(abs(xd - round(xd)), abs(xd), abs(1 - xd))
                    if frac > TOL:
                        record_violation(violations, magnitudes, 18,
                                         f"Scenario {s_id}, flight {fid}: "
                                         f"x_dep[{t}]={xd} not binary",
                                         xd, round(xd), frac)

                S_dep = fl["dep_period"]
                S_arr = fl["arr_period"]
                dmin = fl["delta_min"]
                l_arr = fl["max_arr_delay"]
                for t in range(S_dep - delta + dmin + 1, S_arr + delta + l_arr + 1):
                    xa = get_x_arr(fid, t)
                    frac = min(abs(xa - round(xa)), abs(xa), abs(1 - xa))
                    if frac > TOL:
                        record_violation(violations, magnitudes, 18,
                                         f"Scenario {s_id}, flight {fid}: "
                                         f"x_arr[{t}]={xa} not binary",
                                         xa, round(xa), frac)

            # Constraint 19 (2o): v_dep, v_arr >= 0
            for fl in flights:
                fid = fl["id"]
                v_d = v_dep_sol.get(str(fid), 0.0)
                if v_d < -TOL:
                    record_violation(violations, magnitudes, 19,
                                     f"Scenario {s_id}, flight {fid}: v_dep={v_d} < 0",
                                     v_d, 0.0, -v_d)
                v_a = v_arr_sol.get(str(fid), 0.0)
                if v_a < -TOL:
                    record_violation(violations, magnitudes, 19,
                                     f"Scenario {s_id}, flight {fid}: v_arr={v_a} < 0",
                                     v_a, 0.0, -v_a)

    # =========================================================================
    # Constraint 20 (Tier C, obj consistency): rho-independent envelope check.
    # obj = rho*disp + (1-rho)*delay with rho in [0,1] (rho is not in the
    # instance schema). For any rho the value lies in [min(disp,delay),
    # max(disp,delay)]. disp is computable exactly from `schedule`; delay is
    # computable exactly from per-scenario `v_dep`, `v_arr` and scenario
    # probabilities. Both are non-negative components matching the gurobi_code
    # objective (1a).
    # =========================================================================
    reported_raw = solution.get("objective_value")
    if reported_raw is not None and schedule:
        try:
            reported_obj = float(reported_raw)
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and math.isfinite(reported_obj):
            # First-stage displacement cost (matches gurobi_code obj first term
            # divided by rho).
            disp_cost = 0.0
            for fl in flights:
                fid = fl["id"]
                if fid not in sched:
                    continue
                S_dep = fl["dep_period"]
                sd = sched[fid]["scheduled_dep"]
                # w_dep[t] = 1 iff t <= sd, extended by convention:
                # w[t] = 1 for t <= S_dep - delta, w[t] = 0 for t > S_dep + delta.
                def _w_at(t, S=S_dep, sd=sd, d=delta):
                    if t <= S - d:
                        return 1
                    if t > S + d:
                        return 0
                    return 1 if t <= sd else 0
                for t in range(S_dep - delta + 1, S_dep + delta + 1):
                    g_it = abs(t - S_dep)
                    disp_cost += g_it * (_w_at(t - 1) - _w_at(t))

            # Expected second-stage delay cost (matches gurobi_code obj second
            # term divided by (1-rho)).
            delay_cost = 0.0
            sc_sols = solution.get("scenario_solutions") or []
            scenarios_data = instance.get("scenarios", [])
            prob_by_id = {sc["id"]: sc.get("probability", 0.0) for sc in scenarios_data}
            for sc_sol in sc_sols:
                s_id = sc_sol.get("scenario_id")
                p_s = prob_by_id.get(s_id, 0.0)
                if p_s == 0.0:
                    continue
                v_dep_sol_obj = sc_sol.get("v_dep", {}) or {}
                v_arr_sol_obj = sc_sol.get("v_arr", {}) or {}
                inner = 0.0
                for fl in flights:
                    fid = fl["id"]
                    c_dep = float(fl.get("cost_dep_delay", 0.0))
                    c_arr = float(fl.get("cost_arr_delay", 0.0))
                    try:
                        v_d = float(v_dep_sol_obj.get(str(fid), 0.0))
                    except (TypeError, ValueError):
                        v_d = 0.0
                    try:
                        v_a = float(v_arr_sol_obj.get(str(fid), 0.0))
                    except (TypeError, ValueError):
                        v_a = 0.0
                    inner += c_dep * v_d + c_arr * v_a
                delay_cost += p_s * inner

            obj_lo = min(disp_cost, delay_cost)
            obj_hi = max(disp_cost, delay_cost)
            # 0.1% relative tolerance with 1e-3 absolute floor, applied to the
            # envelope endpoints AND the reported value (whichever is largest).
            scale = max(abs(obj_lo), abs(obj_hi), abs(reported_obj))
            tol = max(1e-3, 1e-3 * scale)

            if reported_obj < obj_lo - tol:
                excess = obj_lo - reported_obj
                record_violation(violations, magnitudes, 20,
                                 f"Objective consistency violated: reported objective_value="
                                 f"{reported_obj} is below envelope lower bound "
                                 f"min(disp,delay)={obj_lo} (disp_cost={disp_cost}, "
                                 f"delay_cost={delay_cost}, tol={tol:.3g}). For any rho in [0,1] "
                                 f"the true value rho*disp+(1-rho)*delay cannot fall below this.",
                                 reported_obj, obj_lo, excess)
            elif reported_obj > obj_hi + tol:
                excess = reported_obj - obj_hi
                record_violation(violations, magnitudes, 20,
                                 f"Objective consistency violated: reported objective_value="
                                 f"{reported_obj} exceeds envelope upper bound "
                                 f"max(disp,delay)={obj_hi} (disp_cost={disp_cost}, "
                                 f"delay_cost={delay_cost}, tol={tol:.3g}). For any rho in [0,1] "
                                 f"the true value rho*disp+(1-rho)*delay cannot exceed this.",
                                 reported_obj, obj_hi, excess)

    return violations, magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for IMSOAN (Wang & Jacquillat 2020)")
    parser.add_argument("--instance_path", required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to candidate solution JSON file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write feasibility result JSON file")
    args = parser.parse_args()

    instance = load_instance(args.instance_path)
    solution = load_solution(args.solution_path)

    violations, magnitudes = check_feasibility(instance, solution)

    violated_indices = sorted(set(idx for idx, _ in violations))
    violation_messages = []
    seen = set()
    for idx, msg in violations:
        key = (idx, msg)
        if key not in seen:
            seen.add(key)
            violation_messages.append(msg)

    result = {
        "feasible": len(violated_indices) == 0,
        "violated_constraints": violated_indices,
        "violations": violation_messages,
        "violation_magnitudes": magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraint violations detected.")
    else:
        print(f"INFEASIBLE - {len(violated_indices)} constraint(s) violated: {violated_indices}")
        for msg in violation_messages[:10]:
            print(f"  {msg}")
        if len(violation_messages) > 10:
            print(f"  ... and {len(violation_messages) - 10} more violations")


if __name__ == "__main__":
    main()
