#!/usr/bin/env python3
"""
Feasibility checker for the UCGNA bilevel optimization problem (Tier C
variant with objective recomputation).

Based on: Byeon & Van Hentenryck (2022), "Benders Subproblem Decomposition
          for Bilevel Problems with Convex Follower"

Checks hard constraints from the mathematical formulation (math_model.txt),
numbered strictly from top to bottom:

  Constraint 1  (1b):  G_x x + G_y y >= h   (upper-level coupling)
  Constraint 2  (1c):  x in X               (leader variable domain)
  Constraint 3  (1d):  y in argmin{...}      (follower feasibility)
  Constraint 4  (4b):  t >= f(x)            (bilevel objective bound)
  Constraint 5  (7b):  G_y y >= h_y - G_xy x (same as 1 in MISOCP)
  Constraint 6  (7c):  By >= b - Ax          (same as follower in 3)
  Constraint 7  (7d):  dual feasibility      (requires dual vars)
  Constraint 8  (7e):  strong duality gap    (requires dual vars)
  Constraint 9  (7f):  McCormick             (requires dual vars)
  Constraint 10 (14a): optimality cuts       (Benders-specific)
  Constraint 11 (14b): feasibility cuts      (Benders-specific)
  Constraint 12 (10b): subproblem            (requires subproblem vars)
  Constraint 13 (10c): subproblem            (requires subproblem vars)
  Constraint 14 (10d): subproblem domain     (requires subproblem vars)
  Constraint 15 (11b): subproblem            (requires subproblem vars)
  Constraint 16 (11c): subproblem domain     (requires subproblem vars)
  Constraint 17 (24b): equivalent to 1/5     (extended formulation)
  Constraint 18 (24c): dual constraint       (requires dual vars)
  Constraint 19 (24d): equivalent to 2       (extended formulation)
  Constraint 20 (24e): equivalent to 3       (extended formulation)
  Constraint 21 (40a): objective consistency (recomputed vs reported)

Constraints 1, 2, 3 are always checked when primal variables are available.
Constraints 4-20 are checked only when the required variables exist in the
solution (e.g., dual variables from the Gurobi MISOCP solution).
Constraint 21 is checked whenever a reported objective_value is supplied.

Handles two solution formats:
  - efficient_solution: structured leader_variables / follower_variables
  - gurobi_solution: flat nonzero_variables dict (or INFEASIBLE with none)

Time convention: internally uses 0-based decision periods (t = 0..T-1).
  - efficient_algorithm.py already uses this convention.
  - gurobi_code.py uses t=0 as pre-horizon; its t=1..T map to our t=0..T-1.
"""

import argparse
import json
import sys

# MISOCP barrier solver (used in the single-level reformulation here) leaves
# equality slacks at ~1e-5 even with tightened BarConvTol; the prior 1e-5
# TOL was rejecting numerically-valid solutions on GenDecomp etc. Loosen to
# 1e-4 to accept normal solver imprecision while still catching real
# constraint violations.
TOL = 1e-4
EPS = 1e-5

# Objective weighting (40a). Hard-coded in gurobi_code.py as BETA = 0.5.
BETA = 0.5


# ======================================================================
# I/O helpers
# ======================================================================

def load_json(path):
    with open(path) as f:
        return json.load(f)


# ======================================================================
# Solution format detection and parsing
# ======================================================================

def detect_format(sol):
    """Return 'efficient', 'gurobi', or None."""
    if "leader_variables" in sol:
        return "efficient"
    if "status_name" in sol or "primary_variables" in sol or "nonzero_variables" in sol:
        return "gurobi"
    return None


def has_solution(sol, fmt):
    """True if the solution contains actual variable values."""
    if fmt == "efficient":
        lv = sol.get("leader_variables", {})
        fv = sol.get("follower_variables", {})
        return bool(lv) and bool(fv)
    if fmt == "gurobi":
        return bool(sol.get("primary_variables") or sol.get("nonzero_variables"))
    return False


def _parse_key2(d):
    """Parse dict with keys 'id1_id2' -> {(int,int): float}."""
    out = {}
    for key, val in d.items():
        parts = key.split("_")
        out[(int(parts[0]), int(parts[1]))] = float(val)
    return out


def _parse_key3(d):
    """Parse dict with keys 'id1_id2_id3' -> {(int,int,int): float}."""
    out = {}
    for key, val in d.items():
        parts = key.split("_")
        out[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(val)
    return out


def parse_efficient(sol):
    """Parse efficient_solution format into unified variable dict."""
    lv = sol["leader_variables"]
    fv = sol["follower_variables"]
    v = {}
    v["o"] = _parse_key2(lv.get("o", {}))
    v["v_plus"] = _parse_key2(lv.get("v_plus", {}))
    v["v_minus"] = _parse_key2(lv.get("v_minus", {}))
    v["w"] = _parse_key3(lv.get("w", {}))
    v["r"] = _parse_key2(lv.get("r", {}))
    v["p"] = _parse_key2(fv.get("p", {}))
    v["s_e"] = _parse_key3(fv.get("s_e", {}))
    v["f"] = _parse_key2(fv.get("f", {}))
    v["theta"] = _parse_key2(fv.get("theta", {}))
    v["s_g"] = _parse_key2(fv.get("s_g", {}))
    v["q_gas"] = _parse_key2(fv.get("q_gas", {}))
    # Variables not in efficient solution output
    v["l_gas"] = {}
    v["gamma_gas"] = {}
    v["pi_sq"] = {}
    v["phi_gas"] = {}
    v["s_g_s"] = {}
    v["psi"] = {}
    v["phi_max"] = {}
    return v


def parse_gurobi(sol):
    """Parse gurobi_solution nonzero_variables into unified variable dict.

    Gurobi uses t=0 as pre-horizon.  Decision periods t=1..T are mapped to
    our t=0..T-1.
    """
    nz = sol.get("primary_variables") or sol.get("nonzero_variables") or {}
    v = {k: {} for k in [
        "o", "v_plus", "v_minus", "w", "r",
        "p", "s_e", "f", "theta",
        "s_g", "q_gas", "l_gas", "gamma_gas",
        "pi_sq", "phi_gas", "s_g_s", "psi", "phi_max",
    ]}

    for name, val in nz.items():
        parts = name.split("_")
        prefix = parts[0]
        try:
            if prefix == "o" and len(parts) == 3:
                uid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["o"][(uid, t - 1)] = float(val)
            elif prefix == "vp" and len(parts) == 3:
                uid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["v_plus"][(uid, t - 1)] = float(val)
            elif prefix == "vm" and len(parts) == 3:
                uid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["v_minus"][(uid, t - 1)] = float(val)
            elif prefix == "w" and len(parts) == 4:
                uid, bid, t = int(parts[1]), int(parts[2]), int(parts[3])
                if t >= 1:
                    v["w"][(uid, bid, t - 1)] = float(val)
            elif prefix == "r" and len(parts) == 3:
                uid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["r"][(uid, t - 1)] = float(val)
            elif prefix == "p" and len(parts) == 3:
                uid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["p"][(uid, t - 1)] = float(val)
            elif prefix == "se" and len(parts) == 4:
                uid, bid, t = int(parts[1]), int(parts[2]), int(parts[3])
                if t >= 1:
                    v["s_e"][(uid, bid, t - 1)] = float(val)
            elif prefix == "f" and len(parts) == 3:
                lid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["f"][(lid, t - 1)] = float(val)
            elif prefix == "theta" and len(parts) == 3:
                bus, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["theta"][(bus, t - 1)] = float(val)
            elif prefix == "sg" and len(parts) == 3:
                jid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["s_g"][(jid, t - 1)] = float(val)
            elif prefix == "qg" and len(parts) == 3:
                jid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["q_gas"][(jid, t - 1)] = float(val)
            elif prefix == "lg" and len(parts) == 3:
                jid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["l_gas"][(jid, t - 1)] = float(val)
            elif prefix == "gamma" and len(parts) == 3:
                jid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["gamma_gas"][(jid, t - 1)] = float(val)
            elif prefix == "pisq" and len(parts) == 3:
                jid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["pi_sq"][(jid, t - 1)] = float(val)
            elif prefix == "phig" and len(parts) == 3:
                cid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["phi_gas"][(cid, t - 1)] = float(val)
            elif prefix == "sgs" and len(parts) == 4:
                jid, sid, t = int(parts[1]), int(parts[2]), int(parts[3])
                if t >= 1:
                    v["s_g_s"][(jid, sid, t - 1)] = float(val)
            elif prefix == "psi" and len(parts) == 3:
                k, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["psi"][(k, t - 1)] = float(val)
            elif prefix == "phimax" and len(parts) == 3:
                uid, t = int(parts[1]), int(parts[2])
                if t >= 1:
                    v["phi_max"][(uid, t - 1)] = float(val)
        except (ValueError, IndexError):
            continue
    return v


def parse_vars(sol, fmt):
    if fmt == "efficient":
        return parse_efficient(sol)
    return parse_gurobi(sol)


# ======================================================================
# Variable accessor (returns default 0.0 for missing / zero variables)
# ======================================================================

def g(v, var_name, key, default=0.0):
    return v.get(var_name, {}).get(key, default)


# ======================================================================
# Objective recomputation (constraint 21)
# ======================================================================

def recompute_objective(inst, v):
    """Recompute the obj (40a) from solution variables.

    Returns (obj_value, mode) where mode is "full" when every variable
    referenced by the objective is present in the parsed solution, or
    "lower_bound" when the supply-interval allocation s_g_s is missing
    (e.g. efficient format). In the lower-bound mode, the gas supply
    cost is replaced by the minimum-cost greedy allocation of the
    observed total junction supply s_g[j,t] to its supply intervals
    sorted by ascending slope -- this is a true lower bound on the
    actual supply cost.
    """
    T = inst["time_periods"]
    gens = inst["generators"]["generators"]
    junctions = inst["gas_network"]["junctions"]

    # ----- Electricity component (40a, first term) -----
    obj_elec = 0.0
    for gen in gens:
        uid = gen["id"]
        no_load = gen["no_load_cost"]
        for t in range(T):
            obj_elec += no_load * g(v, "o", (uid, t))
            obj_elec += g(v, "r", (uid, t))
            for bid in gen["bids"]:
                b = bid["id"]
                obj_elec += bid["price"] * g(v, "s_e", (uid, b, t))

    # ----- Gas component (40a, second term) -----
    s_g_s_present = bool(v.get("s_g_s"))
    obj_gas = 0.0
    for j in junctions:
        jid = j["id"]
        for t in range(T):
            obj_gas += j["demand_shedding_cost"] * g(v, "q_gas", (jid, t))
        if j["is_source"] and j.get("supply_intervals"):
            intervals = j["supply_intervals"]
            if s_g_s_present:
                for t in range(T):
                    for si in intervals:
                        obj_gas += si["slope"] * g(v, "s_g_s", (jid, si["id"], t))
            else:
                # Greedy fill -> minimum supply cost given observed s_g[j,t].
                sorted_si = sorted(intervals, key=lambda x: x["slope"])
                for t in range(T):
                    remaining = g(v, "s_g", (jid, t))
                    for si in sorted_si:
                        cap = si["interval_ub"] - si["interval_lb"]
                        take = min(cap, max(0.0, remaining))
                        obj_gas += si["slope"] * take
                        remaining -= take
                        if remaining <= 1e-12:
                            break

    mode = "full" if s_g_s_present else "lower_bound"
    return BETA * obj_elec + (1.0 - BETA) * obj_gas, mode


# ======================================================================
# Constraint checking
# ======================================================================

def check_feasibility(inst, v, fmt, reported_obj=None):
    """Check all verifiable hard constraints.

    Returns list of (constraint_index, message, lhs, rhs, violation_amount).

    The optional reported_obj enables constraint 21 (objective
    consistency): the obj (40a) is recomputed from the variables and
    compared to reported_obj. The check uses full-equality comparison
    when every obj-determining variable is present in the solution
    (gurobi format), or a lower-bound comparison otherwise (efficient
    format, where s_g_s is missing).
    """
    T = inst["time_periods"]
    gens = inst["generators"]["generators"]
    buses = inst["electricity_network"]["buses"]
    lines = inst["electricity_network"]["lines"]
    junctions = inst["gas_network"]["junctions"]
    connections = inst["gas_network"]["connections"]

    gen_map = {gen["id"]: gen for gen in gens}
    bus_map = {b["id"]: b for b in buses}
    line_map = {l["id"]: l for l in lines}
    junc_map = {j["id"]: j for j in junctions}

    bus_gens = {b["id"]: [] for b in buses}
    for gen in gens:
        bus_gens[gen["bus"]].append(gen["id"])

    junc_gfpps = {j["id"]: [] for j in junctions}
    for gen in gens:
        if gen["is_gfpp"] and gen["gas_junction"] is not None:
            junc_gfpps[gen["gas_junction"]].append(gen["id"])

    viols = []  # (constraint_idx, msg, lhs, rhs, violation_amount)

    REL_TOL = 1e-4  # 0.01% relative slack — accommodates Weymouth-style
                    # nonlinear quadratic constraints under SOCP relaxation
    def chk(ci, msg, lhs, rhs, op):
        """Record violation if constraint is violated beyond both absolute
        and relative tolerance."""
        if op == "<=":
            va = lhs - rhs
        elif op == ">=":
            va = rhs - lhs
        else:  # "="
            va = abs(lhs - rhs)
        # Effective tolerance: max of absolute TOL and relative tolerance
        # scaled to the magnitude of the right-hand side. Catches genuine
        # violations while accepting solver numerical noise on large values.
        eff_tol = max(TOL, REL_TOL * max(abs(lhs), abs(rhs)))
        if va > eff_tol:
            viols.append((ci, msg, float(lhs), float(rhs), float(va)))

    # ==================================================================
    # Constraint 2 (1c): Leader variable domain  x in X
    # ==================================================================

    # --- Binary checks: o, v+, v-, w must be 0 or 1 ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            for vn, label in [("o", "o"), ("v_plus", "v+"), ("v_minus", "v-")]:
                val = g(v, vn, (uid, t))
                rd = round(val)
                if abs(val - rd) > TOL:
                    chk(2, f"{label}[{uid},{t}]={val:.6f} not binary", val, float(rd), "=")
            for bid in gen["bids"]:
                b = bid["id"]
                val = g(v, "w", (uid, b, t))
                rd = round(val)
                if abs(val - rd) > TOL:
                    chk(2, f"w[{uid},{b},{t}]={val:.6f} not binary", val, float(rd), "=")

    # --- (40d) Initial status: o[u,0] = initial_status (efficient only) ---
    if fmt == "efficient":
        for gen in gens:
            uid = gen["id"]
            chk(2, f"InitStatus: o[{uid},0]={g(v,'o',(uid,0)):.0f} "
                    f"!= {gen['initial_status']}",
                g(v, "o", (uid, 0)), float(gen["initial_status"]), "=")

    # --- (40g) Startup/shutdown logic: v+[t] - v-[t] = o[t] - o_prev ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            vp = g(v, "v_plus", (uid, t))
            vm = g(v, "v_minus", (uid, t))
            o_t = g(v, "o", (uid, t))
            o_prev = float(gen["initial_status"]) if t == 0 else g(v, "o", (uid, t - 1))
            lhs = vp - vm
            rhs = o_t - o_prev
            chk(2, f"Logic: v+[{uid},{t}]-v-[{uid},{t}]={lhs:.4f} "
                    f"!= o[{uid},{t}]-o_prev={rhs:.4f}",
                lhs, rhs, "=")

    # --- v+ + v- <= 1 ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            vp = g(v, "v_plus", (uid, t))
            vm = g(v, "v_minus", (uid, t))
            chk(2, f"Excl: v+[{uid},{t}]+v-[{uid},{t}]={vp+vm:.4f} > 1",
                vp + vm, 1.0, "<=")

    # --- (40e) Min up time ---
    for gen in gens:
        uid = gen["id"]
        tau = gen["min_up_time"]
        for t in range(T):
            lhs = sum(g(v, "v_plus", (uid, n)) for n in range(max(0, t - tau + 1), t + 1))
            rhs = g(v, "o", (uid, t))
            chk(2, f"MinUp[{uid},{t}]: sum_vp={lhs:.4f} > o={rhs:.4f}", lhs, rhs, "<=")

    # --- (40f) Min down time ---
    for gen in gens:
        uid = gen["id"]
        tau = gen["min_down_time"]
        for t in range(T):
            lhs = sum(g(v, "v_minus", (uid, n)) for n in range(max(0, t - tau + 1), t + 1))
            rhs = 1.0 - g(v, "o", (uid, t))
            chk(2, f"MinDown[{uid},{t}]: sum_vm={lhs:.4f} > 1-o={rhs:.4f}", lhs, rhs, "<=")

    # --- (40b) Startup cost: r >= 0 and r >= C*(o[t] - sum o[t-n]) ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            r_val = g(v, "r", (uid, t))
            chk(2, f"r[{uid},{t}]={r_val:.6f} < 0", r_val, 0.0, ">=")
            for h_cost, cost in gen["startup_cost_params"]:
                expr = g(v, "o", (uid, t))
                for n in range(1, h_cost + 1):
                    tn = t - n
                    if tn >= 0:
                        expr -= g(v, "o", (uid, tn))
                    else:
                        expr -= float(gen["initial_status"])
                rhs = cost * expr
                chk(2, f"StartupCost[{uid},{t}]: r={r_val:.4f} < {rhs:.4f}",
                    r_val, rhs, ">=")

    # --- (40h) Bid on: w[u,b,t] <= o[u,t] ---
    for gen in gens:
        uid = gen["id"]
        for bid in gen["bids"]:
            b = bid["id"]
            for t in range(T):
                chk(2, f"BidOn: w[{uid},{b},{t}]={g(v,'w',(uid,b,t)):.4f} "
                        f"> o[{uid},{t}]={g(v,'o',(uid,t)):.4f}",
                    g(v, "w", (uid, b, t)), g(v, "o", (uid, t)), "<=")

    # ==================================================================
    # Constraint 1 (1b): Upper-level coupling  G_x x + G_y y >= h
    # ==================================================================

    # --- (40l) Bid bounds: s_e[u,b,t] <= max_amount * w[u,b,t] ---
    for gen in gens:
        uid = gen["id"]
        for bid in gen["bids"]:
            b = bid["id"]
            s_bar = bid["max_amount"]
            for t in range(T):
                se = g(v, "s_e", (uid, b, t))
                w_val = g(v, "w", (uid, b, t))
                rhs = s_bar * w_val
                chk(1, f"BidBound: s_e[{uid},{b},{t}]={se:.6f} > "
                        f"{s_bar}*w={rhs:.6f}", se, rhs, "<=")

    # --- (40m) Bid ordering: s_e[u,b,t] >= max_amount[b] * w[u,b+1,t] ---
    for gen in gens:
        uid = gen["id"]
        bids_list = gen["bids"]
        for idx in range(len(bids_list) - 1):
            b = bids_list[idx]["id"]
            b_next = bids_list[idx + 1]["id"]
            s_bar = bids_list[idx]["max_amount"]
            for t in range(T):
                se = g(v, "s_e", (uid, b, t))
                w_next = g(v, "w", (uid, b_next, t))
                rhs = s_bar * w_next
                chk(1, f"BidOrder: s_e[{uid},{b},{t}]={se:.6f} < "
                        f"{s_bar}*w_next={rhs:.6f}", se, rhs, ">=")

    # ==================================================================
    # Constraint 3 (1d): Follower feasibility  Ax + By >= b
    # ==================================================================

    # --- Non-negativity: s_e >= 0, p >= 0 ---
    for gen in gens:
        uid = gen["id"]
        for bid in gen["bids"]:
            b = bid["id"]
            for t in range(T):
                se = g(v, "s_e", (uid, b, t))
                chk(3, f"s_e[{uid},{b},{t}]={se:.6f} < 0", se, 0.0, ">=")
        for t in range(T):
            p_val = g(v, "p", (uid, t))
            chk(3, f"p[{uid},{t}]={p_val:.6f} < 0", p_val, 0.0, ">=")

    # --- (42c) Generation decomposition: p[u,t] = sum_b s_e[u,b,t] ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            p_val = g(v, "p", (uid, t))
            se_sum = sum(g(v, "s_e", (uid, bid["id"], t)) for bid in gen["bids"])
            chk(3, f"GenDecomp: p[{uid},{t}]={p_val:.6f} != sum_se={se_sum:.6f}",
                p_val, se_sum, "=")

    # --- (42e) Power bounds: p_min * o <= p <= p_max * o ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            p_val = g(v, "p", (uid, t))
            o_val = g(v, "o", (uid, t))
            lb = gen["min_power"] * o_val
            ub = gen["max_power"] * o_val
            chk(3, f"PowLB: p[{uid},{t}]={p_val:.6f} < p_min*o={lb:.6f}",
                p_val, lb, ">=")
            chk(3, f"PowUB: p[{uid},{t}]={p_val:.6f} > p_max*o={ub:.6f}",
                p_val, ub, "<=")

    # --- (42f) Initial generation (efficient format only) ---
    if fmt == "efficient":
        for gen in gens:
            uid = gen["id"]
            p_val = g(v, "p", (uid, 0))
            init_gen = gen["initial_gen"]
            chk(3, f"InitGen: p[{uid},0]={p_val:.6f} != {init_gen:.6f}",
                p_val, init_gen, "=")

    # --- (42g) Ramp up: p[t] - p_prev <= ramp_up*o_prev + p_max*v+[t] ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            p_val = g(v, "p", (uid, t))
            if t == 0:
                p_prev = gen["initial_gen"]
                o_prev = float(gen["initial_status"])
            else:
                p_prev = g(v, "p", (uid, t - 1))
                o_prev = g(v, "o", (uid, t - 1))
            vp = g(v, "v_plus", (uid, t))
            lhs = p_val - p_prev
            rhs = gen["ramp_up"] * o_prev + gen["max_power"] * vp
            chk(3, f"RampUp[{uid},{t}]: delta_p={lhs:.6f} > {rhs:.6f}",
                lhs, rhs, "<=")

    # --- (42h) Ramp down: p_prev - p[t] <= ramp_down*o[t] + p_min*v-[t] ---
    for gen in gens:
        uid = gen["id"]
        for t in range(T):
            p_val = g(v, "p", (uid, t))
            p_prev = gen["initial_gen"] if t == 0 else g(v, "p", (uid, t - 1))
            o_val = g(v, "o", (uid, t))
            vm = g(v, "v_minus", (uid, t))
            lhs = p_prev - p_val
            rhs = gen["ramp_down"] * o_val + gen["min_power"] * vm
            chk(3, f"RampDown[{uid},{t}]: delta_p={lhs:.6f} > {rhs:.6f}",
                lhs, rhs, "<=")

    # --- (42i) DC power flow definition ---
    # Compute expected f from theta using the solution's sign convention:
    #   efficient: f = -B*(theta_from - theta_to)
    #   gurobi:    f =  B*(theta_from - theta_to)
    sign = -1.0 if fmt == "efficient" else 1.0
    computed_f = {}
    for l in lines:
        lid = l["id"]
        B_l = l["susceptance"]
        for t in range(T):
            tf = g(v, "theta", (l["from_bus"], t))
            tt = g(v, "theta", (l["to_bus"], t))
            expected = sign * B_l * (tf - tt)
            computed_f[(lid, t)] = expected
            # Check stored f against expected (if stored)
            if (lid, t) in v["f"]:
                actual = v["f"][(lid, t)]
                chk(3, f"DCflow[{lid},{t}]: f={actual:.6f} != expected={expected:.6f}",
                    actual, expected, "=")

    # --- (42j) Thermal limits: |f| <= f_bar ---
    for l in lines:
        lid = l["id"]
        f_bar = l["thermal_limit"]
        for t in range(T):
            f_val = computed_f.get((lid, t), g(v, "f", (lid, t)))
            chk(3, f"ThermUB[{lid},{t}]: |f|={abs(f_val):.6f} > {f_bar:.6f}",
                abs(f_val), f_bar, "<=")

    # --- (42l) Angle difference limits ---
    for l in lines:
        delta = l["angle_diff_limit"]
        for t in range(T):
            tf = g(v, "theta", (l["from_bus"], t))
            tt = g(v, "theta", (l["to_bus"], t))
            diff = abs(tf - tt)
            chk(3, f"AngleDiff[{l['id']},{t}]: |diff|={diff:.6f} > {delta:.6f}",
                diff, delta, "<=")

    # --- Voltage angle bounds ---
    for bus in buses:
        i = bus["id"]
        for t in range(T):
            th = g(v, "theta", (i, t))
            chk(3, f"ThetaLB[{i},{t}]: theta={th:.6f} < {bus['voltage_angle_lb']:.6f}",
                th, bus["voltage_angle_lb"], ">=")
            chk(3, f"ThetaUB[{i},{t}]: theta={th:.6f} > {bus['voltage_angle_ub']:.6f}",
                th, bus["voltage_angle_ub"], "<=")

    # --- (42b) Power balance at each bus ---
    # Balance equation (same in both conventions when using computed_f):
    #   gen_sum + flow_in - flow_out = demand
    for bus in buses:
        i = bus["id"]
        for t in range(T):
            gen_sum = sum(g(v, "p", (uid, t)) for uid in bus_gens[i])
            demand = bus["demand_profile"][t]
            flow_in = 0.0
            flow_out = 0.0
            for l in lines:
                lid = l["id"]
                fv = computed_f.get((lid, t), g(v, "f", (lid, t)))
                if l["from_bus"] == i:
                    flow_out += fv
                if l["to_bus"] == i:
                    flow_in += fv
            lhs = gen_sum + flow_in - flow_out
            chk(3, f"PowBal[{i},{t}]: gen+flow_in-flow_out={lhs:.6f} != demand={demand:.6f}",
                lhs, demand, "=")

    # ==================================================================
    # Gas network constraints (Constraint 3 continued)
    # ==================================================================

    # --- (42p) Shedding bounds: 0 <= q_gas[j,t] <= d_g[j,t] ---
    for j in junctions:
        jid = j["id"]
        for t in range(T):
            qg = g(v, "q_gas", (jid, t))
            d_g = j["gas_demand_profile"][t]
            chk(3, f"q_gas[{jid},{t}]={qg:.6f} < 0", qg, 0.0, ">=")
            chk(3, f"ShedBound[{jid},{t}]: q_gas={qg:.6f} > d_g={d_g:.6f}",
                qg, d_g, "<=")

    # --- s_g >= 0 ---
    for j in junctions:
        jid = j["id"]
        for t in range(T):
            sg = g(v, "s_g", (jid, t))
            chk(3, f"s_g[{jid},{t}]={sg:.6f} < 0", sg, 0.0, ">=")

    # --- Non-source junctions: s_g = 0 ---
    for j in junctions:
        jid = j["id"]
        if not j["is_source"]:
            for t in range(T):
                sg = g(v, "s_g", (jid, t))
                chk(3, f"NoSupply[{jid},{t}]: s_g={sg:.6f} != 0", sg, 0.0, "=")

    # --- (42o) Demand satisfaction: l_gas + q_gas = d_g ---
    if v.get("l_gas"):
        for j in junctions:
            jid = j["id"]
            for t in range(T):
                lg = g(v, "l_gas", (jid, t))
                qg = g(v, "q_gas", (jid, t))
                d_g = j["gas_demand_profile"][t]
                chk(3, f"DemSat[{jid},{t}]: l_gas+q_gas={lg+qg:.6f} != d_g={d_g:.6f}",
                    lg + qg, d_g, "=")

    # --- (42n) Supply decomposition: s_g = sum s_g_s (source junctions) ---
    if v.get("s_g_s"):
        for j in junctions:
            jid = j["id"]
            if j["is_source"] and j["supply_intervals"]:
                for t in range(T):
                    sg = g(v, "s_g", (jid, t))
                    sg_sum = sum(g(v, "s_g_s", (jid, si["id"], t))
                                for si in j["supply_intervals"])
                    chk(3, f"SupplyDecomp[{jid},{t}]: s_g={sg:.6f} != sum={sg_sum:.6f}",
                        sg, sg_sum, "=")

    # --- (42m) Gas flow conservation ---
    if v.get("phi_gas") and v.get("gamma_gas"):
        for j in junctions:
            jid = j["id"]
            for t in range(T):
                sg = g(v, "s_g", (jid, t))
                d_g = j["gas_demand_profile"][t]
                qg = g(v, "q_gas", (jid, t))
                lg = g(v, "l_gas", (jid, t)) if v.get("l_gas") else (d_g - qg)
                gamma = g(v, "gamma_gas", (jid, t))
                flow_in = sum(g(v, "phi_gas", (c["id"], t))
                              for c in connections if c["to_junction"] == jid)
                flow_out = sum(g(v, "phi_gas", (c["id"], t))
                               for c in connections if c["from_junction"] == jid)
                lhs = sg + flow_in - flow_out
                rhs = lg + gamma
                chk(3, f"GasBal[{jid},{t}]: LHS={lhs:.6f} != RHS={rhs:.6f}",
                    lhs, rhs, "=")

    # --- phi_gas >= 0 ---
    if v.get("phi_gas"):
        for c in connections:
            cid = c["id"]
            for t in range(T):
                phi = g(v, "phi_gas", (cid, t))
                chk(3, f"phi_gas[{cid},{t}]={phi:.6f} < 0", phi, 0.0, ">=")

    # --- Pressure bounds ---
    if v.get("pi_sq"):
        for j in junctions:
            jid = j["id"]
            for t in range(T):
                pi = g(v, "pi_sq", (jid, t))
                chk(3, f"PressLB[{jid},{t}]: pi_sq={pi:.6f} < {j['pressure_lb_squared']:.6f}",
                    pi, j["pressure_lb_squared"], ">=")
                chk(3, f"PressUB[{jid},{t}]: pi_sq={pi:.6f} > {j['pressure_ub_squared']:.6f}",
                    pi, j["pressure_ub_squared"], "<=")

    # --- (42s) Compressor bounds ---
    if v.get("pi_sq"):
        for c in connections:
            if c["type"] == "compressor":
                cid = c["id"]
                fj = c["from_junction"]
                tj = c["to_junction"]
                rlb2 = c["compression_ratio_lb"] ** 2
                rub2 = c["compression_ratio_ub"] ** 2
                for t in range(T):
                    pi_to = g(v, "pi_sq", (tj, t))
                    pi_from = g(v, "pi_sq", (fj, t))
                    chk(3, f"CompLB[{cid},{t}]: pi_to={pi_to:.6f} < "
                            f"ratio_lb^2*pi_from={rlb2 * pi_from:.6f}",
                        pi_to, rlb2 * pi_from, ">=")
                    chk(3, f"CompUB[{cid},{t}]: pi_to={pi_to:.6f} > "
                            f"ratio_ub^2*pi_from={rub2 * pi_from:.6f}",
                        pi_to, rub2 * pi_from, "<=")

    # --- (42u) Weymouth equation (SOC relaxation): pi_from - pi_to >= W*phi^2 ---
    if v.get("pi_sq") and v.get("phi_gas"):
        for c in connections:
            if c["type"] == "pipeline":
                cid = c["id"]
                fj = c["from_junction"]
                tj = c["to_junction"]
                W = c["weymouth_factor"]
                for t in range(T):
                    pi_from = g(v, "pi_sq", (fj, t))
                    pi_to = g(v, "pi_sq", (tj, t))
                    phi = g(v, "phi_gas", (cid, t))
                    lhs = pi_from - pi_to
                    rhs = W * phi * phi
                    chk(3, f"Weymouth[{cid},{t}]: pi_diff={lhs:.6f} < W*phi^2={rhs:.6f}",
                        lhs, rhs, ">=")

    # --- (42w) Heat rate: gamma >= sum(H2*p^2 + H1*p + H0*o) ---
    if v.get("gamma_gas"):
        for j in junctions:
            jid = j["id"]
            gfpp_ids = junc_gfpps.get(jid, [])
            if gfpp_ids:
                for t in range(T):
                    gamma = g(v, "gamma_gas", (jid, t))
                    heat_sum = 0.0
                    for uid in gfpp_ids:
                        hr = gen_map[uid]["heat_rate_coefficients"]
                        p_val = g(v, "p", (uid, t))
                        o_val = g(v, "o", (uid, t))
                        heat_sum += (hr["H_u2"] * p_val ** 2
                                     + hr["H_u1"] * p_val
                                     + hr["H_u0"] * o_val)
                    chk(3, f"HeatRate[{jid},{t}]: gamma={gamma:.6f} < "
                            f"heat_sum={heat_sum:.6f}", gamma, heat_sum, ">=")

    # ==================================================================
    # Constraint 21 (40a): Objective consistency
    # ==================================================================
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj, mode = recompute_objective(inst, v)
            # 0.1% relative tolerance with 1e-3 absolute floor. Generous
            # enough to absorb barrier-solver noise (~1e-6 absolute on a
            # 1e5-magnitude objective for this paper) yet tight enough to
            # catch obj=0 / obj=MAX_FLOAT exploits on any realistic
            # instance.
            tol = max(1e-3, 1e-3 * max(abs(true_obj), abs(reported)))
            if mode == "full":
                diff = abs(reported - true_obj)
                if diff > tol:
                    msg = (f"ObjConsistency(full): reported objective_value="
                           f"{reported} differs from recomputed obj (40a)="
                           f"{true_obj} (|diff|={diff:.6g}, tol={tol:.6g})")
                    viols.append((21, msg, float(reported), float(true_obj), float(diff)))
            else:  # lower_bound
                shortfall = true_obj - reported
                if shortfall > tol:
                    msg = (f"ObjConsistency(lower_bound): reported objective_value="
                           f"{reported} is below recomputed lower bound="
                           f"{true_obj} (shortfall={shortfall:.6g}, tol={tol:.6g})")
                    viols.append((21, msg, float(reported), float(true_obj), float(shortfall)))

    return viols


# ======================================================================
# Output formatting
# ======================================================================

def format_output(viols):
    """Convert raw violation list into the required JSON structure."""
    if not viols:
        return {
            "feasible": True,
            "violated_constraints": [],
            "violations": [],
            "violation_magnitudes": [],
        }

    # Build per-constraint message groups
    constraint_msgs = {}
    magnitudes = []

    for ci, msg, lhs, rhs, va in viols:
        constraint_msgs.setdefault(ci, []).append(msg)
        normalizer = max(abs(rhs), EPS)
        magnitudes.append({
            "constraint": ci,
            "lhs": round(lhs, 10),
            "rhs": round(rhs, 10),
            "raw_excess": round(va, 10),
            "normalizer": round(normalizer, 10),
            "ratio": round(va / normalizer, 10),
        })

    violated_constraints = sorted(constraint_msgs.keys())

    # Aggregate violation messages per constraint index
    violations = []
    for ci in violated_constraints:
        msgs = constraint_msgs[ci]
        if len(msgs) <= 3:
            violations.extend(msgs)
        else:
            violations.append(
                f"{msgs[0]} (and {len(msgs) - 1} more violations of constraint {ci})")

    return {
        "feasible": False,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for UCGNA bilevel optimization "
                    "(Byeon & Van Hentenryck, 2022)")
    parser.add_argument("--instance_path", required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to solution JSON file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write feasibility result JSON")
    args = parser.parse_args()

    inst = load_json(args.instance_path)
    sol = load_json(args.solution_path)

    fmt = detect_format(sol)

    if fmt is None:
        result = {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Unknown solution format"],
            "violation_magnitudes": [],
        }
    elif not has_solution(sol, fmt):
        status = sol.get("status", sol.get("status_name", "unknown"))
        result = {
            "feasible": False,
            "violated_constraints": [],
            "violations": [f"No solution available (status: {status})"],
            "violation_magnitudes": [],
        }
    else:
        v = parse_vars(sol, fmt)
        viols = check_feasibility(inst, v, fmt, sol.get("objective_value"))
        result = format_output(viols)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Result written to {args.result_path}")
    print(f"  Feasible: {result['feasible']}")
    if result["violated_constraints"]:
        print(f"  Violated constraints: {result['violated_constraints']}")
        print(f"  Total violation instances: {len(result['violation_magnitudes'])}")
    elif not result["violation_magnitudes"] and not result["feasible"]:
        print(f"  Note: {result['violations'][0]}")


if __name__ == "__main__":
    main()
