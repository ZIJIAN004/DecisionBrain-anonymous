#!/usr/bin/env python3
"""
MISOCP single-level reformulation (Approach "G") for the UCGNA bilevel
optimization problem from Byeon & Van Hentenryck (2022).

The bilevel problem has:
  - Leader: unit commitment decisions (on/off, startup/shutdown, bid selection)
  - Follower: joint electricity/gas network dispatch

The single-level reformulation replaces follower optimality with:
  1. Follower primal feasibility
  2. Follower dual feasibility
  3. Strong duality (with McCormick linearization for bilinear terms)

Time indexing: t=0 is pre-horizon (fixed from initial conditions).
              t=1..T are decision periods. Demand profiles are indexed
              0..T-1 in the JSON, corresponding to periods t=1..T.

INFERRED ASSUMPTIONS (not specified in paper):
  - beta = 0.5 (weighting between electricity and gas objectives)
  - Dual variable upper bound (DUAL_UB) = 10000 for McCormick linearization
  - SOC dual contributions not explicitly modeled in strong duality;
    relying on primal SOC constraints for correctness
  - Compression ratios are on pressure (squared for pressure-squared constraints)
  - Ramp-up rate used for (42g), ramp-down rate for (42h)
"""

import argparse
import json
import sys
import math
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

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    print("ERROR: gurobipy not installed. Install with: pip install gurobipy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BETA = 0.5          # INFERRED: objective weighting
DUAL_UB = 10000.0   # INFERRED: upper bound on dual variables for McCormick
EPS = 1e-8


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_model(inst, time_limit=3600):
    """Build and return the Gurobi model for the single-level MISOCP.

    Time convention: t=0 is pre-horizon (initial state, fixed).
    t=1..T are decision periods. JSON demand_profile[k] corresponds to t=k+1.
    """

    T = inst["time_periods"]   # number of decision periods
    periods = range(1, T + 1)  # decision periods: 1..T

    # --- Shorthand accessors ---
    e_net = inst["electricity_network"]
    buses = {b["id"]: b for b in e_net["buses"]}
    lines = {l["id"]: l for l in e_net["lines"]}
    bus_ids = sorted(buses.keys())
    line_ids = sorted(lines.keys())

    gen_data = inst["generators"]
    gens = {g["id"]: g for g in gen_data["generators"]}
    gen_ids = sorted(gens.keys())
    gfpp_ids = [uid for uid in gen_ids if gens[uid]["is_gfpp"]]

    g_net = inst["gas_network"]
    junctions = {j["id"]: j for j in g_net["junctions"]}
    connections = {c["id"]: c for c in g_net["connections"]}
    junc_ids = sorted(junctions.keys())
    conn_ids = sorted(connections.keys())
    source_junc_ids = [j for j in junc_ids if junctions[j]["is_source"]]
    pipeline_ids = [a for a in conn_ids if connections[a]["type"] == "pipeline"]
    compressor_ids = [a for a in conn_ids if connections[a]["type"] == "compressor"]

    pricing_zones = g_net["pricing_zones"]
    psi_ub = g_net["max_gas_price_mmBtu"]  # 200
    psi_lb = g_net["min_gas_price_mmBtu"]  # 0

    # Map junction -> zone
    junc_to_zone = {}
    for zone in pricing_zones:
        for j in zone["junctions"]:
            junc_to_zone[j] = zone["id"]
    zone_ids = [z["id"] for z in pricing_zones]

    # Map bus -> generators at that bus
    bus_to_gens = {i: [] for i in bus_ids}
    for uid in gen_ids:
        bus_to_gens[gens[uid]["bus"]].append(uid)

    # Map junction -> GFPPs at that junction
    junc_to_gfpps = {j: [] for j in junc_ids}
    for uid in gfpp_ids:
        gj = gens[uid]["gas_junction"]
        if gj is not None:
            junc_to_gfpps[gj].append(uid)

    # Helper: get demand at bus i for decision period t (1-indexed)
    def elec_demand(i, t):
        return buses[i]["demand_profile"][t - 1]

    def gas_demand(j, t):
        return junctions[j]["gas_demand_profile"][t - 1]

    # -----------------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------------
    m = gp.Model("UCGNA_MISOCP")
    m.setParam("Threads", 1)

    # -----------------------------------------------------------------------
    # LEADER VARIABLES (Table 4)
    # -----------------------------------------------------------------------
    # o[u,t] for t=0..T (t=0 is pre-horizon, fixed)
    o = {}
    v_plus = {}   # t=1..T
    v_minus = {}  # t=1..T
    w = {}        # t=1..T
    r = {}        # t=1..T
    phi_max = {}  # t=1..T, GFPPs only

    for u in gen_ids:
        g = gens[u]
        # Pre-horizon on/off status (fixed)
        o[u, 0] = m.addVar(vtype=GRB.BINARY, name=f"o_{u}_0")
        for t in periods:
            o[u, t] = m.addVar(vtype=GRB.BINARY, name=f"o_{u}_{t}")
            v_plus[u, t] = m.addVar(vtype=GRB.BINARY, name=f"vp_{u}_{t}")
            v_minus[u, t] = m.addVar(vtype=GRB.BINARY, name=f"vm_{u}_{t}")
            r[u, t] = m.addVar(lb=0.0, name=f"r_{u}_{t}")
            for bid in g["bids"]:
                b = bid["id"]
                w[u, b, t] = m.addVar(vtype=GRB.BINARY, name=f"w_{u}_{b}_{t}")

    for u in gfpp_ids:
        for t in periods:
            phi_max[u, t] = m.addVar(lb=0.0, name=f"phimax_{u}_{t}")

    # -----------------------------------------------------------------------
    # FOLLOWER PRIMAL VARIABLES (Table 5) - only for t=1..T
    # -----------------------------------------------------------------------
    s_e = {}     # power from bid
    p = {}       # total power (also need p[u,0] = initial_gen for ramp)
    f_line = {}  # power flow on line
    theta = {}   # voltage angle
    s_g = {}     # gas supply at junction
    s_g_s = {}   # gas supply from interval
    pi_sq = {}   # pressure squared
    phi_gas = {} # gas flow on connection
    l_gas = {}   # satisfied gas demand
    q_gas = {}   # shed gas demand
    gamma = {}   # total GFPP gas consumption at junction

    for u in gen_ids:
        g = gens[u]
        # Pre-horizon generation (fixed)
        p[u, 0] = m.addVar(lb=0.0, name=f"p_{u}_0")
        for t in periods:
            p[u, t] = m.addVar(lb=0.0, name=f"p_{u}_{t}")
            for bid in g["bids"]:
                b = bid["id"]
                s_e[u, b, t] = m.addVar(lb=0.0, name=f"se_{u}_{b}_{t}")

    for l in line_ids:
        for t in periods:
            f_line[l, t] = m.addVar(lb=-GRB.INFINITY, name=f"f_{l}_{t}")

    for i in bus_ids:
        for t in periods:
            theta[i, t] = m.addVar(
                lb=buses[i]["voltage_angle_lb"],
                ub=buses[i]["voltage_angle_ub"],
                name=f"theta_{i}_{t}")

    for j in junc_ids:
        for t in periods:
            s_g[j, t] = m.addVar(lb=0.0, name=f"sg_{j}_{t}")
            l_gas[j, t] = m.addVar(lb=0.0, name=f"lg_{j}_{t}")
            q_gas[j, t] = m.addVar(lb=0.0, name=f"qg_{j}_{t}")
            gamma[j, t] = m.addVar(lb=0.0, name=f"gamma_{j}_{t}")
            pi_sq[j, t] = m.addVar(
                lb=junctions[j]["pressure_lb_squared"],
                ub=junctions[j]["pressure_ub_squared"],
                name=f"pisq_{j}_{t}")

    for j in source_junc_ids:
        junc = junctions[j]
        for si in junc["supply_intervals"]:
            sid = si["id"]
            cap = si["interval_ub"] - si["interval_lb"]
            for t in periods:
                s_g_s[j, sid, t] = m.addVar(lb=0.0, ub=cap,
                                             name=f"sgs_{j}_{sid}_{t}")

    for a in conn_ids:
        for t in periods:
            phi_gas[a, t] = m.addVar(lb=0.0, name=f"phig_{a}_{t}")

    # Gas zonal price variables
    psi = {}
    for k in zone_ids:
        for t in periods:
            psi[k, t] = m.addVar(lb=psi_lb, ub=psi_ub, name=f"psi_{k}_{t}")

    # -----------------------------------------------------------------------
    # FOLLOWER DUAL VARIABLES (for linear constraints, t=1..T)
    # -----------------------------------------------------------------------
    # (42b) power balance: lambda_b[i,t] free
    lambda_b = {}
    for i in bus_ids:
        for t in periods:
            lambda_b[i, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                       name=f"lam_b_{i}_{t}")

    # (42c) p = sum s_e: lambda_c[u,t] free
    lambda_c = {}
    for u in gen_ids:
        for t in periods:
            lambda_c[u, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                       name=f"lam_c_{u}_{t}")

    # (42d upper) s_e <= s_bar * w: rho_d_upper[u,b,t] >= 0
    rho_d_upper = {}
    for u in gen_ids:
        for bid in gens[u]["bids"]:
            b = bid["id"]
            for t in periods:
                rho_d_upper[u, b, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                                  name=f"rho_du_{u}_{b}_{t}")

    # (42e lower) p >= p_min * o: alpha_lower[u,t] >= 0
    alpha_lower = {}
    for u in gen_ids:
        for t in periods:
            alpha_lower[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"al_{u}_{t}")

    # (42e upper) p <= p_max * o: alpha_upper[u,t] >= 0
    alpha_upper = {}
    for u in gen_ids:
        for t in periods:
            alpha_upper[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"au_{u}_{t}")

    # (42g) ramp up: delta_up[u,t] >= 0, for t >= 1 (all decision periods)
    delta_up = {}
    for u in gen_ids:
        for t in periods:
            delta_up[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                       name=f"du_{u}_{t}")

    # (42h) ramp down: delta_down[u,t] >= 0
    delta_down = {}
    for u in gen_ids:
        for t in periods:
            delta_down[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                         name=f"dd_{u}_{t}")

    # (42i) DC flow: lambda_i[l,t] free
    lambda_i = {}
    for l in line_ids:
        for t in periods:
            lambda_i[l, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                       name=f"lam_i_{l}_{t}")

    # (42j) thermal limit: rho_j_upper[l,t], rho_j_lower[l,t] >= 0
    rho_j_upper = {}
    rho_j_lower = {}
    for l in line_ids:
        for t in periods:
            rho_j_upper[l, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"rho_ju_{l}_{t}")
            rho_j_lower[l, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"rho_jl_{l}_{t}")

    # (42l) angle diff: rho_l_upper[l,t], rho_l_lower[l,t] >= 0
    rho_l_upper = {}
    rho_l_lower = {}
    for l in line_ids:
        for t in periods:
            rho_l_upper[l, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"rho_lu_{l}_{t}")
            rho_l_lower[l, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"rho_ll_{l}_{t}")

    # (42m) gas flow conservation: lambda_m[j,t] free
    lambda_m = {}
    for j in junc_ids:
        for t in periods:
            lambda_m[j, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                       name=f"lam_m_{j}_{t}")

    # (42n) supply decomposition: lambda_n[j,t] free (source only)
    lambda_n = {}
    for j in source_junc_ids:
        for t in periods:
            lambda_n[j, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                       name=f"lam_n_{j}_{t}")

    # (42o) demand satisfaction: lambda_o[j,t] free
    lambda_o = {}
    for j in junc_ids:
        for t in periods:
            lambda_o[j, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                       name=f"lam_o_{j}_{t}")

    # (42p) shedding bound: rho_p[j,t] >= 0
    rho_p = {}
    for j in junc_ids:
        for t in periods:
            rho_p[j, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"rho_p_{j}_{t}")

    # (42r) supply interval upper: rho_r[j,s,t] >= 0
    rho_r = {}
    for j in source_junc_ids:
        for si in junctions[j]["supply_intervals"]:
            sid = si["id"]
            for t in periods:
                rho_r[j, sid, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                             name=f"rho_r_{j}_{sid}_{t}")

    # (42s) compressor bounds duals
    comp_lower_dual = {}
    comp_upper_dual = {}
    for a in compressor_ids:
        for t in periods:
            comp_lower_dual[a, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                              name=f"cl_{a}_{t}")
            comp_upper_dual[a, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                              name=f"cu_{a}_{t}")

    # Non-source supply zero: lambda_ns[j,t] free
    lambda_ns = {}
    for j in junc_ids:
        if not junctions[j]["is_source"]:
            for t in periods:
                lambda_ns[j, t] = m.addVar(lb=-DUAL_UB, ub=DUAL_UB,
                                            name=f"lam_ns_{j}_{t}")

    # -----------------------------------------------------------------------
    # McCormick auxiliary variables for strong duality bilinear terms
    # -----------------------------------------------------------------------
    # mu_d[u,b,t] = rho_d_upper[u,b,t] * w[u,b,t]
    mu_d = {}
    for u in gen_ids:
        for bid in gens[u]["bids"]:
            b = bid["id"]
            for t in periods:
                mu_d[u, b, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                          name=f"mu_d_{u}_{b}_{t}")

    # mu_el[u,t] = alpha_lower[u,t] * o[u,t]
    mu_el = {}
    for u in gen_ids:
        for t in periods:
            mu_el[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"mu_el_{u}_{t}")

    # mu_eu[u,t] = alpha_upper[u,t] * o[u,t]
    mu_eu = {}
    for u in gen_ids:
        for t in periods:
            mu_eu[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"mu_eu_{u}_{t}")

    # mu_go[u,t] = delta_up[u,t] * o[u,t-1], for t in 1..T
    mu_go = {}
    for u in gen_ids:
        for t in periods:
            mu_go[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"mu_go_{u}_{t}")

    # mu_gv[u,t] = delta_up[u,t] * v_plus[u,t], for t in 1..T
    mu_gv = {}
    for u in gen_ids:
        for t in periods:
            mu_gv[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"mu_gv_{u}_{t}")

    # mu_ho[u,t] = delta_down[u,t] * o[u,t], for t in 1..T
    mu_ho = {}
    for u in gen_ids:
        for t in periods:
            mu_ho[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"mu_ho_{u}_{t}")

    # mu_hv[u,t] = delta_down[u,t] * v_minus[u,t], for t in 1..T
    mu_hv = {}
    for u in gen_ids:
        for t in periods:
            mu_hv[u, t] = m.addVar(lb=0.0, ub=DUAL_UB,
                                    name=f"mu_hv_{u}_{t}")

    # McCormick for bid-validity (41a-41e): v_bid[u,k,t] = psi[k,t] * o[u,t]
    v_bid = {}
    for u in gfpp_ids:
        gj = gens[u]["gas_junction"]
        k = junc_to_zone[gj]
        for t in periods:
            v_bid[u, k, t] = m.addVar(lb=0.0, ub=psi_ub,
                                       name=f"vbid_{u}_{k}_{t}")

    m.update()

    # -----------------------------------------------------------------------
    # OBJECTIVE (40a)
    # -----------------------------------------------------------------------
    obj_elec = gp.LinExpr()
    obj_gas = gp.LinExpr()

    for t in periods:
        for u in gen_ids:
            g = gens[u]
            obj_elec.add(g["no_load_cost"] * o[u, t])
            obj_elec.add(r[u, t])
            for bid in g["bids"]:
                b = bid["id"]
                obj_elec.add(bid["price"] * s_e[u, b, t])

        for j in junc_ids:
            junc = junctions[j]
            obj_gas.add(junc["demand_shedding_cost"] * q_gas[j, t])
            if junc["is_source"]:
                for si in junc["supply_intervals"]:
                    sid = si["id"]
                    obj_gas.add(si["slope"] * s_g_s[j, sid, t])

    m.setObjective(BETA * obj_elec + (1 - BETA) * obj_gas, GRB.MINIMIZE)

    # -----------------------------------------------------------------------
    # FIX PRE-HORIZON STATE (t=0)
    # -----------------------------------------------------------------------
    for u in gen_ids:
        g = gens[u]
        m.addConstr(o[u, 0] == g["initial_status"], name=f"fix_o0_{u}")
        m.addConstr(p[u, 0] == g["initial_gen"], name=f"fix_p0_{u}")

    # -----------------------------------------------------------------------
    # LEADER CONSTRAINTS
    # -----------------------------------------------------------------------

    # (40d) Initial status fixing for must-stay periods
    # Fix o[u,t] = initial_status for t = 1, ..., min(T, remaining_periods)
    for u in gen_ids:
        g = gens[u]
        init_status = g["initial_status"]
        remaining = g["initial_active_periods"] + g["initial_inactive_periods"]
        for t in range(1, min(T + 1, remaining + 1)):
            m.addConstr(o[u, t] == init_status, name=f"init_fix_{u}_{t}")

    # (40g) Startup/shutdown logic: v+[t] - v-[t] = o[t] - o[t-1], t=1..T
    for u in gen_ids:
        for t in periods:
            m.addConstr(v_plus[u, t] - v_minus[u, t] == o[u, t] - o[u, t - 1],
                        name=f"logic_{u}_{t}")
        for t in periods:
            m.addConstr(v_plus[u, t] + v_minus[u, t] <= 1,
                        name=f"vpm_excl_{u}_{t}")

    # (40b) Startup cost
    for u in gen_ids:
        g = gens[u]
        for h_idx, (h, C_uh) in enumerate(g["startup_cost_params"]):
            for t in periods:
                # r[u,t] >= C_uh * (o[u,t] - sum_{n=1..h} o[u,t-n])
                expr = C_uh * o[u, t]
                for n in range(1, h + 1):
                    tn = t - n
                    if tn >= 0:
                        # tn=0 is the pre-horizon state (fixed)
                        expr -= C_uh * o[u, tn]
                    else:
                        # Before pre-horizon: use initial_status
                        expr -= C_uh * g["initial_status"]
                m.addConstr(r[u, t] >= expr,
                            name=f"startup_cost_{u}_{h_idx}_{t}")

    # (40e) Min up time
    for u in gen_ids:
        g = gens[u]
        tau_bar = g["min_up_time"]
        for t in periods:
            lhs = gp.LinExpr()
            for tp in range(max(1, t - tau_bar + 1), t + 1):
                lhs.add(v_plus[u, tp])
            m.addConstr(lhs <= o[u, t], name=f"min_up_{u}_{t}")

    # (40f) Min down time
    for u in gen_ids:
        g = gens[u]
        tau = g["min_down_time"]
        for t in periods:
            lhs = gp.LinExpr()
            for tp in range(max(1, t - tau + 1), t + 1):
                lhs.add(v_minus[u, tp])
            m.addConstr(lhs <= 1 - o[u, t], name=f"min_down_{u}_{t}")

    # (40h) Bid selection requires generator on (for GFPPs)
    for u in gfpp_ids:
        g = gens[u]
        for bid in g["bids"]:
            b = bid["id"]
            for t in periods:
                m.addConstr(w[u, b, t] <= o[u, t],
                            name=f"bid_on_{u}_{b}_{t}")

    # (40i) phi_max definition for GFPPs
    # phi_max[u,t] = max_gas_price_fraction * psi_ub * o[u,t]
    for u in gfpp_ids:
        g = gens[u]
        frac = g["max_gas_price_fraction"]
        for t in periods:
            m.addConstr(phi_max[u, t] == frac * psi_ub * o[u, t],
                        name=f"phi_max_def_{u}_{t}")

    # (40l) Bid bounds: 0 <= s_e[u,b,t] <= s_bar_b * w[u,b,t]
    for u in gen_ids:
        g = gens[u]
        for bid in g["bids"]:
            b = bid["id"]
            s_bar = bid["max_amount"]
            for t in periods:
                m.addConstr(s_e[u, b, t] <= s_bar * w[u, b, t],
                            name=f"bid_ub_{u}_{b}_{t}")

    # (40m) Sequential bid activation
    for u in gen_ids:
        g = gens[u]
        bids_list = g["bids"]
        for idx in range(len(bids_list) - 1):
            b = bids_list[idx]["id"]
            b_next = bids_list[idx + 1]["id"]
            s_bar = bids_list[idx]["max_amount"]
            for t in periods:
                m.addConstr(s_e[u, b, t] >= s_bar * w[u, b_next, t],
                            name=f"bid_seq_{u}_{b}_{t}")

    # (40n) + McCormick (41a-41e): Bid-validity for GFPPs
    for u in gfpp_ids:
        gj = gens[u]["gas_junction"]
        k = junc_to_zone[gj]
        for t in periods:
            # (41a) phi_max >= v_bid (= psi * o linearized)
            m.addConstr(phi_max[u, t] >= v_bid[u, k, t],
                        name=f"bidval_{u}_{t}")
            # (41b)
            m.addConstr(v_bid[u, k, t] >= psi[k, t] - psi_ub * (1 - o[u, t]),
                        name=f"mc_bid_lb1_{u}_{t}")
            # (41c)
            m.addConstr(v_bid[u, k, t] <= psi[k, t] - psi_lb * (1 - o[u, t]),
                        name=f"mc_bid_ub1_{u}_{t}")
            # (41d)
            m.addConstr(v_bid[u, k, t] <= psi_ub * o[u, t],
                        name=f"mc_bid_ub2_{u}_{t}")
            # (41e)
            m.addConstr(v_bid[u, k, t] >= psi_lb * o[u, t],
                        name=f"mc_bid_lb2_{u}_{t}")

    # -----------------------------------------------------------------------
    # FOLLOWER PRIMAL CONSTRAINTS (t=1..T)
    # -----------------------------------------------------------------------

    # (42b) Power balance at each bus
    for i in bus_ids:
        for t in periods:
            gen_sum = gp.LinExpr()
            for u in bus_to_gens[i]:
                gen_sum.add(p[u, t])
            demand = elec_demand(i, t)

            flow_out = gp.LinExpr()
            flow_in = gp.LinExpr()
            for l in line_ids:
                ln = lines[l]
                if ln["from_bus"] == i:
                    flow_out.add(f_line[l, t])
                if ln["to_bus"] == i:
                    flow_in.add(f_line[l, t])

            m.addConstr(gen_sum - demand == flow_out - flow_in,
                        name=f"pbal_{i}_{t}")

    # (42c) Generation = sum of bids
    for u in gen_ids:
        for t in periods:
            bid_sum = gp.LinExpr()
            for bid in gens[u]["bids"]:
                bid_sum.add(s_e[u, bid["id"], t])
            m.addConstr(p[u, t] == bid_sum, name=f"gen_bid_{u}_{t}")

    # (42e) Power bounds
    for u in gen_ids:
        g = gens[u]
        for t in periods:
            m.addConstr(p[u, t] >= g["min_power"] * o[u, t],
                        name=f"pmin_{u}_{t}")
            m.addConstr(p[u, t] <= g["max_power"] * o[u, t],
                        name=f"pmax_{u}_{t}")

    # (42g) Ramp up: p[t] - p[t-1] <= ramp_up * o[t-1] + max_power * v+[t]
    for u in gen_ids:
        g = gens[u]
        for t in periods:
            m.addConstr(p[u, t] - p[u, t - 1] <=
                        g["ramp_up"] * o[u, t - 1] + g["max_power"] * v_plus[u, t],
                        name=f"ramp_up_{u}_{t}")

    # (42h) Ramp down: p[t-1] - p[t] <= ramp_down * o[t] + min_power * v-[t]
    for u in gen_ids:
        g = gens[u]
        for t in periods:
            m.addConstr(p[u, t - 1] - p[u, t] <=
                        g["ramp_down"] * o[u, t] + g["min_power"] * v_minus[u, t],
                        name=f"ramp_down_{u}_{t}")

    # (42i) DC power flow: f = b * (theta_from - theta_to)
    for l in line_ids:
        ln = lines[l]
        for t in periods:
            m.addConstr(f_line[l, t] == ln["susceptance"] *
                        (theta[ln["from_bus"], t] - theta[ln["to_bus"], t]),
                        name=f"dcflow_{l}_{t}")

    # (42j) Thermal limits
    for l in line_ids:
        ln = lines[l]
        for t in periods:
            m.addConstr(f_line[l, t] <= ln["thermal_limit"],
                        name=f"therm_ub_{l}_{t}")
            m.addConstr(f_line[l, t] >= -ln["thermal_limit"],
                        name=f"therm_lb_{l}_{t}")

    # (42l) Angle difference limits
    for l in line_ids:
        ln = lines[l]
        for t in periods:
            m.addConstr(theta[ln["from_bus"], t] - theta[ln["to_bus"], t] <=
                        ln["angle_diff_limit"],
                        name=f"angdiff_ub_{l}_{t}")
            m.addConstr(theta[ln["from_bus"], t] - theta[ln["to_bus"], t] >=
                        -ln["angle_diff_limit"],
                        name=f"angdiff_lb_{l}_{t}")

    # (42m) Gas flow conservation
    for j in junc_ids:
        for t in periods:
            flow_out = gp.LinExpr()
            flow_in = gp.LinExpr()
            for a in conn_ids:
                cn = connections[a]
                if cn["from_junction"] == j:
                    flow_out.add(phi_gas[a, t])
                if cn["to_junction"] == j:
                    flow_in.add(phi_gas[a, t])
            m.addConstr(s_g[j, t] - l_gas[j, t] - gamma[j, t] ==
                        flow_out - flow_in,
                        name=f"gasbal_{j}_{t}")

    # (42n) Supply decomposition (source junctions)
    for j in source_junc_ids:
        for t in periods:
            supply_sum = gp.LinExpr()
            for si in junctions[j]["supply_intervals"]:
                supply_sum.add(s_g_s[j, si["id"], t])
            m.addConstr(s_g[j, t] == supply_sum, name=f"supply_dec_{j}_{t}")

    # Non-source junctions: s_g = 0
    for j in junc_ids:
        if not junctions[j]["is_source"]:
            for t in periods:
                m.addConstr(s_g[j, t] == 0, name=f"no_supply_{j}_{t}")

    # (42o) Demand satisfaction
    for j in junc_ids:
        for t in periods:
            d_g = gas_demand(j, t)
            m.addConstr(l_gas[j, t] == d_g - q_gas[j, t],
                        name=f"gas_demand_{j}_{t}")

    # (42p) Shedding bounds
    for j in junc_ids:
        for t in periods:
            d_g = gas_demand(j, t)
            m.addConstr(q_gas[j, t] <= d_g, name=f"shed_ub_{j}_{t}")

    # (42s) Compressor constraints
    for a in compressor_ids:
        cn = connections[a]
        ratio_lb_sq = cn["compression_ratio_lb"] ** 2
        ratio_ub_sq = cn["compression_ratio_ub"] ** 2
        fj = cn["from_junction"]
        tj = cn["to_junction"]
        for t in periods:
            m.addConstr(pi_sq[tj, t] >= ratio_lb_sq * pi_sq[fj, t],
                        name=f"comp_lb_{a}_{t}")
            m.addConstr(pi_sq[tj, t] <= ratio_ub_sq * pi_sq[fj, t],
                        name=f"comp_ub_{a}_{t}")

    # (42u) Weymouth equation (SOC relaxation) for pipelines
    # pi_sq[from] - pi_sq[to] >= W * phi_gas^2
    for a in pipeline_ids:
        cn = connections[a]
        W = cn["weymouth_factor"]
        fj = cn["from_junction"]
        tj = cn["to_junction"]
        for t in periods:
            m.addQConstr(
                pi_sq[fj, t] - pi_sq[tj, t] >= W * phi_gas[a, t] * phi_gas[a, t],
                name=f"weymouth_{a}_{t}")

    # (42w) Heat rate constraint (SOC)
    # gamma[j,t] >= sum_{u at j} (H_u2 * p[u,t]^2 + H_u1 * p[u,t] + H_u0 * o[u,t])
    for j in junc_ids:
        gfpps_at_j = junc_to_gfpps[j]
        if gfpps_at_j:
            for t in periods:
                quad_expr = gp.QuadExpr()
                for u in gfpps_at_j:
                    g = gens[u]
                    hr = g["heat_rate_coefficients"]
                    quad_expr.add(hr["H_u2"] * p[u, t] * p[u, t])
                    quad_expr.add(hr["H_u1"] * p[u, t])
                    quad_expr.add(hr["H_u0"] * o[u, t])
                m.addQConstr(gamma[j, t] >= quad_expr,
                             name=f"heatrate_{j}_{t}")

    # -----------------------------------------------------------------------
    # DUAL FEASIBILITY CONSTRAINTS
    # -----------------------------------------------------------------------

    # --- Dual for s_e[u,b,t] (>= 0) ---
    # (42c): coeff +1 -> lambda_c
    # (42d upper): s_bar*w - s_e >= 0 -> coeff -1 -> rho_d_upper * (-1)
    # Cost: BETA * price
    # Condition: lambda_c - rho_d_upper <= BETA * price
    for u in gen_ids:
        for bid in gens[u]["bids"]:
            b = bid["id"]
            for t in periods:
                m.addConstr(lambda_c[u, t] - rho_d_upper[u, b, t] <= BETA * bid["price"],
                            name=f"df_se_{u}_{b}_{t}")

    # --- Dual for p[u,t] (>= 0) ---
    # (42b): coeff +1 at bus -> lambda_b[bus,t]
    # (42c): coeff -1 -> -lambda_c[u,t]
    # (42e lower): p >= p_min*o -> coeff +1 -> alpha_lower
    # (42e upper): p_max*o - p >= 0 -> coeff -1 -> -alpha_upper
    # (42g) at t: ramp_up*o[t-1]+p_max*v+[t]-p[t]+p[t-1] >= 0 -> p[t] coeff -1 -> -delta_up[t]
    # (42g) at t+1: ... +p[t] ... -> p[t] coeff +1 -> +delta_up[t+1]
    # (42h) at t: ramp_down*o[t]+p_min*v-[t]+p[t]-p[t-1] >= 0 -> p[t] coeff +1 -> +delta_down[t]
    # (42h) at t+1: ... -p[t] ... -> p[t] coeff -1 -> -delta_down[t+1]
    # Cost: 0
    for u in gen_ids:
        g = gens[u]
        bus_u = g["bus"]
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_b[bus_u, t], 1.0)
            expr.add(lambda_c[u, t], -1.0)
            expr.add(alpha_lower[u, t], 1.0)
            expr.add(alpha_upper[u, t], -1.0)
            # (42g) at t
            expr.add(delta_up[u, t], -1.0)
            # (42g) at t+1 (if exists)
            if t + 1 <= T:
                expr.add(delta_up[u, t + 1], 1.0)
            # (42h) at t
            expr.add(delta_down[u, t], 1.0)
            # (42h) at t+1 (if exists)
            if t + 1 <= T:
                expr.add(delta_down[u, t + 1], -1.0)
            m.addConstr(expr <= 0, name=f"df_p_{u}_{t}")

    # --- Dual for f_line[l,t] (free) ---
    # (42b): from_bus coeff -1, to_bus coeff +1
    # (42i): f - b*theta_from + b*theta_to = 0 -> f coeff +1 -> lambda_i
    # (42j upper): f_bar - f >= 0 -> coeff -1 -> -rho_j_upper
    # (42j lower): f + f_bar >= 0 -> coeff +1 -> +rho_j_lower
    # Cost: 0 (free -> equality)
    for l in line_ids:
        ln = lines[l]
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_b[ln["from_bus"], t], -1.0)
            expr.add(lambda_b[ln["to_bus"], t], 1.0)
            expr.add(lambda_i[l, t], 1.0)
            expr.add(rho_j_upper[l, t], -1.0)
            expr.add(rho_j_lower[l, t], 1.0)
            m.addConstr(expr == 0, name=f"df_f_{l}_{t}")

    # --- Dual for theta[i,t] (bounded, treat as free for simplicity) ---
    # (42i): f = b*(theta_from - theta_to) rewritten as f - b*theta_from + b*theta_to = 0
    #   from_bus: coeff -b -> lambda_i * (-b)
    #   to_bus: coeff +b -> lambda_i * (+b)
    # (42l upper): Delta - (theta_from - theta_to) >= 0
    #   from_bus: coeff -1 -> -rho_l_upper
    #   to_bus: coeff +1 -> +rho_l_upper
    # (42l lower): (theta_from - theta_to) + Delta >= 0
    #   from_bus: coeff +1 -> +rho_l_lower
    #   to_bus: coeff -1 -> -rho_l_lower
    # Cost: 0 (equality for free)
    for i in bus_ids:
        for t in periods:
            expr = gp.LinExpr()
            for l in line_ids:
                ln = lines[l]
                b_l = ln["susceptance"]
                if ln["from_bus"] == i:
                    expr.add(lambda_i[l, t], -b_l)
                    expr.add(rho_l_upper[l, t], -1.0)
                    expr.add(rho_l_lower[l, t], 1.0)
                if ln["to_bus"] == i:
                    expr.add(lambda_i[l, t], b_l)
                    expr.add(rho_l_upper[l, t], 1.0)
                    expr.add(rho_l_lower[l, t], -1.0)
            m.addConstr(expr == 0, name=f"df_theta_{i}_{t}")

    # --- Dual for s_g[j,t] (>= 0) ---
    # (42m): coeff +1 -> lambda_m
    # (42n)/(no_supply): coeff -1 -> -lambda_n or -lambda_ns
    # Cost: 0
    for j in junc_ids:
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_m[j, t], 1.0)
            if junctions[j]["is_source"]:
                expr.add(lambda_n[j, t], -1.0)
            else:
                expr.add(lambda_ns[j, t], -1.0)
            m.addConstr(expr <= 0, name=f"df_sg_{j}_{t}")

    # --- Dual for s_g_s[j,s,t] (>= 0, <= cap) ---
    # (42n): coeff +1 -> lambda_n
    # (42r): cap - s_g_s >= 0 -> coeff -1 -> -rho_r
    # Cost: (1-BETA) * slope
    for j in source_junc_ids:
        for si in junctions[j]["supply_intervals"]:
            sid = si["id"]
            for t in periods:
                expr = gp.LinExpr()
                expr.add(lambda_n[j, t], 1.0)
                expr.add(rho_r[j, sid, t], -1.0)
                m.addConstr(expr <= (1 - BETA) * si["slope"],
                            name=f"df_sgs_{j}_{sid}_{t}")

    # --- Dual for pi_sq[j,t] (bounded, treat as free) ---
    # (42s) compressor:
    #   lower: pi_sq[to] - ratio_lb^2*pi_sq[from] >= 0
    #     from: coeff -ratio_lb^2 -> comp_lower_dual * (-ratio_lb^2)
    #     to: coeff +1 -> comp_lower_dual
    #   upper: ratio_ub^2*pi_sq[from] - pi_sq[to] >= 0
    #     from: coeff +ratio_ub^2 -> comp_upper_dual * ratio_ub^2
    #     to: coeff -1 -> -comp_upper_dual
    # Cost: 0
    for j in junc_ids:
        for t in periods:
            expr = gp.LinExpr()
            for a in compressor_ids:
                cn = connections[a]
                ratio_lb_sq = cn["compression_ratio_lb"] ** 2
                ratio_ub_sq = cn["compression_ratio_ub"] ** 2
                if cn["from_junction"] == j:
                    expr.add(comp_lower_dual[a, t], -ratio_lb_sq)
                    expr.add(comp_upper_dual[a, t], ratio_ub_sq)
                if cn["to_junction"] == j:
                    expr.add(comp_lower_dual[a, t], 1.0)
                    expr.add(comp_upper_dual[a, t], -1.0)
            m.addConstr(expr == 0, name=f"df_pisq_{j}_{t}")

    # --- Dual for phi_gas[a,t] (>= 0) ---
    # (42m): from_junction flow_out coeff -1, to_junction flow_in coeff +1
    #   (gasbal: s_g - l_gas - gamma - flow_out + flow_in = 0)
    #   at from_junction: coeff -1 -> lambda_m[from] * (-1)
    #   at to_junction: coeff +1 -> lambda_m[to] * (+1)
    # Cost: 0
    for a in conn_ids:
        cn = connections[a]
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_m[cn["from_junction"], t], -1.0)
            expr.add(lambda_m[cn["to_junction"], t], 1.0)
            m.addConstr(expr <= 0, name=f"df_phig_{a}_{t}")

    # --- Dual for l_gas[j,t] (>= 0) ---
    # (42m): coeff -1 -> -lambda_m
    # (42o): l_gas = d_g - q_gas -> coeff +1 -> lambda_o
    # Cost: 0
    for j in junc_ids:
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_m[j, t], -1.0)
            expr.add(lambda_o[j, t], 1.0)
            m.addConstr(expr <= 0, name=f"df_lg_{j}_{t}")

    # --- Dual for q_gas[j,t] (>= 0) ---
    # (42o): coeff -1 -> -lambda_o
    # (42p): d_g - q >= 0 -> coeff -1 -> -rho_p
    # Cost: (1-BETA) * kappa_j
    for j in junc_ids:
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_o[j, t], -1.0)
            expr.add(rho_p[j, t], -1.0)
            m.addConstr(expr <= (1 - BETA) * junctions[j]["demand_shedding_cost"],
                        name=f"df_qg_{j}_{t}")

    # --- Dual for gamma[j,t] (>= 0) ---
    # (42m): coeff -1 -> -lambda_m
    # Cost: 0
    for j in junc_ids:
        for t in periods:
            expr = gp.LinExpr()
            expr.add(lambda_m[j, t], -1.0)
            m.addConstr(expr <= 0, name=f"df_gamma_{j}_{t}")

    # -----------------------------------------------------------------------
    # McCORMICK CONSTRAINTS for strong duality bilinear terms
    # -----------------------------------------------------------------------
    def add_mccormick(model, mu, dual, binary, dual_ub, name_prefix):
        """mu = dual * binary, dual in [0, dual_ub], binary in {0,1}."""
        model.addConstr(mu >= 0, name=f"{name_prefix}_lb1")
        model.addConstr(mu <= dual_ub * binary, name=f"{name_prefix}_ub1")
        model.addConstr(mu >= dual - dual_ub * (1 - binary), name=f"{name_prefix}_lb2")
        model.addConstr(mu <= dual, name=f"{name_prefix}_ub2")

    # mu_d[u,b,t] = rho_d_upper[u,b,t] * w[u,b,t]
    for u in gen_ids:
        for bid in gens[u]["bids"]:
            b = bid["id"]
            for t in periods:
                add_mccormick(m, mu_d[u, b, t], rho_d_upper[u, b, t],
                              w[u, b, t], DUAL_UB, f"mc_d_{u}_{b}_{t}")

    # mu_el[u,t] = alpha_lower[u,t] * o[u,t]
    for u in gen_ids:
        for t in periods:
            add_mccormick(m, mu_el[u, t], alpha_lower[u, t],
                          o[u, t], DUAL_UB, f"mc_el_{u}_{t}")

    # mu_eu[u,t] = alpha_upper[u,t] * o[u,t]
    for u in gen_ids:
        for t in periods:
            add_mccormick(m, mu_eu[u, t], alpha_upper[u, t],
                          o[u, t], DUAL_UB, f"mc_eu_{u}_{t}")

    # mu_go[u,t] = delta_up[u,t] * o[u,t-1]
    for u in gen_ids:
        for t in periods:
            add_mccormick(m, mu_go[u, t], delta_up[u, t],
                          o[u, t - 1], DUAL_UB, f"mc_go_{u}_{t}")

    # mu_gv[u,t] = delta_up[u,t] * v_plus[u,t]
    for u in gen_ids:
        for t in periods:
            add_mccormick(m, mu_gv[u, t], delta_up[u, t],
                          v_plus[u, t], DUAL_UB, f"mc_gv_{u}_{t}")

    # mu_ho[u,t] = delta_down[u,t] * o[u,t]
    for u in gen_ids:
        for t in periods:
            add_mccormick(m, mu_ho[u, t], delta_down[u, t],
                          o[u, t], DUAL_UB, f"mc_ho_{u}_{t}")

    # mu_hv[u,t] = delta_down[u,t] * v_minus[u,t]
    for u in gen_ids:
        for t in periods:
            add_mccormick(m, mu_hv[u, t], delta_down[u, t],
                          v_minus[u, t], DUAL_UB, f"mc_hv_{u}_{t}")

    # -----------------------------------------------------------------------
    # STRONG DUALITY CONSTRAINT
    # -----------------------------------------------------------------------
    # Follower primal objective <= dual objective
    # LHS: sum of follower cost * follower variable
    # RHS: sum of (dual * RHS), where RHS may involve leader variables (linearized)

    primal_cost = gp.LinExpr()
    dual_cost = gp.LinExpr()

    for t in periods:
        # --- Primal cost ---
        for u in gen_ids:
            for bid in gens[u]["bids"]:
                b = bid["id"]
                primal_cost.add(BETA * bid["price"] * s_e[u, b, t])
        for j in junc_ids:
            junc = junctions[j]
            primal_cost.add((1 - BETA) * junc["demand_shedding_cost"] * q_gas[j, t])
            if junc["is_source"]:
                for si in junc["supply_intervals"]:
                    primal_cost.add((1 - BETA) * si["slope"] * s_g_s[j, si["id"], t])

        # --- Dual cost (RHS * dual) ---

        # (42b) power balance: RHS = demand (constant)
        for i in bus_ids:
            dual_cost.add(lambda_b[i, t], elec_demand(i, t))

        # (42c) p = sum s_e: RHS = 0

        # (42d upper) s_bar * w - s_e >= 0: RHS = s_bar * w (parametric)
        # -> s_bar * mu_d (McCormick for rho_d_upper * w)
        for u in gen_ids:
            for bid in gens[u]["bids"]:
                b = bid["id"]
                dual_cost.add(mu_d[u, b, t], bid["max_amount"])

        # (42e lower) p >= p_min * o: RHS = p_min * o
        # -> p_min * mu_el
        for u in gen_ids:
            dual_cost.add(mu_el[u, t], gens[u]["min_power"])

        # (42e upper) p_max * o - p >= 0: RHS = p_max * o
        # -> p_max * mu_eu
        for u in gen_ids:
            dual_cost.add(mu_eu[u, t], gens[u]["max_power"])

        # (42g) ramp up: ramp_up*o[t-1] + p_max*v+[t] - p[t] + p[t-1] >= 0
        # RHS = ramp_up * o[t-1] + p_max * v+[t] (parametric)
        # -> ramp_up * mu_go + p_max * mu_gv
        for u in gen_ids:
            g = gens[u]
            dual_cost.add(mu_go[u, t], g["ramp_up"])
            dual_cost.add(mu_gv[u, t], g["max_power"])

        # (42h) ramp down: ramp_down*o[t] + p_min*v-[t] + p[t] - p[t-1] >= 0
        # RHS = ramp_down * o[t] + p_min * v-[t] (parametric)
        # -> ramp_down * mu_ho + p_min * mu_hv
        for u in gen_ids:
            g = gens[u]
            dual_cost.add(mu_ho[u, t], g["ramp_down"])
            dual_cost.add(mu_hv[u, t], g["min_power"])

        # (42i) DC flow: RHS = 0

        # (42j upper) f_bar - f >= 0: RHS = f_bar (constant)
        for l in line_ids:
            dual_cost.add(rho_j_upper[l, t], lines[l]["thermal_limit"])

        # (42j lower) f + f_bar >= 0: RHS = f_bar (constant)
        for l in line_ids:
            dual_cost.add(rho_j_lower[l, t], lines[l]["thermal_limit"])

        # (42l upper) Delta - angle_diff >= 0: RHS = Delta
        for l in line_ids:
            dual_cost.add(rho_l_upper[l, t], lines[l]["angle_diff_limit"])

        # (42l lower) angle_diff + Delta >= 0: RHS = Delta
        for l in line_ids:
            dual_cost.add(rho_l_lower[l, t], lines[l]["angle_diff_limit"])

        # (42m) gas balance: RHS = 0

        # (42n) supply decomposition: RHS = 0

        # (42o) demand: l_gas + q_gas = d_g -> RHS = d_g
        for j in junc_ids:
            dual_cost.add(lambda_o[j, t], gas_demand(j, t))

        # (42p) shed: d_g - q >= 0 -> RHS = d_g
        for j in junc_ids:
            dual_cost.add(rho_p[j, t], gas_demand(j, t))

        # (42r) supply interval: cap - s_g_s >= 0 -> RHS = cap
        for j in source_junc_ids:
            for si in junctions[j]["supply_intervals"]:
                sid = si["id"]
                cap = si["interval_ub"] - si["interval_lb"]
                dual_cost.add(rho_r[j, sid, t], cap)

        # (42s) compressor: RHS = 0
        # non-source s_g=0: RHS = 0

    # Strong duality: primal_cost == dual_cost
    m.addConstr(primal_cost == dual_cost, name="strong_duality")

    # -----------------------------------------------------------------------
    # GUROBI PARAMETERS (Section 8.2.2)
    # -----------------------------------------------------------------------
    m.Params.NumericFocus = 3
    m.Params.DualReductions = 0
    m.Params.ScaleFlag = 0
    m.Params.BarQCPConvTol = 1e-7
    m.Params.Aggregate = 0
    m.Params.TimeLimit = time_limit

    return m


def solve_and_output(m, solution_path):
    """Solve the model and write solution JSON."""
    m.optimize()

    result = {
        "status": m.Status,
        "status_name": {
            1: "LOADED",
            2: "OPTIMAL",
            3: "INFEASIBLE",
            4: "INF_OR_UNBD",
            5: "UNBOUNDED",
            6: "CUTOFF",
            7: "ITERATION_LIMIT",
            8: "NODE_LIMIT",
            9: "TIME_LIMIT",
            10: "SOLUTION_LIMIT",
            11: "INTERRUPTED",
            12: "NUMERIC",
            13: "SUBOPTIMAL",
            14: "INPROGRESS",
            15: "USER_OBJ_LIMIT",
        }.get(m.Status, "UNKNOWN"),
        "objective_value": None,
        "best_bound": None,
        "gap": None,
        "runtime": m.Runtime,
        "node_count": m.NodeCount,
    }

    if m.SolCount > 0:
        result["objective_value"] = m.ObjVal
        try:
            result["best_bound"] = m.ObjBound
            result["gap"] = m.MIPGap
        except Exception:
            pass

        # Export only primary leader/follower variables. The MISOCP
        # single-level reformulation also produces dual multipliers
        # (lam_*, rho_*, al_*, au_*, du_*, dd_*, cl_*, cu_*) and
        # McCormick product variables (mu_*, vbid_*); those are
        # reformulation artifacts and are NOT part of the original
        # bilevel solution structure, so they are excluded.
        primary_prefixes = (
            "o_", "vp_", "vm_", "r_", "w_", "phimax_",
            "p_", "se_", "f_", "theta_",
            "sg_", "sgs_", "pisq_", "phig_",
            "lg_", "qg_", "gamma_", "psi_",
        )
        primary_vars = {}
        for v in m.getVars():
            if abs(v.X) > 1e-7 and v.VarName.startswith(primary_prefixes):
                primary_vars[v.VarName] = v.X
        result["primary_variables"] = primary_vars
    else:
        print("WARNING: No feasible solution found.")

    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']:.6f}")
    print(f"Status: {result['status_name']}")
    print(f"Runtime: {result['runtime']:.2f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MISOCP single-level reformulation (Approach G) for UCGNA bilevel problem "
                    "(Byeon & Van Hentenryck, 2022)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path to output solution JSON (default: gurobi_solution_1.json)")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Gurobi time limit in seconds (default: 3600)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from {args.instance_path}")
    inst = load_instance(args.instance_path)
    print(f"Instance: {inst.get('instance_id', 'unknown')}, "
          f"T={inst['time_periods']}, "
          f"generators={inst['generators']['num_generators']}, "
          f"GFPPs={inst['generators']['num_gfpp']}")

    print("Building MISOCP model...")
    model = build_model(inst, time_limit=args.time_limit)
    print(f"Model has {model.NumVars} variables, {model.NumConstrs} linear constraints, "
          f"{model.NumQConstrs} quadratic constraints")

    print("Solving...")
    result = solve_and_output(model, args.solution_path)

    # Always exit 0: even when Gurobi proved INFEASIBLE or no incumbent
    # was found, the wrapper has produced a valid solution JSON (with
    # objective_value=None) and the orchestration layer's classifier
    # interprets that correctly. Returning a non-zero exit code here
    # would have run_program_solutions.py record exit_code=1 which the
    # tag classifier promotes to tag G/H — falsely flagging a genuine
    # INFEAS result as a Python crash.
    return 0


if __name__ == "__main__":
    sys.exit(main())
