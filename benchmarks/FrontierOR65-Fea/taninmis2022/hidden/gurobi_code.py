#!/usr/bin/env python3
"""
MM-SMIP (Multistage Multiscale Stochastic Mixed Integer Programming) model
for power system capacity expansion, solved monolithically with Gurobi.

Based on: Taninmis et al. (2022)
"""

import argparse
import json
import time

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
    with open(path, "r") as f:
        return json.load(f)


def build_scenario_tree_info(instance):
    """Build helper mappings for the scenario tree."""
    tree = instance["scenario_tree"]
    sp = instance["scenario_parameters"]

    node_by_id = {n["id"]: n for n in tree}

    # Compute demand growth factor and cost multiplier along each node's path
    high_growth = sp["demand_growth_high_percent"]
    low_growth = sp["demand_growth_low_percent"]
    cost_var = sp["cost_variation_percent"]

    # For each node, determine its ancestor path from root
    def ancestor_path(nid):
        path = []
        cur = nid
        while cur is not None:
            path.append(cur)
            cur = node_by_id[cur]["parent_id"]
        path.reverse()
        return path

    node_demand_growth = {}   # node_id -> cumulative demand growth factor
    node_cost_mult = {}       # node_id -> cumulative cost multiplier

    for node in tree:
        nid = node["id"]
        path = ancestor_path(nid)
        growth = 1.0
        cost_mult = 1.0
        for i, pid in enumerate(path):
            if i == 0:
                continue  # root: no growth
            parent = node_by_id[path[i - 1]]
            children = parent["children"]
            # First child = high demand growth / cost increase
            # Second child = low demand growth / cost decrease
            if len(children) >= 2:
                child_idx = children.index(pid)
                if child_idx == 0:
                    growth *= (1.0 + high_growth)
                    cost_mult *= (1.0 + cost_var)
                else:
                    growth *= (1.0 + low_growth)
                    cost_mult *= (1.0 - cost_var)
            else:
                # single child fallback
                growth *= (1.0 + low_growth)

        node_demand_growth[nid] = growth
        node_cost_mult[nid] = cost_mult

    return node_by_id, node_demand_growth, node_cost_mult


def build_model(instance, time_limit=None):
    """Build the monolithic MM-SMIP Gurobi model."""
    system = instance["system"]
    sp = instance["scenario_parameters"]
    gp_params = instance["global_parameters"]
    rep_days = instance["representative_days"]
    ll_scenarios = instance["lower_level_scenarios"]

    generators = system["generators"]
    lines = system["transmission_lines"]
    storage_units = system["storage"]
    loads = system["loads"]
    buses = system["buses"]

    node_by_id, node_demand_growth, node_cost_mult = build_scenario_tree_info(instance)
    tree = instance["scenario_tree"]

    num_hours = sp["hours_per_day"]
    num_rep_days = sp["num_representative_days"]
    num_ll_scenarios = sp["num_lower_level_scenarios_per_day"]

    RM = gp_params["reserve_margin_requirement"]
    SR_pct = gp_params["spinning_reserve_percent"]
    derating = gp_params["derating_factors"]

    # Separate existing and candidate assets
    existing_gens = [g for g in generators if g["existing"]]
    candidate_gens = [g for g in generators if not g["existing"]]
    existing_lines = [l for l in lines if l["existing"]]
    candidate_lines = [l for l in lines if not l["existing"]]
    existing_storage = [s for s in storage_units if s["existing"]]
    candidate_storage = [s for s in storage_units if not s["existing"]]

    all_gens = generators
    all_lines = lines
    all_storage = storage_units

    # Load by bus
    load_by_bus = {ld["bus"]: ld["annual_peak_load_MW"] for ld in loads}

    # Line connectivity helpers: lines incident to each bus
    # For each line, track (from_bus, to_bus)
    line_from_to = {l["id"]: (l["from_bus"], l["to_bus"]) for l in all_lines}

    # Generators at each bus
    gens_at_bus = {}
    for b in buses:
        gens_at_bus[b] = [g for g in all_gens if g["bus"] == b]

    # Storage at each bus
    storage_at_bus = {}
    for b in buses:
        storage_at_bus[b] = [s for s in all_storage if s["bus"] == b]

    # Lines connected to each bus
    lines_at_bus = {}
    for b in buses:
        lines_at_bus[b] = [l for l in all_lines
                           if l["from_bus"] == b or l["to_bus"] == b]

    # Node IDs
    node_ids = [n["id"] for n in tree]

    # Representative day IDs
    day_ids = [d["id"] for d in rep_days]
    day_by_id = {d["id"]: d for d in rep_days}

    # Hours
    hours = list(range(num_hours))  # 0..23

    # Scenario indices for lower level
    ll_scen_ids = list(range(num_ll_scenarios))  # 0..19

    # =========================================================================
    # Create Gurobi model
    # =========================================================================
    model = gp.Model("MM_SMIP_Capacity_Expansion")
    model.setParam("Threads", 1)
    if time_limit is not None:
        model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)
    # Disable dual reductions so infeasible vs unbounded is distinguished
    model.setParam("DualReductions", 0)
    # Use barrier for the root LP (much faster for large models)
    model.setParam("Method", 2)
    # Focus on finding feasible solutions
    model.setParam("MIPFocus", 1)

    # =========================================================================
    # UPPER-LEVEL VARIABLES
    # =========================================================================

    # x[n, g_id] = binary build decision for candidate generator g at node n
    x_gen = {}
    for n in node_ids:
        for g in candidate_gens:
            x_gen[n, g["id"]] = model.addVar(
                vtype=GRB.BINARY, name=f"x_gen_{n}_{g['id']}")

    # x[n, l_id] = binary build decision for candidate line l at node n
    x_line = {}
    for n in node_ids:
        for l in candidate_lines:
            x_line[n, l["id"]] = model.addVar(
                vtype=GRB.BINARY, name=f"x_line_{n}_{l['id']}")

    # x[n, s_id] = binary build decision for candidate storage s at node n
    x_stor = {}
    for n in node_ids:
        for s in candidate_storage:
            x_stor[n, s["id"]] = model.addVar(
                vtype=GRB.BINARY, name=f"x_stor_{n}_{s['id']}")

    # kappa[n, g_id] = cumulative available capacity [MW] for candidate gen
    kappa_gen = {}
    for n in node_ids:
        for g in candidate_gens:
            kappa_gen[n, g["id"]] = model.addVar(
                lb=0.0, ub=g["Pmax_MW"], vtype=GRB.CONTINUOUS,
                name=f"kappa_gen_{n}_{g['id']}")

    kappa_line = {}
    for n in node_ids:
        for l in candidate_lines:
            kappa_line[n, l["id"]] = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                name=f"kappa_line_{n}_{l['id']}")

    kappa_stor = {}
    for n in node_ids:
        for s in candidate_storage:
            kappa_stor[n, s["id"]] = model.addVar(
                lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
                name=f"kappa_stor_{n}_{s['id']}")

    # =========================================================================
    # LOWER-LEVEL VARIABLES (for each node, rep day, ll scenario, hour)
    # =========================================================================

    # alpha[n,k,p,h,g] = commitment status (binary)
    alpha = {}
    # gamma[n,k,p,h,g] = startup indicator (binary)
    gamma = {}
    # p_gen[n,k,p,h,g] = power output (continuous)
    p_gen = {}
    # s_res[n,k,p,h,g] = spinning reserve contribution (continuous)
    s_res = {}

    # Flow variables: bidirectional
    # f_fwd[n,k,p,h,l] = flow from from_bus to to_bus
    # f_bwd[n,k,p,h,l] = flow from to_bus to from_bus
    f_fwd = {}
    f_bwd = {}

    # Storage variables
    # r[n,k,p,h,s] = stored energy
    # u[n,k,p,h,s] = withdrawal (discharge)
    # v[n,k,p,h,s] = injection (charge)
    r_stor = {}
    u_stor = {}
    v_stor = {}

    print("Creating lower-level variables...")

    for n in node_ids:
        for k in day_ids:
            for p in ll_scen_ids:
                for h in hours:
                    # Generator variables
                    for g in all_gens:
                        gid = g["id"]
                        key = (n, k, p, h, gid)
                        alpha[key] = model.addVar(
                            vtype=GRB.BINARY, name=f"alpha_{n}_{k}_{p}_{h}_{gid}")
                        gamma[key] = model.addVar(
                            vtype=GRB.BINARY, name=f"gamma_{n}_{k}_{p}_{h}_{gid}")
                        p_gen[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"p_{n}_{k}_{p}_{h}_{gid}")
                        s_res[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"s_{n}_{k}_{p}_{h}_{gid}")

                    # Flow variables for all lines
                    for l in all_lines:
                        lid = l["id"]
                        key = (n, k, p, h, lid)
                        f_fwd[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"f_fwd_{n}_{k}_{p}_{h}_{lid}")
                        f_bwd[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"f_bwd_{n}_{k}_{p}_{h}_{lid}")

                    # Storage variables
                    for s in all_storage:
                        sid = s["id"]
                        key = (n, k, p, h, sid)
                        r_stor[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"r_{n}_{k}_{p}_{h}_{sid}")
                        u_stor[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"u_{n}_{k}_{p}_{h}_{sid}")
                        v_stor[key] = model.addVar(
                            lb=0.0, vtype=GRB.CONTINUOUS,
                            name=f"v_{n}_{k}_{p}_{h}_{sid}")

    model.update()
    print("Variables created. Adding constraints...")

    # =========================================================================
    # UPPER-LEVEL CONSTRAINTS
    # =========================================================================

    # (1b)-(1d) Cumulative capacity tracking
    for n in node_ids:
        node = node_by_id[n]
        parent_id = node["parent_id"]

        for g in candidate_gens:
            gid = g["id"]
            if parent_id is None:
                # Root: kappa = a_g * x  (eq 1b, with kappa^0 = 0)
                model.addConstr(
                    kappa_gen[n, gid] == g["Pmax_MW"] * x_gen[n, gid],
                    name=f"cum_gen_{n}_{gid}")
            else:
                model.addConstr(
                    kappa_gen[n, gid] == kappa_gen[parent_id, gid] + g["Pmax_MW"] * x_gen[n, gid],
                    name=f"cum_gen_{n}_{gid}")

        for l in candidate_lines:
            lid = l["id"]
            if parent_id is None:
                model.addConstr(
                    kappa_line[n, lid] == x_line[n, lid],
                    name=f"cum_line_{n}_{lid}")
            else:
                model.addConstr(
                    kappa_line[n, lid] == kappa_line[parent_id, lid] + x_line[n, lid],
                    name=f"cum_line_{n}_{lid}")

        for s in candidate_storage:
            sid = s["id"]
            if parent_id is None:
                model.addConstr(
                    kappa_stor[n, sid] == x_stor[n, sid],
                    name=f"cum_stor_{n}_{sid}")
            else:
                model.addConstr(
                    kappa_stor[n, sid] == kappa_stor[parent_id, sid] + x_stor[n, sid],
                    name=f"cum_stor_{n}_{sid}")

    # (1e) Capacity upper bound (max once: kappa <= 1, already in variable bounds)

    # (1f) Reserve margin constraint: single region (all buses)
    for n in node_ids:
        # Total peak demand at node n
        peak_demand_n = sum(load_by_bus.get(b, 0.0) for b in buses) * node_demand_growth[n]

        # Derated capacity of existing generators
        existing_cap = sum(
            derating.get(g["type"], 1.0) * g["Pmax_MW"]
            for g in existing_gens
        )

        # Derated capacity of candidate generators (depends on kappa)
        cand_cap_expr = gp.LinExpr(existing_cap)
        for g in candidate_gens:
            gid = g["id"]
            df = derating.get(g["type"], 1.0)
            cand_cap_expr += df * kappa_gen[n, gid]

        model.addConstr(
            cand_cap_expr >= (1.0 + RM) * peak_demand_n,
            name=f"reserve_margin_{n}")

    # =========================================================================
    # LOWER-LEVEL CONSTRAINTS
    # =========================================================================

    print("Adding lower-level constraints...")

    for n in node_ids:
        for k in day_ids:
            day = day_by_id[k]
            load_profile = day["load_profile"]
            solar_profile = day["solar_profile"]
            ll_scens = ll_scenarios[str(k)]

            for pidx in ll_scen_ids:
                scen = ll_scens[pidx]
                demand_mult = scen["demand_multipliers"]
                solar_mult = scen["solar_multipliers"]
                wind_pf = scen["wind_power_factors"]

                for h in hours:
                    # Demand at each bus for this (n, k, p, h)
                    growth = node_demand_growth[n]

                    # ---- Generator constraints ----
                    for g in all_gens:
                        gid = g["id"]
                        key = (n, k, pidx, h, gid)
                        is_candidate = not g["existing"]
                        gtype = g["type"]

                        # Effective Pmax considering renewables
                        pmax = g["Pmax_MW"]
                        if gtype == "solar":
                            eff_pmax = pmax * solar_profile[h] * solar_mult[h]
                        elif gtype == "wind":
                            eff_pmax = pmax * wind_pf[h]
                        else:
                            eff_pmax = pmax

                        # (2b) Commitment linked to capacity for candidates
                        if is_candidate:
                            model.addConstr(
                                alpha[key] <= kappa_gen[n, gid] / pmax,
                                name=f"link_alpha_{n}_{k}_{pidx}_{h}_{gid}")

                        # (2d) Startup indicator: gamma >= alpha_h - alpha_{h-1}
                        if h >= 1:
                            prev_key = (n, k, pidx, h - 1, gid)
                            model.addConstr(
                                gamma[key] >= alpha[key] - alpha[prev_key],
                                name=f"startup_{n}_{k}_{pidx}_{h}_{gid}")
                        else:
                            # h=0: alpha_{-1} = 0
                            model.addConstr(
                                gamma[key] >= alpha[key],
                                name=f"startup_{n}_{k}_{pidx}_{h}_{gid}")

                        # (2e) Output + reserve upper bound
                        if is_candidate:
                            # Candidate: p + s <= kappa^n_g (eq 2e)
                            model.addConstr(
                                p_gen[key] + s_res[key] <= kappa_gen[n, gid],
                                name=f"pmax_{n}_{k}_{pidx}_{h}_{gid}")
                        else:
                            # Existing generator: p + s <= Pmax * alpha (eq 2e)
                            model.addConstr(
                                p_gen[key] + s_res[key] <= eff_pmax * alpha[key],
                                name=f"pmax_{n}_{k}_{pidx}_{h}_{gid}")

                        # (2f) Minimum power output
                        model.addConstr(
                            p_gen[key] >= g["Pmin_MW"] * alpha[key],
                            name=f"pmin_{n}_{k}_{pidx}_{h}_{gid}")

                        # (2h) Ramp limits
                        if h >= 1:
                            prev_key = (n, k, pidx, h - 1, gid)
                            ramp = g["ramp_MW_per_h"]
                            model.addConstr(
                                p_gen[key] - p_gen[prev_key] <= ramp,
                                name=f"ramp_up_{n}_{k}_{pidx}_{h}_{gid}")
                            model.addConstr(
                                p_gen[prev_key] - p_gen[key] <= ramp,
                                name=f"ramp_dn_{n}_{k}_{pidx}_{h}_{gid}")

                    # (2c) Min up/down time constraints
                    for g in all_gens:
                        gid = g["id"]
                        min_on = g["min_on_h"]
                        min_off = g["min_off_h"]

                        if h >= 1:
                            # Min up time: if unit starts at h (alpha_h -
                            # alpha_{h-1} = 1), then alpha must be 1 for
                            # min_on hours.
                            if min_on > 0:
                                for tau in range(h, min(h + min_on, num_hours)):
                                    startup_diff = (alpha[(n, k, pidx, h, gid)]
                                                    - alpha[(n, k, pidx, h - 1, gid)])
                                    model.addConstr(
                                        alpha[(n, k, pidx, tau, gid)] >= startup_diff,
                                        name=f"minon_{n}_{k}_{pidx}_{h}_{tau}_{gid}")

                            # Min down time: if unit shuts down at h
                            # (alpha_{h-1} - alpha_h = 1), then alpha must
                            # be 0 for min_off hours.
                            if min_off > 0:
                                for tau in range(h, min(h + min_off, num_hours)):
                                    shutdown_diff = (alpha[(n, k, pidx, h - 1, gid)]
                                                     - alpha[(n, k, pidx, h, gid)])
                                    model.addConstr(
                                        1 - alpha[(n, k, pidx, tau, gid)] >= shutdown_diff,
                                        name=f"minoff_{n}_{k}_{pidx}_{h}_{tau}_{gid}")

                    # ---- Flow constraints (2m) ----
                    for l in all_lines:
                        lid = l["id"]
                        fkey = (n, k, pidx, h, lid)
                        is_cand_line = not l["existing"]

                        if is_cand_line:
                            cap_expr_fwd = l["flow_limit_MW"] * kappa_line[n, lid]
                            cap_expr_bwd = l["flow_limit_MW"] * kappa_line[n, lid]
                        else:
                            cap_expr_fwd = l["flow_limit_MW"]
                            cap_expr_bwd = l["flow_limit_MW"]

                        model.addConstr(
                            f_fwd[fkey] <= cap_expr_fwd,
                            name=f"fcap_fwd_{n}_{k}_{pidx}_{h}_{lid}")
                        model.addConstr(
                            f_bwd[fkey] <= cap_expr_bwd,
                            name=f"fcap_bwd_{n}_{k}_{pidx}_{h}_{lid}")

                    # ---- Storage constraints (2i)-(2k) ----
                    for s in all_storage:
                        sid = s["id"]
                        skey = (n, k, pidx, h, sid)
                        eff = s["efficiency"]
                        is_cand_stor = not s["existing"]

                        # (2i) Storage dynamics
                        if h == 0:
                            # r^1 = 0
                            model.addConstr(
                                r_stor[skey] == 0,
                                name=f"stor_init_{n}_{k}_{pidx}_{sid}")
                        else:
                            prev_skey = (n, k, pidx, h - 1, sid)
                            model.addConstr(
                                r_stor[skey] == r_stor[prev_skey]
                                + eff * v_stor[prev_skey]
                                - u_stor[prev_skey],
                                name=f"stor_dyn_{n}_{k}_{pidx}_{h}_{sid}")

                        # (2j) Withdrawal <= stored energy
                        model.addConstr(
                            u_stor[skey] <= r_stor[skey],
                            name=f"stor_with_{n}_{k}_{pidx}_{h}_{sid}")

                        # (2k) Storage capacity limit
                        if is_cand_stor:
                            model.addConstr(
                                r_stor[skey] <= s["capacity_MW"] * kappa_stor[n, sid],
                                name=f"stor_cap_{n}_{k}_{pidx}_{h}_{sid}")
                            # Also limit charge/discharge to capacity
                            model.addConstr(
                                u_stor[skey] <= s["capacity_MW"] * kappa_stor[n, sid],
                                name=f"stor_dis_cap_{n}_{k}_{pidx}_{h}_{sid}")
                            model.addConstr(
                                v_stor[skey] <= s["capacity_MW"] * kappa_stor[n, sid],
                                name=f"stor_chg_cap_{n}_{k}_{pidx}_{h}_{sid}")
                        else:
                            model.addConstr(
                                r_stor[skey] <= s["capacity_MW"],
                                name=f"stor_cap_{n}_{k}_{pidx}_{h}_{sid}")
                            model.addConstr(
                                u_stor[skey] <= s["capacity_MW"],
                                name=f"stor_dis_cap_{n}_{k}_{pidx}_{h}_{sid}")
                            model.addConstr(
                                v_stor[skey] <= s["capacity_MW"],
                                name=f"stor_chg_cap_{n}_{k}_{pidx}_{h}_{sid}")

                    # ---- Power balance (2l) ----
                    for b in buses:
                        demand_b = (load_by_bus.get(b, 0.0) * growth
                                    * load_profile[h] * demand_mult[h])

                        # Generation at bus b
                        gen_expr = gp.LinExpr()
                        for g in gens_at_bus[b]:
                            gen_expr += p_gen[(n, k, pidx, h, g["id"])]

                        # Storage at bus b: withdrawal - injection
                        stor_expr = gp.LinExpr()
                        for s in storage_at_bus[b]:
                            sid = s["id"]
                            stor_expr += u_stor[(n, k, pidx, h, sid)]
                            stor_expr -= v_stor[(n, k, pidx, h, sid)]

                        # Net flow into bus b
                        flow_expr = gp.LinExpr()
                        for l in lines_at_bus[b]:
                            lid = l["id"]
                            fb = l["from_bus"]
                            tb = l["to_bus"]
                            loss = l["loss_factor"]
                            fkey = (n, k, pidx, h, lid)
                            if tb == b:
                                # Inflow via forward direction
                                flow_expr += (1.0 - loss) * f_fwd[fkey]
                                # Outflow via backward direction
                                flow_expr -= f_bwd[fkey]
                            else:
                                # fb == b
                                # Outflow via forward direction
                                flow_expr -= f_fwd[fkey]
                                # Inflow via backward direction
                                flow_expr += (1.0 - loss) * f_bwd[fkey]

                        model.addConstr(
                            gen_expr + stor_expr + flow_expr == demand_b,
                            name=f"balance_{n}_{k}_{pidx}_{h}_{b}")

                    # ---- Spinning reserve (2g) ----
                    for b in buses:
                        demand_b = (load_by_bus.get(b, 0.0) * growth
                                    * load_profile[h] * demand_mult[h])
                        sr_req = SR_pct * demand_b

                        sr_expr = gp.LinExpr()
                        for g in gens_at_bus[b]:
                            sr_expr += s_res[(n, k, pidx, h, g["id"])]

                        model.addConstr(
                            sr_expr >= sr_req,
                            name=f"spin_res_{n}_{k}_{pidx}_{h}_{b}")

    print("Constraints added. Building objective...")

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================

    obj = gp.LinExpr()

    # --- Investment costs ---
    for n in node_ids:
        node = node_by_id[n]
        pi_n = node["probability"]
        cm = node_cost_mult[n]

        for g in candidate_gens:
            gid = g["id"]
            base_cost = g["capital_cost_per_kW"] * g["Pmax_MW"] * 1000.0
            obj += pi_n * cm * base_cost * x_gen[n, gid]

        for l in candidate_lines:
            lid = l["id"]
            base_cost = l["capital_cost_per_kW"] * l["flow_limit_MW"] * 1000.0
            obj += pi_n * cm * base_cost * x_line[n, lid]

        for s in candidate_storage:
            sid = s["id"]
            base_cost = s["capital_cost_per_kW"] * s["capacity_MW"] * 1000.0
            obj += pi_n * cm * base_cost * x_stor[n, sid]

    # --- Expected operational costs ---
    # For each node n: pi_n * (1/num_rep_days_weight) * (1/S) * sum over k,p of OC
    # Assume each representative day has equal weight: 1/num_rep_days
    # and each lower-level scenario has equal probability: 1/S
    #
    # The quadratic cost GC(p) = a + b*p + c*p^2 is linearized as
    # a + (b + c*Pmax)*p.  This is the secant approximation at p=Pmax
    # and a valid upper bound on the convex quadratic.  The c_cost
    # coefficients are very small relative to b_cost (< 3% error at
    # Pmax), so the approximation is tight.  Linearization allows the
    # barrier solver to handle the LP relaxation, which is critical for
    # solving the 1.5M-variable model in reasonable time.
    for n in node_ids:
        node = node_by_id[n]
        pi_n = node["probability"]
        weight = pi_n / (num_rep_days * num_ll_scenarios)

        for k in day_ids:
            for pidx in ll_scen_ids:
                for h in hours:
                    for g in all_gens:
                        gid = g["id"]
                        key = (n, k, pidx, h, gid)

                        # Startup cost
                        obj += weight * g["startup_cost"] * gamma[key]

                        # Generation cost: a*alpha + (b + c*Pmax)*p
                        # Linearized from: a*alpha + b*p + c*p^2
                        obj += weight * g["a_cost"] * alpha[key]
                        linearized_b = g["b_cost"] + g["c_cost"] * g["Pmax_MW"]
                        obj += weight * linearized_b * p_gen[key]

    model.setObjective(obj, GRB.MINIMIZE)
    model.update()

    print(f"Model built: {model.NumVars} variables, {model.NumConstrs} constraints.")

    return model


def extract_solution(model, instance):
    """Extract solution from solved model."""
    result = {}

    if model.Status == GRB.OPTIMAL:
        result["status"] = "optimal"
        result["objective_value"] = model.ObjVal
    elif model.Status == GRB.TIME_LIMIT:
        result["status"] = "time_limit"
        if model.SolCount > 0:
            result["objective_value"] = model.ObjVal
            result["best_bound"] = model.ObjBound
            result["mip_gap"] = model.MIPGap
        else:
            result["objective_value"] = None
            result["best_bound"] = model.ObjBound if hasattr(model, "ObjBound") else None
    elif model.Status == GRB.INFEASIBLE:
        result["status"] = "infeasible"
        result["objective_value"] = None
    elif model.Status == GRB.UNBOUNDED:
        result["status"] = "unbounded"
        result["objective_value"] = None
    else:
        result["status"] = f"gurobi_status_{model.Status}"
        result["objective_value"] = model.ObjVal if model.SolCount > 0 else None

    result["solve_time_seconds"] = model.Runtime
    result["num_variables"] = model.NumVars
    result["num_constraints"] = model.NumConstrs
    result["node_count"] = int(model.NodeCount) if hasattr(model, "NodeCount") else None

    # Extract investment decisions if solution exists
    if model.SolCount > 0:
        investments = {}
        tree = instance["scenario_tree"]
        candidate_gens = [g for g in instance["system"]["generators"] if not g["existing"]]
        candidate_lines = [l for l in instance["system"]["transmission_lines"] if not l["existing"]]
        candidate_storage = [s for s in instance["system"]["storage"] if not s["existing"]]

        for node in tree:
            n = node["id"]
            node_inv = {}

            gen_decisions = {}
            for g in candidate_gens:
                gid = g["id"]
                var = model.getVarByName(f"x_gen_{n}_{gid}")
                if var is not None:
                    gen_decisions[gid] = round(var.X)
            node_inv["generators"] = gen_decisions

            line_decisions = {}
            for l in candidate_lines:
                lid = l["id"]
                var = model.getVarByName(f"x_line_{n}_{lid}")
                if var is not None:
                    line_decisions[lid] = round(var.X)
            node_inv["lines"] = line_decisions

            stor_decisions = {}
            for s in candidate_storage:
                sid = s["id"]
                var = model.getVarByName(f"x_stor_{n}_{sid}")
                if var is not None:
                    stor_decisions[sid] = round(var.X)
            node_inv["storage"] = stor_decisions

            investments[str(n)] = node_inv

        result["investments"] = investments

        # ---- Operational variables (lower-level dispatch) ------------------
        # Written so feasibility_check.py can verify constraints (2b)-(2o)
        # (commitment, dispatch, ramp, reserve, power balance, flow, storage)
        # against the original problem definition. Keys are
        # ``"{n},{k},{pidx},{h},{var_id}"`` exactly matching the lookup format
        # used by feasibility_check.py.
        sp = instance["scenario_parameters"]
        rep_days = instance["representative_days"]
        ll_scenarios = instance["lower_level_scenarios"]
        generators = instance["system"]["generators"]
        lines = instance["system"]["transmission_lines"]
        storage_units = instance["system"]["storage"]

        num_hours = sp["hours_per_day"]
        num_ll = sp["num_lower_level_scenarios_per_day"]
        node_ids = [node["id"] for node in tree]
        day_ids = [d["id"] for d in rep_days]

        alpha_out, gamma_out, p_gen_out, s_res_out = {}, {}, {}, {}
        f_fwd_out, f_bwd_out = {}, {}
        u_stor_out, v_stor_out, r_stor_out = {}, {}, {}
        for n in node_ids:
            for k in day_ids:
                for p in range(num_ll):
                    for h in range(num_hours):
                        for g in generators:
                            gid = g["id"]
                            key = f"{n},{k},{p},{h},{gid}"
                            v = model.getVarByName(f"alpha_{n}_{k}_{p}_{h}_{gid}")
                            if v is not None:
                                alpha_out[key] = round(v.X)
                            v = model.getVarByName(f"gamma_{n}_{k}_{p}_{h}_{gid}")
                            if v is not None:
                                gamma_out[key] = round(v.X)
                            v = model.getVarByName(f"p_{n}_{k}_{p}_{h}_{gid}")
                            if v is not None:
                                p_gen_out[key] = float(v.X)
                            v = model.getVarByName(f"s_{n}_{k}_{p}_{h}_{gid}")
                            if v is not None:
                                s_res_out[key] = float(v.X)
                        for l in lines:
                            lid = l["id"]
                            key = f"{n},{k},{p},{h},{lid}"
                            v = model.getVarByName(f"f_fwd_{n}_{k}_{p}_{h}_{lid}")
                            if v is not None:
                                f_fwd_out[key] = float(v.X)
                            v = model.getVarByName(f"f_bwd_{n}_{k}_{p}_{h}_{lid}")
                            if v is not None:
                                f_bwd_out[key] = float(v.X)
                        for s in storage_units:
                            sid = s["id"]
                            key = f"{n},{k},{p},{h},{sid}"
                            v = model.getVarByName(f"u_{n}_{k}_{p}_{h}_{sid}")
                            if v is not None:
                                u_stor_out[key] = float(v.X)
                            v = model.getVarByName(f"v_{n}_{k}_{p}_{h}_{sid}")
                            if v is not None:
                                v_stor_out[key] = float(v.X)
                            v = model.getVarByName(f"r_{n}_{k}_{p}_{h}_{sid}")
                            if v is not None:
                                r_stor_out[key] = float(v.X)
        result["alpha"] = alpha_out
        result["gamma"] = gamma_out
        result["p_gen"] = p_gen_out
        result["s_res"] = s_res_out
        result["f_fwd"] = f_fwd_out
        result["f_bwd"] = f_bwd_out
        result["u_stor"] = u_stor_out
        result["v_stor"] = v_stor_out
        result["r_stor"] = r_stor_out

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MM-SMIP Power System Capacity Expansion via Gurobi (monolithic)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json",
                        help="Path for output solution JSON")
    parser.add_argument("--time_limit", type=int, default=None,
                        help="Solver time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print(f"Loading instance from {args.instance_path}")
    instance = load_instance(args.instance_path)

    print("Building model...")
    t0 = time.time()
    model = build_model(instance, time_limit=args.time_limit)
    build_time = time.time() - t0
    print(f"Model built in {build_time:.1f} seconds.")

    print("Solving...")
    model.optimize()

    print("Extracting solution...")
    solution = extract_solution(model, instance)
    solution["build_time_seconds"] = build_time
    solution["instance_path"] = args.instance_path

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.2f}")
    print(f"Status: {solution['status']}")


if __name__ == "__main__":
    main()
