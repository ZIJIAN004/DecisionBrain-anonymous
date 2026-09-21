#!/usr/bin/env python3
"""
Feasibility checker for the MM-SMIP Power System Capacity Expansion model.

Checks all hard constraints from the mathematical formulation (Taninmis et al., 2022):
  Upper-level: (1b)-(1g)  ->  Constraints 1-6
  Lower-level: (2b)-(2o)  ->  Constraints 7-20
  Objective consistency   ->  Constraint 21 (envelope check: investment cost is
      exact from first-stage variables; operational cost is bounded
      [ops_LB, ops_UB] using demand, renewable capacity and per-MWh variable
      cost ranges, since second-stage operational variables may be absent)

Usage:
  python feasibility_check.py \
      --instance_path instance_1.json \
      --solution_path efficient_solution_1.json \
      --result_path efficient_feasi_result_1.json
"""

import argparse
import json
import math

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def safe_float(val):
    """Convert to float, replacing inf/nan with large finite values for JSON compatibility."""
    v = float(val)
    if math.isinf(v):
        return 1e30 if v > 0 else -1e30
    if math.isnan(v):
        return 0.0
    return v


def record_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, raw_excess):
    """Record a single constraint violation."""
    normalizer = max(abs(rhs), EPS)
    ratio = raw_excess / normalizer if not math.isinf(raw_excess) else 1e30
    violations_list.append((constraint_idx, message))
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": safe_float(lhs),
        "rhs": safe_float(rhs),
        "raw_excess": safe_float(raw_excess),
        "normalizer": safe_float(normalizer),
        "ratio": safe_float(ratio),
    })


def _effective_tol(lhs, rhs):
    """Tolerance that accommodates solver numerical noise. Combines an
    absolute floor (1e-4, looser than the prior 1e-5 which rejected
    gurobi MIPGap-level dust) with a relative-magnitude floor for cases
    where the constraint operates on values much larger than 1."""
    ABS_TOL = 1e-4
    REL_TOL = 1e-4  # 0.01% relative slack
    return max(ABS_TOL, REL_TOL * max(abs(lhs), abs(rhs)))


def check_le(lhs, rhs):
    """Check lhs <= rhs. Returns (violated, raw_excess)."""
    excess = lhs - rhs
    return excess > _effective_tol(lhs, rhs), max(excess, 0.0)


def check_ge(lhs, rhs):
    """Check lhs >= rhs. Returns (violated, raw_excess)."""
    excess = rhs - lhs
    return excess > _effective_tol(lhs, rhs), max(excess, 0.0)


def check_eq(lhs, rhs):
    """Check lhs == rhs. Returns (violated, raw_excess)."""
    excess = abs(lhs - rhs)
    return excess > _effective_tol(lhs, rhs), excess


def build_ancestor_path(node_by_id, nid):
    """Build path from root to node nid."""
    path = []
    cur = nid
    while cur is not None:
        path.append(cur)
        cur = node_by_id[cur]["parent_id"]
    path.reverse()
    return path


def build_demand_growth(tree, sp):
    """Compute demand growth factor for each node."""
    node_by_id = {n["id"]: n for n in tree}
    dg_high = 1.0 + sp["demand_growth_high_percent"]
    dg_low = 1.0 + sp["demand_growth_low_percent"]
    demand_growth = {}
    demand_growth[tree[0]["id"]] = 1.0  # root

    for n in tree:
        nid = n["id"]
        children = n["children"]
        if len(children) >= 2:
            demand_growth[children[0]] = demand_growth[nid] * dg_high
            demand_growth[children[1]] = demand_growth[nid] * dg_low
        elif len(children) == 1:
            demand_growth[children[0]] = demand_growth[nid] * dg_high

    return demand_growth


def build_cost_mult(tree, sp):
    """Cumulative investment-cost multiplier along each node's path.

    Mirrors gurobi_code.py's build_scenario_tree_info: first child increases
    cost by cost_variation_percent, second child decreases it. Root = 1.0.
    """
    cost_var = sp["cost_variation_percent"]
    cost_mult = {tree[0]["id"]: 1.0}
    for n in tree:
        nid = n["id"]
        children = n["children"]
        if len(children) >= 2:
            cost_mult[children[0]] = cost_mult[nid] * (1.0 + cost_var)
            cost_mult[children[1]] = cost_mult[nid] * (1.0 - cost_var)
        elif len(children) == 1:
            cost_mult[children[0]] = cost_mult[nid] * (1.0 + cost_var)
    return cost_mult


def get_effective_pmax(gen, day_data, ll_scen, hour_h):
    """Effective Pmax for a generator considering renewable profiles."""
    pmax = gen["Pmax_MW"]
    gtype = gen["type"]
    if gtype == "solar":
        return pmax * day_data["solar_profile"][hour_h] * ll_scen["solar_multipliers"][hour_h]
    elif gtype == "wind":
        return pmax * ll_scen["wind_power_factors"][hour_h]
    return pmax


def check_feasibility(instance, solution):
    """Check all hard constraints. Returns (feasible, violated_constraints, violations, violation_magnitudes)."""

    violations_list = []  # list of (constraint_idx, message)
    magnitudes_list = []  # list of magnitude dicts

    # -------------------------------------------------------------------------
    # Extract instance data
    # -------------------------------------------------------------------------
    system = instance["system"]
    sp = instance["scenario_parameters"]
    gp_params = instance["global_parameters"]
    tree = instance["scenario_tree"]
    rep_days = instance["representative_days"]
    ll_scenarios = instance["lower_level_scenarios"]

    generators = system["generators"]
    lines = system["transmission_lines"]
    storage_units = system["storage"]
    loads = system["loads"]
    buses = system["buses"]

    node_by_id = {n["id"]: n for n in tree}
    node_ids = [n["id"] for n in tree]

    existing_gens = [g for g in generators if g["existing"]]
    candidate_gens = [g for g in generators if not g["existing"]]
    candidate_lines = [l for l in lines if not l["existing"]]
    candidate_storage = [s for s in storage_units if not s["existing"]]

    num_hours = sp["hours_per_day"]
    num_rep_days = sp["num_representative_days"]
    num_ll_scenarios = sp["num_lower_level_scenarios_per_day"]

    RM = gp_params["reserve_margin_requirement"]
    SR_pct = gp_params["spinning_reserve_percent"]
    derating = gp_params["derating_factors"]

    load_by_bus = {}
    for ld in loads:
        load_by_bus[ld["bus"]] = load_by_bus.get(ld["bus"], 0.0) + ld["annual_peak_load_MW"]

    day_by_id = {d["id"]: d for d in rep_days}
    day_ids = [d["id"] for d in rep_days]
    hours = list(range(num_hours))
    ll_scen_ids = list(range(num_ll_scenarios))

    demand_growth = build_demand_growth(tree, sp)

    # Generator/storage/line lookups
    gen_by_id = {g["id"]: g for g in generators}
    line_by_id = {l["id"]: l for l in lines}
    stor_by_id = {s["id"]: s for s in storage_units}

    gens_at_bus = {b: [g for g in generators if g["bus"] == b] for b in buses}
    storage_at_bus = {b: [s for s in storage_units if s["bus"] == b] for b in buses}

    # -------------------------------------------------------------------------
    # Extract solution variables
    # -------------------------------------------------------------------------
    investments = solution.get("investments", None)

    # Check if solution has any variable data at all
    has_investments = investments is not None and len(investments) > 0

    # Check for operational variables
    alpha_sol = solution.get("alpha", None)
    gamma_sol = solution.get("gamma", None)
    p_sol = solution.get("p_gen", solution.get("p", None))
    s_sol = solution.get("s_res", solution.get("s", None))
    f_fwd_sol = solution.get("f_fwd", None)
    f_bwd_sol = solution.get("f_bwd", None)
    f_sol = solution.get("f", None)  # alternative flow format
    u_sol = solution.get("u_stor", solution.get("u", None))
    v_sol = solution.get("v_stor", solution.get("v", None))
    r_sol = solution.get("r_stor", solution.get("r", None))

    has_operational = all(x is not None for x in [alpha_sol, p_sol])

    # If solution has no variable data at all, there is nothing to check.
    # Return feasible=None to indicate "no solution available" rather than
    # a genuine feasibility failure.
    if not has_investments and not has_operational:
        status = solution.get("status", None)
        obj = solution.get("objective_value", None)
        if status and ("infeasible" in str(status).lower() or str(status) == "gurobi_status_4"):
            reason = "Solver reported infeasible/unbounded status; no decision variables available."
        elif obj is None and not has_investments:
            reason = "Solution contains no decision variables and null objective; cannot verify feasibility."
        else:
            reason = "Solution contains only an objective value but no decision variable values; cannot verify any constraint."

        return None, [], [reason], []


    # -------------------------------------------------------------------------
    # Build kappa (cumulative capacity) from investment decisions
    # -------------------------------------------------------------------------
    # x_gen[n][gid], x_line[n][lid], x_stor[n][sid] = 0 or 1
    x_gen = {}
    x_line = {}
    x_stor = {}
    kappa_gen = {}
    kappa_line = {}
    kappa_stor = {}

    if has_investments:
        for n in node_ids:
            nstr = str(n)
            node_inv = investments.get(nstr, investments.get(n, {}))
            x_gen[n] = {}
            x_line[n] = {}
            x_stor[n] = {}

            gen_inv = node_inv.get("generators", {})
            for g in candidate_gens:
                gid = g["id"]
                x_gen[n][gid] = gen_inv.get(gid, 0)

            line_inv = node_inv.get("lines", {})
            for l in candidate_lines:
                lid = l["id"]
                x_line[n][lid] = line_inv.get(lid, 0)

            stor_inv = node_inv.get("storage", {})
            for s in candidate_storage:
                sid = s["id"]
                x_stor[n][sid] = stor_inv.get(sid, 0)

        # Compute kappa from x using constraints (1b)-(1d)
        for n in node_ids:
            node = node_by_id[n]
            parent_id = node["parent_id"]
            kappa_gen[n] = {}
            kappa_line[n] = {}
            kappa_stor[n] = {}

            for g in candidate_gens:
                gid = g["id"]
                if parent_id is None:
                    kappa_gen[n][gid] = x_gen[n].get(gid, 0)
                else:
                    kappa_gen[n][gid] = kappa_gen[parent_id].get(gid, 0) + x_gen[n].get(gid, 0)

            for l in candidate_lines:
                lid = l["id"]
                if parent_id is None:
                    kappa_line[n][lid] = x_line[n].get(lid, 0)
                else:
                    kappa_line[n][lid] = kappa_line[parent_id].get(lid, 0) + x_line[n].get(lid, 0)

            for s in candidate_storage:
                sid = s["id"]
                if parent_id is None:
                    kappa_stor[n][sid] = x_stor[n].get(sid, 0)
                else:
                    kappa_stor[n][sid] = kappa_stor[parent_id].get(sid, 0) + x_stor[n].get(sid, 0)

    # -------------------------------------------------------------------------
    # Sanity check: trivial all-zero candidate investments + missing operational
    # data + suspiciously low objective. Reserve-margin (1f) alone can pass if
    # existing capacity already covers (1+RM)*peak_demand, so the upper-level
    # checks would otherwise mark such a claim feasible. Refuse it because the
    # operational cost (running existing generators against demand for every
    # hour of every scenario) is implicit in the objective and cannot
    # legitimately be near zero unless demand itself is zero.
    if has_investments and not has_operational:
        all_zero = True
        for n in node_ids:
            for g in candidate_gens:
                if x_gen[n].get(g["id"], 0):
                    all_zero = False
                    break
            if not all_zero:
                break
            for l in candidate_lines:
                if x_line[n].get(l["id"], 0):
                    all_zero = False
                    break
            if not all_zero:
                break
            for s in candidate_storage:
                if x_stor[n].get(s["id"], 0):
                    all_zero = False
                    break
            if not all_zero:
                break

        if all_zero:
            obj = solution.get("objective_value", None)
            try:
                obj_val = float(obj) if obj is not None else None
            except (TypeError, ValueError):
                obj_val = None
            min_var_cost = min(
                (g.get("variable_cost_per_MWh", 0.0) for g in existing_gens
                 if g.get("variable_cost_per_MWh", 0.0) > 0),
                default=0.0,
            )
            total_demand_MW = sum(load_by_bus.get(b, 0.0) for b in buses)
            total_hours = (num_hours * num_rep_days * num_ll_scenarios
                           * len(node_ids))
            cost_lb = min_var_cost * total_demand_MW * total_hours * 0.01
            if obj_val is not None and cost_lb > 0 and obj_val < cost_lb:
                msg = (
                    f"Trivial all-zero candidate investment with objective={obj_val} "
                    f"below 1% of implicit operational cost lower bound={cost_lb:.0f} "
                    f"(min_var_cost={min_var_cost}/MWh × demand={total_demand_MW}MW "
                    f"× hours={total_hours}); operational decisions not provided so "
                    f"feasibility cannot be verified at the lower level."
                )
                violations_list.append((-1, msg))
                magnitudes_list.append({
                    "constraint": -1, "lhs": obj_val, "rhs": cost_lb,
                    "raw_excess": cost_lb - obj_val,
                    "normalizer": max(cost_lb, EPS),
                    "ratio": (cost_lb - obj_val) / max(cost_lb, EPS),
                })

    # =========================================================================
    # CONSTRAINT CHECKS - UPPER LEVEL
    # =========================================================================

    if has_investments:
        # --- Constraint 1: (1b) kappa^n_g = kappa^{p(n)}_g + a_g * x^n_g ---
        # This is an equality constraint. kappa is derived from x, so if we
        # derived kappa ourselves it will always hold. But if the solution
        # provides kappa directly, we should verify. Since the solution only
        # provides x (investments), and we compute kappa from x, constraint 1
        # is automatically satisfied by construction. We still include the
        # check for completeness with the solution-provided kappa if available.
        sol_kappa_gen = solution.get("kappa_gen", None)
        if sol_kappa_gen is not None:
            for n in node_ids:
                node = node_by_id[n]
                parent_id = node["parent_id"]
                for g in candidate_gens:
                    gid = g["id"]
                    a_g = g["Pmax_MW"]
                    nstr = str(n)
                    sol_k = sol_kappa_gen.get(nstr, {}).get(gid, None)
                    if sol_k is not None:
                        parent_k = 0.0
                        if parent_id is not None:
                            pstr = str(parent_id)
                            parent_k = sol_kappa_gen.get(pstr, {}).get(gid, 0.0)
                        expected = parent_k + a_g * x_gen[n].get(gid, 0)
                        violated, excess = check_eq(sol_k, expected)
                        if violated:
                            record_violation(violations_list, magnitudes_list, 1,
                                             f"(1b) Capacity tracking violated for generator {gid} at node {n}: "
                                             f"kappa={sol_k}, expected={expected}",
                                             sol_k, expected, excess)

        # --- Constraint 2: (1c) kappa^n_l = kappa^{p(n)}_l + a_l * x^n_l ---
        # Same logic as above - satisfied by construction from x.

        # --- Constraint 3: (1d) kappa^n_s = kappa^{p(n)}_s + a_s * x^n_s ---
        # Same logic - satisfied by construction from x.

        # --- Constraint 4: (1e) kappa^n_g <= a_g, kappa^n_l <= a_l, kappa^n_s <= a_s ---
        # Since kappa is normalized (bounded by 1 in gurobi code, a_g cancels),
        # check cumulative build <= 1 (at most once).
        for n in node_ids:
            for g in candidate_gens:
                gid = g["id"]
                k_val = kappa_gen[n].get(gid, 0)
                # kappa / a_g <= 1, or equivalently kappa <= a_g
                # In our representation, kappa counts builds (0 or 1 each), so kappa <= 1
                violated, excess = check_le(k_val, 1.0)
                if violated:
                    record_violation(violations_list, magnitudes_list, 4,
                                     f"(1e) Capacity upper bound violated for generator {gid} at node {n}: "
                                     f"cumulative builds={k_val} > 1",
                                     k_val, 1.0, excess)

            for l in candidate_lines:
                lid = l["id"]
                k_val = kappa_line[n].get(lid, 0)
                violated, excess = check_le(k_val, 1.0)
                if violated:
                    record_violation(violations_list, magnitudes_list, 4,
                                     f"(1e) Capacity upper bound violated for line {lid} at node {n}: "
                                     f"cumulative builds={k_val} > 1",
                                     k_val, 1.0, excess)

            for s in candidate_storage:
                sid = s["id"]
                k_val = kappa_stor[n].get(sid, 0)
                violated, excess = check_le(k_val, 1.0)
                if violated:
                    record_violation(violations_list, magnitudes_list, 4,
                                     f"(1e) Capacity upper bound violated for storage {sid} at node {n}: "
                                     f"cumulative builds={k_val} > 1",
                                     k_val, 1.0, excess)

        # --- Constraint 5: (1f) Reserve margin ---
        # sum_{j:J_hat(j)=j_hat} (sum_{g in G_j} DF_g*a_g + sum_{g in G'_j} DF_g*kappa^n_g*a_g)
        #   >= (1 + RM) * PK^n_{j_hat}
        # Single region (all buses), PK = sum of peak loads * demand_growth
        for n in node_ids:
            total_peak = sum(load_by_bus.get(b, 0.0) for b in buses) * demand_growth[n]
            rhs = (1.0 + RM) * total_peak

            # Derated existing capacity
            lhs = sum(derating.get(g["type"], 1.0) * g["Pmax_MW"] for g in existing_gens)
            # Derated candidate capacity
            for g in candidate_gens:
                gid = g["id"]
                df = derating.get(g["type"], 1.0)
                k_val = kappa_gen[n].get(gid, 0)
                lhs += df * g["Pmax_MW"] * k_val

            violated, excess = check_ge(lhs, rhs)
            if violated:
                record_violation(violations_list, magnitudes_list, 5,
                                 f"(1f) Reserve margin violated at node {n}: "
                                 f"derated capacity={lhs:.2f} MW < required={(rhs):.2f} MW",
                                 lhs, rhs, excess)

        # --- Constraint 6: (1g) Binary and non-negativity ---
        for n in node_ids:
            for g in candidate_gens:
                gid = g["id"]
                xv = x_gen[n].get(gid, 0)
                if not (xv == 0 or xv == 1 or abs(xv) < TOL or abs(xv - 1) < TOL):
                    excess = min(abs(xv), abs(xv - 1))
                    record_violation(violations_list, magnitudes_list, 6,
                                     f"(1g) Binary violated for x_gen[{n}][{gid}]={xv}",
                                     xv, round(xv), excess)
                kv = kappa_gen[n].get(gid, 0)
                if kv < -TOL:
                    record_violation(violations_list, magnitudes_list, 6,
                                     f"(1g) Non-negativity violated for kappa_gen[{n}][{gid}]={kv}",
                                     kv, 0.0, abs(kv))

            for l in candidate_lines:
                lid = l["id"]
                xv = x_line[n].get(lid, 0)
                if not (xv == 0 or xv == 1 or abs(xv) < TOL or abs(xv - 1) < TOL):
                    excess = min(abs(xv), abs(xv - 1))
                    record_violation(violations_list, magnitudes_list, 6,
                                     f"(1g) Binary violated for x_line[{n}][{lid}]={xv}",
                                     xv, round(xv), excess)

            for s in candidate_storage:
                sid = s["id"]
                xv = x_stor[n].get(sid, 0)
                if not (xv == 0 or xv == 1 or abs(xv) < TOL or abs(xv - 1) < TOL):
                    excess = min(abs(xv), abs(xv - 1))
                    record_violation(violations_list, magnitudes_list, 6,
                                     f"(1g) Binary violated for x_stor[{n}][{sid}]={xv}",
                                     xv, round(xv), excess)

    # =========================================================================
    # CONSTRAINT CHECKS - LOWER LEVEL
    # =========================================================================

    if has_operational:
        for n in node_ids:
            for k in day_ids:
                day_data = day_by_id[k]
                load_profile = day_data["load_profile"]
                ll_scens = ll_scenarios[str(k)]

                for pidx in ll_scen_ids:
                    scen = ll_scens[pidx]
                    demand_mult = scen["demand_multipliers"]

                    for h in hours:
                        growth = demand_growth[n]

                        for g in generators:
                            gid = g["id"]
                            is_candidate = not g["existing"]
                            key = f"{n},{k},{pidx},{h},{gid}"

                            a_val = float(alpha_sol.get(key, 0.0))
                            g_val = float(gamma_sol.get(key, 0.0)) if gamma_sol else 0.0
                            p_val = float(p_sol.get(key, 0.0)) if p_sol else 0.0
                            s_val = float(s_sol.get(key, 0.0)) if s_sol else 0.0

                            eff_pmax = get_effective_pmax(g, day_data, scen, h)

                            # --- Constraint 7: (2b) alpha <= kappa/a for candidates ---
                            if is_candidate and has_investments:
                                k_frac = kappa_gen[n].get(gid, 0)
                                violated, excess = check_le(a_val, k_frac)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 7,
                                                     f"(2b) Commitment exceeds capacity for {gid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"alpha={a_val} > kappa_frac={k_frac}",
                                                     a_val, k_frac, excess)

                            # --- Constraint 9: (2d) gamma >= alpha_h - alpha_{h-1} ---
                            if h >= 1:
                                prev_key = f"{n},{k},{pidx},{h-1},{gid}"
                                a_prev = float(alpha_sol.get(prev_key, 0.0))
                                rhs_val = a_val - a_prev
                                violated, excess = check_ge(g_val, rhs_val)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 9,
                                                     f"(2d) Startup indicator violated for {gid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"gamma={g_val} < alpha_h - alpha_h-1={rhs_val}",
                                                     g_val, rhs_val, excess)
                            else:
                                # h=0: alpha_{-1} = 0
                                violated, excess = check_ge(g_val, a_val)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 9,
                                                     f"(2d) Startup indicator violated for {gid} at n={n},k={k},p={pidx},h=0: "
                                                     f"gamma={g_val} < alpha={a_val}",
                                                     g_val, a_val, excess)

                            # --- Constraint 10: (2e) p + s <= Pmax*alpha or kappa ---
                            if is_candidate and has_investments:
                                k_frac = kappa_gen[n].get(gid, 0)
                                cap = k_frac * g["Pmax_MW"]
                                # For renewables, also limited by eff_pmax * alpha
                                if g["type"] in ("solar", "wind"):
                                    cap = min(cap, eff_pmax * a_val)
                                else:
                                    cap = eff_pmax * a_val
                            else:
                                cap = eff_pmax * a_val

                            lhs_val = p_val + s_val
                            violated, excess = check_le(lhs_val, cap)
                            if violated:
                                record_violation(violations_list, magnitudes_list, 10,
                                                 f"(2e) Max output violated for {gid} at n={n},k={k},p={pidx},h={h}: "
                                                 f"p+s={lhs_val:.4f} > cap={cap:.4f}",
                                                 lhs_val, cap, excess)

                            # --- Constraint 11: (2f) Pmin*alpha <= p ---
                            pmin_val = g["Pmin_MW"]
                            lhs_val = pmin_val * a_val
                            violated, excess = check_le(lhs_val, p_val)
                            if violated:
                                record_violation(violations_list, magnitudes_list, 11,
                                                 f"(2f) Min output violated for {gid} at n={n},k={k},p={pidx},h={h}: "
                                                 f"Pmin*alpha={lhs_val:.4f} > p={p_val:.4f}",
                                                 lhs_val, p_val, excess)

                            # --- Constraint 13: (2h) Ramp limits ---
                            if h >= 1:
                                prev_key = f"{n},{k},{pidx},{h-1},{gid}"
                                p_prev = float(p_sol.get(prev_key, 0.0))
                                ramp = g["ramp_MW_per_h"]
                                # Ramp up: p_h - p_{h-1} <= RU
                                ru_lhs = p_val - p_prev
                                violated, excess = check_le(ru_lhs, ramp)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 13,
                                                     f"(2h) Ramp-up violated for {gid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"p_h-p_h-1={ru_lhs:.4f} > RU={ramp}",
                                                     ru_lhs, ramp, excess)
                                # Ramp down: p_{h-1} - p_h <= RD
                                rd_lhs = p_prev - p_val
                                violated, excess = check_le(rd_lhs, ramp)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 13,
                                                     f"(2h) Ramp-down violated for {gid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"p_h-1-p_h={rd_lhs:.4f} > RD={ramp}",
                                                     rd_lhs, ramp, excess)

                        # --- Constraint 8: (2c) Min up/down time ---
                        for g in generators:
                            gid = g["id"]
                            min_on = g["min_on_h"]
                            min_off = g["min_off_h"]

                            if h >= 1:
                                key_h = f"{n},{k},{pidx},{h},{gid}"
                                key_hm1 = f"{n},{k},{pidx},{h-1},{gid}"
                                a_h = float(alpha_sol.get(key_h, 0.0))
                                a_hm1 = float(alpha_sol.get(key_hm1, 0.0))
                                startup = a_h - a_hm1  # positive if started

                                # Min up time: if started at h, must stay on for min_on hours
                                if min_on > 0 and startup > TOL:
                                    for tau in range(h, min(h + min_on, num_hours)):
                                        key_tau = f"{n},{k},{pidx},{tau},{gid}"
                                        a_tau = float(alpha_sol.get(key_tau, 0.0))
                                        # alpha_tau >= alpha_h - alpha_{h-1}
                                        violated, excess = check_ge(a_tau, startup)
                                        if violated:
                                            record_violation(violations_list, magnitudes_list, 8,
                                                             f"(2c) Min up-time violated for {gid} at n={n},k={k},p={pidx}: "
                                                             f"started at h={h}, alpha[{tau}]={a_tau:.4f} < startup_diff={startup:.4f}",
                                                             a_tau, startup, excess)

                                # Min down time: if shut down at h, must stay off for min_off hours
                                shutdown = a_hm1 - a_h  # positive if shut down
                                if min_off > 0 and shutdown > TOL:
                                    for tau in range(h, min(h + min_off, num_hours)):
                                        key_tau = f"{n},{k},{pidx},{tau},{gid}"
                                        a_tau = float(alpha_sol.get(key_tau, 0.0))
                                        # 1 - alpha_tau >= shutdown
                                        lhs_val = 1.0 - a_tau
                                        violated, excess = check_ge(lhs_val, shutdown)
                                        if violated:
                                            record_violation(violations_list, magnitudes_list, 8,
                                                             f"(2c) Min down-time violated for {gid} at n={n},k={k},p={pidx}: "
                                                             f"shutdown at h={h}, 1-alpha[{tau}]={lhs_val:.4f} < shutdown_diff={shutdown:.4f}",
                                                             lhs_val, shutdown, excess)

                        # --- Constraint 12: (2g) Spinning reserve ---
                        for b in buses:
                            demand_b = load_by_bus.get(b, 0.0) * growth * load_profile[h] * demand_mult[h]
                            sr_req = SR_pct * demand_b

                            sr_total = 0.0
                            for g in gens_at_bus[b]:
                                key = f"{n},{k},{pidx},{h},{g['id']}"
                                sr_total += float(s_sol.get(key, 0.0)) if s_sol else 0.0

                            violated, excess = check_ge(sr_total, sr_req)
                            if violated:
                                record_violation(violations_list, magnitudes_list, 12,
                                                 f"(2g) Spinning reserve violated at bus {b}, n={n},k={k},p={pidx},h={h}: "
                                                 f"reserve={sr_total:.4f} < required={sr_req:.4f}",
                                                 sr_total, sr_req, excess)

                        # --- Constraint 14: (2i) Storage dynamics ---
                        for s in storage_units:
                            sid = s["id"]
                            eff = s["efficiency"]
                            key = f"{n},{k},{pidx},{h},{sid}"
                            r_val = float(r_sol.get(key, 0.0)) if r_sol else 0.0
                            u_val = float(u_sol.get(key, 0.0)) if u_sol else 0.0
                            v_val = float(v_sol.get(key, 0.0)) if v_sol else 0.0

                            if h == 0:
                                # r^1 = 0
                                violated, excess = check_eq(r_val, 0.0)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 14,
                                                     f"(2i) Initial storage violated for {sid} at n={n},k={k},p={pidx}: "
                                                     f"r[0]={r_val} != 0",
                                                     r_val, 0.0, excess)
                            else:
                                prev_key = f"{n},{k},{pidx},{h-1},{sid}"
                                r_prev = float(r_sol.get(prev_key, 0.0)) if r_sol else 0.0
                                v_prev = float(v_sol.get(prev_key, 0.0)) if v_sol else 0.0
                                u_prev = float(u_sol.get(prev_key, 0.0)) if u_sol else 0.0
                                expected = r_prev + eff * v_prev - u_prev
                                violated, excess = check_eq(r_val, expected)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 14,
                                                     f"(2i) Storage dynamics violated for {sid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"r={r_val:.4f} != expected={expected:.4f}",
                                                     r_val, expected, excess)

                        # --- Constraint 15: (2j) u <= r ---
                        for s in storage_units:
                            sid = s["id"]
                            key = f"{n},{k},{pidx},{h},{sid}"
                            r_val = float(r_sol.get(key, 0.0)) if r_sol else 0.0
                            u_val = float(u_sol.get(key, 0.0)) if u_sol else 0.0
                            violated, excess = check_le(u_val, r_val)
                            if violated:
                                record_violation(violations_list, magnitudes_list, 15,
                                                 f"(2j) Withdrawal exceeds stored energy for {sid} at n={n},k={k},p={pidx},h={h}: "
                                                 f"u={u_val:.4f} > r={r_val:.4f}",
                                                 u_val, r_val, excess)

                        # --- Constraint 16: (2k) r <= kappa_s (candidate storage) ---
                        if has_investments:
                            for s in candidate_storage:
                                sid = s["id"]
                                key = f"{n},{k},{pidx},{h},{sid}"
                                r_val = float(r_sol.get(key, 0.0)) if r_sol else 0.0
                                cap = s["capacity_MW"] * kappa_stor[n].get(sid, 0)
                                violated, excess = check_le(r_val, cap)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 16,
                                                     f"(2k) Storage capacity violated for {sid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"r={r_val:.4f} > cap={cap:.4f}",
                                                     r_val, cap, excess)

                        # --- Constraint 17: (2l) Power balance ---
                        for b in buses:
                            demand_b = load_by_bus.get(b, 0.0) * growth * load_profile[h] * demand_mult[h]

                            # Generation
                            gen_sum = 0.0
                            for g in gens_at_bus[b]:
                                key = f"{n},{k},{pidx},{h},{g['id']}"
                                gen_sum += float(p_sol.get(key, 0.0))

                            # Storage: u - v
                            stor_sum = 0.0
                            for s in storage_at_bus[b]:
                                sid = s["id"]
                                key = f"{n},{k},{pidx},{h},{sid}"
                                u_val = float(u_sol.get(key, 0.0)) if u_sol else 0.0
                                v_val = float(v_sol.get(key, 0.0)) if v_sol else 0.0
                                stor_sum += u_val - v_val

                            # Flow: need to handle both f_fwd/f_bwd and generic f formats
                            flow_sum = 0.0
                            for l in lines:
                                lid = l["id"]
                                loss = l["loss_factor"]
                                fkey = f"{n},{k},{pidx},{h},{lid}"

                                if f_fwd_sol is not None and f_bwd_sol is not None:
                                    fwd = float(f_fwd_sol.get(fkey, 0.0))
                                    bwd = float(f_bwd_sol.get(fkey, 0.0))
                                    if l["to_bus"] == b:
                                        flow_sum += (1.0 - loss) * fwd - bwd
                                    elif l["from_bus"] == b:
                                        flow_sum += (1.0 - loss) * bwd - fwd
                                elif f_sol is not None:
                                    # Signed flow: positive = from_bus -> to_bus
                                    f_val = float(f_sol.get(fkey, 0.0))
                                    if l["to_bus"] == b:
                                        flow_sum += (1.0 - loss) * f_val
                                    elif l["from_bus"] == b:
                                        flow_sum -= f_val

                            lhs_val = gen_sum + stor_sum + flow_sum
                            violated, excess = check_eq(lhs_val, demand_b)
                            if violated:
                                record_violation(violations_list, magnitudes_list, 17,
                                                 f"(2l) Power balance violated at bus {b}, n={n},k={k},p={pidx},h={h}: "
                                                 f"supply={lhs_val:.4f} != demand={demand_b:.4f}",
                                                 lhs_val, demand_b, excess)

                        # --- Constraint 18: (2m) Flow limits ---
                        for l in lines:
                            lid = l["id"]
                            is_cand = not l["existing"]
                            fkey = f"{n},{k},{pidx},{h},{lid}"

                            if is_cand and has_investments:
                                cap = l["flow_limit_MW"] * kappa_line[n].get(lid, 0)
                            else:
                                cap = l["flow_limit_MW"]

                            if f_fwd_sol is not None and f_bwd_sol is not None:
                                fwd = float(f_fwd_sol.get(fkey, 0.0))
                                bwd = float(f_bwd_sol.get(fkey, 0.0))
                                violated, excess = check_le(fwd, cap)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 18,
                                                     f"(2m) Forward flow limit violated for {lid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"f_fwd={fwd:.4f} > cap={cap:.4f}",
                                                     fwd, cap, excess)
                                violated, excess = check_le(bwd, cap)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 18,
                                                     f"(2m) Backward flow limit violated for {lid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"f_bwd={bwd:.4f} > cap={cap:.4f}",
                                                     bwd, cap, excess)
                            elif f_sol is not None:
                                f_val = float(f_sol.get(fkey, 0.0))
                                violated, excess = check_le(abs(f_val), cap)
                                if violated:
                                    record_violation(violations_list, magnitudes_list, 18,
                                                     f"(2m) Flow limit violated for {lid} at n={n},k={k},p={pidx},h={h}: "
                                                     f"|f|={abs(f_val):.4f} > cap={cap:.4f}",
                                                     abs(f_val), cap, excess)

                        # --- Constraint 19: (2n) Binary constraints ---
                        for g in generators:
                            gid = g["id"]
                            key = f"{n},{k},{pidx},{h},{gid}"
                            a_val = float(alpha_sol.get(key, 0.0))
                            if not (abs(a_val) < TOL or abs(a_val - 1.0) < TOL):
                                excess = min(abs(a_val), abs(a_val - 1.0))
                                record_violation(violations_list, magnitudes_list, 19,
                                                 f"(2n) Binary violated for alpha[{gid}] at n={n},k={k},p={pidx},h={h}: "
                                                 f"alpha={a_val}",
                                                 a_val, round(a_val), excess)
                            if gamma_sol:
                                g_val = float(gamma_sol.get(key, 0.0))
                                if not (abs(g_val) < TOL or abs(g_val - 1.0) < TOL):
                                    excess = min(abs(g_val), abs(g_val - 1.0))
                                    record_violation(violations_list, magnitudes_list, 19,
                                                     f"(2n) Binary violated for gamma[{gid}] at n={n},k={k},p={pidx},h={h}: "
                                                     f"gamma={g_val}",
                                                     g_val, round(g_val), excess)

                        # --- Constraint 20: (2o) Non-negativity ---
                        for g in generators:
                            gid = g["id"]
                            key = f"{n},{k},{pidx},{h},{gid}"
                            p_val = float(p_sol.get(key, 0.0))
                            if p_val < -TOL:
                                record_violation(violations_list, magnitudes_list, 20,
                                                 f"(2o) Non-negativity violated for p[{gid}] at n={n},k={k},p={pidx},h={h}: p={p_val}",
                                                 p_val, 0.0, abs(p_val))
                            if s_sol:
                                s_val = float(s_sol.get(key, 0.0))
                                if s_val < -TOL:
                                    record_violation(violations_list, magnitudes_list, 20,
                                                     f"(2o) Non-negativity violated for s[{gid}] at n={n},k={k},p={pidx},h={h}: s={s_val}",
                                                     s_val, 0.0, abs(s_val))

                        for s in storage_units:
                            sid = s["id"]
                            key = f"{n},{k},{pidx},{h},{sid}"
                            if r_sol:
                                r_val = float(r_sol.get(key, 0.0))
                                if r_val < -TOL:
                                    record_violation(violations_list, magnitudes_list, 20,
                                                     f"(2o) Non-negativity violated for r[{sid}] at n={n},k={k},p={pidx},h={h}: r={r_val}",
                                                     r_val, 0.0, abs(r_val))
                            if u_sol:
                                u_val = float(u_sol.get(key, 0.0))
                                if u_val < -TOL:
                                    record_violation(violations_list, magnitudes_list, 20,
                                                     f"(2o) Non-negativity violated for u[{sid}] at n={n},k={k},p={pidx},h={h}: u={u_val}",
                                                     u_val, 0.0, abs(u_val))
                            if v_sol:
                                v_val = float(v_sol.get(key, 0.0))
                                if v_val < -TOL:
                                    record_violation(violations_list, magnitudes_list, 20,
                                                     f"(2o) Non-negativity violated for v[{sid}] at n={n},k={k},p={pidx},h={h}: v={v_val}",
                                                     v_val, 0.0, abs(v_val))

                        for l in lines:
                            lid = l["id"]
                            fkey = f"{n},{k},{pidx},{h},{lid}"
                            if f_fwd_sol:
                                fwd = float(f_fwd_sol.get(fkey, 0.0))
                                if fwd < -TOL:
                                    record_violation(violations_list, magnitudes_list, 20,
                                                     f"(2o) Non-negativity violated for f_fwd[{lid}] at n={n},k={k},p={pidx},h={h}: f={fwd}",
                                                     fwd, 0.0, abs(fwd))
                            if f_bwd_sol:
                                bwd = float(f_bwd_sol.get(fkey, 0.0))
                                if bwd < -TOL:
                                    record_violation(violations_list, magnitudes_list, 20,
                                                     f"(2o) Non-negativity violated for f_bwd[{lid}] at n={n},k={k},p={pidx},h={h}: f={bwd}",
                                                     bwd, 0.0, abs(bwd))

    # =========================================================================
    # CONSTRAINT 21: Objective consistency (Tier C anti-exploit envelope)
    # =========================================================================
    # The solution carries first-stage investment variables only -- second-stage
    # dispatch variables (alpha/gamma/p/s/...) may be absent. So we cannot fully
    # recompute the objective. Instead bound it:
    #   obj_LB = inv_exact + ops_LB
    #   obj_UB = inv_exact + ops_UB
    # where inv_exact is computed from x_*, and ops_{LB,UB} use demand,
    # renewable capacity, and per-MWh variable-cost ranges. Reject if
    # reported_obj falls outside [obj_LB, obj_UB] beyond tolerance. This catches
    # both the obj=0 (under-report) and obj=MAX_FLOAT (over-report) exploits.
    if has_investments:
        reported_obj_raw = solution.get("objective_value", None)
        try:
            reported_obj = float(reported_obj_raw) if reported_obj_raw is not None else None
        except (TypeError, ValueError):
            reported_obj = None
        if reported_obj is not None and math.isfinite(reported_obj) is False:
            # NaN/+-inf as reported: treat as infinite magnitude, always rejected.
            # Keep reported_obj for the message.
            pass

        if reported_obj is not None:
            # ---- Investment cost (EXACT from x_gen / x_line / x_stor) -----
            cost_var = sp.get("cost_variation_percent", 0.0)
            cost_mult = build_cost_mult(tree, sp)
            inv_cost_exact = 0.0
            for n in node_ids:
                node = node_by_id[n]
                pi_n = float(node.get("probability", 1.0))
                cm = cost_mult.get(n, 1.0)
                for g in candidate_gens:
                    gid = g["id"]
                    base = g["capital_cost_per_kW"] * g["Pmax_MW"] * 1000.0
                    inv_cost_exact += pi_n * cm * base * x_gen[n].get(gid, 0)
                for l in candidate_lines:
                    lid = l["id"]
                    base = l["capital_cost_per_kW"] * l["flow_limit_MW"] * 1000.0
                    inv_cost_exact += pi_n * cm * base * x_line[n].get(lid, 0)
                for s in candidate_storage:
                    sid = s["id"]
                    base = s["capital_cost_per_kW"] * s["capacity_MW"] * 1000.0
                    inv_cost_exact += pi_n * cm * base * x_stor[n].get(sid, 0)

            # ---- Operational cost lower bound ------------------------------
            # Per (n,k,p,h): demand D_{nkph} must be met by generation (>=
            # demand even with losses), of which at most R_{nkph} can come
            # from renewables (b=c=0). The non-renewable portion D-R costs at
            # least min_b_nonrenew per MWh, where min_b_nonrenew is the min
            # linearized marginal cost (b + c*Pmax) over available
            # non-renewable generators (existing + built candidates at node n).
            # If only renewables exist at node n, min_b_nonrenew defaults to 0
            # and the LB collapses to zero -- correct (no marginal cost floor).
            nonrenew_marginal_global = [
                g["b_cost"] + g["c_cost"] * g["Pmax_MW"]
                for g in generators if g["type"] not in ("solar", "wind")
            ]
            ops_lb = 0.0
            ops_ub = 0.0
            for n in node_ids:
                node = node_by_id[n]
                pi_n = float(node.get("probability", 1.0))
                weight = pi_n / max(1, num_rep_days * num_ll_scenarios)
                growth = demand_growth[n]
                total_peak = sum(load_by_bus.get(b, 0.0) for b in buses)

                # Available non-renewable generators at this node = existing
                # non-renewables + built candidate non-renewables.
                avail_nonrenew_marg = [
                    g["b_cost"] + g["c_cost"] * g["Pmax_MW"]
                    for g in existing_gens if g["type"] not in ("solar", "wind")
                ]
                for g in candidate_gens:
                    if g["type"] in ("solar", "wind"):
                        continue
                    if kappa_gen[n].get(g["id"], 0) >= 1 - TOL:
                        avail_nonrenew_marg.append(
                            g["b_cost"] + g["c_cost"] * g["Pmax_MW"])
                min_b_nonrenew = min(avail_nonrenew_marg) if avail_nonrenew_marg else 0.0

                # Built/existing renewable units at this node, by type.
                avail_renew = list(existing_gens)  # any existing renewables
                avail_renew = [g for g in avail_renew if g["type"] in ("solar", "wind")]
                for g in candidate_gens:
                    if g["type"] in ("solar", "wind"):
                        if kappa_gen[n].get(g["id"], 0) >= 1 - TOL:
                            avail_renew.append(g)

                # Per-period max generation cost (for the upper bound). Uses
                # the same linearized cost as gurobi_code.py:
                #   per-hour cost upper bound = sum_g (a_g + (b+c*Pmax)*Pmax_g + SC_g)
                per_period_max = 0.0
                for g in generators:
                    per_period_max += (
                        g["a_cost"]
                        + (g["b_cost"] + g["c_cost"] * g["Pmax_MW"]) * g["Pmax_MW"]
                        + g["startup_cost"]
                    )

                # Iterate (k, p, h) only for the LB; UB uses summed period count.
                for k in day_ids:
                    day_data = day_by_id[k]
                    lp = day_data["load_profile"]
                    solar_profile = day_data["solar_profile"]
                    ll_scens = ll_scenarios[str(k)]
                    for pidx in ll_scen_ids:
                        scen = ll_scens[pidx]
                        dm = scen["demand_multipliers"]
                        sm = scen["solar_multipliers"]
                        wpf = scen["wind_power_factors"]
                        for h in hours:
                            d_h = total_peak * growth * lp[h] * dm[h]
                            # Max renewable supply this hour (full capacity
                            # times available profile/factor).
                            r_h = 0.0
                            for g in avail_renew:
                                if g["type"] == "solar":
                                    r_h += g["Pmax_MW"] * solar_profile[h] * sm[h]
                                elif g["type"] == "wind":
                                    r_h += g["Pmax_MW"] * wpf[h]
                            need_nonrenew = max(0.0, d_h - r_h)
                            ops_lb += weight * min_b_nonrenew * need_nonrenew

                # Operational UB: each (k,p,h) has cost <= per_period_max.
                ops_ub += weight * num_rep_days * num_ll_scenarios * num_hours * per_period_max

            obj_lb = inv_cost_exact + ops_lb
            obj_ub = inv_cost_exact + ops_ub

            # Tolerance: 0.1% relative + 1.0 absolute floor. The LB is already
            # conservative (max renewable supply, no a_cost, no startup cost),
            # and the UB is comically loose (whole fleet at Pmax every hour),
            # so this tolerance only guards against floating-point drift.
            tol_lb = max(1.0, 1e-3 * abs(obj_lb))
            tol_ub = max(1.0, 1e-3 * abs(obj_ub))

            rep_finite = math.isfinite(reported_obj)
            if (not rep_finite) or (reported_obj < obj_lb - tol_lb):
                deficit = (obj_lb - reported_obj) if rep_finite else float("inf")
                record_violation(
                    violations_list, magnitudes_list, 21,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} below lower bound={obj_lb:.6g} "
                    f"(inv_exact={inv_cost_exact:.6g}, ops_LB={ops_lb:.6g}, "
                    f"deficit={deficit:.6g}, tol={tol_lb:.3g})",
                    reported_obj, obj_lb, deficit,
                )
            elif (not rep_finite) or (reported_obj > obj_ub + tol_ub):
                excess = (reported_obj - obj_ub) if rep_finite else float("inf")
                record_violation(
                    violations_list, magnitudes_list, 21,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} above upper bound={obj_ub:.6g} "
                    f"(inv_exact={inv_cost_exact:.6g}, ops_UB={ops_ub:.6g}, "
                    f"excess={excess:.6g}, tol={tol_ub:.3g})",
                    reported_obj, obj_ub, excess,
                )

    # =========================================================================
    # Aggregate results
    # =========================================================================
    violated_set = sorted(set(v[0] for v in violations_list))
    # Aggregate messages per constraint index
    msg_by_idx = {}
    for idx, msg in violations_list:
        if idx not in msg_by_idx:
            msg_by_idx[idx] = []
        msg_by_idx[idx].append(msg)

    # Build concise violation messages (one per constraint index)
    concise_msgs = []
    for idx in violated_set:
        msgs = msg_by_idx[idx]
        if len(msgs) == 1:
            concise_msgs.append(msgs[0])
        else:
            concise_msgs.append(f"{msgs[0]} (and {len(msgs)-1} more similar violations)")

    feasible = len(violated_set) == 0
    return feasible, violated_set, concise_msgs, magnitudes_list


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MM-SMIP Power System Capacity Expansion")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    feasible, violated_constraints, violations, violation_magnitudes = check_feasibility(instance, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if feasible is None:
        status = "NO_SOLUTION"
    elif feasible:
        status = "FEASIBLE"
    else:
        status = "INFEASIBLE"
    print(f"Result: {status}")
    if feasible is not None and not feasible:
        print(f"Violated constraints: {violated_constraints}")
        for msg in violations:
            print(f"  - {msg}")
    elif feasible is None:
        for msg in violations:
            print(f"  - {msg}")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    main()
