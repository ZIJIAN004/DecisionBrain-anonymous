#!/usr/bin/env python3
"""
Gurobi implementation of the Unit Commitment (UC) mathematical model.

Paper: Knueven, Ostrowski, Watson (2020)
       "On Mixed Integer Programming Formulations for the Unit Commitment Problem"

This program implements the standard 3-binary-variable (3-bin) UC formulation
as the base mathematical model described in the paper. It uses:
  - Binary variables u(g,t), v(g,t), w(g,t) for on/off, startup, shutdown
  - Logical linking constraints
  - Minimum up/down time constraints (Rajan and Takriti 2005 style)
  - Generation limits
  - Startup/shutdown ramping constraints
  - Piecewise linear production costs (disaggregated)
  - Downtime-dependent startup costs (matching formulation)
  - Power balance and reserve constraints

NOT SPECIFIED IN PAPER (SLIDES): The exact constraint forms for min up/down time,
ramping, generation limits, and startup costs are not written out in the slides.
We use the standard forms from the literature referenced in the paper (Rajan-Takriti
2005, Carrion-Arroyo 2006 style) as inferred assumptions.
"""

import argparse
import json
import math
import sys

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve UC problem with Gurobi (standard 3-bin formulation)."
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    return parser.parse_args()


def compute_startup_windows(startup_costs):
    """
    Compute [lo, hi] downtime windows for each startup cost tier.

    The matching formulation assigns a startup to tier k if the unit was offline
    for lo_k to hi_k periods. The last tier (cold) has hi=None (no upper bound).

    Inferred assumption: consecutive tiers partition the downtime axis, where
    lo_k = prev_hi + 1 and hi_k = max_downtime_hours for tier k.
    """
    windows = []
    prev_max = 0
    for tier in startup_costs:
        lo = prev_max + 1
        hi = tier["max_downtime_hours"]  # None for the last (cold) tier
        windows.append((lo, hi))
        if hi is not None:
            prev_max = hi
    return windows


def build_and_solve(data, time_limit):
    """
    Build and solve the standard 3-bin UC MIP using Gurobi.

    Returns the Gurobi model and key variable dictionaries.
    """
    generators = data["generators"]
    demand = data["demand"]
    reserve = data.get("reserve_requirement", [0.0] * len(demand))
    T = len(demand)
    G = len(generators)

    # Pre-process generator parameters
    gen = []
    for gd in generators:
        g = {
            "id": gd["id"],
            "p_max": gd["p_max"],
            "p_min": gd["p_min"],
            "RU": gd["ramp_up_limit"],
            "RD": gd["ramp_down_limit"],
            "SU": gd["startup_ramp_limit"],    # max power at first period online
            "SD": gd["shutdown_ramp_limit"],   # max power at last period online
            "UT": int(gd["min_up_time"]),
            "DT": int(gd["min_down_time"]),
            "NL": gd["no_load_cost"],
            "init": int(gd["initial_status"]),  # >0: on for n periods, <0: off for |n| periods
            "p0": gd["initial_power"],
            "tiers": gd["startup_costs"],
            "segs": gd["production_cost_segments"],
            "windows": compute_startup_windows(gd["startup_costs"]),
        }
        gen.append(g)

    # -------------------------------------------------------------------------
    # Create Gurobi model
    # -------------------------------------------------------------------------
    model = gp.Model("UC_Standard_3bin")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 1e-4)   # 0.01% optimality gap (as in paper)
    model.setParam("OutputFlag", 1)

    # -------------------------------------------------------------------------
    # Decision Variables (indexed t=1..T)
    # -------------------------------------------------------------------------
    # Binary status, startup, shutdown
    u = model.addVars(G, range(1, T + 1), vtype=GRB.BINARY, name="u")
    v = model.addVars(G, range(1, T + 1), vtype=GRB.BINARY, name="v")
    w = model.addVars(G, range(1, T + 1), vtype=GRB.BINARY, name="w")

    # Power output and available power (for reserve)
    p = model.addVars(G, range(1, T + 1), lb=0.0, name="p")
    pb = model.addVars(G, range(1, T + 1), lb=0.0, name="pb")

    # Piecewise segment power pl[g, t, l]
    pl = {}
    for g_idx in range(G):
        n_segs = len(gen[g_idx]["segs"])
        for t in range(1, T + 1):
            for l in range(n_segs):
                pl[(g_idx, t, l)] = model.addVar(lb=0.0, name=f"pl_{g_idx}_{t}_{l}")

    # Startup cost tier indicators delta[g, t, k]
    delta = {}
    for g_idx in range(G):
        n_tiers = len(gen[g_idx]["tiers"])
        for t in range(1, T + 1):
            for k in range(n_tiers):
                delta[(g_idx, t, k)] = model.addVar(lb=0.0, ub=1.0,
                                                      name=f"delta_{g_idx}_{t}_{k}")

    model.update()

    # -------------------------------------------------------------------------
    # Enforce initial up/down time obligations
    # -------------------------------------------------------------------------
    # Inferred assumption: if init_status > 0 (on for n periods), the unit must
    # remain on for max(0, UT - n) more periods; similarly for off.
    for g_idx in range(G):
        g = gen[g_idx]
        init = g["init"]
        if init > 0:
            must_on = max(0, g["UT"] - init)
            for t in range(1, min(must_on, T) + 1):
                u[g_idx, t].lb = 1.0
        elif init < 0:
            must_off = max(0, g["DT"] - abs(init))
            for t in range(1, min(must_off, T) + 1):
                u[g_idx, t].ub = 0.0

    model.update()

    # -------------------------------------------------------------------------
    # Helper: get initial-condition values as Python scalars for t=0
    # -------------------------------------------------------------------------
    def u0(g_idx):
        return 1 if gen[g_idx]["init"] > 0 else 0

    def p0(g_idx):
        return gen[g_idx]["p0"] if gen[g_idx]["init"] > 0 else 0.0

    # -------------------------------------------------------------------------
    # Logical linking: u[g,t] - u[g,t-1] = v[g,t] - w[g,t]
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        for t in range(1, T + 1):
            u_prev = u[g_idx, t - 1] if t > 1 else u0(g_idx)
            model.addConstr(
                u[g_idx, t] - u_prev == v[g_idx, t] - w[g_idx, t],
                name=f"link_{g_idx}_{t}"
            )

    # -------------------------------------------------------------------------
    # Minimum up-time constraints (Rajan-Takriti 2005 style)
    # sum_{s=max(1,t-UT+1)}^{t} v[g,s] <= u[g,t]
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        UT = gen[g_idx]["UT"]
        for t in range(1, T + 1):
            start = max(1, t - UT + 1)
            model.addConstr(
                gp.quicksum(v[g_idx, s] for s in range(start, t + 1)) <= u[g_idx, t],
                name=f"minup_{g_idx}_{t}"
            )

    # -------------------------------------------------------------------------
    # Minimum down-time constraints (Rajan-Takriti 2005 style)
    # sum_{s=max(1,t-DT+1)}^{t} w[g,s] <= 1 - u[g,t]
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        DT = gen[g_idx]["DT"]
        for t in range(1, T + 1):
            start = max(1, t - DT + 1)
            model.addConstr(
                gp.quicksum(w[g_idx, s] for s in range(start, t + 1)) <= 1 - u[g_idx, t],
                name=f"mindn_{g_idx}_{t}"
            )

    # -------------------------------------------------------------------------
    # Generation limits
    # p_min * u[g,t] <= p[g,t] <= pb[g,t] <= p_max * u[g,t]
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        g = gen[g_idx]
        for t in range(1, T + 1):
            model.addConstr(p[g_idx, t] >= g["p_min"] * u[g_idx, t],
                            name=f"pmin_{g_idx}_{t}")
            model.addConstr(pb[g_idx, t] <= g["p_max"] * u[g_idx, t],
                            name=f"pmax_{g_idx}_{t}")
            model.addConstr(p[g_idx, t] <= pb[g_idx, t],
                            name=f"p_le_pb_{g_idx}_{t}")

    # -------------------------------------------------------------------------
    # Ramping constraints with startup/shutdown ramp rates
    # Ramp-up:   p[g,t] - p[g,t-1] <= RU * u[g,t-1] + SU * v[g,t]
    # Ramp-down: p[g,t-1] - p[g,t] <= RD * u[g,t]   + SD * w[g,t]
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        g = gen[g_idx]
        for t in range(1, T + 1):
            p_prev = p[g_idx, t - 1] if t > 1 else p0(g_idx)
            u_prev = u[g_idx, t - 1] if t > 1 else u0(g_idx)

            model.addConstr(
                p[g_idx, t] - p_prev <= g["RU"] * u_prev + g["SU"] * v[g_idx, t],
                name=f"rampup_{g_idx}_{t}"
            )
            model.addConstr(
                p_prev - p[g_idx, t] <= g["RD"] * u[g_idx, t] + g["SD"] * w[g_idx, t],
                name=f"rampdn_{g_idx}_{t}"
            )

    # -------------------------------------------------------------------------
    # Piecewise linear production cost disaggregation
    # sum_l pl[g,t,l] = p[g,t] - p_min * u[g,t]
    # 0 <= pl[g,t,l] <= (P^l - P^{l-1}) * u[g,t]   (standard upper bound)
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        g = gen[g_idx]
        segs = g["segs"]
        n_segs = len(segs)
        for t in range(1, T + 1):
            # Output above minimum = sum of segment outputs
            model.addConstr(
                gp.quicksum(pl[(g_idx, t, l)] for l in range(n_segs))
                == p[g_idx, t] - g["p_min"] * u[g_idx, t],
                name=f"pwsum_{g_idx}_{t}"
            )
            for l, seg in enumerate(segs):
                seg_width = seg["output_mw_end"] - seg["output_mw_start"]
                model.addConstr(
                    pl[(g_idx, t, l)] <= seg_width * u[g_idx, t],
                    name=f"pwub_{g_idx}_{t}_{l}"
                )

    # -------------------------------------------------------------------------
    # Downtime-dependent startup costs (matching formulation)
    #
    # delta[g,t,k] = 1 iff generator g starts at time t with startup tier k.
    # Partition: sum_k delta[g,t,k] = v[g,t]
    # Window:    delta[g,t,k] <= sum_{i=lo_k}^{hi_k} w[g,t-i]  (for k < last tier)
    #            (unit must have shut down in the window [lo_k, hi_k] periods ago)
    #
    # Inferred assumption: matching formulation assigns the cheapest eligible tier.
    # The last (cold) tier is the residual and has no window upper bound.
    # Pre-horizon shutdowns are treated as a constant contribution if they fall
    # within a tier's window (based on initial_status).
    # -------------------------------------------------------------------------
    for g_idx in range(G):
        g = gen[g_idx]
        tiers = g["tiers"]
        windows = g["windows"]
        n_tiers = len(tiers)
        init = g["init"]

        for t in range(1, T + 1):
            # Partition constraint
            model.addConstr(
                gp.quicksum(delta[(g_idx, t, k)] for k in range(n_tiers))
                == v[g_idx, t],
                name=f"tiersum_{g_idx}_{t}"
            )

            # Window upper bounds for all but the last (cold) tier
            for k in range(n_tiers - 1):
                lo, hi = windows[k]
                w_terms = []
                for i in range(lo, hi + 1):
                    prev_t = t - i
                    if prev_t >= 1:
                        w_terms.append(w[g_idx, prev_t])
                    elif prev_t <= 0 and init < 0:
                        # Pre-horizon shutdown: unit was off for |init| periods before t=1.
                        # At period t, that shutdown occurred (t-1) + |init| periods ago.
                        shutdown_ago = (t - 1) + abs(init)
                        if lo <= shutdown_ago <= hi:
                            w_terms.append(1.0)  # constant: pre-horizon shutdown in window

                if w_terms:
                    model.addConstr(
                        delta[(g_idx, t, k)] <= gp.quicksum(w_terms),
                        name=f"tierwin_{g_idx}_{t}_{k}"
                    )
                else:
                    # No valid shutdown in this window: tier k is infeasible at this t
                    model.addConstr(
                        delta[(g_idx, t, k)] == 0,
                        name=f"tierinf_{g_idx}_{t}_{k}"
                    )

    # -------------------------------------------------------------------------
    # Power balance (copper-plate, equality)
    # sum_g p[g,t] = D[t]
    # -------------------------------------------------------------------------
    for t in range(1, T + 1):
        model.addConstr(
            gp.quicksum(p[g_idx, t] for g_idx in range(G)) == demand[t - 1],
            name=f"balance_{t}"
        )

    # -------------------------------------------------------------------------
    # Reserve requirement
    # sum_g pb[g,t] >= D[t] + R[t]
    # -------------------------------------------------------------------------
    for t in range(1, T + 1):
        model.addConstr(
            gp.quicksum(pb[g_idx, t] for g_idx in range(G))
            >= demand[t - 1] + reserve[t - 1],
            name=f"reserve_{t}"
        )

    # -------------------------------------------------------------------------
    # Objective function
    # min sum_g sum_t [ NL_g * u[g,t]
    #                  + sum_l f^l * pl[g,t,l]
    #                  + sum_k SC_k * delta[g,t,k] ]
    # -------------------------------------------------------------------------
    obj_terms = []
    for g_idx in range(G):
        g = gen[g_idx]
        for t in range(1, T + 1):
            obj_terms.append(g["NL"] * u[g_idx, t])
            for l, seg in enumerate(g["segs"]):
                obj_terms.append(seg["marginal_cost_per_mwh"] * pl[(g_idx, t, l)])
            for k, tier in enumerate(g["tiers"]):
                obj_terms.append(tier["cost"] * delta[(g_idx, t, k)])

    model.setObjective(gp.quicksum(obj_terms), GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    model.optimize()

    return model, u, v, w, p, pb, pl, delta, gen, G, T


def extract_solution(model, u, v, w, p, pb, pl, delta, gen, G, T):
    """Extract solution values from Gurobi model."""
    status = model.Status

    # Attempt to get best objective value
    obj_val = None
    if model.SolCount > 0:
        obj_val = model.ObjVal

    solution = {
        "objective_value": obj_val,
        "status": status,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "solve_time": model.Runtime,
        "schedule": {}
    }

    if model.SolCount > 0:
        # Expose only the 1-bin original variables (u, p, pb); v, w are 3-bin
        # internal auxiliaries derivable from u as v(t)=max(0,u(t)-u(t-1)),
        # w(t)=max(0,u(t-1)-u(t)). pl and delta are piecewise/tier auxiliaries.
        for g_idx in range(G):
            g_id = gen[g_idx]["id"]
            solution["schedule"][g_id] = {
                "u": [round(u[g_idx, t].X) for t in range(1, T + 1)],
                "p": [p[g_idx, t].X for t in range(1, T + 1)],
                "pb": [pb[g_idx, t].X for t in range(1, T + 1)],
            }

    return solution


def main():
    args = parse_args()
    install_gurobi_logger(args.log_path)

    with open(args.instance_path) as f:
        data = json.load(f)

    model, u, v, w, p, pb, pl, delta, gen, G, T = build_and_solve(data, args.time_limit)
    solution = extract_solution(model, u, v, w, p, pb, pl, delta, gen, G, T)

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']:.4f}")
        print(f"MIP gap: {solution['mip_gap']:.6%}")
    else:
        print("No feasible solution found within the time limit.")

    print(f"Solution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
