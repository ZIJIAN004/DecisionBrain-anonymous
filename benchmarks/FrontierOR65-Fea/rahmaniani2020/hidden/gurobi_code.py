#!/usr/bin/env python3
"""
Direct Gurobi MIP formulation of the Stochastic Capacitated Facility Location (SFL)
problem from Rahmaniani et al. (2020), Appendix D.2, Equations (D.4)-(D.7).

  SFL := min  sum_i f_i y_i + sum_s sum_i sum_j p_s c_{ij} x_{ij}^s
         s.t. sum_i x_{ij}^s >= d_j^s          for all j in M, s in S     (D.5)
              sum_j x_{ij}^s <= u_i y_i          for all i in N, s in S     (D.6)
              sum_i u_i y_i  >= max_s sum_j d_j^s                           (D.7)
              y_i in {0,1}, x_{ij}^s >= 0
"""

import json
import argparse
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


def main():
    parser = argparse.ArgumentParser(
        description="Solve SFL via direct Gurobi MIP (Rahmaniani et al. 2020)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path for the output solution JSON.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ------------------------------------------------------------------ #
    # Load instance data
    # ------------------------------------------------------------------ #
    with open(args.instance_path, "r") as fh:
        data = json.load(fh)

    num_facilities = data["num_facilities"]          # |N|
    num_customers = data["num_customers"]            # |M|
    num_scenarios = data["num_scenarios"]             # |S|

    f_cost = data["facilities"]["fixed_costs"]        # f_i
    cap = data["facilities"]["capacities"]            # u_i
    c = data["routing_costs"]                         # c[i][j]

    prob = data["scenarios"]["probabilities"]          # p_s
    demand = data["scenarios"]["demands"]              # demand[s][j] = d_j^s

    N = range(num_facilities)
    M = range(num_customers)
    S = range(num_scenarios)

    # max total demand across scenarios (for complete recourse constraint D.7)
    D_max = max(sum(demand[s][j] for j in M) for s in S)

    # ------------------------------------------------------------------ #
    # Build Gurobi model
    # ------------------------------------------------------------------ #
    model = gp.Model("SFL")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)  # single-thread as in the paper

    # Decision variables
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")
    x = model.addVars(num_facilities, num_customers, num_scenarios,
                       lb=0.0, name="x")

    # Objective (D.4)
    model.setObjective(
        gp.quicksum(f_cost[i] * y[i] for i in N)
        + gp.quicksum(
            prob[s] * c[i][j] * x[i, j, s]
            for s in S for i in N for j in M
        ),
        GRB.MINIMIZE,
    )

    # Demand satisfaction (D.5): sum_i x_{ij}^s >= d_j^s
    for j in M:
        for s in S:
            model.addConstr(
                gp.quicksum(x[i, j, s] for i in N) >= demand[s][j],
                name=f"demand_{j}_{s}",
            )

    # Capacity (D.6): sum_j x_{ij}^s <= u_i y_i
    for i in N:
        for s in S:
            model.addConstr(
                gp.quicksum(x[i, j, s] for j in M) <= cap[i] * y[i],
                name=f"capacity_{i}_{s}",
            )

    # Complete recourse (D.7): sum_i u_i y_i >= max_s sum_j d_j^s
    model.addConstr(
        gp.quicksum(cap[i] * y[i] for i in N) >= D_max,
        name="recourse",
    )

    # ------------------------------------------------------------------ #
    # Optimize
    # ------------------------------------------------------------------ #
    model.optimize()

    # ------------------------------------------------------------------ #
    # Extract and save solution
    # ------------------------------------------------------------------ #
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["y"] = [int(round(y[i].X)) for i in N]
        result["status"] = model.Status
        result["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
    else:
        result["objective_value"] = None
        result["status"] = model.Status

    with open(args.solution_path, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"Solution written to {args.solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {result['objective_value']}")


if __name__ == "__main__":
    main()
