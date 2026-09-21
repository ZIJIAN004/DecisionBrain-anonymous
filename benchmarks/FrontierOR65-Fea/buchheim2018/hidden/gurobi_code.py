#!/usr/bin/env python3
"""
Gurobi implementation of the Quadratic Shortest Path Problem (QSPP)
from Buchheim & Traversi (2018), "Quadratic Combinatorial Optimization
Using Separable Underestimators", INFORMS J. Computing 30(3):424-437.

Model (19):
  min  sum_(a,b in A) Q_ab x_a x_b + sum_(a in A) L_a x_a
  s.t. flow conservation for all intermediate nodes
       source outflow = 1
       sink inflow = 1
       x_a in (0,1) for all a in A
"""

import argparse
import json
import os
import sys
from collections import defaultdict

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


def load_instance(instance_path):
    """Load QSPP instance from JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def build_and_solve(data, time_limit):
    """Build and solve the QSPP model using Gurobi."""
    num_arcs = data["num_arcs"]
    num_nodes = data["num_nodes"]
    source = data["source_node"]
    target = data["target_node"]
    arcs = data["arcs"]
    linear_costs = data["linear_costs"]
    Q = data["quadratic_costs"]

    # Build adjacency: outgoing and incoming arcs for each node
    delta_plus = defaultdict(list)   # outgoing arcs
    delta_minus = defaultdict(list)  # incoming arcs
    for arc in arcs:
        aid = arc["id"]
        delta_plus[arc["from_node"]].append(aid)
        delta_minus[arc["to_node"]].append(aid)

    # Create model
    model = gp.Model("QSPP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # Suppress output for cleaner runs; remove if debugging is needed
    model.setParam("OutputFlag", 1)

    # Decision variables: x_a in {0,1} for each arc a
    x = model.addVars(num_arcs, vtype=GRB.BINARY, name="x")

    # Objective: min sum_{a,b} Q_{ab} x_a x_b + sum_a L_a x_a
    obj = gp.QuadExpr()
    # Linear part
    for a in range(num_arcs):
        obj += linear_costs[a] * x[a]
    # Quadratic part
    for a in range(num_arcs):
        for b in range(a, num_arcs):
            if Q[a][b] != 0:
                if a == b:
                    # x_a^2 = x_a for binary, so diagonal contributes Q[a][a]*x_a
                    # But Q diagonal is 0 per the instance data. Include for generality.
                    obj += Q[a][a] * x[a]
                else:
                    # Q is symmetric: Q[a][b]*x_a*x_b + Q[b][a]*x_b*x_a = 2*Q[a][b]*x_a*x_b
                    # Gurobi expects the combined coefficient for x_a*x_b when a != b
                    obj += (Q[a][b] + Q[b][a]) * x[a] * x[b]

    model.setObjective(obj, GRB.MINIMIZE)

    # Flow conservation constraints
    all_nodes = set(range(num_nodes))
    for i in all_nodes:
        out_arcs = delta_plus.get(i, [])
        in_arcs = delta_minus.get(i, [])
        if i == source:
            # sum_{a in delta+(s)} x_a = 1
            model.addConstr(
                gp.quicksum(x[a] for a in out_arcs) == 1,
                name=f"source_{i}"
            )
        elif i == target:
            # sum_{a in delta-(t)} x_a = 1
            model.addConstr(
                gp.quicksum(x[a] for a in in_arcs) == 1,
                name=f"sink_{i}"
            )
        else:
            # Flow conservation: out - in = 0
            model.addConstr(
                gp.quicksum(x[a] for a in out_arcs)
                - gp.quicksum(x[a] for a in in_arcs) == 0,
                name=f"flow_{i}"
            )

    # Optimize
    model.optimize()

    # Extract solution
    result = {}
    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = model.Status
        result["status_str"] = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }.get(model.Status, str(model.Status))
        result["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        # Record active arcs in the solution
        sol_arcs = []
        for a in range(num_arcs):
            if x[a].X > 0.5:
                sol_arcs.append(arcs[a])
        result["solution_arcs"] = sol_arcs
    else:
        result["objective_value"] = None
        result["status"] = model.Status
        result["status_str"] = "NO_SOLUTION_FOUND"
        result["solution_arcs"] = []

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Solve QSPP using Gurobi (Buchheim & Traversi 2018)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the problem instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path where the final solution JSON file will be written."
    )
    parser.add_argument(
        "--time_limit", type=int, required=True,
        help="Maximum solver runtime in seconds."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    result = build_and_solve(data, args.time_limit)

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
        print(f"Status: {result['status_str']}")
    else:
        print("No feasible solution found within the time limit.")


if __name__ == "__main__":
    main()
