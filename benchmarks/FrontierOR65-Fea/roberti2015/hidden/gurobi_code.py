"""
Gurobi implementation of the Fixed Charge Transportation Problem (FCTP)
using the standard MIP formulation F0 from:

Roberti, Bartolini, Mingozzi (2014)
"The Fixed Charge Transportation Problem: An Exact Algorithm Based on a
New Integer Programming Formulation", Management Science.

Formulation F0:
  min  sum_{i,j} (c_{ij} * x_{ij} + f_{ij} * y_{ij})
  s.t. sum_j x_{ij} = a_i,        for all i in S           (supply)
       sum_i x_{ij} = b_j,        for all j in T           (demand)
       x_{ij} <= m_{ij} * y_{ij}, for all (i,j) in A       (linking)
       x_{ij} >= 0,               for all (i,j) in A
       y_{ij} in {0,1},           for all (i,j) in A

where m_{ij} = min(a_i, b_j).
"""

import argparse
import json
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
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_fctp(data, time_limit):
    m = data["num_sources"]
    n = data["num_sinks"]
    a = data["supply"]
    b = data["demand"]
    f = data["fixed_cost"]
    c = data["variable_cost"]
    cap = data["capacity"]  # m_{ij} = min(a_i, b_j), precomputed

    model = gp.Model("FCTP_F0")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = {}
    y = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(lb=0.0, ub=cap[i][j], vtype=GRB.CONTINUOUS,
                                   name=f"x_{i}_{j}")
            y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    model.update()

    # Objective: min sum (c_ij * x_ij + f_ij * y_ij)
    model.setObjective(
        gp.quicksum(c[i][j] * x[i, j] + f[i][j] * y[i, j]
                     for i in range(m) for j in range(n)),
        GRB.MINIMIZE
    )

    # Supply constraints: sum_j x_{ij} = a_i
    for i in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(n)) == a[i],
            name=f"supply_{i}"
        )

    # Demand constraints: sum_i x_{ij} = b_j
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(m)) == b[j],
            name=f"demand_{j}"
        )

    # Linking constraints: x_{ij} <= m_{ij} * y_{ij}
    for i in range(m):
        for j in range(n):
            model.addConstr(
                x[i, j] <= cap[i][j] * y[i, j],
                name=f"link_{i}_{j}"
            )

    model.optimize()

    # Extract solution
    solution = {}

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        solution["mip_gap"] = model.MIPGap

        # Extract flow and arc usage
        flow = []
        arcs_used = []
        for i in range(m):
            flow_row = []
            arc_row = []
            for j in range(n):
                flow_row.append(x[i, j].X)
                arc_row.append(int(round(y[i, j].X)))
            flow.append(flow_row)
            arcs_used.append(arc_row)

        solution["flow"] = flow
        solution["arcs_used"] = arcs_used
        solution["num_sources"] = m
        solution["num_sinks"] = n
    else:
        solution["objective_value"] = None
        solution["status"] = "infeasible_or_no_solution"

    if model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        solution["status"] = "time_limit_with_feasible"

    return solution


def main():
    parser = argparse.ArgumentParser(description="FCTP Solver using Gurobi (F0 formulation)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    solution = solve_fctp(data, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if solution["objective_value"] is not None:
        print(f"Objective value: {solution['objective_value']}")
        print(f"Status: {solution['status']}")


if __name__ == "__main__":
    main()
