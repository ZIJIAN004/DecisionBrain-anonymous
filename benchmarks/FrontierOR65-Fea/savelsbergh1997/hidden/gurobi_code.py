"""
Gurobi implementation of the standard Integer Programming formulation for the
Generalized Assignment Problem (GAP) - Maximization form.

Reference: Savelsbergh (1997), "A Branch-and-Price Algorithm for the Generalized
Assignment Problem", Operations Research 45(6):831-841.

Formulation 1 (Standard IP):
  max  sum_{i,j} p_{ij} x_{ij}
  s.t. sum_{i} x_{ij} = 1,          for all j  (each job assigned to exactly one agent)
       sum_{j} w_{ij} x_{ij} <= c_i, for all i  (capacity constraints)
       x_{ij} in {0, 1}
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
    """Load a GAP instance from a JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    return data


def solve_gap(instance_path, solution_path, time_limit):
    """Solve the GAP using Gurobi with the standard IP formulation."""
    data = load_instance(instance_path)

    m = data["num_agents"]
    n = data["num_jobs"]
    profits = data["profits"]    # profits[i][j]
    weights = data["weights"]    # weights[i][j]
    capacities = data["capacities"]  # capacities[i]

    # Create model
    model = gp.Model("GAP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables: x[i][j] = 1 if job j is assigned to agent i
    x = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}",
                                   obj=profits[i][j])

    model.setAttr("ModelSense", GRB.MAXIMIZE)
    model.update()

    # Constraint (1): Each job is assigned to exactly one agent
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(m)) == 1,
            name=f"assign_{j}"
        )

    # Constraint (2): Capacity constraint for each agent
    for i in range(m):
        model.addConstr(
            gp.quicksum(weights[i][j] * x[i, j] for j in range(n)) <= capacities[i],
            name=f"capacity_{i}"
        )

    # Optimize
    model.optimize()

    # Extract solution
    solution = {
        "problem": data.get("problem", "GAP"),
        "objective": "maximize",
        "num_agents": m,
        "num_jobs": n,
        "solver": "Gurobi",
        "formulation": "Standard IP (Formulation 1)",
    }

    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution["objective_value"] = obj_val
        solution["status"] = model.Status
        solution["status_description"] = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
        }.get(model.Status, f"STATUS_{model.Status}")
        solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        solution["best_bound"] = model.ObjBound

        # Extract assignment
        assignment = {}
        for j in range(n):
            for i in range(m):
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
                    break
        solution["assignment"] = assignment

        # Verify capacity constraints
        agent_loads = [0] * m
        for j_str, i in assignment.items():
            j = int(j_str)
            agent_loads[i] += weights[i][j]
        solution["agent_loads"] = agent_loads
        solution["capacities"] = capacities
        solution["capacity_feasible"] = all(
            agent_loads[i] <= capacities[i] for i in range(m)
        )
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["status_description"] = "NO_SOLUTION_FOUND"

    # Write solution
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {obj_val}")
        print(f"Status: {solution['status_description']}")
    else:
        print("No feasible solution found.")


def main():
    parser = argparse.ArgumentParser(
        description="Solve GAP using Gurobi (Standard IP Formulation)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file.")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_gap(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
