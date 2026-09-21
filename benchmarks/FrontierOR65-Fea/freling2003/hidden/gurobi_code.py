"""
Gurobi implementation of the Multiperiod Single-Sourcing Problem (MPSSP)
formulation (P_0) from Freling, Romeijn, Romero Morales, Wagelmans (2003).

minimize  sum_{t,i,j} c_{ijt} * x_{ij}  +  sum_{t,i} h_{it} * I_{it}
subject to
  sigma_t * sum_j d_j * x_{ij} + I_{it} <= b_{it} + I_{i,t-1},   for all i,t   (1)
  sum_i x_{ij} = 1,                                              for all j
  x_{ij} in {0,1},                                               for all i,j
  I_{i0} = 0,                                                    for all i
  I_{it} >= 0,                                                   for all i,t
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
    """Load the MPSSP instance from a JSON file."""
    with open(instance_path, "r") as f:
        data = json.load(f)
    return data


def solve_mpssp(instance_path, solution_path, time_limit):
    """Build and solve the MPSSP using Gurobi."""
    data = load_instance(instance_path)

    m = data["parameters"]["num_facilities"]
    n = data["parameters"]["num_customers"]
    T = data["parameters"]["num_periods"]

    sigma = data["seasonal_factors"]           # length T
    demands = data["demands"]                  # length n  (d_j)
    capacities = data["capacities"]            # m x T     (b_{it})
    holding_costs = data["holding_costs"]      # m x T     (h_{it})
    transport_costs = data["transportation_costs"]  # m x n x T  (c_{ijt})

    # ----------------------------------------------------------------
    # Build the Gurobi model
    # ----------------------------------------------------------------
    model = gp.Model("MPSSP_P0")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # x[i,j] : binary assignment variable
    x = {}
    for i in range(m):
        for j in range(n):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # I[i,t] : inventory at facility i at end of period t (t=1..T, 0-indexed: 0..T-1)
    I = {}
    for i in range(m):
        for t in range(T):
            I[i, t] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"I_{i}_{t}")

    model.update()

    # ----------------------------------------------------------------
    # Objective: min sum_{t,i,j} c_{ijt} * x_{ij} + sum_{t,i} h_{it} * I_{it}
    # ----------------------------------------------------------------
    obj = gp.LinExpr()

    # Transportation costs:  sum_t sum_i sum_j  c_{ijt} * x_{ij}
    for i in range(m):
        for j in range(n):
            cost_ij = sum(transport_costs[i][j][t] for t in range(T))
            obj += cost_ij * x[i, j]

    # Inventory holding costs:  sum_t sum_i  h_{it} * I_{it}
    for i in range(m):
        for t in range(T):
            obj += holding_costs[i][t] * I[i, t]

    model.setObjective(obj, GRB.MINIMIZE)

    # ----------------------------------------------------------------
    # Constraints
    # ----------------------------------------------------------------

    # (1) Capacity / flow balance constraints:
    #   sigma_t * sum_j d_j * x_{ij} + I_{it} <= b_{it} + I_{i,t-1}
    #   with I_{i,0} = 0
    for i in range(m):
        for t in range(T):
            lhs = gp.LinExpr()
            # sigma_t * sum_j d_j * x_{ij}
            for j in range(n):
                lhs += sigma[t] * demands[j] * x[i, j]
            # + I_{it}
            lhs += I[i, t]
            # RHS: b_{it} + I_{i,t-1}
            if t == 0:
                rhs = capacities[i][t]  # I_{i,0} = 0
            else:
                rhs = capacities[i][t] + I[i, t - 1]
            model.addConstr(lhs <= rhs, name=f"cap_{i}_{t}")

    # Assignment constraints: sum_i x_{ij} = 1 for all j
    for j in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(m)) == 1,
            name=f"assign_{j}",
        )

    # ----------------------------------------------------------------
    # Optimize
    # ----------------------------------------------------------------
    model.optimize()

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    solution = {}
    if model.SolCount > 0:
        obj_val = model.ObjVal
        solution["objective_value"] = obj_val

        # Extract assignment
        assignment = {}
        for j in range(n):
            for i in range(m):
                if x[i, j].X > 0.5:
                    assignment[str(j)] = i
                    break
            else:
                assignment[str(j)] = -1  # should not happen

        solution["assignment"] = assignment

        # Extract inventory levels
        inventory = {}
        for i in range(m):
            inventory[str(i)] = [I[i, t].X for t in range(T)]
        solution["inventory"] = inventory

        # Solver status info
        solution["status"] = model.Status
        solution["mip_gap"] = model.MIPGap if model.Status != GRB.OPTIMAL else 0.0
        solution["runtime"] = model.Runtime
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["runtime"] = model.Runtime
        solution["message"] = "No feasible solution found within time limit."

    # Write solution
    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if model.SolCount > 0:
        print(f"Objective value: {obj_val}")
    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Solve MPSSP (P_0) with Gurobi."
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the problem instance.",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default="gurobi_solution_1.json",
        help="Path where the final solution JSON file will be written.",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=3600,
        help="Maximum solver runtime in seconds.",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve_mpssp(args.instance_path, args.solution_path, args.time_limit)
