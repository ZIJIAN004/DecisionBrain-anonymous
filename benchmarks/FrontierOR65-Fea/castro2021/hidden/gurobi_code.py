"""
Gurobi implementation of the Minimum Convex Cost Flow in Bipartite Networks (MCCFBN)
problem from Castro & Nasini (2021).

Model (Equations 1-4 from the paper):
    min   sum_{i in I} sum_{j in J} f_{ij}(x_{ij})
    s.t.  sum_{i in I} x_{ij} = d_j,         for all j in J     (demand satisfaction)
          sum_{j in J} x_{ij} <= s_i,         for all i in I     (supply capacity)
          0 <= x_{ij} <= u_{ij},              for all i in I, j in J  (arc bounds)

Cost functions:
    - Linear:    f_{ij}(x) = c_{ij} * x
    - Quadratic: f_{ij}(x) = c_{ij} * x + q_{ij} * x^2
"""

import argparse
import json
import os
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


def build_and_solve(data, time_limit):
    n = data["n"]  # number of supply nodes
    m = data["m"]  # number of demand nodes

    supplies = data["supplies"]
    demands = data["demands"]
    linear_costs = data["linear_costs"]  # n x m matrix
    quadratic_costs = data["quadratic_costs"]  # n x m matrix
    arc_capacity = data["arc_capacity"]  # scalar upper bound for all arcs
    cost_type = data.get("cost_type", "linear_integer")

    # Determine if we have individual arc capacities or a single scalar.
    # The instance provides a single "arc_capacity" value applied to all arcs.
    # Paper Eq. (4): 0 <= x_{ij} <= u_{ij}
    u = [[arc_capacity for _ in range(m)] for _ in range(n)]  # n x m matrix per Eq. (4)

    model = gp.Model("MCCFBN")
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)  # single thread as in paper
    # Paper uses optimality tolerance of 1e-4
    model.setParam("OptimalityTol", 1e-4)
    model.setParam("BarConvTol", 1e-4)

    # Decision variables: x[i][j] = flow from supply i to demand j
    x = {}
    for i in range(n):
        for j in range(m):
            x[i, j] = model.addVar(
                lb=0.0,
                ub=u[i][j],
                name=f"x_{i}_{j}"
            )

    model.update()

    # Objective: min sum_{i,j} f_{ij}(x_{ij})
    obj = gp.QuadExpr()
    has_quadratic = False
    for i in range(n):
        for j in range(m):
            c_ij = linear_costs[i][j]
            q_ij = quadratic_costs[i][j]
            obj += c_ij * x[i, j]
            if q_ij != 0:
                obj += q_ij * x[i, j] * x[i, j]
                has_quadratic = True

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (2): sum_{i in I} x_{ij} = d_j, for all j in J
    for j in range(m):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(n)) == demands[j],
            name=f"demand_{j}"
        )

    # Constraint (3): sum_{j in J} x_{ij} <= s_i, for all i in I
    for i in range(n):
        model.addConstr(
            gp.quicksum(x[i, j] for j in range(m)) <= supplies[i],
            name=f"supply_{i}"
        )

    # Use barrier method (interior-point) to match the paper's approach
    if has_quadratic:
        model.setParam("Method", 2)  # barrier
        model.setParam("BarHomogeneous", 0)
    else:
        # For linear problems, let Gurobi choose, but prefer barrier
        model.setParam("Method", 2)

    # Disable crossover to match paper setting (no crossover for BlockIP)
    model.setParam("Crossover", 0)

    model.optimize()

    result = {
        "objective_value": None,
        "status": None,
        "flows": None
    }

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        # Barrier (interior-point) without crossover leaves ~all n*m variables
        # with positive dust values just above the prior 1e-8 threshold; for
        # n=200, m=500000 (l41) that's 100M+ dict entries → 10+GB RAM → OOM
        # during solution extraction (gurobi already solved). Raise the
        # threshold to 1e-3 — dust below this is below the BarConvTol that
        # the checker also uses, so it carries no meaningful flow.
        FLOW_THRESHOLD = 1e-3
        flows = {}
        for i in range(n):
            for j in range(m):
                val = x[i, j].X
                if val > FLOW_THRESHOLD:
                    flows[f"x_{i}_{j}"] = val
        result["flows"] = flows
    else:
        result["status"] = "infeasible_or_no_solution"
        result["objective_value"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for MCCFBN (Castro & Nasini 2021)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to write the solution JSON file."
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

    with open(args.solution_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    if result["objective_value"] is not None:
        print(f"Objective value: {result['objective_value']}")
    else:
        print("No feasible solution found.")


if __name__ == "__main__":
    main()
