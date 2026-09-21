"""
Gurobi implementation of the Maximally Diverse Grouping Problem (MDGP).

Based on: Lai, Hao, Fu & Yue (2021), "Neighborhood decomposition based variable
neighborhood search and tabu search for maximally diverse grouping",
European Journal of Operational Research.

The model is a Quadratic Binary Programming (QBP) formulation:
  Maximize  sum_{g=1}^{m} sum_{i<j} d_{ij} * X_{ig} * X_{jg}
  s.t.      sum_{g=1}^{m} X_{ig} = 1,  for all i        (assignment)
            L_g <= sum_{i} X_{ig} <= U_g, for all g      (capacity)
            X_{ig} in {0, 1}
"""

import argparse
import json
import numpy as np
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

    N = data['N']
    m = data['m']
    capacities = data['group_capacities']
    L = [cap['L_g'] for cap in capacities]
    U = [cap['U_g'] for cap in capacities]

    # Reconstruct full distance matrix from upper triangular entries
    dist_upper = data['distances_upper_triangular']
    D = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            D[i][j] = dist_upper[idx]
            D[j][i] = dist_upper[idx]
            idx += 1

    return N, m, L, U, D


def _construct_feasible_start(N, m, L, U):
    """Construct a feasible assignment to provide Gurobi with a warm start."""
    import random as _rnd
    assignment = [0] * N
    vertices = list(range(N))
    _rnd.shuffle(vertices)
    idx = 0
    group_sizes = [0] * m
    # Phase 1: fill each group to its lower bound
    for g in range(m):
        for _ in range(L[g]):
            assignment[vertices[idx]] = g
            group_sizes[g] += 1
            idx += 1
    # Phase 2: distribute remaining vertices respecting upper bounds
    for k in range(idx, N):
        v = vertices[k]
        for g in range(m):
            if group_sizes[g] < U[g]:
                assignment[v] = g
                group_sizes[g] += 1
                break
    return assignment


def solve_mdgp(N, m, L, U, D, time_limit):
    model = gp.Model("MDGP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    # Suppress output for cleaner solution files
    model.setParam("OutputFlag", 1)

    # Decision variables: X[i, g] = 1 if vertex i assigned to group g
    X = model.addVars(N, m, vtype=GRB.BINARY, name="X")

    # Provide a feasible warm-start solution so Gurobi has an incumbent
    # immediately, preventing "no feasible solution" on timeout.
    init_assign = _construct_feasible_start(N, m, L, U)
    for i in range(N):
        for g in range(m):
            X[i, g].Start = 1.0 if init_assign[i] == g else 0.0

    # Objective: maximize sum of within-group pairwise distances
    # Use quadratic objective: sum_{g} sum_{i<j} d_{ij} * X_{i,g} * X_{j,g}
    obj = gp.QuadExpr()
    for g in range(m):
        for i in range(N - 1):
            for j in range(i + 1, N):
                if D[i][j] > 0:
                    obj.add(X[i, g] * X[j, g], D[i][j])

    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraint (2): each vertex assigned to exactly one group
    for i in range(N):
        model.addConstr(gp.quicksum(X[i, g] for g in range(m)) == 1,
                        name=f"assign_{i}")

    # Constraint (3): group capacity bounds
    for g in range(m):
        model.addConstr(gp.quicksum(X[i, g] for i in range(N)) >= L[g],
                        name=f"cap_lb_{g}")
        model.addConstr(gp.quicksum(X[i, g] for i in range(N)) <= U[g],
                        name=f"cap_ub_{g}")

    model.optimize()

    # Extract solution
    objective_value = None
    assignment = {}

    if model.SolCount > 0:
        objective_value = model.ObjVal
        for i in range(N):
            for g in range(m):
                if X[i, g].X > 0.5:
                    assignment[i] = g
                    break
    else:
        # No feasible solution found
        objective_value = None

    return objective_value, assignment, model.Status


def main():
    parser = argparse.ArgumentParser(description="Gurobi solver for MDGP")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to write the solution JSON file')
    parser.add_argument('--time_limit', type=int, required=True,
                        help='Maximum solver runtime in seconds')
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    N, m, L, U, D = load_instance(args.instance_path)
    objective_value, assignment, status = solve_mdgp(N, m, L, U, D, args.time_limit)

    # Build solution output
    solution = {
        "objective_value": objective_value,
        "solver_status": status,
        "N": N,
        "m": m,
    }

    if assignment:
        # Group assignments: vertex -> group (0-indexed)
        solution["assignment"] = {str(i): g for i, g in assignment.items()}
        # Groups: group -> list of vertices
        groups = {}
        for i, g in assignment.items():
            groups.setdefault(g, []).append(i)
        solution["groups"] = {str(g): sorted(verts) for g, verts in groups.items()}

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Objective value: {objective_value}")
    print(f"Solution written to: {args.solution_path}")


if __name__ == "__main__":
    main()
