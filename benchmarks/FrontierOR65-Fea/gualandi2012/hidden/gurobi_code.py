#!/usr/bin/env python3
"""
Gurobi implementation of the Minimum Graph Coloring Problem (Min-GCP)
using Formulation 1 of Gualandi & Malucelli (2012), Section 2, Eqs. (1)-(4).
"""

import argparse
import json
import random
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
    """Load problem instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)
    n = data['graph']['num_vertices']
    edges = [tuple(e) for e in data['graph']['edges']]
    return n, edges, data


def build_adjacency(n, edges):
    """Build adjacency list representation."""
    adj_list = defaultdict(set)
    for u, v in edges:
        adj_list[u].add(v)
        adj_list[v].add(u)
    return adj_list


def greedy_coloring(n, adj_list, order=None):
    """Greedy sequential coloring; returns (coloring dict, #colors)."""
    if order is None:
        order = list(range(n))
    color = {}
    for v in order:
        used = set(color[u] for u in adj_list[v] if u in color)
        c = 1
        while c in used:
            c += 1
        color[v] = c
    num_colors = max(color.values()) if color else 0
    return color, num_colors


def find_maximal_clique(n, adj_list):
    """Greedy maximal clique heuristic (lower bound on chi(G))."""
    vertices = sorted(range(n), key=lambda v: len(adj_list[v]), reverse=True)
    best = []
    for start in vertices[:min(n, 50)]:
        clique = [start]
        candidates = list(adj_list[start])
        random.shuffle(candidates)
        for v in candidates:
            if all(v in adj_list[u] for u in clique):
                clique.append(v)
        if len(clique) > len(best):
            best = clique
    return sorted(best)


def collect_cliques(n, adj_list):
    """
    Collect a family \\mathscr{C} of cliques of G used to post the redundant
    alldifferent constraints (Eq. 2, Section 2.1 preprocessing).
    Only cliques of size >= 3 are kept (size-2 already covered by edge
    constraints (1)).
    """
    cliques = []
    seen = set()
    for start in range(n):
        clique = [start]
        cands = sorted(adj_list[start], key=lambda v: len(adj_list[v]),
                       reverse=True)
        for v in cands:
            if all(v in adj_list[u] for u in clique):
                clique.append(v)
        key = tuple(sorted(clique))
        if len(clique) >= 3 and key not in seen:
            seen.add(key)
            cliques.append(sorted(clique))
    return cliques


def solve_formulation1(n, edges, adj_list, chi_lo, chi_hi, time_limit):
    """
    Formulation 1 (Gualandi & Malucelli 2012, Eqs. 1-4):

        minimize  x_0
        s.t.  x_i != x_j                             for {i,j} in E       (1)
              alldifferent({x_i : i in C})           for C in \\mathscr{C} (2)
              x_0 = max({x_i : i in V})                                    (3)
              x_0 <= x_0*                                                  (4)
              x_i in K = {1, ..., chi_hi}, x_0 in {chi_lo, ..., chi_hi}

    Encoded for Gurobi with assignment auxiliaries z_{i,k} = 1 iff x_i = k:
      - sum_k z_{i,k} = 1
      - x_i = sum_k k * z_{i,k}
      - (1): z_{i,k} + z_{j,k} <= 1 for {i,j} in E, k in K
      - (2): sum_{i in C} z_{i,k} <= 1 for C in \\mathscr{C}, k in K
      - (3): x_0 = max_i x_i via addGenConstrMax
      - (4): x_0 <= chi_hi via the upper-bound domain of x_0
    """
    K = list(range(1, chi_hi + 1))

    model = gp.Model("MinGCP_Formulation1")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)

    z = {}
    for i in range(n):
        for k in K:
            z[i, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")

    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype=GRB.INTEGER, lb=1, ub=chi_hi,
                            name=f"x_{i}")

    x0 = model.addVar(vtype=GRB.INTEGER, lb=chi_lo, ub=chi_hi, name="x_0")

    model.update()

    for i in range(n):
        model.addConstr(gp.quicksum(z[i, k] for k in K) == 1,
                        name=f"assign_{i}")
        model.addConstr(x[i] == gp.quicksum(k * z[i, k] for k in K),
                        name=f"link_{i}")

    for (u, v) in edges:
        for k in K:
            model.addConstr(z[u, k] + z[v, k] <= 1,
                            name=f"edge_{u}_{v}_k{k}")

    cliques = collect_cliques(n, adj_list)
    for ci, C in enumerate(cliques):
        for k in K:
            model.addConstr(gp.quicksum(z[i, k] for i in C) <= 1,
                            name=f"clique_{ci}_k{k}")

    model.addGenConstrMax(x0, [x[i] for i in range(n)], name="x0_max")

    model.setObjective(x0, GRB.MINIMIZE)

    model.optimize()

    if model.SolCount > 0:
        coloring = {i: int(round(x[i].X)) for i in range(n)}
        num_colors = int(round(x0.X))
        model.dispose()
        return num_colors, coloring

    model.dispose()
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Solve Min-GCP via Gurobi using Formulation 1 (Eqs. 1-4)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    import time
    overall_start = time.time()

    n, edges, _ = load_instance(args.instance_path)
    adj_list = build_adjacency(n, edges)

    clique = find_maximal_clique(n, adj_list)
    chi_lo = max(1, len(clique))

    color_assign, chi_hi = greedy_coloring(n, adj_list)
    for _ in range(20):
        order = list(range(n))
        random.shuffle(order)
        ca, nc = greedy_coloring(n, adj_list, order)
        if nc < chi_hi:
            chi_hi = nc
            color_assign = ca

    elapsed = time.time() - overall_start
    solve_time = max(1, int(args.time_limit - elapsed))

    num_colors, coloring = solve_formulation1(
        n, edges, adj_list, chi_lo, chi_hi, solve_time
    )

    if num_colors is None:
        num_colors = chi_hi
        coloring = color_assign

    valid = all(coloring.get(u) != coloring.get(v) for (u, v) in edges)
    if not valid:
        coloring, num_colors = greedy_coloring(n, adj_list)

    color_classes = defaultdict(list)
    for v, c in coloring.items():
        color_classes[c].append(v)

    solution = {
        "objective_value": num_colors,
        "num_vertices": n,
        "num_edges": len(edges),
        "chromatic_number_lower_bound": chi_lo,
        "heuristic_upper_bound": chi_hi,
        "coloring": coloring,
        "color_classes": {str(c): sorted(vs) for c, vs in color_classes.items()},
        "valid": valid
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Objective value (chromatic number): {num_colors}")
    print(f"Lower bound (clique number): {chi_lo}")


if __name__ == "__main__":
    main()
