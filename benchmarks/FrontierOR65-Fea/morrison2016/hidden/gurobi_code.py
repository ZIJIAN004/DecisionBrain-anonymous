#!/usr/bin/env python3
"""
Graph coloring via set-covering formulation (Morrison et al., 2016).

Formulation:
    minimize   sum_{S in S} x_S
    subject to sum_{S: v in S} x_S >= 1   for all v in V
               x_S in {0, 1}               for all S in S

where S is the family of ALL maximal independent sets in G.
"""

import argparse
import json
import time

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


def build_adjacency(num_vertices, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def enumerate_maximal_independent_sets(num_vertices, adj):
    """
    Enumerate all maximal independent sets using Bron-Kerbosch on the
    complement graph. An independent set in G is a clique in the complement
    of G, so we run Bron-Kerbosch with pivoting on the complement.
    """
    # Build complement adjacency
    comp_adj = [set() for _ in range(num_vertices)]
    for v in range(num_vertices):
        for u in range(num_vertices):
            if u != v and u not in adj[v]:
                comp_adj[v].add(u)

    maximal_independent_sets = []

    def bron_kerbosch(R, P, X):
        """Bron-Kerbosch with pivoting (finds maximal cliques in complement = maximal independent sets in G)."""
        if not P and not X:
            maximal_independent_sets.append(frozenset(R))
            return

        # Choose pivot to maximize |P intersect N(pivot)| in complement graph
        pivot = max(P | X, key=lambda v: len(P & comp_adj[v]))
        candidates = P - comp_adj[pivot]

        for v in list(candidates):
            neighbors_v = comp_adj[v]
            bron_kerbosch(
                R | {v},
                P & neighbors_v,
                X & neighbors_v,
            )
            P = P - {v}
            X = X | {v}

    bron_kerbosch(set(), set(range(num_vertices)), set())
    return maximal_independent_sets


def solve(instance_path, solution_path, time_limit):
    """Load instance, enumerate maximal independent sets, solve IP."""
    with open(instance_path, "r") as f:
        instance = json.load(f)

    instance_id = instance["instance_id"]
    graph = instance["graph"]
    num_vertices = graph["num_vertices"]
    edges = graph["edges"]

    if solution_path is None:
        solution_path = f"gurobi_solution_{instance_id}.json"

    adj = build_adjacency(num_vertices, edges)

    # Enumerate all maximal independent sets
    t0 = time.time()
    mis_list = enumerate_maximal_independent_sets(num_vertices, adj)
    enum_time = time.time() - t0
    num_sets = len(mis_list)

    print(f"Enumerated {num_sets} maximal independent sets in {enum_time:.2f}s")

    # Build vertex-to-set incidence: for each vertex, which sets contain it
    vertex_to_sets = [[] for _ in range(num_vertices)]
    for idx, s in enumerate(mis_list):
        for v in s:
            vertex_to_sets[v].append(idx)

    # Build Gurobi model
    model = gp.Model("graph_coloring_set_cover")
    model.setParam("Threads", 1)
    # Pass a slightly-shorter TimeLimit to gurobi so model.optimize() returns
    # cleanly before the outer wrapper's 30s grace SIGKILL. Plan F 1h run found
    # incumbent obj=35 (jsonl) but solution.json was missing — the final LP at
    # node ~80 ran past 3600s and the wrapper killed before json.dump.
    _internal_tl = max(60, time_limit - 120) if time_limit > 240 else time_limit
    model.setParam("TimeLimit", _internal_tl)
    model.setParam("OutputFlag", 1)
    # Solver hints: l11 (n=300, |E|≈111K dense graph coloring) was 1h TLE
    # with no incumbent in prior runs; MIPFocus=1 prioritizes finding any
    # feasible incumbent over closing the LP-gap, and NoRelHeurTime gives
    # gurobi a root-node heuristic budget before B&B starts. Other 4
    # instances already OPT-ed; expected minor slowdown is acceptable.
    model.setParam("MIPFocus", 1)
    model.setParam("NoRelHeurTime", min(60.0, time_limit * 0.05))

    # One binary variable per maximal independent set
    x = model.addVars(num_sets, vtype=GRB.BINARY, name="x")

    # Objective: minimize number of sets used
    model.setObjective(gp.quicksum(x[i] for i in range(num_sets)), GRB.MINIMIZE)

    # Covering constraints: every vertex must be in at least one chosen set
    for v in range(num_vertices):
        model.addConstr(
            gp.quicksum(x[i] for i in vertex_to_sets[v]) >= 1,
            name=f"cover_{v}",
        )

    model.optimize()

    # Extract solution
    solution = {
        "instance_id": instance_id,
        "problem": "graph_coloring",
        "formulation": "set_covering_maximal_independent_sets",
        "num_maximal_independent_sets": num_sets,
        "enumeration_time_seconds": round(enum_time, 4),
    }

    if model.SolCount > 0:
        obj_val = round(model.ObjVal)
        selected_sets = []
        for i in range(num_sets):
            if x[i].X > 0.5:
                selected_sets.append(sorted(mis_list[i]))

        # Build vertex-to-color mapping from selected sets
        coloring = {}
        for color_idx, s in enumerate(selected_sets):
            for v in s:
                if v not in coloring:
                    coloring[v] = color_idx

        solution["objective_value"] = obj_val
        solution["status"] = model.Status
        solution["status_name"] = (
            "OPTIMAL" if model.Status == GRB.OPTIMAL else "TIME_LIMIT"
        )
        solution["mip_gap"] = model.MIPGap if hasattr(model, "MIPGap") else None
        solution["selected_sets"] = selected_sets
        solution["coloring"] = {str(v): coloring[v] for v in sorted(coloring)}
        solution["solve_time_seconds"] = round(model.Runtime, 4)
    else:
        solution["objective_value"] = None
        solution["status"] = model.Status
        solution["status_name"] = "NO_SOLUTION"
        solution["solve_time_seconds"] = round(model.Runtime, 4)

    with open(solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {solution_path}")
    if solution["objective_value"] is not None:
        print(f"Chromatic number (objective): {solution['objective_value']}")

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Graph coloring via set-covering over maximal independent sets (Morrison et al., 2016)"
    )
    parser.add_argument(
        "--instance_path",
        type=str,
        required=True,
        help="Path to instance JSON file",
    )
    parser.add_argument(
        "--solution_path",
        type=str,
        default=None,
        help="Path to write solution JSON (default: gurobi_solution_<id>.json)",
    )
    parser.add_argument(
        "--time_limit",
        type=int,
        default=300,
        help="Gurobi time limit in seconds (default: 300)",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)
    solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
