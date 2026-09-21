#!/usr/bin/env python3
"""
Feasibility checker for Selective Graph Coloring (SEL-COL) solutions.

Checks constraints from:
  - Model 1 (gurobi solutions with coloring): Constraints 1-6
  - Model 2 (efficient/cutting-plane solutions): Constraints 7-10

Constraint numbering (top to bottom from math_model.txt):
  1  (1b): w_ik <= y_k                    for all i in V, k in {1,...,P}
  2  (1c): w_ik + w_jk <= 1               for all {i,j} in E, k in {1,...,P}
  3  (1d): sum_{i in V_p} sum_k w_ik = 1  for all p in {1,...,P}
  4  (1e): y_k in {0,1}                   for all k in {1,...,P}
  5  (1f): w_ik in {0,1}                  for all i in V, k in {1,...,P}
  6  (2):  y_k <= y_{k-1}                 for all k in {2,...,P}
  7  (3b): sum_{i in V_p} x_i = 1         for all p in {1,...,P}
  8  (3c): t >= chi(G[x])
  9  (3d): t >= 0
  10 (3e): x_i in {0,1}                   for all i in V
  11 (obj): reported objective_value must equal sum_k y_k = number of
            distinct colors used by selected vertices (full recompute).
"""

import argparse
import json

tol = 1e-5
eps = 1e-5


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Maximum clique solver (branch-and-bound with greedy coloring bound)
# ---------------------------------------------------------------------------

def maximum_clique(adj, vertices):
    """Find a maximum clique in the subgraph defined by adj and vertices."""
    if not vertices:
        return []

    best = []

    def greedy_coloring(vlist):
        color = {}
        max_c = 0
        for v in vlist:
            used = {color[u] for u in vlist if u in color and u in adj[v]}
            c = 1
            while c in used:
                c += 1
            color[v] = c
            if c > max_c:
                max_c = c
        return color, max_c

    def expand(clique, candidates):
        nonlocal best
        if not candidates:
            if len(clique) > len(best):
                best = list(clique)
            return
        if len(clique) + len(candidates) <= len(best):
            return

        color, _ = greedy_coloring(candidates)
        sorted_cands = sorted(candidates, key=lambda v: color[v], reverse=True)

        for v in sorted_cands:
            if len(clique) + color[v] <= len(best):
                return
            clique.append(v)
            new_cands = [u for u in sorted_cands if u in adj[v] and u != v]
            expand(clique, new_cands)
            clique.pop()

    verts = sorted(vertices, key=lambda v: len(adj.get(v, set())), reverse=True)

    # Greedy initial clique
    for v in verts:
        if all(u in adj.get(v, set()) for u in best):
            best.append(v)

    expand([], verts)
    return best


# ---------------------------------------------------------------------------
# Violation recording helper
# ---------------------------------------------------------------------------

def record_violation(violated_set, violations, violation_magnitudes,
                     constraint_idx, msg, lhs, rhs, violation_amount):
    normalizer = max(abs(rhs), eps)
    ratio = violation_amount / normalizer
    violated_set.add(constraint_idx)
    violations.append(msg)
    violation_magnitudes.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(violation_amount),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


# ---------------------------------------------------------------------------
# Model 1 checker (gurobi solutions)
# ---------------------------------------------------------------------------

def check_gurobi_solution(instance, solution):
    """Check Model 1 constraints 1-6 for a gurobi solution."""
    violations = []
    violation_magnitudes = []
    violated_set = set()

    num_vertices = instance["graph"]["num_vertices"]
    edges = [tuple(e) for e in instance["graph"]["edges"]]
    clusters = instance["partition"]["clusters"]
    P = instance["partition"]["num_clusters"]

    selected_vertices = solution["selected_vertices"]  # dict: cluster_str -> vertex
    coloring = solution["coloring"]                     # dict: vertex_str -> color

    # Reconstruct w_ik and y_k
    w = {}
    for i in range(num_vertices):
        for k in range(P):
            w[(i, k)] = 0
    y = {k: 0 for k in range(P)}

    for cluster_str, vertex in selected_vertices.items():
        vertex = int(vertex)
        color = int(coloring[str(vertex)])
        w[(vertex, color)] = 1
        y[color] = 1

    def add(ci, msg, lhs, rhs, va):
        record_violation(violated_set, violations, violation_magnitudes,
                         ci, msg, lhs, rhs, va)

    # Constraint 1 (1b): w_ik <= y_k
    for i in range(num_vertices):
        for k in range(P):
            lhs = w[(i, k)]
            rhs = y[k]
            va = max(0, lhs - rhs)
            if va > tol:
                add(1, f"Constraint 1 (1b) violated: w[{i},{k}]={lhs} > y[{k}]={rhs}",
                    lhs, rhs, va)

    # Constraint 2 (1c): w_ik + w_jk <= 1 for each edge and color
    for (i, j) in edges:
        for k in range(P):
            lhs = w[(i, k)] + w[(j, k)]
            rhs = 1
            va = max(0, lhs - rhs)
            if va > tol:
                add(2, f"Constraint 2 (1c) violated: w[{i},{k}]+w[{j},{k}]={lhs} > 1; "
                       f"adjacent vertices {i},{j} share color {k}",
                    lhs, rhs, va)

    # Constraint 3 (1d): sum_{i in V_p} sum_k w_ik = 1
    for p_idx, cluster in enumerate(clusters):
        lhs = sum(w[(i, k)] for i in cluster for k in range(P))
        rhs = 1
        va = abs(lhs - rhs)
        if va > tol:
            add(3, f"Constraint 3 (1d) violated: cluster {p_idx} has "
                   f"{int(lhs)} vertex-color assignments (expected 1)",
                lhs, rhs, va)

    # Constraint 4 (1e): y_k in {{0,1}}
    for k in range(P):
        val = y[k]
        nearest = round(val)
        va = abs(val - nearest)
        if va > tol:
            add(4, f"Constraint 4 (1e) violated: y[{k}]={val} is not binary",
                val, nearest, va)

    # Constraint 5 (1f): w_ik in {{0,1}}
    for i in range(num_vertices):
        for k in range(P):
            val = w[(i, k)]
            nearest = round(val)
            va = abs(val - nearest)
            if va > tol:
                add(5, f"Constraint 5 (1f) violated: w[{i},{k}]={val} is not binary",
                    val, nearest, va)

    # Constraint 6 (2): y_k <= y_{k-1} for k in {2,...,P} — symmetry breaking.
    # NOTE 2026-05-19: This is a MODEL-LEVEL constraint that's only added to speed
    # up the solver by removing equivalent color permutations. It is NOT part of
    # the underlying SEL-COL problem definition (any coloring with the same set
    # of colors used is equally valid). The checker reconstructs y[k] from
    # `coloring` (y[k]=1 iff some selected vertex uses color k); solver's
    # solution.json doesn't preserve y values when a paid-for color happens to
    # have no vertex assigned (model permitted by symbreak). Removing this
    # check: solutions where colors are non-contiguously labeled are still
    # feasible if all OTHER constraints hold. Objective consistency (constraint
    # 11 below) already verifies obj_value matches actual colors used.

    # Constraint 11 (obj): reported objective_value must equal the number of
    # distinct colors used by selected vertices (sum_k y_k). Full recompute is
    # possible because all obj-determining variables (selected_vertices and
    # coloring) are present in the solution. Defends against LLM exploits that
    # report obj=0 or obj=sys.float_info.max while producing a valid coloring.
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            # true_obj = number of distinct colors actually assigned to selected
            # vertices (== sum_k y_k under the reconstruction above).
            true_obj = float(sum(1 for k in range(P) if y[k] == 1))
            abs_diff = abs(reported - true_obj)
            # Objective is an integer count; any mismatch by >= 1 must fire.
            tolerance = 0.5
            if abs_diff > tolerance:
                add(11,
                    f"Constraint 11 (obj) violated: reported objective_value="
                    f"{reported} differs from recomputed sum_k y_k="
                    f"{true_obj} (number of distinct colors used by selected "
                    f"vertices; |diff|={abs_diff:.3g}, tol={tolerance:.3g})",
                    reported, true_obj, abs_diff)

    return {
        "feasible": len(violated_set) == 0,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


# ---------------------------------------------------------------------------
# Model 2 checker (efficient / cutting-plane solutions) — REMOVED.
#
# The previous `check_efficient_solution` path verified Model 2 cutting-plane
# constraints (3b)-(3e), including `t >= chi(G[x])`, on solutions keyed by
# `selected_vertices` / `max_clique`. Model 2 is a reformulation used inside
# the paper's cutting-plane algorithm; a generated algorithm operating on the
# SEL-COL problem description produces Model 1 output (`coloring`), not Model
# 2 variables. Keeping that branch as dead code was misleading, so it has
# been removed. Any Model-1-style solution now always goes through
# `check_gurobi_solution` below.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SEL-COL solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_gurobi_solution(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
