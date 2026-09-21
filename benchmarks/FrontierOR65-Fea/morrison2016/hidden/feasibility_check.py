#!/usr/bin/env python3
"""
Feasibility checker for Graph Coloring via Set Covering with Maximal Independent Sets.
Based on Morrison, Sewell, and Jacobson (2016).

Constraints (from math_model.txt, top to bottom):
  (1) Covering: sum_{S: v in S} x_S >= 1  for all v in V
  (2) Binary domain: x_S in {0, 1} for all S in S
      — S must be a maximal independent set in G.
      — Since the solution encodes selected sets explicitly, binary is automatic.
        We verify that each selected set is a valid maximal independent set.
  (3) Domain checks on the `coloring` field (auto-generated).
  (4) Objective consistency: reported objective_value must equal len(selected_sets),
      which is the recomputed value of sum_{S in S} x_S since each x_S = 1 for a
      listed set. Catches Tier-C exploits that fabricate `objective_value`.
"""

import argparse
import json


def check_feasibility(instance, solution):
    graph = instance["graph"]
    num_vertices = graph["num_vertices"]
    edges = graph["edges"]

    # Build adjacency sets
    adj = [set() for _ in range(num_vertices)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    selected_sets = solution["selected_sets"]  # list of lists of vertex indices

    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # ---------------------------------------------------------------
    # Constraint (1): Covering — sum_{S: v in S} x_S >= 1 for all v
    # LHS = number of selected sets containing v, RHS = 1
    # ---------------------------------------------------------------
    for v in range(num_vertices):
        count = sum(1 for s in selected_sets if v in s)
        lhs = float(count)
        rhs = 1.0
        # >= constraint: violation if rhs > lhs
        violation_amount = rhs - lhs
        if violation_amount > tol:
            violated_constraints.add(1)
            violations.append(
                f"Vertex {v} is not covered: appears in {count} selected set(s), needs >= 1"
            )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer,
            })

    # ---------------------------------------------------------------
    # Constraint (2): Binary domain — x_S in {0,1}, S in S (maximal independent sets)
    # We verify each selected set (x_S = 1) is:
    #   (a) An independent set: no edge between any pair of vertices in S.
    #   (b) A maximal independent set: no vertex outside S can be added
    #       while maintaining independence.
    # ---------------------------------------------------------------

    for idx, s in enumerate(selected_sets):
        s_set = set(s)

        # (a) Independence check: for each edge (u,v), both u and v should
        #     not be in the same selected set.
        #     LHS = number of edges within S, RHS = 0 (equality: must be 0)
        edge_count = 0
        edge_violations_detail = []
        for u, v in edges:
            if u in s_set and v in s_set:
                edge_count += 1
                edge_violations_detail.append((u, v))

        if edge_count > 0:
            lhs = float(edge_count)
            rhs = 0.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                violated_constraints.add(2)
                pairs_str = ", ".join(f"({u},{v})" for u, v in edge_violations_detail[:5])
                if len(edge_violations_detail) > 5:
                    pairs_str += f" ... ({len(edge_violations_detail)} total)"
                violations.append(
                    f"Selected set {idx} is not independent: {edge_count} internal edge(s): {pairs_str}"
                )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

        # (b) Maximality check: for each vertex not in S, it must be adjacent
        #     to at least one vertex in S (otherwise it could be added).
        #     LHS = number of addable vertices, RHS = 0 (equality: must be 0)
        addable = []
        for v in range(num_vertices):
            if v not in s_set:
                if not (adj[v] & s_set):
                    addable.append(v)

        if len(addable) > 0:
            lhs = float(len(addable))
            rhs = 0.0
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                violated_constraints.add(2)
                verts_str = ", ".join(str(v) for v in addable[:10])
                if len(addable) > 10:
                    verts_str += f" ... ({len(addable)} total)"
                violations.append(
                    f"Selected set {idx} is not maximal: vertex(es) {verts_str} could be added"
                )
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer,
                })

    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # coloring: dict[vertex_str -> color_int]. Keys must be valid vertex indices in
    # [0, num_vertices); values must be integer color indices in [0, num_colors_used).
    # Every vertex must be assigned exactly one color, and adjacent vertices must differ.
    coloring = solution.get("coloring", {})
    num_colors_used = len(selected_sets)
    for vk, cv in coloring.items():
        try:
            v = int(vk)
        except (TypeError, ValueError):
            v = -1
        if not (0 <= v < num_vertices):
            violated_constraints.add(3)
            violations.append(
                f"Invalid vertex key in coloring: {vk!r} (valid range 0..{num_vertices-1})"
            )
            violation_magnitudes.append({
                "constraint": 3, "lhs": float(v), "rhs": 0.0,
                "raw_excess": 1.0, "normalizer": max(1.0, eps), "ratio": 1.0,
            })
            continue
        # Color must be a non-negative integer within the number of used colors
        if not isinstance(cv, int) or cv < 0 or cv >= num_colors_used:
            violated_constraints.add(3)
            violations.append(
                f"Invalid color for vertex {v} in coloring: {cv} "
                f"(must be integer in [0,{num_colors_used-1}])"
            )
            violation_magnitudes.append({
                "constraint": 3, "lhs": float(cv if isinstance(cv, (int, float)) else -1),
                "rhs": 0.0, "raw_excess": 1.0, "normalizer": max(1.0, eps), "ratio": 1.0,
            })
    # Every vertex in 0..num_vertices must be in coloring.
    coloring_keys_int = set()
    for vk in coloring.keys():
        try: coloring_keys_int.add(int(vk))
        except (TypeError, ValueError): pass
    missing_v = set(range(num_vertices)) - coloring_keys_int
    if missing_v:
        violated_constraints.add(3)
        violations.append(
            f"coloring is missing {len(missing_v)} vertex assignment(s): "
            f"{sorted(missing_v)[:8]}{'...' if len(missing_v) > 8 else ''}"
        )
        violation_magnitudes.append({
            "constraint": 3, "lhs": float(num_vertices - len(missing_v)),
            "rhs": float(num_vertices), "raw_excess": float(len(missing_v)),
            "normalizer": float(num_vertices), "ratio": float(len(missing_v)) / max(num_vertices, 1),
        })

    # ---------------------------------------------------------------
    # Constraint (4): Objective consistency (Tier-C anti-exploit).
    # Objective = sum_{S in S} x_S = number of color classes used.
    # Since the solution lists exactly the sets with x_S = 1, the true
    # objective equals len(selected_sets). Compare against reported value.
    # ---------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(len(selected_sets))
            abs_diff = abs(reported - true_obj)
            # Objective is an integer count; require integer match (tol = 0.5).
            obj_tol = 0.5
            if abs_diff > obj_tol:
                violated_constraints.add(4)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed len(selected_sets)="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer,
                })

    # ---------------------------------------------------------------
    # Build result
    # ---------------------------------------------------------------
    feasible = len(violated_constraints) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for graph coloring set-covering formulation"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the instance JSON file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the candidate solution JSON file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the feasibility result JSON file")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {'PASS' if result['feasible'] else 'FAIL'}")
    if not result["feasible"]:
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
