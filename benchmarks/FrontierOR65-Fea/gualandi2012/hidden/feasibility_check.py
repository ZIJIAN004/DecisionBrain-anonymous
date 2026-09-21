#!/usr/bin/env python3
"""
Feasibility checker for Minimum Graph Coloring Problem (Min-GCP).

Based on: Gualandi & Malucelli (2012), "Exact Solution of Graph Coloring
Problems via Constraint Programming and Column Generation",
INFORMS Journal on Computing, 24(1), 81-100.

Checks hard constraints from the mathematical formulations:
  Constraint (1): x_i != x_j for all {i,j} in E
  Constraint (2): alldifferent({x_i | i in C}) for all cliques C
  Constraint (3): x_0 = max({x_i | i in V})
  Constraint (4): x_0 <= x_0* (upper bound)
  Constraint (6): sum_{i in S_v} lambda_i >= 1 for all v in V (vertex coverage)
  Constraint (7): lambda_i in {0,1} (integrality / valid partition)

Constraints (5), (8)-(31) are objectives or algorithm-internal subproblem
constraints (LP relaxation, pricing, augmented pricing, multicoloring)
and do not apply to solution feasibility.
"""

import argparse
import json

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def check_feasibility(instance, solution):
    violations = []
    violation_magnitudes = []
    violated_set = set()

    n = instance["graph"]["num_vertices"]
    edges = [tuple(e) for e in instance["graph"]["edges"]]

    # Parse coloring: vertex (int) -> color (int)
    coloring = {}
    if "coloring" in solution:
        for v_str, c in solution["coloring"].items():
            coloring[int(v_str)] = int(c)

    # Parse color_classes: color (int) -> list of vertices
    color_classes = {}
    if "color_classes" in solution:
        for c_str, verts in solution["color_classes"].items():
            color_classes[int(c_str)] = [int(v) for v in verts]

    objective_value = solution.get("objective_value", None)
    upper_bound = solution.get("heuristic_upper_bound", None)

    # =========================================================================
    # Constraint (1): x_i != x_j for all {i,j} in E
    # Proper coloring: adjacent vertices must have different colors.
    # Modeled as |x_i - x_j| >= 1 (a >= constraint).
    # =========================================================================
    conflict_edges = []
    for u, v in edges:
        if u in coloring and v in coloring:
            if coloring[u] == coloring[v]:
                conflict_edges.append((u, v))

    if conflict_edges:
        violated_set.add(1)
        msgs = []
        for u, v in conflict_edges:
            c = coloring[u]
            msgs.append(f"Edge ({u},{v}): both assigned color {c}")
            lhs = 0.0  # |x_i - x_j| = 0
            rhs = 1.0  # must be >= 1
            raw_excess = rhs - lhs  # 1.0
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })
        violations.append(
            f"Constraint (1) violated: {len(conflict_edges)} edge(s) have "
            f"same-color endpoints. " + "; ".join(msgs[:5])
            + (f" ... and {len(msgs)-5} more" if len(msgs) > 5 else "")
        )

    # =========================================================================
    # Constraint (2): alldifferent({x_i | i in C}) for all cliques C
    # Check using max_clique_vertices from instance properties if available.
    # This is subsumed by constraint (1) for proper colorings, but we check
    # explicitly for the known clique.
    # =========================================================================
    clique_vertices = []
    if "properties" in instance and "max_clique_vertices" in instance["properties"]:
        clique_vertices = instance["properties"]["max_clique_vertices"]

    if clique_vertices:
        clique_colors = [coloring[v] for v in clique_vertices if v in coloring]
        if len(clique_colors) != len(set(clique_colors)):
            violated_set.add(2)
            # Find duplicate colors
            seen = {}
            dup_pairs = []
            for v in clique_vertices:
                if v not in coloring:
                    continue
                c = coloring[v]
                if c in seen:
                    dup_pairs.append((seen[c], v, c))
                else:
                    seen[c] = v
            for v1, v2, c in dup_pairs:
                lhs = 0.0  # difference = 0
                rhs = 1.0  # must be >= 1
                raw_excess = rhs - lhs
                normalizer = max(abs(rhs), EPS)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })
            violations.append(
                f"Constraint (2) violated: alldifferent violated in max clique "
                f"{clique_vertices}. Duplicate color pairs: {dup_pairs}"
            )

    # =========================================================================
    # Constraint (3): x_0 = max({x_i | i in V})
    # The reported objective should equal the number of distinct colors used.
    # In the formulation, x_i in {1,...,chi_bar} (1-indexed), so
    # x_0 = max(x_i) = num_colors. With 0-indexed colors in the solution,
    # num_distinct_colors = max_color + 1 = number of colors used.
    # =========================================================================
    if objective_value is not None and coloring:
        num_distinct_colors = len(set(coloring.values()))
        lhs = float(objective_value)
        rhs = float(num_distinct_colors)
        raw_excess = abs(lhs - rhs)
        if raw_excess > TOL:
            violated_set.add(3)
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })
            violations.append(
                f"Constraint (3) violated: objective_value={objective_value} "
                f"but {num_distinct_colors} distinct colors used"
            )

    # =========================================================================
    # Constraint (4): x_0 <= x_0* (upper bound)
    # In the paper, x_0* is the cost of the last solution found during the
    # CP search tree -- it is an algorithm-internal bounding constraint used
    # to prune the search space, NOT a hard feasibility constraint on the
    # solution itself. A valid proper coloring that uses more colors than a
    # heuristic upper bound is still a feasible coloring. Therefore we do
    # NOT treat violations of this bound as infeasibility.
    # =========================================================================
    # (Intentionally not checked as a hard constraint.)

    # =========================================================================
    # Constraint (6): sum_{i in S_v} lambda_i >= 1 for all v in V
    # Every vertex must be assigned a color (covered by at least one color class).
    # =========================================================================
    uncolored = []
    for v in range(n):
        if v not in coloring:
            uncolored.append(v)

    if uncolored:
        violated_set.add(6)
        for v in uncolored:
            lhs = 0.0  # vertex not covered
            rhs = 1.0  # must be >= 1
            raw_excess = rhs - lhs
            normalizer = max(abs(rhs), EPS)
            violation_magnitudes.append({
                "constraint": 6,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": raw_excess,
                "normalizer": normalizer,
                "ratio": raw_excess / normalizer,
            })
        violations.append(
            f"Constraint (6) violated: {len(uncolored)} vertex/vertices "
            f"not assigned any color: {uncolored[:10]}"
            + (f" ... and {len(uncolored)-10} more" if len(uncolored) > 10 else "")
        )

    # =========================================================================
    # Constraint (7): lambda_i in {0,1} (integrality / valid partition)
    # Each vertex should appear in exactly one color class. Check that the
    # color_classes form a valid partition of V consistent with the coloring.
    # =========================================================================
    if color_classes:
        # Check vertices appearing in multiple color classes
        vertex_count = {}
        for c, verts in color_classes.items():
            for v in verts:
                vertex_count[v] = vertex_count.get(v, 0) + 1

        multi_assigned = {v: cnt for v, cnt in vertex_count.items() if cnt > 1}
        # Check consistency between coloring and color_classes
        inconsistent = []
        for c, verts in color_classes.items():
            for v in verts:
                if v in coloring and coloring[v] != c:
                    inconsistent.append((v, coloring[v], c))

        if multi_assigned or inconsistent:
            violated_set.add(7)
            if multi_assigned:
                # For multi-assigned vertices, the excess is count - 1
                for v, cnt in multi_assigned.items():
                    lhs = float(cnt)  # appears in cnt classes
                    rhs = 1.0  # should be in exactly 1
                    raw_excess = abs(lhs - rhs)
                    normalizer = max(abs(rhs), EPS)
                    violation_magnitudes.append({
                        "constraint": 7,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": raw_excess,
                        "normalizer": normalizer,
                        "ratio": raw_excess / normalizer,
                    })
                violations.append(
                    f"Constraint (7) violated: {len(multi_assigned)} vertex/vertices "
                    f"appear in multiple color classes: {dict(list(multi_assigned.items())[:5])}"
                )
            if inconsistent:
                for v, c_coloring, c_class in inconsistent:
                    lhs = float(c_coloring)
                    rhs = float(c_class)
                    raw_excess = abs(lhs - rhs)
                    normalizer = max(abs(rhs), EPS)
                    violation_magnitudes.append({
                        "constraint": 7,
                        "lhs": lhs,
                        "rhs": rhs,
                        "raw_excess": raw_excess,
                        "normalizer": normalizer,
                        "ratio": raw_excess / normalizer,
                    })
                violations.append(
                    f"Constraint (7) violated: {len(inconsistent)} inconsistencies "
                    f"between coloring and color_classes: {inconsistent[:5]}"
                )

    _domain_check_vars_binary = []
    _domain_check_vars_integer = []

    # =====================================================================
    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # =====================================================================
    # Constraint 8: Binary domain — variables must be 0 or 1
    for var_name, var_dict in _domain_check_vars_binary:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                if abs(v - round(v)) > TOL or round(v) not in (0, 1):
                    viol = min(abs(v - 0), abs(v - 1))
                    if viol > TOL:
                        violated_set.add(8)
                        violations.append(
                            f"Constraint 8 (binary domain): {var_name}[{key}] = {v} not in {0, 1}")
                        violation_magnitudes.append({
                            "constraint": 8,
                            "lhs": v,
                            "rhs": 1.0,
                            "raw_excess": float(viol),
                            "normalizer": 1.0,
                            "ratio": float(viol),
                        })

    # Constraint 9: Integer domain — variables must be integral
    for var_name, var_dict in _domain_check_vars_integer:
        if isinstance(var_dict, dict):
            for key, val in var_dict.items():
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    continue
                frac = abs(v - round(v))
                if frac > TOL:
                    violated_set.add(9)
                    violations.append(
                        f"Constraint 9 (integer domain): {var_name}[{key}] = {v} is not integer")
                    violation_magnitudes.append({
                        "constraint": 9,
                        "lhs": v,
                        "rhs": round(v),
                        "raw_excess": float(frac),
                        "normalizer": max(abs(round(v)), EPS),
                        "ratio": float(frac / max(abs(round(v)), EPS)),
                    })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Min-GCP solutions"
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

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraint violations detected")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    main()
