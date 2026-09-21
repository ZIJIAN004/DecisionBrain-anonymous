#!/usr/bin/env python3
"""
Feasibility checker for Minimum Graph Bisection (Armbruster et al. 2012).

Checks the ILP formulation (Equation 1) constraints:
  Constraint 1: sum_{j=2}^{n} f_j * y_{1j} <= F
  Constraint 2: f_1 + sum_{j=2}^{n} f_j * (1 - y_{1j}) <= F
  Constraint 3: Cycle inequalities (odd subset D of cycle C):
                sum_{ij in D} y_{ij} - sum_{ij in C\\D} y_{ij} <= |D| - 1
  Constraint 4: y in {0, 1}^E  (binary domain)
  Constraint 5: solution["objective_value"] equals the cut cost
                sum_{ij in E} w_{ij} * y_{ij}  recomputed from the partition.
                (Without this, an LLM can report any objective and pass C1-C4.)

Node indexing: the paper uses 1-based with node 1 as the star center.
The data uses 0-based with node 0 as the star center.
"""

import argparse
import json
from collections import defaultdict


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_partition(instance, solution):
    """
    Extract a 0/1 partition array from either efficient or gurobi solution format.
    Returns partition array or None if no valid solution exists.
    """
    n = instance["num_nodes"]

    # Efficient solution format: has "partition" list
    if "partition" in solution and isinstance(solution["partition"], list) and len(solution["partition"]) > 0:
        return solution["partition"]

    # Gurobi solution format: has "partition_S" and "partition_complement"
    if "partition_S" in solution and "partition_complement" in solution:
        if len(solution["partition_S"]) == 0 and len(solution["partition_complement"]) == 0:
            return None  # No solution (e.g., INFEASIBLE)
        partition = [None] * n
        for node in solution["partition_S"]:
            partition[node] = 0  # Side containing node 0
        for node in solution["partition_complement"]:
            partition[node] = 1  # Side separated from node 0
        # Check all nodes assigned
        if any(p is None for p in partition):
            return None
        return partition

    return None


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the Minimum Graph Bisection ILP.
    Returns (feasible, violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5

    n = instance["num_nodes"]
    edges = [tuple(e) for e in instance["edges"]]
    node_weights = instance["node_weights"]
    F = instance["bisection_capacity_F"]

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Extract partition
    partition = extract_partition(instance, solution)
    if partition is None:
        # No solution to check — report all structural constraints violated
        violations.append("No valid partition found in solution (e.g., solver returned INFEASIBLE)")
        violated_constraints.add(4)
        violation_magnitudes.append({
            "constraint": 4,
            "lhs": float("nan"),
            "rhs": float("nan"),
            "raw_excess": float("inf"),
            "normalizer": eps,
            "ratio": float("inf"),
        })
        return (False,
                sorted(violated_constraints),
                violations,
                violation_magnitudes)

    # Build augmented edge set (star from node 0 to all others)
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i, j), max(i, j)))
    for j in range(1, n):
        edge_set.add((0, j))

    # Derive y from partition: y_{ij} = 1 if partition[i] != partition[j]
    def y_val(i, j):
        return 1 if partition[i] != partition[j] else 0

    # ------------------------------------------------------------------
    # Constraint 4: Binary domain — y in {0,1}^E
    # Check that partition values are all 0 or 1.
    # ------------------------------------------------------------------
    non_binary = []
    for i in range(n):
        val = partition[i]
        if val not in (0, 1) and not (isinstance(val, float) and (abs(val) < tol or abs(val - 1.0) < tol)):
            non_binary.append(i)

    if non_binary:
        violated_constraints.add(4)
        for node in non_binary:
            val = partition[node]
            violation_amount = min(abs(val - 0), abs(val - 1))
            rhs = 1.0  # closest binary value bound
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Constraint 4: Node {node} has non-binary partition value {val}"
            )
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": float(val),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer),
            })

    # ------------------------------------------------------------------
    # Constraint 1: sum_{j=1}^{n-1} f_j * y_{0,j} <= F
    # Weight of nodes separated from node 0 must not exceed F.
    # ------------------------------------------------------------------
    lhs_1 = sum(node_weights[j] * y_val(0, j) for j in range(1, n))
    rhs_1 = float(F)
    violation_amount_1 = lhs_1 - rhs_1  # positive means violated (LHS > RHS for <=)

    if violation_amount_1 > tol:
        violated_constraints.add(1)
        normalizer_1 = max(abs(rhs_1), eps)
        violations.append(
            f"Constraint 1: Weight of nodes separated from node 0 = {lhs_1} exceeds capacity F = {rhs_1}"
        )
        violation_magnitudes.append({
            "constraint": 1,
            "lhs": float(lhs_1),
            "rhs": float(rhs_1),
            "raw_excess": float(violation_amount_1),
            "normalizer": float(normalizer_1),
            "ratio": float(violation_amount_1 / normalizer_1),
        })

    # ------------------------------------------------------------------
    # Constraint 2: f_0 + sum_{j=1}^{n-1} f_j * (1 - y_{0,j}) <= F
    # Weight of nodes in the same cluster as node 0 must not exceed F.
    # ------------------------------------------------------------------
    lhs_2 = node_weights[0] + sum(node_weights[j] * (1 - y_val(0, j)) for j in range(1, n))
    rhs_2 = float(F)
    violation_amount_2 = lhs_2 - rhs_2

    if violation_amount_2 > tol:
        violated_constraints.add(2)
        normalizer_2 = max(abs(rhs_2), eps)
        violations.append(
            f"Constraint 2: Weight of nodes with node 0 = {lhs_2} exceeds capacity F = {rhs_2}"
        )
        violation_magnitudes.append({
            "constraint": 2,
            "lhs": float(lhs_2),
            "rhs": float(rhs_2),
            "raw_excess": float(violation_amount_2),
            "normalizer": float(normalizer_2),
            "ratio": float(violation_amount_2 / normalizer_2),
        })

    # ------------------------------------------------------------------
    # Constraint 3: Cycle inequalities (odd-subset of cycle)
    # For binary partitions derived from a valid 0/1 assignment, cycle
    # inequalities are always satisfied. We verify via triangle
    # inequalities on the augmented graph (triangles are the shortest
    # cycles and capture all violations for binary solutions).
    #
    # For a triangle (a, b, c) with all three edges in the augmented graph:
    #   |D|=1 forms:  y_{ab} - y_{ac} - y_{bc} <= 0  (and permutations)
    #   |D|=3 form:   y_{ab} + y_{ac} + y_{bc} <= 2
    # ------------------------------------------------------------------
    # Build adjacency for augmented graph
    adj = defaultdict(set)
    for (i, j) in edge_set:
        adj[i].add(j)
        adj[j].add(i)

    constraint_3_violated = False
    # Check triangle inequalities on all triangles in augmented graph
    # To avoid O(n^3), iterate over edges and check common neighbors
    checked_triangles = set()
    for (i, j) in edge_set:
        common = adj[i] & adj[j]
        for k in common:
            tri = tuple(sorted([i, j, k]))
            if tri in checked_triangles:
                continue
            checked_triangles.add(tri)

            a, b, c = tri
            y_ab = y_val(a, b)
            y_ac = y_val(a, c)
            y_bc = y_val(b, c)

            # |D|=1 inequalities (3 forms):
            # y_ab - y_ac - y_bc <= 0
            # y_ac - y_ab - y_bc <= 0
            # y_bc - y_ab - y_ac <= 0
            for (d_val, cd_vals, label) in [
                (y_ab, y_ac + y_bc, f"y_{{{a},{b}}} - y_{{{a},{c}}} - y_{{{b},{c}}}"),
                (y_ac, y_ab + y_bc, f"y_{{{a},{c}}} - y_{{{a},{b}}} - y_{{{b},{c}}}"),
                (y_bc, y_ab + y_ac, f"y_{{{b},{c}}} - y_{{{a},{b}}} - y_{{{a},{c}}}"),
            ]:
                lhs_val = d_val - cd_vals
                rhs_val = 0.0
                excess = lhs_val - rhs_val
                if excess > tol:
                    constraint_3_violated = True
                    normalizer = max(abs(rhs_val), eps)
                    if 3 not in violated_constraints:
                        violated_constraints.add(3)
                    violations.append(
                        f"Constraint 3: Triangle ({a},{b},{c}) |D|=1 inequality violated: {label} = {lhs_val} > 0"
                    )
                    violation_magnitudes.append({
                        "constraint": 3,
                        "lhs": float(lhs_val),
                        "rhs": float(rhs_val),
                        "raw_excess": float(excess),
                        "normalizer": float(normalizer),
                        "ratio": float(excess / normalizer),
                    })

            # |D|=3 inequality: y_ab + y_ac + y_bc <= 2
            lhs_d3 = y_ab + y_ac + y_bc
            rhs_d3 = 2.0
            excess_d3 = lhs_d3 - rhs_d3
            if excess_d3 > tol:
                constraint_3_violated = True
                normalizer_d3 = max(abs(rhs_d3), eps)
                if 3 not in violated_constraints:
                    violated_constraints.add(3)
                violations.append(
                    f"Constraint 3: Triangle ({a},{b},{c}) |D|=3 inequality violated: "
                    f"y_{{{a},{b}}} + y_{{{a},{c}}} + y_{{{b},{c}}} = {lhs_d3} > 2"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": float(lhs_d3),
                    "rhs": float(rhs_d3),
                    "raw_excess": float(excess_d3),
                    "normalizer": float(normalizer_d3),
                    "ratio": float(excess_d3 / normalizer_d3),
                })

    # ------------------------------------------------------------------
    # Constraint 5: objective_value must match the cut cost recomputed
    # from the partition. Edge weights are non-negative in this benchmark,
    # so the reported objective should equal sum of w_{ij} over cut edges.
    # ------------------------------------------------------------------
    if "objective_value" in solution and solution["objective_value"] is not None:
        edge_weights = instance.get("edge_weights", [])
        if len(edge_weights) == len(edges):
            recomputed_obj = sum(
                float(edge_weights[k])
                for k, (i, j) in enumerate(edges)
                if partition[i] != partition[j]
            )
            try:
                claimed_obj = float(solution["objective_value"])
            except (TypeError, ValueError):
                claimed_obj = None
            if claimed_obj is not None:
                diff = abs(claimed_obj - recomputed_obj)
                # Allow 0.1% relative slack, with a 0.5 absolute floor for
                # integer-weight rounding noise.
                threshold = max(0.5, 1e-3 * abs(recomputed_obj))
                if diff > threshold:
                    violated_constraints.add(5)
                    normalizer = max(abs(recomputed_obj), eps)
                    violations.append(
                        f"Constraint 5: objective_value mismatch — reported "
                        f"{claimed_obj} but recomputed cut cost is "
                        f"{recomputed_obj} (diff {diff:.4g})"
                    )
                    violation_magnitudes.append({
                        "constraint": 5,
                        "lhs": float(claimed_obj),
                        "rhs": float(recomputed_obj),
                        "raw_excess": float(diff),
                        "normalizer": float(normalizer),
                        "ratio": float(diff / normalizer),
                    })

    feasible = len(violated_constraints) == 0
    return (feasible,
            sorted(violated_constraints),
            violations,
            violation_magnitudes)


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Minimum Graph Bisection (Armbruster et al. 2012)"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance"
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution"
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result"
    )
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    feasible, violated_constraints, violations, violation_magnitudes = check_feasibility(
        instance, solution
    )

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasible: {feasible}")
    if not feasible:
        print(f"Violated constraints: {violated_constraints}")
        for v in violations:
            print(f"  - {v}")
    print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    main()
