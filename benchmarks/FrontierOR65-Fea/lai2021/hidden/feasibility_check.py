"""
Feasibility checker for the Maximally Diverse Grouping Problem (MDGP).

Checks hard constraints from the mathematical formulation:
  Constraint (1): sum_{g=1}^{m} X_{ig} = 1, for all i  (each vertex assigned to exactly one group)
  Constraint (2): L_g <= sum_{i=1}^{N} X_{ig} <= U_g, for all g  (group capacity bounds)
  Constraint (3): X_{ig} in {0, 1}  (binary domain)
  Constraint (4): objective consistency -- the reported objective_value must equal the
                  objective recomputed from the assignment, i.e.
                  sum_{g} sum_{i<j, both in g} d_{ij}  (Tier C anti-score-gaming check)

This is the obj-recompute variant of feasibility_check.py. Constraints (1)-(3)
are byte-for-byte identical to the original; constraint (4) is the only addition.
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Feasibility checker for MDGP")
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON instance file')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON solution file')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the JSON feasibility result')
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    with open(args.solution_path, 'r') as f:
        solution = json.load(f)

    N = instance['N']
    m = instance['m']
    capacities = instance['group_capacities']
    L = [cap['L_g'] for cap in capacities]
    U = [cap['U_g'] for cap in capacities]

    assignment = solution.get('assignment', {})

    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # =========================================================================
    # Constraint (1): sum_{g=1}^{m} X_{ig} = 1 for all i = 1..N
    # Each vertex must be assigned to exactly one group.
    # =========================================================================
    # Count how many groups each vertex is assigned to
    vertex_group_count = {}
    for v_str, g in assignment.items():
        v = int(v_str)
        vertex_group_count[v] = vertex_group_count.get(v, 0) + 1

    # Check unassigned vertices
    for i in range(N):
        count = vertex_group_count.get(i, 0)
        if abs(count - 1) > tol:
            violated_constraints.add(1)
            rhs = 1.0
            lhs = float(count)
            violation_amount = abs(lhs - rhs)
            normalizer = max(abs(rhs), eps)
            if count == 0:
                violations.append(f"Vertex {i} is not assigned to any group")
            else:
                violations.append(f"Vertex {i} is assigned to {count} groups instead of 1")
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # =========================================================================
    # Constraint (2): L_g <= sum_{i=1}^{N} X_{ig} <= U_g for all g = 1..m
    # Group capacity bounds.
    # =========================================================================
    # Count group sizes from assignment
    group_sizes = [0] * m
    for v_str, g in assignment.items():
        g_int = int(g)
        if 0 <= g_int < m:
            group_sizes[g_int] += 1

    for g in range(m):
        size = group_sizes[g]
        # Check lower bound: L_g <= size  =>  size >= L_g
        lb_violation = L[g] - size
        if lb_violation > tol:
            violated_constraints.add(2)
            lhs = float(size)
            rhs = float(L[g])
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Group {g} has {size} members, below lower bound {L[g]}")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": lb_violation,
                "normalizer": normalizer,
                "ratio": lb_violation / normalizer
            })

        # Check upper bound: size <= U_g
        ub_violation = size - U[g]
        if ub_violation > tol:
            violated_constraints.add(2)
            lhs = float(size)
            rhs = float(U[g])
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Group {g} has {size} members, above upper bound {U[g]}")
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": ub_violation,
                "normalizer": normalizer,
                "ratio": ub_violation / normalizer
            })

    # =========================================================================
    # Constraint (3): X_{ig} in {0, 1} (binary domain)
    # In the solution representation, assignment maps vertex -> group (integer).
    # Check that all group indices are valid integers in {0, ..., m-1}.
    # =========================================================================
    for v_str, g in assignment.items():
        v = int(v_str)
        g_val = g if isinstance(g, int) else int(g)
        if g_val < 0 or g_val >= m:
            violated_constraints.add(3)
            lhs = float(g_val)
            rhs_lo = 0.0
            rhs_hi = float(m - 1)
            if g_val < 0:
                violation_amount = abs(g_val)
                rhs = rhs_lo
            else:
                violation_amount = g_val - (m - 1)
                rhs = rhs_hi
            normalizer = max(abs(rhs), eps)
            violations.append(
                f"Vertex {v} assigned to invalid group {g_val} (valid range: 0..{m-1})")
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": float(violation_amount),
                "normalizer": normalizer,
                "ratio": float(violation_amount) / normalizer
            })

    # =========================================================================
    # Constraint (4): Objective consistency (Tier C anti-score-gaming check).
    # The MDGP objective is fully determined by the assignment:
    #   obj = sum_{g} sum_{i<j, both assigned to g} d_{ij}
    # Every variable the objective depends on (the assignment) is present in
    # the solution, so a FULL recompute is possible. We reject the solution
    # when the reported objective_value disagrees with the recomputed value.
    # =========================================================================
    reported_obj = solution.get('objective_value')
    dist_upper = instance.get('distances_upper_triangular')
    if reported_obj is not None and dist_upper is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            try:
                # Flat upper-triangular index for pair (i, j), i < j, 0-indexed:
                #   d(0,1), d(0,2), ..., d(0,N-1), d(1,2), ..., d(N-2,N-1)
                def _flat_idx(i, j):
                    if i > j:
                        i, j = j, i
                    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

                # Group membership from the (canonical) assignment dict.
                members_by_group = {}
                for v_str, g in assignment.items():
                    v = int(v_str)
                    # Only in-range vertices contribute a valid pairwise distance.
                    if 0 <= v < N:
                        members_by_group.setdefault(int(g), []).append(v)

                true_obj = 0.0
                for g, mem in members_by_group.items():
                    mem = sorted(mem)
                    for a in range(len(mem)):
                        for b in range(a + 1, len(mem)):
                            true_obj += float(dist_upper[_flat_idx(mem[a], mem[b])])

                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with a 1e-3 absolute floor.
                obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    violated_constraints.add(4)
                    normalizer = max(abs(true_obj), eps)
                    violations.append(
                        f"Objective consistency violated: reported objective_value="
                        f"{reported} differs from recomputed within-group distance sum="
                        f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})")
                    violation_magnitudes.append({
                        "constraint": 4,
                        "lhs": reported,
                        "rhs": true_obj,
                        "raw_excess": abs_diff,
                        "normalizer": normalizer,
                        "ratio": abs_diff / normalizer
                    })
            except Exception:
                # Fail open: never crash where the original would not, and never
                # remove or alter constraint (1)-(3) verdicts. A malformed
                # solution that breaks the recompute is already caught above.
                pass

    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if feasible:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {sorted(violated_constraints)}")
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
