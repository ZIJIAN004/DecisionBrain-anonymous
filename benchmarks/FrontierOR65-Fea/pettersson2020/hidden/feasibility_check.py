"""
Feasibility checker for Multi-Objective Integer Programming (MOIP) solutions.

Checks candidate solutions against the hard constraints from the mathematical
formulation in Pettersson & Ozlen (2020):

  min  f_1(x), ..., f_n(x)
  s.t. Ax <= C              -- Constraint (1): Knapsack capacity
       x in Z^c             -- Constraint (2): Integrality (binary)

Tier-C addition (Constraint 3): objective consistency.
The program reports `objective_value` (count of non-dominated vectors) and
a list of `non_dominated_objective_vectors`. Both are trivially forgeable
by an exploit (e.g. `objective_value = sys.float_info.max` with the
trivial empty selection). This constraint recomputes each claimed
objective vector from `solutions` and `objective_coefficients` and
verifies count parity, so a candidate that lies about either gets
rejected as infeasible.
"""

import argparse
import itertools
import json


def check_feasibility(instance, solution):
    """Return a dict with feasibility verdict and per-constraint details.

    Matches the original `main()` output shape exactly (the eval pipeline
    JSON contract): {feasible, violated_constraints, violations,
    violation_magnitudes}.
    """
    tol = 1e-5
    eps = 1e-5

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    num_items = instance["num_items"]
    weights = instance["weights"]
    capacity = instance["capacity"]
    obj_coeffs = instance["objective_coefficients"]
    num_objectives = instance["num_objectives"]
    variable_type = instance.get("variable_type", "binary")

    solutions = solution["solutions"]
    reported_obj_vectors = solution.get("non_dominated_objective_vectors", [])

    for sol_idx, x in enumerate(solutions):
        # =====================================================================
        # Constraint (1): Ax <= C  (knapsack capacity)
        # sum(weights[i] * x[i]) <= capacity
        # =====================================================================
        lhs = sum(weights[i] * x[i] for i in range(num_items))
        rhs = capacity
        violation_amount = max(0.0, lhs - rhs)
        if violation_amount > tol:
            if 1 not in violated_constraints:
                violated_constraints.add(1)
                violations.append(
                    f"Constraint 1 (capacity) violated: solution {sol_idx} "
                    f"has total weight {lhs} exceeding capacity {rhs}"
                )
            normalizer = max(abs(rhs), eps)
            violation_magnitudes.append({
                "constraint": 1,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(violation_amount / normalizer)
            })

        # =====================================================================
        # Constraint (2): x in Z^c  (integrality / binary)
        # Each variable must be integer; for binary problems, x[i] in {0, 1}
        # =====================================================================
        for i in range(num_items):
            val = x[i]
            if variable_type == "binary":
                # Must be 0 or 1
                violation_amount = min(abs(val - 0), abs(val - 1))
            else:
                # Must be integer
                violation_amount = abs(val - round(val))

            if violation_amount > tol:
                if 2 not in violated_constraints:
                    violated_constraints.add(2)
                    violations.append(
                        f"Constraint 2 (integrality) violated: solution {sol_idx}, "
                        f"variable x[{i}] = {val} is not "
                        f"{'binary' if variable_type == 'binary' else 'integer'}"
                    )
                # For integrality, RHS is the nearest feasible integer value
                if variable_type == "binary":
                    nearest = 0 if abs(val) < abs(val - 1) else 1
                else:
                    nearest = round(val)
                rhs_val = float(nearest)
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 2,
                    "lhs": float(val),
                    "rhs": rhs_val,
                    "raw_excess": float(violation_amount),
                    "normalizer": float(normalizer),
                    "ratio": float(violation_amount / normalizer)
                })

    # =========================================================================
    # Constraint (3): Objective consistency (Tier-C anti-exploit)
    #
    #   (a) `objective_value` must equal len(solutions) (it is the count of
    #       non-dominated objective vectors reported).
    #   (b) len(solutions) must equal len(non_dominated_objective_vectors).
    #   (c) For each i, k:
    #       non_dominated_objective_vectors[i][k]
    #         == sum_j (objective_coefficients[k][j] * solutions[i][j])
    #
    # Tolerance is 0.5 because every quantity here is integer-valued
    # (counts and sums of integer coefficients times binary variables);
    # an exploit lying by even a single unit must trip the check, while
    # legitimate solutions agree exactly.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    n_sols = len(solutions)
    n_vecs = len(reported_obj_vectors)

    def _record_obj_violation(message, lhs_val, rhs_val, raw_excess):
        if 3 not in violated_constraints:
            violated_constraints.add(3)
            violations.append(f"Constraint 3 (objective consistency) violated: {message}")
        try:
            rhs_f = float(rhs_val)
        except (TypeError, ValueError):
            rhs_f = 0.0
        try:
            lhs_f = float(lhs_val)
        except (TypeError, ValueError):
            lhs_f = 0.0
        try:
            excess_f = float(raw_excess)
        except (TypeError, ValueError):
            excess_f = 1.0
        normalizer = max(abs(rhs_f), eps)
        violation_magnitudes.append({
            "constraint": 3,
            "lhs": lhs_f,
            "rhs": rhs_f,
            "raw_excess": excess_f,
            "normalizer": float(normalizer),
            "ratio": float(excess_f / normalizer),
        })

    # (a) reported objective_value must equal the integer count of solutions
    try:
        reported_count = float(reported_obj)
        reported_finite = True
    except (TypeError, ValueError):
        reported_count = float("nan")
        reported_finite = False

    if not reported_finite or reported_count != reported_count:  # missing or NaN
        _record_obj_violation(
            f"objective_value={reported_obj!r} is missing or not numeric "
            f"(expected count of non-dominated vectors = {n_sols})",
            reported_obj if isinstance(reported_obj, (int, float)) else 0.0,
            n_sols,
            abs(n_sols),
        )
    elif abs(reported_count - n_sols) > 0.5:
        _record_obj_violation(
            f"objective_value={reported_count} differs from count of solutions={n_sols}",
            reported_count,
            n_sols,
            abs(reported_count - n_sols),
        )

    # (b) the two reported lists must have the same length
    if n_sols != n_vecs:
        _record_obj_violation(
            f"len(solutions)={n_sols} differs from "
            f"len(non_dominated_objective_vectors)={n_vecs}",
            n_sols,
            n_vecs,
            abs(n_sols - n_vecs),
        )

    # (c) each claimed obj vector must match the recomputed sum from its
    # solution. Only run when shapes line up so we never index out of range.
    if n_sols == n_vecs and n_sols > 0:
        for sol_idx, (x, claimed_vec) in enumerate(zip(solutions, reported_obj_vectors)):
            if not isinstance(claimed_vec, (list, tuple)) or len(claimed_vec) != num_objectives:
                _record_obj_violation(
                    f"solution {sol_idx}: claimed obj vector has wrong length "
                    f"({len(claimed_vec) if hasattr(claimed_vec, '__len__') else 'N/A'} "
                    f"vs expected {num_objectives})",
                    len(claimed_vec) if hasattr(claimed_vec, "__len__") else 0,
                    num_objectives,
                    1.0,
                )
                continue

            # itertools.compress is C-implemented and ~10x faster than a
            # Python multiply-and-sum loop on the binary path, which matters
            # because num_items can be 1e7.
            binary_path = (variable_type == "binary") and all(v in (0, 1) for v in x[:min(64, len(x))])
            for k in range(num_objectives):
                if binary_path:
                    recomputed_k = sum(itertools.compress(obj_coeffs[k], x))
                else:
                    recomputed_k = sum(obj_coeffs[k][j] * x[j] for j in range(num_items))
                try:
                    claimed_k = float(claimed_vec[k])
                except (TypeError, ValueError):
                    _record_obj_violation(
                        f"solution {sol_idx} objective {k}: claimed value "
                        f"{claimed_vec[k]!r} is not numeric (recomputed={recomputed_k})",
                        0.0, float(recomputed_k), 1.0,
                    )
                    continue
                diff = abs(claimed_k - float(recomputed_k))
                if diff > 0.5:
                    _record_obj_violation(
                        f"solution {sol_idx} objective {k}: claimed value "
                        f"{claimed_k} differs from recomputed "
                        f"sum_j(obj_coeffs[{k}][j]*x[j])={recomputed_k} "
                        f"(|diff|={diff})",
                        claimed_k, float(recomputed_k), diff,
                    )

    violated_list = sorted(violated_constraints)
    feasible = len(violated_list) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_list,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for MOIP knapsack solutions"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path, 'r') as f:
        instance = json.load(f)
    with open(args.solution_path, 'r') as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
