"""
Feasibility checker for the Discrete Truss Structure Design problem.

Based on the MILP formulation (5) from:
  Bollapragada, Ghattas, and Hooker (2001)
  "Optimal Design of Truss Structures by Logic-Based Branch and Cut"
  Operations Research, 49(1):42-51

Constraints checked (numbered top-to-bottom from formulation (5)):
  1. Equilibrium equations:  sum_i b[i][j]*s[i][l] = p[j][l]  for all j, l
  2. Compatibility equations: sum_j b[i][j]*d[j][l] = v[i][l]  for all i, l
  3. Hooke's law (linearized): (E_i/h_i)*A_i*v[i][l] = s[i][l]  for all i, l
  4. Exactly one discrete size per bar: A_i in {A_{i1},...,A_{iK}}
  5. Elongation bounds: v_i^L <= v[i][l] <= v_i^U  for all i, l
     (where v_i^L, v_i^U incorporate stress bounds as per paper)
  6. Displacement bounds: d_j^L <= d[j][l] <= d_j^U  for all j, l
  7. Integrality / linking: bars in same linking group must have same area
  8. Objective consistency: reported objective_value must equal
     c * sum_i h_i * A_i within tolerance (full recompute -- all variables
     required by the obj formula are present in the solution).

Note: The solution files store original-formulation variables (A_i, s_il, d_jl).
  Elongation v[i][l] is derived from compatibility: v[i][l] = sum_j b[i][j]*d[j][l].
  Constraints 2 and 3 from the MILP are checked using these derived elongations
  against Hooke's law with the chosen discrete area.
"""

import argparse
import json
import math
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_b_matrix(instance):
    """Build direction cosine matrix b[i][j] consistent with the algorithm code."""
    bars = instance["bars"]
    dofs = instance["degrees_of_freedom"]
    nodes = {n["node_id"]: n for n in instance["nodes"]}

    num_bars = len(bars)
    num_dofs = len(dofs)

    dof_map = {}
    for dof in dofs:
        dof_map[(dof["node"], dof["direction"])] = dof["dof_id"] - 1

    b = np.zeros((num_bars, num_dofs))

    dim = instance.get("dimension", 2)
    directions = ["x", "y"] if dim == 2 else ["x", "y", "z"]

    for bar_idx, bar in enumerate(bars):
        ni = bar["node_i"]
        nj = bar["node_j"]
        node_i = nodes[ni]
        node_j = nodes[nj]

        dx = node_j["x"] - node_i["x"]
        dy = node_j["y"] - node_i["y"]
        dz = 0.0
        if dim == 3:
            dz = node_j.get("z", 0.0) - node_i.get("z", 0.0)

        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            continue

        cos_vals = [dx / length, dy / length]
        if dim == 3:
            cos_vals.append(dz / length)

        for d_idx, direction in enumerate(directions):
            if (ni, direction) in dof_map:
                j = dof_map[(ni, direction)]
                b[bar_idx, j] = -cos_vals[d_idx]
            if (nj, direction) in dof_map:
                j = dof_map[(nj, direction)]
                b[bar_idx, j] = cos_vals[d_idx]

    return b


def extract_solution(instance, solution):
    """
    Extract bar areas, displacements, and bar forces from solution,
    handling both efficient_algorithm and gurobi solution formats.
    """
    num_bars = instance["num_bars"]
    num_dofs = instance["num_free_dofs"]
    num_loads = instance["num_loading_conditions"]

    # Bar areas
    areas = {}
    for entry in solution["bar_areas"]:
        areas[entry["bar_id"]] = entry["area"]

    # Displacements: d[j][l] (0-indexed j, 0-indexed l)
    d = np.zeros((num_dofs, num_loads))
    disp_data = solution["displacements"]

    if isinstance(disp_data, list) and len(disp_data) > 0:
        if isinstance(disp_data[0], dict):
            # Gurobi format: list of {dof_id, load, value}
            for entry in disp_data:
                j = entry["dof_id"] - 1
                l = entry["load"] - 1
                d[j, l] = entry["value"]
        elif isinstance(disp_data[0], list):
            # Efficient algorithm format: list of lists, d[j] = [val_l1, val_l2, ...]
            for j, vals in enumerate(disp_data):
                for l, val in enumerate(vals):
                    d[j, l] = val
        else:
            # Single load, flat list of values per DOF
            for j, val in enumerate(disp_data):
                d[j, 0] = val

    # Bar forces: s[i][l] (0-indexed i, 0-indexed l)
    s = np.zeros((num_bars, num_loads))
    force_data = solution["bar_forces"]

    if isinstance(force_data, list) and len(force_data) > 0:
        if isinstance(force_data[0], dict):
            # Gurobi format: list of {bar_id, load, force}
            for entry in force_data:
                i = entry["bar_id"] - 1
                l = entry["load"] - 1
                s[i, l] = entry["force"]
        elif isinstance(force_data[0], list):
            # Efficient algorithm format: list of lists, s[i] = [val_l1, val_l2, ...]
            for i, vals in enumerate(force_data):
                for l, val in enumerate(vals):
                    s[i, l] = val
        else:
            # Single load, flat list of values per bar
            for i, val in enumerate(force_data):
                s[i, 0] = val

    return areas, d, s


def check_feasibility(instance, solution):
    """
    Check all hard constraints from formulation (5) of the paper.
    Returns (feasible, violated_constraints, violations, violation_magnitudes).
    """
    tol = 1e-5
    eps = 1e-5

    num_bars = instance["num_bars"]
    num_dofs = instance["num_free_dofs"]
    num_loads = instance["num_loading_conditions"]
    bars = instance["bars"]
    discrete_areas = instance["discrete_areas"]
    E = instance["material_properties"]["modulus_of_elasticity"]
    cost_density = instance["material_properties"]["cost_density"]

    # Stress bounds per bar
    stress_lb = []
    stress_ub = []
    if "bar_specific_stress_bounds" in instance:
        for sb in instance["bar_specific_stress_bounds"]:
            stress_lb.append(sb["lower"])
            stress_ub.append(sb["upper"])
    else:
        sl = instance["stress_bounds"]["lower"]
        su = instance["stress_bounds"]["upper"]
        stress_lb = [sl] * num_bars
        stress_ub = [su] * num_bars

    # Displacement bounds
    d_lb_val = instance["displacement_bounds"]["lower"]
    d_ub_val = instance["displacement_bounds"]["upper"]

    # Compute elongation bounds incorporating stress bounds
    # v_i^L = (h_i / E_i) * sigma_i^L
    # v_i^U = (h_i / E_i) * sigma_i^U
    v_lb = np.zeros(num_bars)
    v_ub = np.zeros(num_bars)
    for i, bar in enumerate(bars):
        h_i = bar["length"]
        v_lb[i] = (h_i / E) * stress_lb[i]
        v_ub[i] = (h_i / E) * stress_ub[i]

    # Build direction cosine matrix
    b = build_b_matrix(instance)

    # Build load vector p[j][l]
    p = np.zeros((num_dofs, num_loads))
    for load_idx, lc in enumerate(instance["loading_conditions"]):
        for ld in lc["loads"]:
            dof_idx = ld["dof_id"] - 1
            p[dof_idx, load_idx] = ld["force"]

    # Extract solution
    area_map, d, s = extract_solution(instance, solution)

    # Ordered bar areas (0-indexed)
    A = np.zeros(num_bars)
    for i, bar in enumerate(bars):
        A[i] = area_map[bar["bar_id"]]

    # Derive elongations from compatibility: v[i][l] = sum_j b[i][j] * d[j][l]
    v = np.zeros((num_bars, num_loads))
    for i in range(num_bars):
        for l in range(num_loads):
            v[i, l] = sum(b[i, j] * d[j, l] for j in range(num_dofs))

    # Linking groups
    linking_groups = instance.get("linking_groups", [])

    violated_set = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, message, lhs, rhs, violation_amount):
        violated_set.add(constraint_idx)
        violations.append(message)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # =========================================================================
    # Constraint 1: Equilibrium equations
    #   sum_i b[i][j] * s[i][l] = p[j][l]  for all j, l
    # =========================================================================
    for j in range(num_dofs):
        for l in range(num_loads):
            lhs = sum(b[i, j] * s[i, l] for i in range(num_bars))
            rhs = p[j, l]
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                dof_info = instance["degrees_of_freedom"][j]
                record_violation(
                    1,
                    f"Equilibrium violated at DOF {dof_info['dof_id']} "
                    f"(node {dof_info['node']}, dir {dof_info['direction']}), "
                    f"load {l+1}: LHS={lhs:.6f}, RHS={rhs:.6f}, "
                    f"diff={violation_amount:.6f}",
                    lhs, rhs, violation_amount,
                )

    # =========================================================================
    # Constraint 2: Compatibility equations
    #   sum_j b[i][j] * d[j][l] = v[i][l]  for all i, l
    #
    # Since v[i][l] is derived from compatibility, this is satisfied by
    # construction. However we still check Hooke's law (Constraint 3) which
    # ties together A_i, v[i][l], and s[i][l].
    #
    # We verify compatibility indirectly: the elongation used to check other
    # constraints is computed directly from displacements, so compatibility
    # is inherently satisfied. We include it for completeness by checking
    # that the force s[i][l] is consistent with A_i * v[i][l] via Hooke's law.
    # That check is Constraint 3.
    # =========================================================================
    # Compatibility is satisfied by construction of v from d, so no separate
    # violation is possible here. Constraint 2 is trivially satisfied.

    # =========================================================================
    # Constraint 3: Hooke's law
    #   (E_i / h_i) * A_i * v[i][l] = s[i][l]  for all i, l
    #
    # IMPORTANT NOTE on the MILP formulation (5) vs original formulation (1):
    # The MILP linearises Hooke's law using disaggregated elongation variables
    #   v_{ik,l}, yielding:  (E_i/h_i) * sum_k A_{ik} * v_{ik,l} = s_{i,l}
    # The solution files store only the aggregate A_i, s_{i,l}, and d_{j,l}
    # (not the disaggregated v_{ik,l}), so we check the equivalent original
    # nonlinear form:  (E_i/h_i) * A_i * v_{i,l} = s_{i,l}.
    #
    # For near-zero bars (A_i = A_{i1} ≈ 0, representing effectively absent
    # bars), MILP solver tolerances can cause small residual forces that are
    # negligible relative to the overall force magnitudes but produce large
    # absolute violations in Hooke's law.  To avoid false positives we scale
    # the tolerance by the maximum absolute force across all bars/loads.
    # =========================================================================
    max_abs_force = max(
        (abs(s[i, l]) for i in range(num_bars) for l in range(num_loads)),
        default=1.0,
    )
    hooke_tol = max(tol, 1e-4 * max_abs_force)
    for i in range(num_bars):
        h_i = bars[i]["length"]
        for l in range(num_loads):
            lhs = (E / h_i) * A[i] * v[i, l]
            rhs = s[i, l]
            violation_amount = abs(lhs - rhs)
            if violation_amount > hooke_tol:
                record_violation(
                    3,
                    f"Hooke's law violated for bar {bars[i]['bar_id']}, "
                    f"load {l+1}: (E/h)*A*v={lhs:.6f}, s={rhs:.6f}, "
                    f"diff={violation_amount:.6f}",
                    lhs, rhs, violation_amount,
                )

    # =========================================================================
    # Constraint 4: Exactly one discrete size per bar
    #   A_i in {A_{i1}, ..., A_{iK}}  for all i
    # =========================================================================
    for i in range(num_bars):
        area_val = A[i]
        min_dist = min(abs(area_val - da) for da in discrete_areas)
        if min_dist > tol:
            # Find nearest for reporting
            nearest = min(discrete_areas, key=lambda da: abs(da - area_val))
            violation_amount = min_dist
            record_violation(
                4,
                f"Bar {bars[i]['bar_id']} area {area_val:.6f} is not a "
                f"discrete area (nearest: {nearest})",
                area_val, nearest, violation_amount,
            )

    # =========================================================================
    # Constraint 5: Elongation bounds (incorporating stress bounds)
    #   v_i^L <= v[i][l] <= v_i^U  for all i, l
    # =========================================================================
    for i in range(num_bars):
        for l in range(num_loads):
            vil = v[i, l]
            # Lower bound: v_i^L <= v[i][l]
            if v_lb[i] - vil > tol:
                violation_amount = v_lb[i] - vil
                record_violation(
                    5,
                    f"Elongation lower bound violated for bar "
                    f"{bars[i]['bar_id']}, load {l+1}: "
                    f"v={vil:.6f} < v_L={v_lb[i]:.6f}",
                    vil, v_lb[i], violation_amount,
                )
            # Upper bound: v[i][l] <= v_i^U
            if vil - v_ub[i] > tol:
                violation_amount = vil - v_ub[i]
                record_violation(
                    5,
                    f"Elongation upper bound violated for bar "
                    f"{bars[i]['bar_id']}, load {l+1}: "
                    f"v={vil:.6f} > v_U={v_ub[i]:.6f}",
                    vil, v_ub[i], violation_amount,
                )

    # =========================================================================
    # Constraint 6: Displacement bounds
    #   d_j^L <= d[j][l] <= d_j^U  for all j, l
    # =========================================================================
    if d_lb_val is not None and d_ub_val is not None:
        for j in range(num_dofs):
            for l in range(num_loads):
                djl = d[j, l]
                dof_info = instance["degrees_of_freedom"][j]
                # Lower bound
                if d_lb_val - djl > tol:
                    violation_amount = d_lb_val - djl
                    record_violation(
                        6,
                        f"Displacement lower bound violated at DOF "
                        f"{dof_info['dof_id']} (node {dof_info['node']}, "
                        f"dir {dof_info['direction']}), load {l+1}: "
                        f"d={djl:.6f} < d_L={d_lb_val:.6f}",
                        djl, d_lb_val, violation_amount,
                    )
                # Upper bound
                if djl - d_ub_val > tol:
                    violation_amount = djl - d_ub_val
                    record_violation(
                        6,
                        f"Displacement upper bound violated at DOF "
                        f"{dof_info['dof_id']} (node {dof_info['node']}, "
                        f"dir {dof_info['direction']}), load {l+1}: "
                        f"d={djl:.6f} > d_U={d_ub_val:.6f}",
                        djl, d_ub_val, violation_amount,
                    )

    # =========================================================================
    # Constraint 7: Integrality / Linking constraints
    #   Bars in the same linking group must have the same area.
    # =========================================================================
    for group in linking_groups:
        bar_ids = group["bar_ids"]
        if len(bar_ids) < 2:
            continue
        ref_area = area_map[bar_ids[0]]
        for bid in bar_ids[1:]:
            other_area = area_map[bid]
            violation_amount = abs(ref_area - other_area)
            if violation_amount > tol:
                record_violation(
                    7,
                    f"Linking group {group.get('group_id', '?')}: "
                    f"bar {bar_ids[0]} area={ref_area}, "
                    f"bar {bid} area={other_area} (should be equal)",
                    other_area, ref_area, violation_amount,
                )

    # =========================================================================
    # Constraint 8: Objective consistency (Tier C anti-exploit defense)
    #   reported objective_value must equal  c * sum_i h_i * A_i  within
    #   tolerance. Full recompute: every variable required by the objective
    #   formula (A_i, h_i, c) is available from the solution + instance, so
    #   we can recompute exactly rather than just lower-bound.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            true_obj = float(
                cost_density * sum(bars[i]["length"] * A[i] for i in range(num_bars))
            )
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance, 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(
                    8,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed c*sum_i(h_i*A_i)="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})",
                    reported, true_obj, abs_diff,
                )
        elif reported is not None:
            # Non-finite reported obj (inf/nan) -- always a violation since the
            # truss objective is a finite positive sum of finite quantities.
            true_obj = float(
                cost_density * sum(bars[i]["length"] * A[i] for i in range(num_bars))
            )
            record_violation(
                8,
                f"Objective consistency violated: reported objective_value="
                f"{reported} is not finite; recomputed c*sum_i(h_i*A_i)={true_obj}",
                reported, true_obj, float("inf"),
            )

    feasible = len(violated_set) == 0
    violated_constraints = sorted(violated_set)

    return feasible, violated_constraints, violations, violation_magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for discrete truss design "
                    "(Bollapragada et al. 2001)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path for the output feasibility result JSON")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    feasible, violated_constraints, violations, violation_magnitudes = \
        check_feasibility(instance, solution)

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {'PASS' if feasible else 'FAIL'}")
    if not feasible:
        print(f"Violated constraints: {violated_constraints}")
        for v in violations:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
