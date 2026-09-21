#!/usr/bin/env python3
"""
Feasibility checker for the PP-G2KP model (Furini, Malaguti, Thomopulos 2016).

Checks each hard constraint from the mathematical formulation against a
candidate solution (y variables: items_selected with item_id, plate_dims, copies).

Constraint numbering (top-to-bottom in the formulation section of math_model.txt):
  Constraint 1 (eq 2): Flow-balance for item plates (j in J_bar, j != 0)
  Constraint 2 (eq 3): Flow-balance for non-item plates (j in J \\ J_bar)
  Constraint 3 (eq 4): Original panel used at most once
  Constraint 4 (eq 5): Capacity y_j <= u_j
  Constraint 5 (eq 6): x variables non-negative integer
  Constraint 6 (eq 7): y variables non-negative integer
  Constraint 7 (Tier C): Objective consistency -- reported objective_value
                         must equal the recomputed total profit
                         sum_{j in J_bar} p_j * y_j (eq 1).

Since the solution only provides y values (items_selected), constraints involving
x variables (1, 2, 3, 5) are checked via necessary conditions derivable from the
flow-network structure:
  - Total item area <= panel area (from constraints 1+2+3 combined)
  - Each item fits within panel dimensions
  - Constraint 5 (x integrality): not directly checkable from y-only solution

Constraint 7 is an objective-recomputation defense: the objective (eq 1) is a
deterministic function of the y variables alone, and the solution carries every
y variable, so the true objective can be recomputed exactly and compared against
the program's self-reported objective_value. A mismatch beyond tolerance is a
score-gaming exploit and is reported as a constraint violation.
"""

import json
import argparse

TOL = 1e-5
EPS = 1e-5


def compute_violation(lhs, rhs, op):
    """
    Compute violation_amount for a constraint with given operator.
    op: '<=', '>=', '='
    Returns violation_amount (0 if satisfied within tolerance).
    """
    if op == '<=':
        return max(0.0, lhs - rhs)
    elif op == '>=':
        return max(0.0, rhs - lhs)
    elif op == '=':
        return abs(lhs - rhs)
    return 0.0


def record_violation(violations_list, magnitudes_list, constraint_idx, lhs, rhs, message):
    """Record a violation with normalized magnitude."""
    raw_excess = compute_violation(lhs, rhs, '<=') if lhs > rhs else compute_violation(lhs, rhs, '>=')
    # Use the raw difference
    raw_excess = abs(lhs - rhs) if raw_excess == 0 else raw_excess
    normalizer = max(abs(rhs), EPS)
    ratio = raw_excess / normalizer

    violations_list.append(message)
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw_excess),
        "normalizer": float(normalizer),
        "ratio": float(ratio),
    })


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the PP-G2KP model.
    Returns (feasible, violated_constraints, violations, violation_magnitudes).
    """
    violated_constraints = set()
    violations = []
    magnitudes = []

    L = instance['panel']['length']
    W = instance['panel']['width']
    items = instance['items']
    panel_area = L * W

    # Build item lookup by id
    item_by_id = {it['id']: it for it in items}

    # Parse solution
    items_selected = solution.get('items_selected', [])

    # Aggregate y values per item_id
    y_per_item = {}
    for entry in items_selected:
        iid = entry['item_id']
        copies = entry['copies']
        if iid in y_per_item:
            y_per_item[iid] += copies
        else:
            y_per_item[iid] = copies

    # ===================================================================
    # Constraint 6 (eq 7): y_j >= 0, integer for all j in J_bar
    # ===================================================================
    for entry in items_selected:
        iid = entry['item_id']
        copies = entry['copies']

        # Check non-negativity
        if copies < -TOL:
            lhs = copies
            rhs = 0.0
            violated_constraints.add(6)
            record_violation(violations, magnitudes, 6, lhs, rhs,
                             f"Constraint 6: y for item {iid} is negative ({copies})")

        # Check integrality
        if abs(copies - round(copies)) > TOL:
            lhs = abs(copies - round(copies))
            rhs = 0.0
            violated_constraints.add(6)
            record_violation(violations, magnitudes, 6, lhs, rhs,
                             f"Constraint 6: y for item {iid} is not integer ({copies})")

        # Check item exists in instance
        if iid not in item_by_id:
            violated_constraints.add(6)
            violations.append(
                f"Constraint 6: item {iid} in solution not found in instance")
            magnitudes.append({
                "constraint": 6,
                "lhs": float(iid),
                "rhs": 0.0,
                "raw_excess": 1.0,
                "normalizer": 1.0,
                "ratio": 1.0,
            })
            continue

        # Check plate_dims match item dimensions
        item = item_by_id[iid]
        pdims = entry.get('plate_dims', [])
        if len(pdims) == 2:
            if not (pdims[0] == item['length'] and pdims[1] == item['width']):
                violated_constraints.add(6)
                violations.append(
                    f"Constraint 6: plate_dims {pdims} for item {iid} "
                    f"do not match item dimensions [{item['length']}, {item['width']}]")
                magnitudes.append({
                    "constraint": 6,
                    "lhs": 0.0,
                    "rhs": 0.0,
                    "raw_excess": 1.0,
                    "normalizer": 1.0,
                    "ratio": 1.0,
                })

    # ===================================================================
    # Constraint 4 (eq 5): y_j <= u_j for all j in J_bar
    # ===================================================================
    for iid, total_copies in y_per_item.items():
        if iid not in item_by_id:
            continue  # already flagged above
        item = item_by_id[iid]
        u_j = item['copies']
        lhs = total_copies
        rhs = u_j
        violation_amount = compute_violation(lhs, rhs, '<=')
        if violation_amount > TOL:
            violated_constraints.add(4)
            record_violation(violations, magnitudes, 4, lhs, rhs,
                             f"Constraint 4: item {iid} uses {total_copies} copies "
                             f"but only {u_j} available")

    # ===================================================================
    # Constraints 1-3 (eqs 2-4): flow balance + panel use, using x (cuts)
    #
    # Solution provides cuts_used (x variables) and a plates list with
    # in_J_bar flags, so we check the actual hard constraints rather than
    # derived necessary conditions.
    # ===================================================================
    cuts_used = solution.get('cuts_used', [])
    plates_info = solution.get('plates', [])
    in_J_bar = {p['id']: bool(p.get('in_J_bar', False)) for p in plates_info}
    plate_ids = {p['id'] for p in plates_info}

    y_per_plate = {}
    for entry in items_selected:
        pid = entry.get('plate_id')
        if pid is None:
            continue
        y_per_plate[pid] = y_per_plate.get(pid, 0) + entry['copies']

    flow_in = {pid: 0 for pid in plate_ids}
    flow_out = {pid: 0 for pid in plate_ids}
    for c in cuts_used:
        cnt = c.get('count', 0)
        parent = c.get('parent_plate_id')
        c1 = c.get('child1_plate_id')
        c2 = c.get('child2_plate_id')
        if parent is not None:
            flow_out[parent] = flow_out.get(parent, 0) + cnt
        if c1 is not None:
            flow_in[c1] = flow_in.get(c1, 0) + cnt
        if c2 is not None:
            flow_in[c2] = flow_in.get(c2, 0) + cnt

    # Constraint 1 (eq 2): flow balance for item plates j in J_bar, j != 0
    for pid in plate_ids:
        if pid == 0 or not in_J_bar.get(pid, False):
            continue
        fin = flow_in.get(pid, 0)
        fout = flow_out.get(pid, 0)
        yj = y_per_plate.get(pid, 0)
        lhs = fin - fout - yj
        if lhs < -TOL:
            violated_constraints.add(1)
            record_violation(violations, magnitudes, 1, lhs, 0.0,
                             f"Constraint 1 (eq 2): plate {pid} flow balance "
                             f"in({fin}) - out({fout}) - y({yj}) = {lhs} < 0")

    # Constraint 2 (eq 3): flow balance for non-item plates j in J \\ J_bar
    for pid in plate_ids:
        if pid == 0 or in_J_bar.get(pid, False):
            continue
        fin = flow_in.get(pid, 0)
        fout = flow_out.get(pid, 0)
        lhs = fin - fout
        if lhs < -TOL:
            violated_constraints.add(2)
            record_violation(violations, magnitudes, 2, lhs, 0.0,
                             f"Constraint 2 (eq 3): plate {pid} flow balance "
                             f"in({fin}) - out({fout}) = {lhs} < 0")

    # Constraint 3 (eq 4): original panel (plate 0) used at most once
    panel_lhs = flow_out.get(0, 0)
    if in_J_bar.get(0, False):
        panel_lhs += y_per_plate.get(0, 0)
    if panel_lhs - 1 > TOL:
        violated_constraints.add(3)
        record_violation(violations, magnitudes, 3, panel_lhs, 1.0,
                         f"Constraint 3 (eq 4): panel use {panel_lhs} > 1")

    # ===================================================================
    # Constraint 5 (eq 6): x variables non-negative integer
    # Not directly checkable from solution (only y values provided).
    # No violation recorded.
    # ===================================================================

    # ===================================================================
    # Constraint 7 (Tier C, objective consistency):
    #   The objective (eq 1)  max  sum_{j in J_bar} p_j * y_j  is a
    #   deterministic function of the y variables alone, and the solution
    #   carries every y variable (items_selected -> copies). We therefore
    #   recompute the true objective exactly and reject as infeasible when
    #   the program's self-reported objective_value disagrees beyond a
    #   tight tolerance. This defends against LLM score-gaming exploits
    #   that fabricate objective_value (e.g. 0 or sys.float_info.max).
    #
    #   Full recompute applies because no x / second-stage variable enters
    #   the objective -- p_j * y_j depends only on the items kept.
    #
    #   The check is skipped (rather than firing a false positive) when the
    #   solution references an unknown item type: the recompute cannot be
    #   trusted, and such a solution is already infeasible via constraint 6,
    #   so no verdict is lost.
    # ===================================================================
    reported_obj = solution.get('objective_value')
    if reported_obj is not None:
        obj_checkable = True
        reported = None
        true_obj = 0.0
        try:
            reported = float(reported_obj)
            for iid, total_copies in y_per_item.items():
                if iid not in item_by_id:
                    obj_checkable = False
                    break
                true_obj += float(item_by_id[iid]['profit']) * float(total_copies)
        except (TypeError, ValueError):
            obj_checkable = False

        if obj_checkable and reported is not None:
            abs_diff = abs(reported - true_obj)
            # The objective is an integer (sum of integer profits times
            # integer copies); doubles represent it exactly at this scale.
            # Absolute floor 0.5 catches any integer-level mismatch; the
            # relative term guards huge objectives against solver float noise.
            tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(7)
                record_violation(violations, magnitudes, 7, reported, true_obj,
                                 f"Constraint 7 (objective consistency): reported "
                                 f"objective_value={reported} differs from recomputed "
                                 f"sum_j(p_j*y_j)={true_obj} "
                                 f"(|diff|={abs_diff:.6g}, tol={tol:.6g})")

    # Build final result
    violated_list = sorted(violated_constraints)
    feasible = len(violated_list) == 0

    return feasible, violated_list, violations, magnitudes


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for PP-G2KP model solutions."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    with open(args.instance_path) as f:
        instance = json.load(f)
    with open(args.solution_path) as f:
        solution = json.load(f)

    feasible, violated_constraints, violations, magnitudes = check_feasibility(
        instance, solution
    )

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": magnitudes,
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Feasibility: {feasible}")
    if violated_constraints:
        print(f"Violated constraints: {violated_constraints}")
        for v in violations:
            print(f"  - {v}")
    print(f"Result written to: {args.result_path}")


if __name__ == "__main__":
    main()
