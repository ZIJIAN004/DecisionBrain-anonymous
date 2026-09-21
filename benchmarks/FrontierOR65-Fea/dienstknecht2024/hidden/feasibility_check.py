#!/usr/bin/env python3
"""
Feasibility checker for the Trade-Container Assignment Problem in Construction
Projects (TCAPCP) — period-based formulation (TCAPCP_Per).

Source: Dienstknecht & Briskorn (2024), EJOR 315(1), 324-337.

Checks all hard constraints from the mathematical model that can be evaluated
from the assignment (x) variables provided in the solution. Constraints
involving only internal auxiliary variables (y, z, f) that are not part of the
solution output are checked where they impose requirements on x, specifically
the dispersion limit which constrains the number of connected components
(clusters) of assigned containers.

Constraint numbering follows the paper's TCAPCP_Per formulation, counted
top-to-bottom:
  Constraint 1  = Eq (2):  Demand satisfaction
  Constraint 2  = Eq (3):  Container capacity
  Constraint 3  = Eq (4):  z <= x (source linking) — auxiliary, skip
  Constraint 4  = Eq (5):  y <= x first endpoint — auxiliary, skip
  Constraint 5  = Eq (6):  y <= x second endpoint — auxiliary, skip
  Constraint 6  = Eq (7):  Flow conservation — auxiliary, skip
  Constraint 7  = Eq (8):  Flow capacity forward — auxiliary, skip
  Constraint 8  = Eq (9):  Flow capacity reverse — auxiliary, skip
  Constraint 9  = Eq (10): Dispersion limit (max clusters)
  Constraint 10 = Eq (11): Re-assignment detection (existing container)
  Constraint 11 = Eq (12): Re-assignment detection (new container)
  Constraint 12 = Eq (13): Non-negativity of flows — auxiliary, skip
  Constraint 13 = Eq (19): Domain of r (0 <= r <= 1)
  Constraint 14 = Eq (15): Domain of x (binary, container available)
  Constraint 15 = Eq (16): Domain of y — auxiliary, skip
  Constraint 16 = Eq (17): Domain of z — auxiliary, skip
  Constraint 17 = Eq (18): Symmetry breaking — auxiliary, skip
  Constraint 18 = Eq (1):  Objective consistency (Tier C anti-gaming check)

Constraints 10, 11, 13 all constrain the r variable. Since r is derived from x
and constrained to [0,1], and because in any feasible solution the minimization
objective pushes r to its lower bound, we check that the implied r values are
within [0,1] (Constraint 13) and that the linking constraints (10, 11) hold,
i.e., r >= x_c^{j,p} - x_c^{j,p-1} for existing containers and r >= x_c^{j,p}
for newly available containers. Since r is free in [0,1] and these are only
lower bounds (the solution can always set r=1), the binding feasibility check
is whether these constraints are satisfiable, which they always are for binary x
as long as r is allowed in [0,1]. We still verify them for completeness by
computing the implied minimum r and checking it does not exceed 1.

Constraint 18 (objective consistency) is a Tier C defense added on top of the
original checker. The eval pipeline trusts the solution's self-reported
``objective_value``; LLM-evolved candidates have learned to fabricate it. The
objective Z = sum_{j} sum_{p=s_j+1}^{f_j} r^{j,p} counts trade re-assignments.
The r^{j,p} variables are NOT part of the solution output, but they are purely
auxiliary and fully determined by the x assignments: constraints (11)-(12)
impose lower bounds on each r^{j,p}, the [0,1] domain (19) leaves r free above
those bounds, and the minimization (1) therefore drives every r^{j,p} to
exactly its implied lower bound r_lower(j,p). Hence the objective of any x
assignment is exactly sum_{j,p} r_lower(j,p) — a full, exact recomputation,
not a loose bound. We reject solutions whose reported objective_value disagrees
with that recomputation. This is an append-only addition: it can only ADD
constraint 18 to the set of violations; it never alters constraints 1-17.
"""

import json
import argparse
from collections import deque

TOL = 1e-5
EPS = 1e-5


def load_json(path):
    with open(path, 'r') as fh:
        return json.load(fh)


def build_container_lookup(instance):
    """Build lookup structures for containers."""
    containers = {}
    for c in instance['containers']:
        containers[c['id']] = c
    return containers


def build_trade_lookup(instance):
    """Build lookup structures for trades."""
    trades = {}
    for t in instance['trades']:
        trades[t['id']] = t
    return trades


def get_available_containers(containers, p):
    """Return set of container IDs available in period p (C^p)."""
    return {
        cid for cid, c in containers.items()
        if c['availability_start'] <= p <= c['availability_end']
    }


def get_active_trades(trades, p):
    """Return set of trade IDs active in period p (J^p)."""
    return {
        tid for tid, t in trades.items()
        if t['start_period'] <= p <= t['end_period']
    }


def count_clusters(assigned_containers, containers):
    """
    Count the number of connected components (clusters) among the assigned
    containers using the adjacency graph.

    Returns the number of clusters.
    """
    if not assigned_containers:
        return 0

    assigned_set = set(assigned_containers)
    visited = set()
    num_clusters = 0

    for start in assigned_set:
        if start in visited:
            continue
        num_clusters += 1
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in containers[node]['adjacent_containers']:
                if neighbor in assigned_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return num_clusters


def make_violation(constraint_idx, lhs, rhs, raw_excess):
    """Create a violation magnitude entry."""
    normalizer = max(abs(rhs), EPS)
    return {
        "constraint": constraint_idx,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "raw_excess": float(raw_excess),
        "normalizer": float(normalizer),
        "ratio": float(raw_excess / normalizer),
    }


def check_feasibility(instance, solution):
    """
    Check all hard constraints and return feasibility result.
    """
    P = instance['problem_parameters']['num_periods']
    containers = build_container_lookup(instance)
    trades = build_trade_lookup(instance)

    assignments = solution['assignments']

    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    # Helper: get assigned containers for trade j in period p
    def get_assigned(j, p):
        j_str = str(j)
        p_str = str(p)
        if j_str not in assignments:
            return []
        if p_str not in assignments[j_str]:
            return []
        return assignments[j_str][p_str]

    # =========================================================================
    # Constraint 1 — Eq (2): Demand satisfaction (equality)
    # sum_{c in C^p} x_c^{j,p} = n_j  for all p=1..P, j in J^p
    # =========================================================================
    for p in range(1, P + 1):
        active_trades = get_active_trades(trades, p)
        avail_containers = get_available_containers(containers, p)
        for j in active_trades:
            assigned = get_assigned(j, p)
            n_j = trades[j]['container_demand']
            lhs = len(assigned)
            rhs = n_j
            violation_amount = abs(lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(1)
                violations.append(
                    f"Constraint 1 (Demand): Trade {j} in period {p} "
                    f"has {lhs} containers assigned, needs {rhs}"
                )
                violation_magnitudes.append(make_violation(1, lhs, rhs, violation_amount))

    # =========================================================================
    # Constraint 2 — Eq (3): Container capacity (<=)
    # sum_{j in J^p} x_c^{j,p} <= 1  for all p=1..P, c in C^p
    # =========================================================================
    for p in range(1, P + 1):
        active_trades = get_active_trades(trades, p)
        avail_containers = get_available_containers(containers, p)
        # Count how many trades each container is assigned to in this period
        container_usage = {}
        for j in active_trades:
            assigned = get_assigned(j, p)
            for c in assigned:
                container_usage[c] = container_usage.get(c, 0) + 1
        for c in avail_containers:
            usage = container_usage.get(c, 0)
            lhs = usage
            rhs = 1
            violation_amount = max(0, lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(2)
                trades_using = [
                    j for j in active_trades if c in get_assigned(j, p)
                ]
                violations.append(
                    f"Constraint 2 (Capacity): Container {c} in period {p} "
                    f"assigned to {usage} trades: {trades_using}"
                )
                violation_magnitudes.append(make_violation(2, lhs, rhs, violation_amount))

    # =========================================================================
    # Constraint 9 — Eq (10): Dispersion limit (<=)
    # Number of clusters of trade j in period p <= d_j^max
    # sum_{c in C^p} z_c^{j,p} <= d_j^max
    # The z variables identify cluster sources; the number of sources equals
    # the number of connected components in the subgraph of assigned containers.
    # =========================================================================
    for p in range(1, P + 1):
        active_trades = get_active_trades(trades, p)
        for j in active_trades:
            assigned = get_assigned(j, p)
            if not assigned:
                continue
            num_clusters = count_clusters(assigned, containers)
            d_max = trades[j]['max_dispersion']
            lhs = num_clusters
            rhs = d_max
            violation_amount = max(0, lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(9)
                violations.append(
                    f"Constraint 9 (Dispersion): Trade {j} in period {p} "
                    f"has {num_clusters} clusters, max allowed is {d_max}"
                )
                violation_magnitudes.append(make_violation(9, lhs, rhs, violation_amount))

    # =========================================================================
    # Constraint 10 — Eq (11): Re-assignment detection (existing container) (<=)
    # x_c^{j,p} - x_c^{j,p-1} <= r^{j,p}
    # for p=2..P, j in J^p with s_j < p, c in C^p ∩ C^{p-1}
    #
    # Constraint 11 — Eq (12): Re-assignment detection (new container) (<=)
    # x_c^{j,p} <= r^{j,p}
    # for p=2..P, j in J^p with s_j < p, c in C^p \ C^{p-1}
    #
    # Constraint 13 — Eq (19): Domain of r: 0 <= r^{j,p} <= 1
    #
    # These three constraints together require that the implied minimum value
    # of r^{j,p} (the maximum of all lower bounds from constraints 10 and 11)
    # must not exceed 1. Since x is binary and r can be at most 1, the only
    # way this fails is if the constraints are somehow contradictory, which
    # cannot happen for binary x with r in [0,1]. We verify anyway.
    #
    # The implied minimum r^{j,p} is also exactly the contribution of (j,p) to
    # the objective Z = sum r^{j,p} (the minimization drives r down to it), so
    # we accumulate it into `recomputed_objective` for the Constraint 18 check.
    # The (j,p) pairs visited here — p in [2,P], j active with s_j < p — are
    # exactly the index set p in [s_j+1, f_j] summed over in objective (1).
    # =========================================================================
    recomputed_objective = 0.0
    for p in range(2, P + 1):
        active_trades_p = get_active_trades(trades, p)
        avail_p = get_available_containers(containers, p)
        avail_pm1 = get_available_containers(containers, p - 1)

        for j in active_trades_p:
            s_j = trades[j]['start_period']
            if s_j >= p:
                continue  # trade starts at p, not a re-assignment period

            assigned_p = set(get_assigned(j, p))
            assigned_pm1 = set(get_assigned(j, p - 1))

            # Compute implied minimum r from constraint 10
            r_lower = 0.0
            for c in avail_p & avail_pm1:
                x_cur = 1 if c in assigned_p else 0
                x_prev = 1 if c in assigned_pm1 else 0
                diff = x_cur - x_prev
                if diff > r_lower:
                    r_lower = diff

            # Compute implied minimum r from constraint 11
            for c in avail_p - avail_pm1:
                x_cur = 1 if c in assigned_p else 0
                if x_cur > r_lower:
                    r_lower = x_cur

            # Accumulate the objective contribution of r^{j,p} (its minimum
            # value equals r_lower; the minimization objective realizes it).
            recomputed_objective += r_lower

            # Check constraint 13: r must be <= 1
            lhs = r_lower
            rhs = 1.0
            violation_amount = max(0, lhs - rhs)
            if violation_amount > TOL:
                violated_constraints.add(13)
                violations.append(
                    f"Constraint 13 (r domain): Trade {j} in period {p} "
                    f"requires r = {r_lower}, exceeds upper bound 1"
                )
                violation_magnitudes.append(make_violation(13, lhs, rhs, violation_amount))

    # =========================================================================
    # Constraint 14 — Eq (15): Domain of x (binary, container availability)
    # x_c^{j,p} in {0,1} for all p=1..P, j in J^p, c in C^p
    # This also implicitly requires that assigned containers are in C^p.
    # =========================================================================
    for p in range(1, P + 1):
        active_trades = get_active_trades(trades, p)
        avail_containers = get_available_containers(containers, p)
        for j in active_trades:
            assigned = get_assigned(j, p)
            for c in assigned:
                if c not in avail_containers:
                    violated_constraints.add(14)
                    violations.append(
                        f"Constraint 14 (x domain): Trade {j} in period {p} "
                        f"assigned to container {c} which is not available "
                        f"(available: periods {containers[c]['availability_start']}"
                        f"-{containers[c]['availability_end']})"
                    )
                    # LHS = 1 (assigned), RHS = 0 (not in domain)
                    violation_magnitudes.append(make_violation(14, 1, 0, 1.0))

    # Also check that trades have assignments for all their active periods
    for tid, t in trades.items():
        t_str = str(tid)
        for p in range(t['start_period'], t['end_period'] + 1):
            p_str = str(p)
            if t_str not in assignments or p_str not in assignments[t_str]:
                # Missing assignment for an active period — demand violation
                n_j = t['container_demand']
                if n_j > 0:
                    violated_constraints.add(1)
                    violations.append(
                        f"Constraint 1 (Demand): Trade {tid} has no assignment "
                        f"for active period {p}, needs {n_j} containers"
                    )
                    violation_magnitudes.append(make_violation(1, 0, n_j, n_j))

    # =========================================================================
    # Constraint 18 — Eq (1): Objective consistency (Tier C anti-gaming check)
    # Z = sum_{j} sum_{p=s_j+1}^{f_j} r^{j,p}
    #
    # `recomputed_objective` accumulated above is exactly sum_{j,p} r_lower(j,p),
    # the true objective implied by the x assignments (the r variables are
    # auxiliary, not in the solution, and the minimization drives each to its
    # lower bound). We reject any solution whose self-reported objective_value
    # disagrees with this exact recomputation. The objective is an integer
    # count of re-assignments, so a 0.5 absolute tolerance fires on any
    # mismatch of >= 1 (with a tiny relative floor for very large counts).
    # This is append-only: it can add constraint 18 but never touches 1-17.
    # =========================================================================
    reported_obj = solution.get('objective_value')
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(recomputed_objective)
            abs_diff = abs(reported - true_obj)
            tol = max(0.5, 1e-6 * abs(true_obj))
            if abs_diff > tol:
                violated_constraints.add(18)
                violations.append(
                    f"Constraint 18 (Objective consistency): reported "
                    f"objective_value={reported} differs from recomputed total "
                    f"re-assignments sum_{{j,p}} r_lower(j,p)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={tol:.6g})"
                )
                violation_magnitudes.append(
                    make_violation(18, reported, true_obj, abs_diff)
                )

    # =========================================================================
    # Build result
    # =========================================================================
    feasible = len(violated_constraints) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for TCAPCP (Dienstknecht & Briskorn 2024)"
    )
    parser.add_argument(
        '--instance_path', type=str, required=True,
        help='Path to the JSON file containing the data instance'
    )
    parser.add_argument(
        '--solution_path', type=str, required=True,
        help='Path to the JSON file containing the candidate solution'
    )
    parser.add_argument(
        '--result_path', type=str, required=True,
        help='Path to write the JSON file containing the feasibility result'
    )
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    result = check_feasibility(instance, solution)

    with open(args.result_path, 'w') as fh:
        json.dump(result, fh, indent=2)

    if result['feasible']:
        print(f"FEASIBLE — no constraint violations detected.")
    else:
        print(f"INFEASIBLE — {len(result['violated_constraints'])} constraint(s) violated: "
              f"{result['violated_constraints']}")
        for v in result['violations']:
            print(f"  - {v}")


if __name__ == '__main__':
    main()
