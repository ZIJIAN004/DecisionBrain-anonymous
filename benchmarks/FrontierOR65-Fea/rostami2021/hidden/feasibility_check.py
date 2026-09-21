"""
Feasibility checker for the Stochastic Single-Allocation Hub Location Problem
with Variable Allocation (DEF_V).

Checks ORIGINAL DEF_V constraints (the formulation (25)-(28) in the paper):
  Constraint 1 (tag 25): sum_{k!=i} x^s_{ik} = 1 - z_i, for all i in N, s in S_w
  Constraint 2 (tag 26): x^s_{ik} <= z_k, for all i,k in N (i!=k), s in S_w
  Constraint 3 (tag 27): z_i in {0,1}, for all i in N
  Constraint 4 (tag 28): x^s_{ik} in {0,1}, for all i,k in N, s in S_w

Additionally checks variant-specific constraints:
  Constraint 5 (tag 20): sum_k z_k >= 1  (SAHLP / CSAHLP: at least one hub)
  Constraint 6 (SApHMP only): sum_k z_k = p  (exactly p hubs)
  Constraint 7 (CSAHLP only): sum_i O^s_i * X^s_{ik} <= Gamma_k * z_k, for all k, s

Tier-C objective-consistency check (added to defeat LLM score-gaming exploits
that fabricate `objective_value`):
  Constraint 8: the reported objective_value must equal the obj recomputed
                from (z, x^s) using the DEF_V formula (Section 3.2 / page 8):
                  obj = sum_k f_k z_k
                      + sum_s p_s sum_{i,k, i!=k} c^s_{ik} x^s_{ik}
                      + sum_s p_s sum_{i,j} alpha w^s_{ij}
                            sum_{k,l in N} d_{k,l} X^s_{i,k} X^s_{j,l}
                where c^s_{ik} = d_{ik}(chi O^s_i + delta D^s_i),
                      X^s_{i,i} := z_i, X^s_{i,k} := x^s_{i,k} for k != i.
                Tolerance: max(1e-3, 1e-3 * |true_obj|) (0.1% relative,
                1e-3 absolute floor).  For SApHMP, the f_k z_k term is
                dropped (the fixed-cost layer is replaced by the p-hub
                cardinality constraint).

The flow variables y^s_{ikl} and flow-balance constraints (Eq 9-11) are an
Ernst-Krishnamoorthy linearization of the original quadratic inter-hub
transport terms; they are NOT part of the original DEF_V formulation
(25)-(28) and are therefore NOT checked here. LLM-evolved algorithms that
solve DEF_V without flow linearization still produce valid original-
formulation solutions, which this checker must accept.
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Check feasibility of a candidate solution for the stochastic SAHLP (DEF_V)."
    )
    parser.add_argument('--instance_path', type=str, required=True,
                        help='Path to the JSON file containing the data instance.')
    parser.add_argument('--solution_path', type=str, required=True,
                        help='Path to the JSON file containing the candidate solution.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Path to write the JSON file containing the feasibility result.')
    args = parser.parse_args()

    tol = 1e-5
    eps = 1e-5

    # =========================================================================
    # Load instance data
    # =========================================================================
    with open(args.instance_path, 'r') as f:
        instance = json.load(f)

    n = instance['n']
    num_scenarios = instance['num_scenarios']
    variant = instance['problem_info']['variant']
    scenarios = instance['scenarios']
    p_hubs = instance.get('p_hubs')
    hub_capacities = instance.get('hub_capacities')
    distances = instance['distances']
    fixed_costs = instance['fixed_costs']
    chi = instance['cost_parameters']['chi']
    alpha = instance['cost_parameters']['alpha']
    delta = instance['cost_parameters']['delta']

    N = range(n)
    S = range(num_scenarios)

    # Precompute scenario-dependent demands and outgoing flows O^s_i
    w_s = []
    O_s = []
    D_s = []
    for s in S:
        ws = scenarios[s]['demands']
        w_s.append(ws)
        Os = [sum(ws[i][j] for j in N) for i in N]
        Ds = [sum(ws[j][i] for j in N) for i in N]
        O_s.append(Os)
        D_s.append(Ds)

    # =========================================================================
    # Load candidate solution. Raw solver values (z_values, x_values) are
    # required so the binary-integrality and multi-allocation constraints
    # (Eq 25, 27, 28) are not vacuous.
    # =========================================================================
    with open(args.solution_path, 'r') as f:
        solution = json.load(f)

    def _reject(reason):
        result = {
            "feasible": False,
            "violated_constraints": [],
            "violations": [reason],
            "violation_magnitudes": [],
        }
        with open(args.result_path, 'w') as fout:
            json.dump(result, fout, indent=2)
        print(f"Solution is INFEASIBLE. {reason}")
        print(f"Result written to {args.result_path}")

    for required in ('z_values', 'x_values'):
        if required not in solution:
            _reject(
                f"Solution missing required field '{required}' "
                "(raw solver values needed for Eq 25, 27, 28)."
            )
            return

    # Continuous z values straight from the solver -- do NOT cast to int,
    # otherwise the binary check on Eq 27 becomes vacuous.
    z = [float(solution['z_values'][k]) for k in N]

    # Raw x values: nested dict keyed by str(s) -> str(i) -> str(k). Nested
    # keying lets node i map to multiple k values in the same scenario, so
    # Eq 25 violations are expressible.
    x_raw = solution['x_values']
    x_s = {}
    for s in S:
        s_dict = x_raw.get(str(s), {})
        for i in N:
            i_dict = s_dict.get(str(i), {})
            for k in N:
                if i != k:
                    x_s[(s, i, k)] = float(i_dict.get(str(k), 0.0))

    # Rounded views used only by the capacity / flow helpers; binary
    # integrality itself is verified tolerance-wise by Constraints 3 and 4.
    z_int = [int(round(zv)) for zv in z]

    def X(s, i, k):
        if i == k:
            return z_int[i]
        return int(round(x_s.get((s, i, k), 0.0)))

    # =========================================================================
    # Check constraints
    # =========================================================================
    violated_constraints = set()
    violations = []
    violation_magnitudes = []

    def record_violation(constraint_idx, msg, lhs_val, rhs_val, violation_amount):
        violated_constraints.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs_val), eps)
        ratio = violation_amount / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs_val),
            "rhs": float(rhs_val),
            "raw_excess": float(violation_amount),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # -------------------------------------------------------------------------
    # Constraint 1 (tag 25): sum_{k!=i} x^s_{ik} = 1 - z_i, for all i, s
    # -------------------------------------------------------------------------
    for s in S:
        for i in N:
            lhs = sum(x_s.get((s, i, k), 0) for k in N if k != i)
            rhs = 1 - z[i]
            violation_amount = abs(lhs - rhs)
            if violation_amount > tol:
                record_violation(
                    1,
                    f"Constraint 25 violated: scenario {s}, node {i}: "
                    f"sum_k x^s_{{i={i},k}} = {lhs}, expected 1 - z_{i} = {rhs}",
                    lhs, rhs, violation_amount
                )

    # -------------------------------------------------------------------------
    # Constraint 2 (tag 26): x^s_{ik} <= z_k, for all i,k (i!=k), s
    # -------------------------------------------------------------------------
    for s in S:
        for i in N:
            for k in N:
                if i != k:
                    x_val = x_s.get((s, i, k), 0)
                    z_val = z[k]
                    lhs = x_val
                    rhs = z_val
                    violation_amount = lhs - rhs  # <= constraint: how much LHS exceeds RHS
                    if violation_amount > tol:
                        record_violation(
                            2,
                            f"Constraint 26 violated: scenario {s}, node {i} allocated to "
                            f"non-hub node {k}: x^s_{{i={i},k={k}}} = {x_val}, z_{k} = {z_val}",
                            lhs, rhs, violation_amount
                        )

    # -------------------------------------------------------------------------
    # Constraint 3 (tag 27): z_i in {0,1}, for all i
    # Tolerance check on the raw solver value (Item 1 of reviewer note).
    # -------------------------------------------------------------------------
    for i in N:
        z_val = z[i]
        dist_to_binary = abs(z_val - round(z_val))
        if dist_to_binary > tol:
            record_violation(
                3,
                f"Constraint 27 violated: z_{i} = {z_val} is not binary",
                z_val, round(z_val), dist_to_binary
            )

    # -------------------------------------------------------------------------
    # Constraint 4 (tag 28): x^s_{ik} in {0,1}, for all i,k, s
    # Tolerance check on the raw solver value (Item 1 of reviewer note).
    # -------------------------------------------------------------------------
    for s in S:
        for i in N:
            for k in N:
                if i != k:
                    x_val = x_s.get((s, i, k), 0.0)
                    dist_to_binary = abs(x_val - round(x_val))
                    if dist_to_binary > tol:
                        record_violation(
                            4,
                            f"Constraint 28 violated: x^s_{{s={s},i={i},k={k}}} = {x_val} "
                            f"is not binary",
                            x_val, round(x_val), dist_to_binary
                        )

    # -------------------------------------------------------------------------
    # Constraint 5 (tag 20): sum_k z_k >= 1 (at least one hub)
    # This applies to SAHLP and CSAHLP variants (not SApHMP which uses constraint 6)
    # -------------------------------------------------------------------------
    if variant != "SApHMP":
        lhs = sum(z[k] for k in N)
        rhs = 1
        violation_amount = rhs - lhs  # >= constraint: how much RHS exceeds LHS
        if violation_amount > tol:
            record_violation(
                5,
                f"Constraint 20 violated: sum of z_k = {lhs}, need >= {rhs} (at least one hub)",
                lhs, rhs, violation_amount
            )

    # -------------------------------------------------------------------------
    # Constraint 6 (SApHMP only): sum_k z_k = p
    # -------------------------------------------------------------------------
    if variant == "SApHMP" and p_hubs is not None:
        lhs = sum(z[k] for k in N)
        rhs = p_hubs
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(
                6,
                f"SApHMP constraint violated: number of hubs = {lhs}, expected p = {rhs}",
                lhs, rhs, violation_amount
            )

    # -------------------------------------------------------------------------
    # Constraint 7 (CSAHLP only): sum_i O^s_i * X^s_{ik} <= Gamma_k * z_k
    # -------------------------------------------------------------------------
    if variant == "CSAHLP" and hub_capacities is not None:
        for s in S:
            for k in N:
                lhs = sum(O_s[s][i] * X(s, i, k) for i in N)
                rhs = hub_capacities[k] * z[k]
                violation_amount = lhs - rhs  # <= constraint
                if violation_amount > tol:
                    record_violation(
                        7,
                        f"CSAHLP capacity violated: scenario {s}, hub {k}: "
                        f"total flow = {lhs}, capacity = {rhs}",
                        lhs, rhs, violation_amount
                    )

    # Variable Domain Checks (auto-generated by add_domain_checks.py)
    # hubs: list[int] of node indices in [0, n) without duplicates, consistent with z_values.
    hubs = solution.get("hubs", [])
    seen_hubs = set()
    for h in hubs:
        if not isinstance(h, int) or h < 0 or h >= n:
            record_violation(
                7,
                f"Invalid node index in hubs: {h} (valid range 0..{n-1})",
                float(h if isinstance(h, (int, float)) else -1), 0.0, 1.0,
            )
        elif h in seen_hubs:
            record_violation(
                7,
                f"Duplicate node index in hubs: {h} (binary domain violated)",
                2.0, 1.0, 1.0,
            )
        else:
            seen_hubs.add(h)

    # allocations: dict[scenario_str -> dict[node_str -> hub_int]]
    # Each assigned hub must be in [0, n) and must be one of the opened hubs.
    allocations = solution.get("allocations", {})
    opened_hubs = set(seen_hubs)
    # If hubs list was empty/invalid, fall back to z_values indicator.
    if not opened_hubs:
        for k_idx in range(n):
            if z[k_idx] > 0.5:
                opened_hubs.add(k_idx)
    for s_key, alloc_dict in allocations.items():
        if not isinstance(alloc_dict, dict):
            continue
        for node_key, hub_val in alloc_dict.items():
            try:
                nd = int(node_key)
            except (TypeError, ValueError):
                nd = -1
            if not (0 <= nd < n):
                record_violation(
                    7,
                    f"Invalid node key in allocations[{s_key!r}]: {node_key!r}",
                    float(nd), 0.0, 1.0,
                )
                continue
            if not isinstance(hub_val, int) or hub_val < 0 or hub_val >= n:
                record_violation(
                    7,
                    f"Invalid hub index in allocations[{s_key!r}][{node_key!r}]: {hub_val}",
                    float(hub_val if isinstance(hub_val, (int, float)) else -1),
                    0.0, 1.0,
                )
            elif opened_hubs and hub_val not in opened_hubs:
                record_violation(
                    7,
                    f"allocations[{s_key!r}][{node_key!r}] assigns to {hub_val} "
                    f"but that node is not an opened hub",
                    float(hub_val), 1.0, 1.0,
                )

    # -------------------------------------------------------------------------
    # Constraint 8 (Tier-C obj-consistency): the reported objective_value
    # must match the obj recomputed from (z, x^s) using the DEF_V formula.
    # All variables determining the obj are in the solution, so this is a
    # full recompute (not a lower bound). Tolerance: max(1e-3, 1e-3*|true|).
    #
    # Quadratic inter-hub term simplification: define X^s_{i,k} = z_i if k==i
    # else x^s_{i,k}. Then the four sub-sums in the DEF_V obj collapse to
    #     sum_{k,l in N} d_{k,l} X^s_{i,k} X^s_{j,l}
    # so the inter-hub cost contribution is
    #     alpha * sum_{i,j} w^s_{ij} (sum_l X^s_{j,l} * h^s_i[l])
    # where h^s_i[l] = sum_k X^s_{i,k} d_{k,l}. This is O(n^3) per scenario.
    # -------------------------------------------------------------------------
    reported_obj_raw = solution.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported_obj = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported_obj = None

        if reported_obj is not None:
            # 1. fixed-cost layer (suppressed for SApHMP, per Section 10).
            if variant == "SApHMP":
                setup_cost = 0.0
            else:
                setup_cost = sum(fixed_costs[k] * z[k] for k in N)

            true_obj = float(setup_cost)
            for s in S:
                ps = float(scenarios[s]['probability'])
                ws = w_s[s]
                Os = O_s[s]
                Ds = D_s[s]

                # Build X^s[i] as a length-n vector for this scenario.
                Xs = []
                for i in N:
                    row = [0.0] * n
                    row[i] = z[i]
                    for k in N:
                        if k == i:
                            continue
                        row[k] = x_s.get((s, i, k), 0.0)
                    Xs.append(row)

                # Collection + distribution: sum_{i,k, i!=k} c^s_{ik} x^s_{ik}
                cd_term = 0.0
                for i in N:
                    di = distances[i]
                    coeff_i = chi * Os[i] + delta * Ds[i]
                    for k in N:
                        if i == k:
                            continue
                        xv = x_s.get((s, i, k), 0.0)
                        if xv == 0.0:
                            continue
                        cd_term += di[k] * coeff_i * xv

                # Inter-hub transfer: alpha * sum_{i,j} w^s_{ij}
                #                            * sum_{k,l} d_{k,l} X^s_{i,k} X^s_{j,l}
                ih_term = 0.0
                for i in N:
                    Xi = Xs[i]
                    h = [0.0] * n
                    for k in N:
                        xk = Xi[k]
                        if xk == 0.0:
                            continue
                        dk = distances[k]
                        for l in N:
                            h[l] += xk * dk[l]
                    wi = ws[i]
                    for j in N:
                        wij = wi[j]
                        if wij == 0:
                            continue
                        Xj = Xs[j]
                        acc = 0.0
                        for l in N:
                            xj = Xj[l]
                            if xj == 0.0:
                                continue
                            acc += xj * h[l]
                        ih_term += wij * acc
                ih_term *= alpha

                true_obj += ps * (cd_term + ih_term)

            abs_diff = abs(reported_obj - true_obj)
            # 0.1% relative with 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                record_violation(
                    8,
                    f"Objective consistency violated: reported objective_value="
                    f"{reported_obj} differs from recomputed DEF_V obj="
                    f"{true_obj} (|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                    reported_obj, true_obj, abs_diff,
                )

    # =========================================================================
    # Build result
    # =========================================================================
    feasible = len(violated_constraints) == 0

    # Aggregate violations by constraint index for concise messages
    constraint_msgs = {}
    for mag in violation_magnitudes:
        c = mag["constraint"]
        if c not in constraint_msgs:
            constraint_msgs[c] = []
    for v_msg in violations:
        # Extract constraint index from the corresponding magnitude entry
        pass

    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if feasible:
        print(f"Solution is FEASIBLE. Result written to {args.result_path}")
    else:
        print(f"Solution is INFEASIBLE. {len(violation_magnitudes)} violation(s) found.")
        print(f"Violated constraints: {sorted(violated_constraints)}")
        print(f"Result written to {args.result_path}")


if __name__ == "__main__":
    main()
