"""
Feasibility checker for Cardinality-constrained Mean-CVaR Portfolio Optimization.
Kobayashi, Takano, Nakata (2021) — Problem (7).

Constraints numbered top-to-bottom from the formulation section:
  1. (7b) CVaR: v >= (1/(1-beta)) * sum_s p_s * max(0, -(r^(s))^T x - a)
  2. (7c) Logical linking: z_n = 0 => x_n = 0  (for all n)
  3. Expected return: mu^T x >= mu_bar  (Eq. 6, part of x in X)
  4. Budget: sum x_n = 1  (part of x in X)
  5. Non-negativity: x_n >= 0 for all n  (part of x in X)
  6. Binary: z_n in {0,1} for all n  (part of z in Z_N^k)
  7. Cardinality: sum z_n <= k  (part of z in Z_N^k)
  8. v >= 0  (variable domain)
  9. Objective consistency (Tier C): reported objective_value must equal the
     recomputed objective (7a) = (1/(2*gamma)) * x^T x + a + v. Guards against
     LLM score-gaming exploits that report a fabricated objective_value while
     keeping the routes/decisions feasible.
"""

import argparse
import json
import math

TOL = 1e-5
EPS = 1e-5


def load_instance(path):
    with open(path) as f:
        data = json.load(f)
    N = data["N"]
    S = data["S"]
    k = data["k"]
    beta = data["beta"]
    gamma = data["gamma"]
    mu_bar = data["mu_bar"]
    mu = data["mu"]
    p_s_raw = data["p_s"]
    if isinstance(p_s_raw, list):
        p = p_s_raw
    else:
        p = [p_s_raw] * S
    scenarios = data["scenarios"]
    return N, S, k, beta, gamma, mu_bar, mu, p, scenarios


def load_solution(path):
    with open(path) as f:
        return json.load(f)


def add_violation(violations_list, magnitudes_list, constraint_idx, message, lhs, rhs, violation_amount):
    normalizer = max(abs(rhs), EPS)
    ratio = violation_amount / normalizer
    violations_list.append((constraint_idx, message))
    magnitudes_list.append({
        "constraint": constraint_idx,
        "lhs": lhs,
        "rhs": rhs,
        "raw_excess": violation_amount,
        "normalizer": normalizer,
        "ratio": ratio,
    })


def check_feasibility(instance_path, solution_path):
    N, S, k, beta, gamma, mu_bar, mu, p, scenarios = load_instance(instance_path)
    sol = load_solution(solution_path)

    x = sol.get("x")
    z = sol.get("z")
    a = sol.get("a")
    v = sol.get("v")

    # Handle null/missing solution
    if x is None or z is None:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution contains null values (no feasible solution found)"],
            "violation_magnitudes": [],
        }

    # If a or v not in solution, try to infer or mark infeasible
    if a is None or v is None:
        return {
            "feasible": False,
            "violated_constraints": [],
            "violations": ["Solution missing auxiliary variables a and/or v"],
            "violation_magnitudes": [],
        }

    violations = []  # list of (constraint_idx, message)
    magnitudes = []  # list of dicts

    # --- Constraint 1: (7b) CVaR ---
    # v >= (1/(1-beta)) * sum_s p_s * max(0, -(r^(s))^T x - a)
    cvar_rhs = 0.0
    for s in range(S):
        loss_s = -sum(scenarios[s][n] * x[n] for n in range(N)) - a
        cvar_rhs += p[s] * max(0.0, loss_s)
    cvar_rhs /= (1.0 - beta)
    lhs_val = v
    rhs_val = cvar_rhs
    viol = rhs_val - lhs_val  # >= constraint: violation if RHS > LHS
    if viol > TOL:
        add_violation(violations, magnitudes, 1,
                      f"CVaR constraint violated: v={lhs_val:.6f} < required {rhs_val:.6f}",
                      lhs_val, rhs_val, viol)

    # --- Constraint 2: (7c) Logical linking ---
    # z_n = 0 => x_n = 0, equivalently x_n <= z_n when z_n = 0
    for n in range(N):
        if abs(z[n]) < TOL and x[n] > TOL:
            # x_n should be 0 but isn't; treat as x_n <= 0 violated
            viol_amt = x[n]
            add_violation(violations, magnitudes, 2,
                          f"Linking violated: z[{n}]=0 but x[{n}]={x[n]:.6f}",
                          x[n], 0.0, viol_amt)

    # --- Constraint 3: Expected return ---
    # mu^T x >= mu_bar
    mu_x = sum(mu[n] * x[n] for n in range(N))
    viol = mu_bar - mu_x  # >= constraint: violation if RHS > LHS
    if viol > TOL:
        add_violation(violations, magnitudes, 3,
                      f"Expected return violated: mu^T x={mu_x:.6f} < mu_bar={mu_bar:.6f}",
                      mu_x, mu_bar, viol)

    # --- Constraint 4: Budget ---
    # sum x_n = 1
    sum_x = sum(x[n] for n in range(N))
    viol = abs(sum_x - 1.0)
    if viol > TOL:
        add_violation(violations, magnitudes, 4,
                      f"Budget violated: sum(x)={sum_x:.6f} != 1.0",
                      sum_x, 1.0, viol)

    # --- Constraint 5: Non-negativity ---
    # x_n >= 0 for all n
    for n in range(N):
        if x[n] < -TOL:
            viol_amt = -x[n]
            add_violation(violations, magnitudes, 5,
                          f"Non-negativity violated: x[{n}]={x[n]:.6f} < 0",
                          x[n], 0.0, viol_amt)

    # --- Constraint 6: Binary ---
    # z_n in {0,1} for all n
    for n in range(N):
        dist = min(abs(z[n] - 0.0), abs(z[n] - 1.0))
        if dist > TOL:
            add_violation(violations, magnitudes, 6,
                          f"Binary violated: z[{n}]={z[n]} not in {{0,1}}",
                          z[n], round(z[n]), dist)

    # --- Constraint 7: Cardinality ---
    # sum z_n <= k
    sum_z = sum(z[n] for n in range(N))
    viol = sum_z - k  # <= constraint: violation if LHS > RHS
    if viol > TOL:
        add_violation(violations, magnitudes, 7,
                      f"Cardinality violated: sum(z)={sum_z} > k={k}",
                      float(sum_z), float(k), viol)

    # --- Constraint 8: v >= 0 ---
    if v < -TOL:
        viol_amt = -v
        add_violation(violations, magnitudes, 8,
                      f"v non-negativity violated: v={v:.6f} < 0",
                      v, 0.0, viol_amt)

    # --- Constraint 9: Objective consistency (Tier C: obj recompute) ---
    # (7a) objective = (1/(2*gamma)) * x^T x + a + v.
    # Every variable the objective depends on (x, a, v) is present in the
    # solution and gamma is in the instance, so a full recompute is exact.
    # Reject when the self-reported objective_value disagrees with the
    # recomputed value beyond a 0.1% relative tolerance (1e-3 absolute floor).
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and math.isfinite(reported):
            reg_term = sum(x[n] * x[n] for n in range(N)) / (2.0 * gamma)
            true_obj = reg_term + a + v
            abs_diff = abs(reported - true_obj)
            tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > tol:
                add_violation(violations, magnitudes, 9,
                              f"Objective consistency violated: reported objective_value={reported} "
                              f"differs from recomputed (1/(2*gamma))*x^T x + a + v={true_obj:.6f} "
                              f"(|diff|={abs_diff:.6g}, tol={tol:.6g})",
                              reported, true_obj, abs_diff)
        elif reported is not None:
            # Non-finite reported objective (inf / nan) — a degenerate exploit form.
            reg_term = sum(x[n] * x[n] for n in range(N)) / (2.0 * gamma)
            true_obj = reg_term + a + v
            add_violation(violations, magnitudes, 9,
                          f"Objective consistency violated: reported objective_value={reported} "
                          f"is non-finite; recomputed (1/(2*gamma))*x^T x + a + v={true_obj:.6f}",
                          reported, true_obj, float("inf"))

    # Build result
    violated_indices = sorted(set(entry[0] for entry in violations))
    # Aggregate messages per constraint index
    messages = []
    for idx in violated_indices:
        msgs = [msg for (ci, msg) in violations if ci == idx]
        messages.append("; ".join(msgs))

    feasible = len(violated_indices) == 0

    return {
        "feasible": feasible,
        "violated_constraints": violated_indices,
        "violations": messages,
        "violation_magnitudes": magnitudes if not feasible else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for Cardinality-constrained Mean-CVaR Portfolio Optimization"
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path to solution JSON file")
    parser.add_argument("--result_path", required=True, help="Path for output feasibility result JSON file")
    args = parser.parse_args()

    result = check_feasibility(args.instance_path, args.solution_path)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        for msg in result["violations"]:
            print(f"  - {msg}")
    print(f"Written to {args.result_path}")


if __name__ == "__main__":
    main()
