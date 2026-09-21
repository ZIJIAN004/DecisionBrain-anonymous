"""
Feasibility checker for sparse portfolio selection solutions.

Checks candidate solutions against the hard constraints from:
  Bertsimas and Cory-Wright (2022),
  "A Scalable Algorithm for Sparse Portfolio Selection"

Constraint numbering (first unique appearance, top to bottom in formulation section):
  C1:  e^T x = 1                       (budget constraint, Problem 1)
  C2:  l <= A x <= u                    (linear constraints, Problem 2)
       - C2: mu^T x >= r_bar            (min return, when applicable)
       - C2: x_i >= l_i * z_i           (min investment, when applicable)
       - C2: x_i <= u_i * z_i           (max investment, when applicable)
  C3:  ||x||_0 <= k                     (cardinality on x, Problem 2)
  C4:  x_i = 0 if z_i = 0  forall i    (linking, Problem 3)
  C5:  x_i^2 <= z_i * theta_i  forall i (perspective, Problem 5; gurobi only)
  C6:  x_i >= 0  forall i              (non-negativity of x, domain)
  C7:  z_i in {0,1}  forall i          (binary z, domain)
  C8:  e^T z <= k                       (cardinality on z, domain)
  C9:  theta_i >= 0  forall i          (non-negativity of theta, domain; gurobi only)
  C10: objective_value matches the recomputed objective (Tier C obj-recompute)
"""

import argparse
import json
import numpy as np


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    n = instance["n"]
    k = instance["k"]
    mu = np.array(instance["mu"], dtype=float)

    x = np.array(solution["x"], dtype=float)
    # z and theta are reformulation auxiliaries (perspective MISOCO); the
    # original Problem (4) only has x. We tolerate their absence.
    z = (np.array(solution["z"], dtype=float)
         if "z" in solution and solution["z"] is not None else None)
    has_theta = "theta" in solution and solution["theta"] is not None
    theta = np.array(solution["theta"], dtype=float) if has_theta else None

    constr = instance.get("constraints", {})
    has_min_return = constr.get("has_min_return_constraint", False)
    r_bar = constr.get("r_bar", None)
    has_min_inv = constr.get("has_min_investment_constraint", False)
    l_min = constr.get("l_min_investment", None)
    u_max = constr.get("u_max_investment", None)

    violations = []
    violation_magnitudes = []
    violated_set = set()

    def record(constraint_idx, msg, lhs, rhs, raw_excess):
        violated_set.add(constraint_idx)
        violations.append(msg)
        normalizer = max(abs(rhs), eps)
        ratio = raw_excess / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": float(lhs),
            "rhs": float(rhs),
            "raw_excess": float(raw_excess),
            "normalizer": float(normalizer),
            "ratio": float(ratio),
        })

    # ------------------------------------------------------------------
    # Constraint 1: e^T x = 1 (budget)
    # ------------------------------------------------------------------
    lhs_c1 = float(np.sum(x))
    rhs_c1 = 1.0
    viol_c1 = abs(lhs_c1 - rhs_c1)
    if viol_c1 > tol:
        record(1, f"Budget constraint violated: sum(x)={lhs_c1:.8f}, expected 1.0",
               lhs_c1, rhs_c1, viol_c1)

    # ------------------------------------------------------------------
    # Constraint 2: l <= Ax <= u (linear constraints, when applicable)
    # ------------------------------------------------------------------
    # 2a: min return constraint: mu^T x >= r_bar
    if has_min_return and r_bar is not None:
        lhs_ret = float(mu @ x)
        rhs_ret = float(r_bar)
        viol_ret = rhs_ret - lhs_ret  # >= constraint: violation if RHS > LHS
        if viol_ret > tol:
            record(2, f"Min return violated: mu^T x={lhs_ret:.8f} < r_bar={rhs_ret:.8f}",
                   lhs_ret, rhs_ret, viol_ret)

    # 2b: min investment threshold (semi-continuous on x): if x_i > 0 then x_i >= l_i.
    if has_min_inv and l_min is not None:
        l_arr = np.array(l_min, dtype=float)
        for i in range(n):
            if x[i] > tol:
                lhs_mi = float(x[i])
                rhs_mi = float(l_arr[i])
                viol_mi = rhs_mi - lhs_mi  # >= constraint
                if viol_mi > tol:
                    record(2, f"Min investment violated for asset {i}: x[{i}]={lhs_mi:.8f} < l_min={rhs_mi:.8f}",
                           lhs_mi, rhs_mi, viol_mi)

    # 2c: max investment cap on x: x_i <= u_i.
    if u_max is not None:
        u_arr = np.array(u_max, dtype=float)
        for i in range(n):
            if x[i] > tol:
                lhs_ui = float(x[i])
                rhs_ui = float(u_arr[i])
                viol_ui = lhs_ui - rhs_ui  # <= constraint
                if viol_ui > tol:
                    record(2, f"Max investment violated for asset {i}: x[{i}]={lhs_ui:.8f} > u_max={rhs_ui:.8f}",
                           lhs_ui, rhs_ui, viol_ui)

    # 2d: general linear constraints: l_lin <= A x <= u_lin
    A_lin = constr.get("A_lin", None)
    l_lin = constr.get("l_lin", None)
    u_lin = constr.get("u_lin", None)
    if A_lin is not None:
        A_mat = np.array(A_lin, dtype=float)
        Ax = A_mat @ x
        m_lin = Ax.shape[0]
        if l_lin is not None:
            l_arr_lin = np.array(l_lin, dtype=float)
            for j in range(m_lin):
                viol_lj = float(l_arr_lin[j] - Ax[j])
                if viol_lj > tol:
                    record(2, f"General linear lower bound violated at row {j}: (Ax)[{j}]={float(Ax[j]):.8f} < l[{j}]={float(l_arr_lin[j]):.8f}",
                           float(Ax[j]), float(l_arr_lin[j]), viol_lj)
        if u_lin is not None:
            u_arr_lin = np.array(u_lin, dtype=float)
            for j in range(m_lin):
                viol_uj = float(Ax[j] - u_arr_lin[j])
                if viol_uj > tol:
                    record(2, f"General linear upper bound violated at row {j}: (Ax)[{j}]={float(Ax[j]):.8f} > u[{j}]={float(u_arr_lin[j]):.8f}",
                           float(Ax[j]), float(u_arr_lin[j]), viol_uj)

    # ------------------------------------------------------------------
    # Constraint 3: ||x||_0 <= k (cardinality)
    # ------------------------------------------------------------------
    nnz_x = int(np.sum(np.abs(x) > tol))
    lhs_c3 = float(nnz_x)
    rhs_c3 = float(k)
    viol_c3 = lhs_c3 - rhs_c3  # <= constraint
    if viol_c3 > tol:
        record(3, f"Cardinality violated: ||x||_0={nnz_x} > k={k}",
               lhs_c3, rhs_c3, viol_c3)

    # ------------------------------------------------------------------
    # NOTE: Constraints 4 (x_i=0 if z_i=0), 5 (perspective x_i^2 <= z_i*theta_i),
    # 7 (z binary), 8 (e^T z <= k via z), and 9 (theta >= 0) all enforce the
    # auxiliary variables z and theta of the perspective MISOCO
    # reformulation (Problem 35). They are NOT constraints of the original
    # Problem (4), whose only decision variable is x. Per project rule
    # they are intentionally NOT enforced so that any solver that does
    # not introduce z / theta is not falsely flagged as infeasible.
    # Cardinality is verified above (C3) directly from x.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 6: x_i >= 0 (non-negativity)
    # ------------------------------------------------------------------
    for i in range(n):
        if x[i] < -tol:
            lhs_c6 = float(x[i])
            rhs_c6 = 0.0
            viol_c6 = -x[i]  # >= constraint: violation = RHS - LHS = -x[i]
            record(6, f"Non-negativity violated: x[{i}]={x[i]:.8f} < 0",
                   lhs_c6, rhs_c6, viol_c6)

    # ------------------------------------------------------------------
    # Constraint 10: objective consistency (Tier C obj-recompute)
    # The objective (Problem 34 / 4 equivalent) is fully determined by x:
    #   obj = (1/2) x^T Sigma x + (1/(2*gamma)) ||x||_2^2 - kappa * mu^T x
    # where Sigma = F F^T + diag(idiosyncratic_variance).
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            try:
                F = np.array(instance["factor_loadings"], dtype=float)
                idio = np.array(instance["idiosyncratic_variance"], dtype=float)
                gamma = float(instance["gamma"])
                kappa = float(instance["kappa"])
            except (KeyError, TypeError, ValueError):
                F = idio = None
                gamma = kappa = None
            if F is not None and idio is not None and gamma is not None and gamma > 0:
                Fx = F.T @ x
                xSx = float(Fx @ Fx + np.sum(idio * x * x))
                ridge = float(x @ x) / (2.0 * gamma)
                ret = float(mu @ x)
                true_obj = 0.5 * xSx + ridge - kappa * ret
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with 1e-6 absolute floor
                obj_tol = max(1e-6, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    record(
                        10,
                        f"Objective consistency violated: reported objective_value="
                        f"{reported} differs from recomputed "
                        f"(1/2)*x^T Sigma x + (1/(2*gamma))*||x||^2 - kappa*mu^T x="
                        f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})",
                        reported, true_obj, abs_diff,
                    )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------
    feasible = len(violated_set) == 0
    violated_constraints = sorted(violated_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for sparse portfolio selection (Bertsimas 2022)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance = json.load(f)
    with open(args.solution_path, "r") as f:
        solution = json.load(f)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    print(f"Result: {status}")
    if not result["feasible"]:
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")
    print(f"Written to {args.result_path}")


if __name__ == "__main__":
    main()
