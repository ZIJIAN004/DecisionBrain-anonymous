"""
Gurobi implementation of the MISOCO formulation (Problem 35) from:
  Bertsimas and Cory-Wright (2022),
  "A Scalable Algorithm for Sparse Portfolio Selection"

Problem (35):
  min_{z in Z_k^n, x in R^n_+, theta in R^n_+}
      (1/2) x^T Sigma x  +  (1/(2*gamma)) * e^T theta  -  kappa * mu^T x
  s.t.
      e^T z <= k,
      e^T x = 1,
      l <= A x <= u          (if any linear constraints),
      x_i^2 <= z_i * theta_i  for all i in [n]    (perspective constraints)

The covariance matrix is reconstructed from the factor model:
  Sigma = F @ F^T + diag(idiosyncratic_variance)
where F = factor_loadings (n x r).

Usage:
  python gurobi_code.py --instance_path instance_1.json \
                        --solution_path gurobi_solution_1.json \
                        --time_limit 300
"""

import argparse
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass


def project_x_to_support(x_raw, z_raw, k, n):
    """Hard-project x onto the cardinality-feasible set with budget sum=1.

    The big-M perspective formulation can leak small nonzero x_i (~1e-4) at
    time-out when the binary z_i has not been fully driven to 0. The original
    Problem (4) requires ||x||_0 <= k, which the leaked x violates. We project
    by:
      1. Pick the support: indices with z_i >= 0.5 (or top-k by |x_i| if z is
         degenerate or violates the cardinality budget itself).
      2. Zero x outside the support; clamp to non-negativity.
      3. Renormalize so sum(x) = 1.
    Returns (x_proj, sorted_support_indices).
    """
    if z_raw is not None:
        support = sorted(int(i) for i in np.where(z_raw >= 0.5)[0])
    else:
        support = []
    if len(support) == 0 or len(support) > k:
        # Fall back to magnitude-based top-k
        order = np.argsort(-np.abs(x_raw))
        support = sorted(int(i) for i in order[:k])

    x_proj = np.zeros(n, dtype=float)
    for i in support:
        x_proj[i] = max(float(x_raw[i]), 0.0)
    s = x_proj.sum()
    if s > 0:
        x_proj /= s
    return x_proj, support


def evaluate_objective(x, Sigma, gamma, kappa, mu):
    """Objective with theta_i = x_i^2 (optimal given z_i=1 in the support)."""
    quad = 0.5 * float(x @ Sigma @ x)
    persp = (1.0 / (2.0 * gamma)) * float((x ** 2).sum())
    ret = kappa * float(mu @ x)
    return quad + persp - ret


def main():
    parser = argparse.ArgumentParser(
        description="Solve sparse portfolio selection via MISOCO (Gurobi)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to write the JSON solution file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ------------------------------------------------------------------ #
    # Load instance
    # ------------------------------------------------------------------ #
    with open(args.instance_path, "r") as fh:
        inst = json.load(fh)

    n = inst["n"]                                          # number of assets
    k = inst["k"]                                          # cardinality budget
    gamma = inst["gamma"]                                  # ridge regularizer
    kappa = inst["kappa"]                                  # return weight (0 or 1)
    mu = np.array(inst["mu"], dtype=float)                 # expected returns (n,)
    F = np.array(inst["factor_loadings"], dtype=float)     # factor loadings  (n, r)
    eps_var = np.array(inst["idiosyncratic_variance"], dtype=float)  # (n,)

    # ------------------------------------------------------------------ #
    # Build covariance matrix: Sigma = F @ F^T + diag(eps_var)
    # ------------------------------------------------------------------ #
    Sigma = F @ F.T + np.diag(eps_var)   # (n, n)  positive definite

    # ------------------------------------------------------------------ #
    # Read optional constraints
    # ------------------------------------------------------------------ #
    constr = inst.get("constraints", {})
    has_min_return = constr.get("has_min_return_constraint", False)
    r_bar = constr.get("r_bar", None)
    has_min_inv = constr.get("has_min_investment_constraint", False)
    l_min = constr.get("l_min_investment", None)    # per-asset lower bounds
    u_max = constr.get("u_max_investment", None)    # per-asset upper bounds
    A_lin = inst.get("A", None)                     # (m, n) linear constraint matrix
    l_lin = inst.get("l", None)                     # (m,) linear lower bounds
    u_lin = inst.get("u", None)                     # (m,) linear upper bounds

    # ------------------------------------------------------------------ #
    # Build Gurobi model
    # ------------------------------------------------------------------ #
    model = gp.Model("SparsePortfolio_MISOCO")
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 1)

    # Decision variables
    x = model.addVars(n, lb=0.0, name="x")          # portfolio weights
    z = model.addVars(n, vtype=GRB.BINARY, name="z") # support indicators
    theta = model.addVars(n, lb=0.0, name="theta")   # auxiliary perspective vars

    # ------------------------------------------------------------------ #
    # Objective: (1/2) x^T Sigma x  +  (1/(2*gamma)) sum(theta)  -  kappa * mu^T x
    # ------------------------------------------------------------------ #
    # Build quadratic objective using the factor structure for efficiency:
    #   x^T Sigma x = x^T (F F^T + diag(eps)) x
    #               = ||F^T x||_2^2  +  eps^T (x o x)
    # We use auxiliary variables v = F^T x (r-dimensional) to keep the
    # objective quadratic rather than constructing the dense n x n Sigma.
    # For correctness on any instance size we use the dense Sigma here.
    obj = gp.QuadExpr()
    # Quadratic term: (1/2) x^T Sigma x
    for i in range(n):
        for j in range(n):
            coeff = Sigma[i, j]
            if abs(coeff) > 1e-16:
                obj += 0.5 * coeff * x[i] * x[j]
    # Regularization / perspective term: (1/(2*gamma)) sum theta_i
    obj += (1.0 / (2.0 * gamma)) * gp.quicksum(theta[i] for i in range(n))
    # Return term: -kappa * mu^T x
    obj -= kappa * gp.quicksum(mu[i] * x[i] for i in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    # ------------------------------------------------------------------ #
    # Constraints
    # ------------------------------------------------------------------ #
    # Cardinality: sum(z) <= k
    model.addConstr(
        gp.quicksum(z[i] for i in range(n)) <= k,
        name="cardinality"
    )
    # Budget: sum(x) = 1
    model.addConstr(
        gp.quicksum(x[i] for i in range(n)) == 1.0,
        name="budget"
    )
    # Perspective constraints: x_i^2 <= z_i * theta_i  (rotated SOC)
    for i in range(n):
        model.addQConstr(
            x[i] * x[i] <= z[i] * theta[i],
            name=f"persp_{i}"
        )

    # Optional: minimum return constraint  mu^T x >= r_bar
    if has_min_return and r_bar is not None:
        model.addConstr(
            gp.quicksum(mu[i] * x[i] for i in range(n)) >= r_bar,
            name="min_return"
        )

    # Optional: minimum investment  x_i >= l_i * z_i  (semi-continuous)
    if has_min_inv and l_min is not None:
        l_arr = np.array(l_min, dtype=float)
        for i in range(n):
            model.addConstr(x[i] >= l_arr[i] * z[i], name=f"min_inv_{i}")

    # Optional: maximum investment  x_i <= u_i * z_i
    if u_max is not None:
        u_arr = np.array(u_max, dtype=float)
        for i in range(n):
            model.addConstr(x[i] <= u_arr[i] * z[i], name=f"max_inv_{i}")

    # General linear constraints: l <= A x <= u   (math_model.txt eq (35))
    if A_lin is not None:
        A_arr = np.array(A_lin, dtype=float)
        l_arr_lin = np.array(l_lin, dtype=float) if l_lin is not None else None
        u_arr_lin = np.array(u_lin, dtype=float) if u_lin is not None else None
        for j in range(A_arr.shape[0]):
            lhs = gp.quicksum(A_arr[j, i] * x[i] for i in range(n))
            if l_arr_lin is not None:
                model.addConstr(lhs >= l_arr_lin[j], name=f"lin_lb_{j}")
            if u_arr_lin is not None:
                model.addConstr(lhs <= u_arr_lin[j], name=f"lin_ub_{j}")

    # ------------------------------------------------------------------ #
    # Solve
    # ------------------------------------------------------------------ #
    model.optimize()

    # ------------------------------------------------------------------ #
    # Extract and save solution
    # ------------------------------------------------------------------ #
    if model.SolCount > 0:
        x_raw = np.array([x[i].X for i in range(n)])
        z_raw = np.array([z[i].X for i in range(n)])
        solver_obj_val = float(model.ObjVal)

        # Hard projection onto cardinality-feasible set: needed because at
        # time-out the big-M perspective lets x_i leak ~1e-4 even when z_i is
        # supposed to be 0, which would violate ||x||_0 <= k in the original
        # problem. The projected x exactly satisfies cardinality and budget.
        x_sol, support = project_x_to_support(x_raw, z_raw, k, n)
        obj_val = evaluate_objective(x_sol, Sigma, gamma, kappa, mu)
    else:
        x_sol = None
        support = None
        solver_obj_val = None
        obj_val = None

    solution = {
        "instance_id": inst.get("instance_id", ""),
        "objective_value": obj_val,           # objective evaluated on projected x
        # The original Problem (4) decision variable is x. z and theta are
        # auxiliary variables of the perspective MISOCO reformulation
        # (Problem 35) and are NOT emitted — feasibility_check verifies only
        # the original problem against x.
        "x": x_sol.tolist() if x_sol is not None else None,
        "support": support,                   # indices with x_i > 0, len <= k
        "solver_obj_val": solver_obj_val,     # Gurobi's ObjVal on the un-projected x
        "solver_status": model.Status,
        "solver_status_str": {
            1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE",
            4: "INF_OR_UNBD", 5: "UNBOUNDED", 9: "TIME_LIMIT",
        }.get(model.Status, f"STATUS_{model.Status}"),
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "runtime_s": model.Runtime,
    }

    with open(args.solution_path, "w") as fh:
        json.dump(solution, fh, indent=2)

    print(f"Status  : {solution['solver_status_str']}")
    print(f"Obj val : {obj_val}")
    print(f"Runtime : {model.Runtime:.2f}s")


if __name__ == "__main__":
    main()
