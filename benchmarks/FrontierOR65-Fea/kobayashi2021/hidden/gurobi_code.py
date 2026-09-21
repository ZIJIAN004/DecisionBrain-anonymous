"""
Gurobi MIO implementation of Cardinality-constrained Mean-CVaR Portfolio Optimization.
Kobayashi, Takano, Nakata (2021).

Uses the BigM formulation (Eq. 32) with linearized CVaR (lifting representation, Eqs. 10b-10d).
"""

import argparse
import json
import sys

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


def load_instance(path):
    with open(path) as f:
        data = json.load(f)
    N = data["N"]
    S = data["S"]
    k = data["k"]
    beta = data["beta"]
    gamma = data["gamma"]
    mu_bar = data["mu_bar"]
    mu = data["mu"]                  # length N
    # p_s may be stored as scalar (uniform) or list
    p_s_raw = data["p_s"]
    if isinstance(p_s_raw, list):
        p = p_s_raw
    else:
        p = [p_s_raw] * S
    scenarios = data["scenarios"]    # S x N
    # Optional generic linear constraints A x <= b (feasible portfolio set X)
    A = data.get("A")
    b = data.get("b")
    return N, S, k, beta, gamma, mu_bar, mu, p, scenarios, A, b


def build_and_solve(instance_path, solution_path, time_limit):
    N, S, k, beta, gamma, mu_bar, mu, p, scenarios, A, b = load_instance(instance_path)

    m = gp.Model("MeanCVaR_BigM")
    m.setParam("Threads", 1)
    m.setParam("TimeLimit", time_limit)
    # Use primal-dual interior point for QP (consistent with paper)
    # Gurobi handles MIQP automatically; set OutputFlag for cleaner runs
    m.setParam("OutputFlag", 1)

    # ----------------------------------------------------------------
    # Decision variables
    # ----------------------------------------------------------------
    x = m.addVars(N, lb=0.0, name="x")          # portfolio weights
    z = m.addVars(N, vtype=GRB.BINARY, name="z") # asset selection
    a = m.addVar(lb=-GRB.INFINITY, name="a")      # VaR auxiliary
    v = m.addVar(lb=0.0, name="v")                # CVaR upper bound
    # Lifting variables q_s >= 0 for linearized CVaR (Eqs. 10c-10d)
    q = m.addVars(S, lb=0.0, name="q")

    # ----------------------------------------------------------------
    # Objective: (1/(2*gamma)) * x^T x + a + v   (Eq. 7a)
    # ----------------------------------------------------------------
    obj = gp.QuadExpr()
    for n in range(N):
        obj += (1.0 / (2.0 * gamma)) * x[n] * x[n]
    obj += a + v
    m.setObjective(obj, GRB.MINIMIZE)

    # ----------------------------------------------------------------
    # CVaR constraint via lifting (Eqs. 10b, 10c, 10d)
    # ----------------------------------------------------------------
    # (10b): v >= 1/(1-beta) * sum_s p_s * q_s
    m.addConstr(
        v >= (1.0 / (1.0 - beta)) * gp.quicksum(p[s] * q[s] for s in range(S)),
        name="cvar_ub"
    )
    # (10c): q_s >= -(r^(s))^T x - a  for all s
    for s in range(S):
        r_s = scenarios[s]
        m.addConstr(
            q[s] >= -gp.quicksum(r_s[n] * x[n] for n in range(N)) - a,
            name=f"q_lb_{s}"
        )
    # (10d): q_s >= 0 already enforced by lb=0

    # ----------------------------------------------------------------
    # Budget constraint: sum x_n = 1  (Eq. 1)
    # ----------------------------------------------------------------
    m.addConstr(gp.quicksum(x[n] for n in range(N)) == 1.0, name="budget")

    # ----------------------------------------------------------------
    # Expected return constraint: mu^T x >= mu_bar  (Eq. 6)
    # ----------------------------------------------------------------
    m.addConstr(
        gp.quicksum(mu[n] * x[n] for n in range(N)) >= mu_bar,
        name="return"
    )

    # ----------------------------------------------------------------
    # Generic linear portfolio constraints: A x <= b  (feasible set X)
    # ----------------------------------------------------------------
    if A is not None and b is not None:
        for i in range(len(b)):
            m.addConstr(
                gp.quicksum(A[i][n] * x[n] for n in range(N)) <= b[i],
                name=f"Ax_b_{i}"
            )

    # ----------------------------------------------------------------
    # BigM cardinality linking: 0 <= x_n <= z_n  (Eq. 32)
    # (valid since x_n <= 1 from budget and x_n >= 0; z_n in {0,1})
    # ----------------------------------------------------------------
    for n in range(N):
        m.addConstr(x[n] <= z[n], name=f"bigm_{n}")

    # ----------------------------------------------------------------
    # Cardinality constraint: sum z_n <= k  (Eq. 4)
    # ----------------------------------------------------------------
    m.addConstr(gp.quicksum(z[n] for n in range(N)) <= k, name="cardinality")

    # ----------------------------------------------------------------
    # Optimize
    # ----------------------------------------------------------------
    m.optimize()

    # ----------------------------------------------------------------
    # Extract solution
    # ----------------------------------------------------------------
    status = m.Status
    has_solution = m.SolCount > 0

    if not has_solution:
        print("No feasible solution found within time limit.", file=sys.stderr)
        result = {
            "objective_value": None,
            "status": status,
            "x": None,
            "z": None,
        }
    else:
        obj_val = m.ObjVal
        x_sol = [x[n].X for n in range(N)]
        z_sol = [int(round(z[n].X)) for n in range(N)]
        a_sol = a.X
        v_sol = v.X
        result = {
            "objective_value": obj_val,
            "status": status,
            "x": x_sol,
            "z": z_sol,
            "a": a_sol,
            "v": v_sol,
            "gap": m.MIPGap if status != GRB.OPTIMAL else 0.0,
        }
        print(f"Objective value: {obj_val:.8f}")
        selected = [n for n in range(N) if z_sol[n] == 1]
        print(f"Selected assets ({len(selected)}): {selected}")

    with open(solution_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Solution written to {solution_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MIO for Cardinality-constrained Mean-CVaR Portfolio Optimization"
    )
    parser.add_argument("--instance_path", required=True, help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True, help="Path for output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True, help="Solver time limit in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    build_and_solve(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
