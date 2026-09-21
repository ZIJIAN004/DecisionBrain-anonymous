"""
MILP2 formulation for the Rank-One Quadratic Assignment Problem (QAP-R1)
using Gurobi, following Wang et al. (2020).

Minimize f(X) = (AX)(BX) + CX where X is a permutation matrix.
"""

import argparse
import json
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

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
    with open(path, "r") as f:
        data = json.load(f)
    n = data["n"]
    A = np.array(data["A"], dtype=float)
    B = np.array(data["B"], dtype=float)
    C = np.array(data["C"], dtype=float)
    return n, A, B, C


def nonneg_transform(n, A, B, C):
    """
    Lemma 1 transformation to make A', B', C^0 all nonneg.
    Returns A', B', C^0, and the constant offset to recover the true objective.
    """
    # a_l = max{|a_ij| : a_ij < 0}, 0 if all nonneg
    neg_A = A[A < 0]
    a_l = float(np.max(np.abs(neg_A))) if neg_A.size > 0 else 0.0

    neg_B = B[B < 0]
    b_l = float(np.max(np.abs(neg_B))) if neg_B.size > 0 else 0.0

    A_prime = A + a_l
    B_prime = B + b_l

    alpha = n * a_l
    beta = n * b_l

    C_prime = C - alpha * B - beta * A

    neg_C_prime = C_prime[C_prime < 0]
    c_l = float(np.max(np.abs(neg_C_prime))) if neg_C_prime.size > 0 else 0.0

    C0 = C_prime + c_l

    # true_obj = milp2_obj + constant_offset
    constant_offset = -alpha * beta - n * c_l

    return A_prime, B_prime, C0, constant_offset


def solve_lap_bounds(cost_matrix):
    """Compute min and max of cost_matrix * X over permutation matrices X."""
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    L = float(cost_matrix[row_ind, col_ind].sum())

    row_ind_max, col_ind_max = linear_sum_assignment(-cost_matrix)
    U = float(cost_matrix[row_ind_max, col_ind_max].sum())

    return L, U


def solve_milp2(n, A_prime, B_prime, C0, constant_offset, time_limit):
    """
    Build and solve the MILP2 formulation.
    Returns (true_objective, assignment) where assignment[i] = j means row i -> col j.
    """
    L_b, U_b = solve_lap_bounds(B_prime)
    L_a, U_a = solve_lap_bounds(A_prime)

    # Special case: if L_b == U_b, reduces to LAP
    if L_b == U_b:
        cost = L_b * A_prime + C0
        row_ind, col_ind = linear_sum_assignment(cost)
        milp2_obj = float(cost[row_ind, col_ind].sum())
        true_obj = milp2_obj + constant_offset
        assignment = list(col_ind)
        return true_obj, assignment

    t_b = int(math.floor(math.log2(U_b - L_b))) + 1

    model = gp.Model("MILP2")

    # Solver settings
    model.setParam("MIPGap", 1e-08)
    model.setParam("Presolve", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", 1)

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    v = model.addVars(t_b, vtype=GRB.BINARY, name="v")
    z = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
    w = model.addVars(t_b, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")

    # Objective: minimize L_b * z + sum_{k=0}^{t_b-1} 2^k * w[k] + C0 X
    obj = L_b * z
    obj += gp.quicksum(2**k * w[k] for k in range(t_b))
    obj += gp.quicksum(C0[i, j] * x[i, j] for i in range(n) for j in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint (i): row assignment
    for i in range(n):
        model.addConstr(gp.quicksum(x[i, j] for j in range(n)) == 1, name=f"row_{i}")

    # Constraint (ii): column assignment
    for j in range(n):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n)) == 1, name=f"col_{j}")

    # Constraint (iii): A'X = z
    model.addConstr(
        gp.quicksum(A_prime[i, j] * x[i, j] for i in range(n) for j in range(n)) == z,
        name="def_z",
    )

    # Constraint (iv): B'X = L_b + sum 2^k v_k
    model.addConstr(
        gp.quicksum(B_prime[i, j] * x[i, j] for i in range(n) for j in range(n))
        == L_b + gp.quicksum(2**k * v[k] for k in range(t_b)),
        name="bin_repr_bx",
    )

    # McCormick envelope constraints for w_k = z * v_k
    for k in range(t_b):
        # (v) w_k <= U_a * v_k
        model.addConstr(w[k] - U_a * v[k] <= 0, name=f"mc_ub1_{k}")
        # (vi) w_k >= L_a * v_k
        model.addConstr(w[k] - L_a * v[k] >= 0, name=f"mc_lb1_{k}")
        # (vii) z - w_k + U_a * v_k <= U_a
        model.addConstr(z - w[k] + U_a * v[k] <= U_a, name=f"mc_ub2_{k}")
        # (viii) z - w_k + L_a * v_k >= L_a
        model.addConstr(z - w[k] + L_a * v[k] >= L_a, name=f"mc_lb2_{k}")

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError("No feasible solution found within the time limit.")

    milp2_obj = model.ObjVal
    true_obj = milp2_obj + constant_offset

    # Extract assignment
    assignment = [0] * n
    for i in range(n):
        for j in range(n):
            if x[i, j].X > 0.5:
                assignment[i] = j
                break

    return true_obj, assignment


def main():
    parser = argparse.ArgumentParser(
        description="Solve QAP-R1 using MILP2 formulation with Gurobi."
    )
    parser.add_argument("--instance_path", type=str, required=True, help="Path to instance JSON file.")
    parser.add_argument("--solution_path", type=str, default="gurobi_solution_1.json", help="Path to output solution JSON file.")
    parser.add_argument("--time_limit", type=int, default=3600, help="Time limit in seconds.")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    n, A, B, C = load_instance(args.instance_path)
    A_prime, B_prime, C0, constant_offset = nonneg_transform(n, A, B, C)
    true_obj, assignment = solve_milp2(n, A_prime, B_prime, C0, constant_offset, args.time_limit)

    solution = {
        "objective_value": true_obj,
        "assignment": assignment,
    }

    with open(args.solution_path, "w") as f:
        json.dump(solution, f, indent=2)

    print(f"Objective value: {true_obj}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
