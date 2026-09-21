"""
CDLP (Choice-Based Deterministic Linear Programming) via Column Generation
============================================================================
Paper: Bront, Mendez-Diaz, Vulcano (2009)
"A Column Generation Algorithm for Choice-Based Network Revenue Management"
Operations Research 57(3):769-784

This program solves the CDLP formulation (Equation (3) in the paper) using
column generation (Section 4). The column generation subproblem (Equation (6))
is solved first by a greedy heuristic (Section 4.2.2), and if that fails,
by an exact MIP reformulation (Section 4.2.1).

Output: optimal CDLP objective value and the primal/dual solutions.
"""

import argparse
import json
import time
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


def load_instance(path):
    """Load problem instance from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def build_problem_data(data):
    """
    Extract and precompute all problem parameters from the instance JSON.
    Returns a dict with all needed arrays/values.
    """
    n = len(data["products"])  # number of products
    m = len(data["network"]["legs"])  # number of legs (resources)
    L = len(data["segments"])  # number of segments
    T = data["booking_horizon"]["T"]
    lam = data["lambda"]  # overall arrival probability per period

    # Product revenues (0-indexed)
    r = np.array([p["fare"] for p in data["products"]], dtype=float)

    # Incidence matrix A: m x n, A[i][j] = 1 if leg i is used by product j
    A = np.zeros((m, n), dtype=float)
    for j, prod in enumerate(data["products"]):
        for leg_id in prod["legs_used"]:
            leg_idx = leg_id - 1  # convert 1-indexed to 0-indexed
            A[leg_idx, j] = 1.0

    # Capacities
    c = np.array([leg["capacity"] for leg in data["network"]["legs"]], dtype=float)

    # Segment data
    segments = []
    for seg in data["segments"]:
        seg_info = {
            "lambda_l": seg["lambda_l"],
            "consideration_set": [pid - 1 for pid in seg["consideration_set"]],  # 0-indexed
            "v": {},  # preference weights: product_0idx -> weight
            "v0": seg["no_purchase_preference"]
        }
        for idx, pid in enumerate(seg["consideration_set"]):
            seg_info["v"][pid - 1] = seg["preference_vector"][idx]
        segments.append(seg_info)

    # Compute p_l = lambda_l / lambda
    p_l = np.array([seg["lambda_l"] / lam for seg in segments])

    return {
        "n": n, "m": m, "L": L, "T": T, "lam": lam,
        "r": r, "A": A, "c": c,
        "segments": segments, "p_l": p_l
    }


def compute_choice_probs(S_set, prob_data):
    """
    Compute P_j(S) for all j in S, using the MNL model with overlapping segments.
    S_set: set of 0-indexed product indices
    Returns: dict {j: P_j(S)} for j in S_set
    """
    segments = prob_data["segments"]
    p_l = prob_data["p_l"]
    n = prob_data["n"]

    P = {}
    for j in range(n):
        if j not in S_set:
            P[j] = 0.0
            continue
        total = 0.0
        for l_idx, seg in enumerate(segments):
            if j in seg["v"]:
                # Compute denominator for this segment
                denom = seg["v0"]
                for h in S_set:
                    if h in seg["v"]:
                        denom += seg["v"][h]
                P_lj = seg["v"][j] / denom
                total += p_l[l_idx] * P_lj
        P[j] = total
    return P


def compute_R_and_Q(S_set, prob_data):
    """
    Compute R(S) = sum_{j in S} r_j * P_j(S)  (expected revenue)
    and Q(S) = A * P(S)  (capacity consumption vector)
    """
    r = prob_data["r"]
    A = prob_data["A"]
    n = prob_data["n"]

    P = compute_choice_probs(S_set, prob_data)

    R_S = sum(r[j] * P[j] for j in S_set)

    P_vec = np.array([P.get(j, 0.0) for j in range(n)])
    Q_S = A @ P_vec

    return R_S, Q_S


def greedy_heuristic(pi, sigma, prob_data):
    """
    Greedy heuristic for the column generation subproblem (Section 4.2.2).
    Returns (S_set, reduced_cost) where S_set is the set of products to offer,
    and reduced_cost is the subproblem objective value.
    """
    n = prob_data["n"]
    r = prob_data["r"]
    A = prob_data["A"]
    segments = prob_data["segments"]
    lam = prob_data["lam"]

    # Step 1: For all products j such that r_j - A_j^T pi <= 0, set y_j = 0
    w = np.array([r[j] - A[:, j] @ pi for j in range(n)])
    S_prime = set()
    for j in range(n):
        if w[j] > 0:
            S_prime.add(j)

    if not S_prime:
        return set(), -sigma

    # Step 3: Compute j1* = argmax over S' of sum_l (r_j - A_j^T pi) * v_lj / (v_lj + v_l0)
    best_val = -np.inf
    best_j = None
    for j in S_prime:
        val = 0.0
        for seg in segments:
            if j in seg["v"]:
                vlj = seg["v"][j]
                val += w[j] * vlj / (vlj + seg["v0"])
        if val > best_val:
            best_val = val
            best_j = j

    S = {best_j}
    S_prime.discard(best_j)

    # Helper: compute Value(S) = subproblem objective without -sigma
    def compute_value(S_set):
        val = 0.0
        for j in S_set:
            for l_idx, seg in enumerate(segments):
                if j in seg["v"]:
                    denom = seg["v0"]
                    for h in S_set:
                        if h in seg["v"]:
                            denom += seg["v"][h]
                    val += w[j] * seg["lambda_l"] * seg["v"][j] / denom
        return val

    # Step 4: Repeat adding products
    changed = True
    while changed and S_prime:
        changed = False
        current_val = compute_value(S)

        # Find best product to add from S'
        # Compute for each j in S': the objective of S union {j}
        best_new_val = -np.inf
        best_new_j = None
        for j in S_prime:
            # Compute objective: sum_l lambda_l * (sum_{i in C_l cap (S union {j})} w_i * v_li) / (sum_{i in C_l cap (S union {j})} v_li + v_l0)
            candidate = S | {j}
            new_val = 0.0
            for l_idx, seg in enumerate(segments):
                num = 0.0
                denom = seg["v0"]
                for h in candidate:
                    if h in seg["v"]:
                        num += w[h] * seg["v"][h]
                        denom += seg["v"][h]
                new_val += seg["lambda_l"] * num / denom
            if new_val > best_new_val:
                best_new_val = new_val
                best_new_j = j

        # Following paper step 4(a)-(b): use Value(S union {j*})
        if best_new_j is not None:
            candidate_val = compute_value(S | {best_new_j})
            if candidate_val > current_val:
                S.add(best_new_j)
                S_prime.discard(best_new_j)
                changed = True

    reduced_cost = compute_value(S) - sigma
    return S, reduced_cost


def exact_mip_subproblem(pi, sigma, prob_data, time_limit=300):
    """
    Exact MIP reformulation for the column generation subproblem (Section 4.2.1).
    Returns (S_set, reduced_cost).
    """
    n = prob_data["n"]
    r = prob_data["r"]
    A = prob_data["A"]
    segments = prob_data["segments"]
    L = prob_data["L"]

    w = np.array([r[j] - A[:, j] @ pi for j in range(n)])

    # Compute K >= 1/v_min
    all_v = []
    for seg in segments:
        all_v.append(seg["v0"])
        for v_val in seg["v"].values():
            if v_val > 0:
                all_v.append(v_val)
    v_min = min(all_v)
    K = 1.0 / v_min + 1.0  # add margin

    model = gp.Model("subproblem_mip")
    model.setParam("Threads", 1)
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    # Variables
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    x = model.addVars(L, lb=0.0, name="x")

    # z[l,j] = x_l * y_j (linearized)
    z = {}
    for l_idx, seg in enumerate(segments):
        for j in seg["consideration_set"]:
            z[l_idx, j] = model.addVar(lb=0.0, name=f"z_{l_idx}_{j}")

    model.update()

    # Objective: max sum_l sum_{j in C_l} lambda_l * (r_j - A_j^T pi) * v_lj * z_lj
    obj = gp.LinExpr()
    for l_idx, seg in enumerate(segments):
        for j in seg["consideration_set"]:
            coeff = seg["lambda_l"] * w[j] * seg["v"][j]
            obj += coeff * z[l_idx, j]
    model.setObjective(obj, GRB.MAXIMIZE)

    # Constraints
    for l_idx, seg in enumerate(segments):
        # x_l * v_l0 + sum_{i in C_l} v_li * z_li = 1
        constr = seg["v0"] * x[l_idx]
        for j in seg["consideration_set"]:
            constr += seg["v"][j] * z[l_idx, j]
        model.addConstr(constr == 1.0, name=f"norm_{l_idx}")

        for j in seg["consideration_set"]:
            # x_l - z_lj <= K - K * y_j
            model.addConstr(x[l_idx] - z[l_idx, j] <= K - K * y[j],
                            name=f"lin1_{l_idx}_{j}")
            # z_lj <= x_l
            model.addConstr(z[l_idx, j] <= x[l_idx],
                            name=f"lin2_{l_idx}_{j}")
            # z_lj <= K * y_j
            model.addConstr(z[l_idx, j] <= K * y[j],
                            name=f"lin3_{l_idx}_{j}")

    model.optimize()

    if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        if model.SolCount > 0:
            S_set = set()
            for j in range(n):
                if y[j].X > 0.5:
                    S_set.add(j)
            obj_val = model.ObjVal
            return S_set, obj_val - sigma
        else:
            return set(), -sigma
    else:
        return set(), -sigma


def solve_cdlp_column_generation(prob_data, time_limit):
    """
    Solve the CDLP via column generation (Section 4 of the paper).
    Returns the optimal objective value and dual prices.
    """
    n = prob_data["n"]
    m = prob_data["m"]
    T = prob_data["T"]
    lam = prob_data["lam"]
    c = prob_data["c"]
    segments = prob_data["segments"]

    start_time = time.time()

    # A.2. Initialization: single column containing one product per segment
    # Pick the first product of each segment (0-indexed, in labeling order)
    init_products = set()
    for seg in segments:
        first_product = min(seg["consideration_set"])
        init_products.add(first_product)

    columns = [frozenset(init_products)]  # list of frozensets
    column_set = {columns[0]}  # for duplicate checking

    # Precompute R(S) and Q(S) for each column
    R_vals = []
    Q_vals = []
    R_S, Q_S = compute_R_and_Q(init_products, prob_data)
    R_vals.append(R_S)
    Q_vals.append(Q_S)

    iteration = 0
    best_obj = 0.0
    best_pi = np.zeros(m)
    best_sigma = 0.0
    best_t_vals = {}

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        iteration += 1
        k = len(columns)

        # Solve reduced LP (Equation (4))
        master = gp.Model("CDLP_master")
        master.setParam("Threads", 1)
        master.setParam("OutputFlag", 0)
        remaining_time = max(1, time_limit - (time.time() - start_time))
        master.setParam("TimeLimit", remaining_time)

        t_vars = master.addVars(k, lb=0.0, name="t")
        master.update()

        # Objective: max sum_S lambda * R(S) * t(S)
        obj = gp.LinExpr()
        for idx in range(k):
            obj += lam * R_vals[idx] * t_vars[idx]
        master.setObjective(obj, GRB.MAXIMIZE)

        # Capacity constraints: sum_S lambda * Q_i(S) * t(S) <= c_i
        cap_constrs = []
        for i in range(m):
            constr = gp.LinExpr()
            for idx in range(k):
                constr += lam * Q_vals[idx][i] * t_vars[idx]
            cap_constrs.append(master.addConstr(constr <= c[i], name=f"cap_{i}"))

        # Time constraint: sum_S t(S) <= T
        time_constr_expr = gp.LinExpr()
        for idx in range(k):
            time_constr_expr += t_vars[idx]
        time_constr = master.addConstr(time_constr_expr <= T, name="time")

        master.optimize()

        if master.status != GRB.OPTIMAL:
            break

        best_obj = master.ObjVal

        # Get dual prices
        pi = np.array([cap_constrs[i].Pi for i in range(m)])
        sigma = time_constr.Pi

        best_pi = pi.copy()
        best_sigma = sigma
        best_t_vals = {}
        for idx in range(k):
            if t_vars[idx].X > 1e-8:
                best_t_vals[idx] = t_vars[idx].X

        master.dispose()

        # Check time
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        # Solve column generation subproblem
        # First try greedy heuristic
        S_greedy, rc_greedy = greedy_heuristic(pi, sigma, prob_data)

        if rc_greedy > 1e-8 and len(S_greedy) > 0:
            new_col = frozenset(S_greedy)
            if new_col not in column_set:
                columns.append(new_col)
                column_set.add(new_col)
                R_S, Q_S = compute_R_and_Q(S_greedy, prob_data)
                R_vals.append(R_S)
                Q_vals.append(Q_S)
                continue

        # If greedy fails, try exact MIP
        elapsed = time.time() - start_time
        remaining = max(1, time_limit - elapsed)
        S_exact, rc_exact = exact_mip_subproblem(pi, sigma, prob_data,
                                                  time_limit=remaining)

        if rc_exact > 1e-8 and len(S_exact) > 0:
            new_col = frozenset(S_exact)
            if new_col not in column_set:
                columns.append(new_col)
                column_set.add(new_col)
                R_S, Q_S = compute_R_and_Q(S_exact, prob_data)
                R_vals.append(R_S)
                Q_vals.append(Q_S)
                continue

        # No entering column found -> optimal
        break

    # Build solution details
    solution_columns = []
    for idx, t_val in best_t_vals.items():
        solution_columns.append({
            "offer_set": sorted([j + 1 for j in columns[idx]]),  # 1-indexed
            "time_allocated": t_val
        })

    return {
        "objective_value": best_obj,
        "dual_prices_pi": best_pi.tolist(),
        "dual_price_sigma": best_sigma,
        "num_iterations": iteration,
        "num_columns_generated": len(columns),
        "active_columns": solution_columns
    }


def main():
    parser = argparse.ArgumentParser(
        description="Solve CDLP via Column Generation (Bront et al. 2009)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # Load instance
    data = load_instance(args.instance_path)

    # Build problem data structures
    prob_data = build_problem_data(data)

    print(f"Problem: {data.get('description', 'N/A')}")
    print(f"  Products: {prob_data['n']}, Legs: {prob_data['m']}, "
          f"Segments: {prob_data['L']}, T: {prob_data['T']}")
    print(f"  Capacities: {prob_data['c'].tolist()}")
    print(f"  Lambda: {prob_data['lam']}")
    print(f"  Time limit: {args.time_limit}s")

    # Solve CDLP via column generation
    start = time.time()
    result = solve_cdlp_column_generation(prob_data, args.time_limit)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  CDLP Objective Value: {result['objective_value']:.4f}")
    print(f"  Dual prices (pi): {result['dual_prices_pi']}")
    print(f"  Dual price (sigma): {result['dual_price_sigma']:.4f}")
    print(f"  Column generation iterations: {result['num_iterations']}")
    print(f"  Total columns generated: {result['num_columns_generated']}")
    print(f"  Elapsed time: {elapsed:.2f}s")
    print(f"\n  Active offer sets:")
    for col in result["active_columns"]:
        print(f"    S = {col['offer_set']}, t(S) = {col['time_allocated']:.4f}")

    # Save solution
    solution = {
        "objective_value": result["objective_value"],
        "instance_id": data.get("instance_id", "unknown"),
        "solver": "Gurobi (column generation)",
        "method": "CDLP",
        "elapsed_time_seconds": elapsed,
        "dual_prices_pi": result["dual_prices_pi"],
        "dual_price_sigma": result["dual_price_sigma"],
        "num_iterations": result["num_iterations"],
        "num_columns_generated": result["num_columns_generated"],
        "active_columns": result["active_columns"]
    }

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"\nSolution saved to {args.solution_path}")


if __name__ == "__main__":
    main()
