"""
Feasibility checker for the Undirected Traveling Purchaser Problem (TPP).

Based on: Laporte, Riera-Ledesma, and Salazar-González (2003)
"A Branch-and-Cut Algorithm for the Undirected Traveling Purchaser Problem"
Operations Research 51(6):940–951

Checks hard constraints (2)–(9) from the mathematical formulation, plus an
objective-consistency check (constraint index 10).

Constraint (10) — Objective consistency (Tier C defense):
    Recompute the objective value (1) directly from the solution variables
        w = sum_{e in E} c_e x_e + sum_{p_k in K} sum_{v_i in M_k} b_ki z_ki
    and reject the solution as infeasible when the reported
    `objective_value` disagrees with the recomputed value. The TPP solution
    carries every variable the objective depends on — the traversed edges
    `tour_edges` (x_e) and the `purchases` (z_ki) — so this is an exact
    full recompute, not a lower bound. This blocks LLM score-gaming
    exploits that fabricate `objective_value` while keeping the
    routes/purchases technically feasible.
"""

import argparse
import json
from collections import defaultdict


def load_instance(instance_path):
    """Load a TPP instance from JSON file."""
    with open(instance_path, 'r') as f:
        data = json.load(f)

    n_vertices = data["num_vertices"]
    n_products = data["num_products"]
    depot = data["depot"]
    supply_type = data["supply_type"]

    V = list(range(n_vertices))
    M = [v for v in V if v != depot]

    # travel_costs: key "i,j" (i<j) -> cost
    travel_costs = {}
    for key, cost in data["travel_costs"].items():
        i, j = map(int, key.split(","))
        travel_costs[(min(i, j), max(i, j))] = cost

    # product_demands: list indexed by product index
    demands = data["product_demands"]

    # product_markets: key str(product) -> list of market indices
    product_markets = {}
    for k_str, markets in data["product_markets"].items():
        product_markets[int(k_str)] = markets

    # purchase_costs: key "market,product" -> cost
    # Store as purchase_costs[product][market] = cost
    purchase_costs = {}
    for key, cost in data["purchase_costs"].items():
        parts = key.split(",")
        market = int(parts[0])
        product = int(parts[1])
        if product not in purchase_costs:
            purchase_costs[product] = {}
        purchase_costs[product][market] = cost

    # supplies: key "product,market" -> quantity
    # Store as supplies[product][market] = quantity
    supplies = {}
    if supply_type == "unlimited":
        for k in range(n_products):
            supplies[k] = {}
            for i in product_markets.get(k, []):
                supplies[k][i] = demands[k]
    else:
        for key, val in data.get("supplies", {}).items():
            parts = key.split(",")
            product = int(parts[0])
            market = int(parts[1])
            if product not in supplies:
                supplies[product] = {}
            supplies[product][market] = val

    return {
        "n_vertices": n_vertices,
        "n_markets": data["num_markets"],
        "n_products": n_products,
        "depot": depot,
        "supply_type": supply_type,
        "V": V,
        "M": M,
        "travel_costs": travel_costs,
        "demands": demands,
        "product_markets": product_markets,
        "purchase_costs": purchase_costs,
        "supplies": supplies,
    }


def load_solution(solution_path):
    """Load a TPP solution from JSON file."""
    with open(solution_path, 'r') as f:
        return json.load(f)


def compute_mandatory_sets(inst):
    """Compute M* (mandatory markets) and K* (products without market choice)."""
    depot = inst["depot"]
    n_products = inst["n_products"]
    demands = inst["demands"]
    product_markets = inst["product_markets"]
    supplies = inst["supplies"]

    # M* = {v_0} union {v_i in M : exists p_k such that sum_{j in M_k \ {i}} q_kj < d_k}
    M_star = {depot}
    for i in inst["M"]:
        for k in range(n_products):
            if i in product_markets.get(k, []):
                total_without_i = sum(
                    supplies.get(k, {}).get(j, 0)
                    for j in product_markets.get(k, []) if j != i
                )
                if total_without_i < demands[k]:
                    M_star.add(i)
                    break

    # K* = {p_k : sum_{i in M_k} q_ki = d_k}
    K_star = set()
    for k in range(n_products):
        total = sum(supplies.get(k, {}).get(i, 0) for i in product_markets.get(k, []))
        if total == demands[k]:
            K_star.add(k)

    return M_star, K_star


def check_feasibility(inst, sol):
    """
    Check all hard constraints (2)-(9) of the TPP formulation, plus the
    objective-consistency constraint (10).

    Returns a dict with feasibility result.
    """
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    depot = inst["depot"]
    n_vertices = inst["n_vertices"]
    n_products = inst["n_products"]
    V = inst["V"]
    M = inst["M"]
    demands = inst["demands"]
    product_markets = inst["product_markets"]
    supplies = inst["supplies"]
    travel_costs = inst["travel_costs"]
    purchase_costs = inst["purchase_costs"]

    M_star, K_star = compute_mandatory_sets(inst)

    # --- Extract solution variables ---

    # y_i: whether market i is visited (1 if visited, 0 otherwise)
    visited_markets = set(sol.get("visited_markets", []))
    y = {}
    y[depot] = 1  # depot always visited
    for i in M:
        y[i] = 1 if i in visited_markets else 0

    # x_e: edge variables from tour_edges
    tour_edges = sol.get("tour_edges", [])
    x = defaultdict(int)
    for edge in tour_edges:
        e = (min(edge[0], edge[1]), max(edge[0], edge[1]))
        x[e] = 1

    # z_ki: purchase quantities
    # sol["purchases"] is {str(product): {str(market): quantity}}
    z = {}
    for k in range(n_products):
        z[k] = {}
    for k_str, market_dict in sol.get("purchases", {}).items():
        k = int(k_str)
        for m_str, qty in market_dict.items():
            m = int(m_str)
            z[k][m] = float(qty)

    def add_violation(constraint_idx, message, lhs, rhs):
        """Record a constraint violation."""
        if constraint_idx == 2:
            # = constraint: violation = |lhs - rhs|
            violation_amount = abs(lhs - rhs)
        elif constraint_idx == 3:
            # >= constraint: violation = max(0, rhs - lhs)
            violation_amount = max(0.0, rhs - lhs)
        elif constraint_idx == 4:
            # = constraint: violation = |lhs - rhs|
            violation_amount = abs(lhs - rhs)
        elif constraint_idx == 5:
            # <= constraint: violation = max(0, lhs - rhs)
            violation_amount = max(0.0, lhs - rhs)
        elif constraint_idx == 6:
            # binary: violation = deviation from {0,1}
            violation_amount = abs(lhs - rhs)
        elif constraint_idx == 7:
            # binary: violation = deviation from {0,1}
            violation_amount = abs(lhs - rhs)
        elif constraint_idx == 8:
            # = 1 constraint: violation = |lhs - 1|
            violation_amount = abs(lhs - rhs)
        elif constraint_idx == 9:
            # >= 0 constraint: violation = max(0, -lhs)
            violation_amount = max(0.0, rhs - lhs)
        else:
            violation_amount = abs(lhs - rhs)

        if violation_amount > tol:
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violated_constraints_set.add(constraint_idx)
            violations.append(message)
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": lhs,
                "rhs": rhs,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": ratio,
            })

    # =========================================================================
    # Constraint (2): Degree constraints
    #   sum_{e in delta({v_i})} x_e = 2 * y_i   for all v_i in V
    # =========================================================================
    for v in V:
        degree = 0
        for e, val in x.items():
            if val > tol and (e[0] == v or e[1] == v):
                degree += val
        lhs = degree
        rhs = 2.0 * y.get(v, 0)
        if abs(lhs - rhs) > tol:
            add_violation(2,
                          f"Degree constraint violated at vertex {v}: "
                          f"degree={lhs}, 2*y_{v}={rhs}",
                          lhs, rhs)

    # =========================================================================
    # Constraint (3): Connectivity constraints
    #   sum_{e in delta(S)} x_e >= 2 * y_i   for all S subset M, v_i in S
    #
    # Instead of enumerating all subsets, we check connectivity of the tour
    # restricted to visited vertices + depot. If the subgraph induced by
    # visited vertices (with edges from x) is connected, then all connectivity
    # constraints are satisfied.
    # =========================================================================
    visited_with_depot = visited_markets | {depot}
    if len(visited_with_depot) > 1:
        # Build adjacency for visited vertices
        adj = defaultdict(set)
        for e, val in x.items():
            if val > tol and e[0] in visited_with_depot and e[1] in visited_with_depot:
                adj[e[0]].add(e[1])
                adj[e[1]].add(e[0])

        # BFS from depot
        reachable = set()
        queue = [depot]
        reachable.add(depot)
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        # Check if all visited markets are reachable from depot
        unreachable = visited_with_depot - reachable
        if unreachable:
            # For each unreachable vertex, the connectivity constraint is violated
            # S = unreachable set, delta(S) has no edges, but y_i=1 for i in S
            # So sum x_e in delta(S) = 0 < 2 * y_i = 2
            for v in unreachable:
                lhs = 0.0
                rhs = 2.0 * y.get(v, 0)
                if rhs > tol:
                    add_violation(3,
                                  f"Connectivity violated: vertex {v} not reachable "
                                  f"from depot in tour subgraph",
                                  lhs, rhs)

    # =========================================================================
    # Constraint (4): Product demand satisfaction
    #   sum_{v_i in M_k} z_ki = d_k   for all p_k in K
    # =========================================================================
    for k in range(n_products):
        total_purchased = sum(z.get(k, {}).values())
        lhs = total_purchased
        rhs = float(demands[k])
        if abs(lhs - rhs) > tol:
            add_violation(4,
                          f"Demand not satisfied for product {k}: "
                          f"purchased={lhs}, demand={rhs}",
                          lhs, rhs)

    # =========================================================================
    # Constraint (5): Purchase only at visited markets with supply limits
    #   z_ki <= q_ki * y_i   for all p_k in K, v_i in M_k
    # =========================================================================
    for k in range(n_products):
        for m, qty in z.get(k, {}).items():
            if qty > tol:
                q_ki = supplies.get(k, {}).get(m, 0)
                y_i = y.get(m, 0)
                lhs = float(qty)
                rhs = float(q_ki * y_i)
                if lhs - rhs > tol:
                    if y_i == 0:
                        msg = (f"Product {k} purchased at unvisited market {m}: "
                               f"z={lhs}, q*y={rhs}")
                    else:
                        msg = (f"Product {k} purchase exceeds supply at market {m}: "
                               f"z={lhs}, q*y={rhs}")
                    add_violation(5, msg, lhs, rhs)

    # =========================================================================
    # Constraint (6): Binary edge variables
    #   x_e in {0, 1}   for all e in E
    # =========================================================================
    for e, val in x.items():
        if abs(val - 0) > tol and abs(val - 1) > tol:
            nearest = 0.0 if val < 0.5 else 1.0
            add_violation(6,
                          f"Edge {e} has non-binary value x={val}",
                          float(val), nearest)

    # =========================================================================
    # Constraint (7): Binary vertex selection for non-mandatory markets
    #   y_i in {0, 1}   for all v_i in M \ M*
    # =========================================================================
    for i in M:
        if i not in M_star:
            yi = y.get(i, 0)
            if abs(yi - 0) > tol and abs(yi - 1) > tol:
                nearest = 0.0 if yi < 0.5 else 1.0
                add_violation(7,
                              f"Market {i} (non-mandatory) has non-binary y={yi}",
                              float(yi), nearest)

    # =========================================================================
    # Constraint (8): Mandatory market visits
    #   y_i = 1   for all v_i in M*
    # =========================================================================
    for i in M_star:
        yi = y.get(i, 0)
        lhs = float(yi)
        rhs = 1.0
        if abs(lhs - rhs) > tol:
            add_violation(8,
                          f"Mandatory market {i} not visited: y={yi}",
                          lhs, rhs)

    # =========================================================================
    # Constraint (9): Non-negativity of purchase quantities
    #   z_ki >= 0   for all p_k in K, v_i in M_k
    # =========================================================================
    for k in range(n_products):
        for m in product_markets.get(k, []):
            z_ki = z.get(k, {}).get(m, 0.0)
            lhs = float(z_ki)
            rhs = 0.0
            if lhs < -tol:
                add_violation(9,
                              f"Negative purchase: z_{k},{m} = {lhs}",
                              lhs, rhs)

    # =========================================================================
    # Constraint (10): Objective consistency  [Tier C defense]
    #   reported objective_value must equal the objective (1) recomputed
    #   from the solution variables:
    #     w = sum_{e in E} c_e x_e + sum_{p_k in K} sum_{v_i in M_k} b_ki z_ki
    #   Travel costs are read with the same key convention as load_instance
    #   (travel_costs[(min,max)]); purchase costs with purchase_costs[k][m].
    #   The TPP solution carries every objective-determining variable
    #   (tour_edges -> x_e, purchases -> z_ki), so this is an exact full
    #   recompute. Rejecting on mismatch blocks fabricated objective values.
    # =========================================================================
    reported_obj = sol.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_route = sum(
                travel_costs.get(e, 0)
                for e, val in x.items() if val > tol
            )
            true_purchase = sum(
                purchase_costs.get(k, {}).get(m, 0) * zq
                for k in z
                for m, zq in z[k].items()
            )
            true_obj = float(true_route + true_purchase)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                add_violation(10,
                              f"Objective consistency violated: reported "
                              f"objective_value={reported} differs from recomputed "
                              f"sum_e(c_e*x_e)+sum_ki(b_ki*z_ki)={true_obj} "
                              f"(routing={true_route}, purchase={true_purchase}, "
                              f"|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                              reported, true_obj)

    # Build result
    feasible = len(violated_constraints_set) == 0
    violated_constraints = sorted(violated_constraints_set)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Undirected Traveling Purchaser Problem (TPP)."
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON file containing the data instance.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON file containing the candidate solution.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON file containing the feasibility result.")
    args = parser.parse_args()

    inst = load_instance(args.instance_path)
    sol = load_solution(args.solution_path)
    result = check_feasibility(inst, sol)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE.")
        print(f"Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
