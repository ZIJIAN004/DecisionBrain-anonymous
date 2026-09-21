#!/usr/bin/env python3
"""
Feasibility checker for S2L-CVRP (Stochastic 2D Loading CVRP).

Checks hard constraints from the mathematical formulation:
  Constraint 1 (Eq 2): depot degree = 2K
  Constraint 2 (Eq 3): each customer has degree 2
  Constraint 3 (Eq 4): Rounded Capacity Inequalities (RCI)
  Constraint 4 (Eq 5): Infeasible path inequalities
  Constraint 5 (Eq 6): Binary integrality of edge variables
  Constraint 6 (obj envelope): reported objective must lie within
      [routing_cost, routing_cost + c_f * n_customers], where
      routing_cost is recomputed exactly from the routes and the
      distance matrix and F(x) is bounded by 0 <= F(x) <= c_f * n.
"""

import argparse
import json
import math
import itertools
from collections import defaultdict


# ---------------------------------------------------------------------------
# Tolerance parameters
# ---------------------------------------------------------------------------
TOL = 1e-5
EPS = 1e-5


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def compute_expected_area_weight(customers):
    """
    Compute expected area a_tilde_j and expected weight q_tilde_j for each customer.
    a_tilde_j = sum_{i in I_j} sum_r p_i^r * h_i^r * w_i^r
    q_tilde_j = sum_{i in I_j} sum_r p_i^r * q_i^r
    """
    result = {}
    for cust in customers:
        cid = cust["id"]
        a_tilde = 0.0
        q_tilde = 0.0
        for item in cust["items"]:
            for r in item["realizations"]:
                p = r["probability"]
                a_tilde += p * r["height"] * r["width"]
                q_tilde += p * r["weight"]
        result[cid] = (a_tilde, q_tilde)
    return result


# ---------------------------------------------------------------------------
# Reconstruct edge variables from routes
# ---------------------------------------------------------------------------

def reconstruct_edges(routes):
    """
    Given routes (lists of customer IDs), reconstruct x_{jk} edge variables.
    Each route is 0 -> c1 -> c2 -> ... -> cm -> 0.
    Returns dict {(j,k): 1} with j < k.
    """
    edges = {}
    for route in routes:
        full_path = [0] + list(route) + [0]
        for i in range(len(full_path) - 1):
            u, v = full_path[i], full_path[i + 1]
            key = (min(u, v), max(u, v))
            edges[key] = edges.get(key, 0) + 1
    return edges


# ---------------------------------------------------------------------------
# Bottom-left heuristic for 2D packing with unloading constraints
# ---------------------------------------------------------------------------

def bottom_left_heuristic_with_unloading(items_with_delivery_order, H, W):
    """
    Bottom-left heuristic for 2OPP with unloading constraints.

    items_with_delivery_order: list of (item_h, item_w, delivery_position)
    H: vehicle height, W: vehicle width.

    Items are packed in reverse delivery order (last delivered first, deepest).
    Unloading from y=H end; later-delivered items must not block earlier ones.

    Returns True if all items packed feasibly.
    """
    sorted_items = sorted(items_with_delivery_order, key=lambda t: -t[2])

    placements = []  # (x, y, w, h, delivery_pos)

    for (ih, iw, dpos) in sorted_items:
        if ih > H or iw > W:
            return False

        placed = False
        y_candidates = sorted(set([0] + [p[1] + p[3] for p in placements]))
        x_candidates = sorted(set([0] + [p[0] + p[2] for p in placements]))

        best_pos = None
        for y in y_candidates:
            if y + ih > H:
                continue
            for x in x_candidates:
                if x + iw > W:
                    continue
                # Check no overlap
                overlap = False
                for (px, py, pw, ph, pd) in placements:
                    if x < px + pw and x + iw > px and y < py + ph and y + ih > py:
                        overlap = True
                        break
                if overlap:
                    continue

                # Check unloading: items with delivery_pos > dpos are still on
                # truck when dpos is delivered; they must not block from y=H exit.
                unloading_ok = True
                for (px, py, pw, ph, pd) in placements:
                    if pd > dpos:
                        if (px < x + iw and px + pw > x and
                                py + ph > y + ih and py < H):
                            unloading_ok = False
                            break
                if not unloading_ok:
                    continue

                if best_pos is None or (y, x) < (best_pos[1], best_pos[0]):
                    best_pos = (x, y)

            if best_pos is not None and best_pos[1] == y:
                break

        if best_pos is None:
            return False

        placements.append((best_pos[0], best_pos[1], iw, ih, dpos))

    return True


def check_packing_feasibility_heuristic(items_by_customer_ordered, H, W):
    """
    Check packing feasibility with unloading using bottom-left heuristic.
    items_by_customer_ordered: list of (customer_id, [(h, w)]) in delivery order.
    Returns True if heuristic finds a feasible packing.
    """
    if not items_by_customer_ordered:
        return True

    total_area = sum(h * w for _, items in items_by_customer_ordered for h, w in items)
    if total_area > H * W:
        return False

    for _, items in items_by_customer_ordered:
        for h, w in items:
            if h > H or w > W:
                return False

    # Build items with delivery positions
    pack_items = []
    for pos_idx, (cid, cust_items) in enumerate(items_by_customer_ordered):
        delivery_pos = pos_idx + 1
        for h, w in cust_items:
            pack_items.append((h, w, delivery_pos))

    return bottom_left_heuristic_with_unloading(pack_items, H, W)


# ---------------------------------------------------------------------------
# Scenario enumeration
# ---------------------------------------------------------------------------

def enumerate_scenarios_for_route(route_customer_ids, customers_by_id):
    """
    Enumerate all scenarios for customers on a route via Cartesian product
    of item realizations.
    Returns list of (probability, [(cust_id, [(h, w, weight)])]).
    """
    item_realizations = []
    for cid in route_customer_ids:
        cust = customers_by_id[cid]
        for item in cust["items"]:
            reals = []
            for r in item["realizations"]:
                reals.append((r["height"], r["width"], r["weight"],
                              r["probability"], cid))
            item_realizations.append(reals)

    if not item_realizations:
        return [(1.0, [])]

    scenarios = []
    for combo in itertools.product(*item_realizations):
        prob = 1.0
        cust_items = defaultdict(list)
        for h, w, weight, p, cid in combo:
            prob *= p
            cust_items[cid].append((h, w, weight))
        scenario_data = [(cid, cust_items[cid]) for cid in route_customer_ids
                         if cid in cust_items]
        scenarios.append((prob, scenario_data))

    return scenarios


def is_route_always_infeasible(route, customers_by_id, H, W, Q):
    """
    Check if a route is in R^{inf}: infeasible under ALL scenarios and
    both delivery orderings.
    """
    # Cap scenario count to avoid exponential blowup
    n_scenarios = 1
    for cid in route:
        cust = customers_by_id[cid]
        for item in cust["items"]:
            n_scenarios *= len(item["realizations"])
    if n_scenarios > 50000:
        return False  # conservative: assume not always infeasible

    scenarios = enumerate_scenarios_for_route(route, customers_by_id)

    for prob, scenario_items in scenarios:
        # Check weight
        total_weight = sum(w for _, items in scenario_items for _, _, w in items)
        if total_weight > Q:
            continue  # infeasible under this scenario due to weight

        # Check packing feasibility in both orderings
        for ordering in [scenario_items, list(reversed(scenario_items))]:
            items_ordered = [(cid, [(h, w) for h, w, _ in items])
                             for cid, items in ordering]
            if check_packing_feasibility_heuristic(items_ordered, H, W):
                return False  # at least one scenario is feasible

    return True  # all scenarios infeasible


# ---------------------------------------------------------------------------
# Violation recording helpers
# ---------------------------------------------------------------------------

def make_violation_entry(constraint_idx, lhs, rhs, violation_amount):
    normalizer = max(abs(rhs), EPS)
    return {
        "constraint": constraint_idx,
        "lhs": lhs,
        "rhs": rhs,
        "raw_excess": violation_amount,
        "normalizer": normalizer,
        "ratio": violation_amount / normalizer,
    }


# ---------------------------------------------------------------------------
# Main feasibility check
# ---------------------------------------------------------------------------

def check_feasibility(instance, solution):
    customers = instance["customers"]
    vehicle = instance["vehicle"]
    H = vehicle["H"]
    W = vehicle["W"]
    Q = vehicle["Q"]
    K = vehicle["K"]
    n_customers = len(customers)

    customers_by_id = {c["id"]: c for c in customers}
    customer_ids = set(c["id"] for c in customers)
    expected = compute_expected_area_weight(customers)

    routes = solution["routes"]

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    # Reconstruct edges from routes
    edges = reconstruct_edges(routes)

    # -----------------------------------------------------------------------
    # Constraint 1 (Eq 2): sum_{j in C} x_{0j} = 2K
    # -----------------------------------------------------------------------
    depot_degree = 0
    for (j, k), val in edges.items():
        if j == 0 or k == 0:
            depot_degree += val
    rhs_c1 = 2 * K
    violation_amount_c1 = abs(depot_degree - rhs_c1)
    if violation_amount_c1 > TOL:
        violated_constraint_set.add(1)
        violations.append(
            f"Constraint 1 (depot degree): depot degree = {depot_degree}, "
            f"expected 2K = {rhs_c1}"
        )
        violation_magnitudes.append(
            make_violation_entry(1, depot_degree, rhs_c1, violation_amount_c1)
        )

    # -----------------------------------------------------------------------
    # Constraint 2 (Eq 3): degree = 2 for each customer j in C
    # -----------------------------------------------------------------------
    # Count degree per customer from the edge set
    degree = defaultdict(int)
    for (j, k), val in edges.items():
        if j > 0:
            degree[j] += val
        if k > 0:
            degree[k] += val

    # Check each customer
    for cid in sorted(customer_ids):
        deg = degree.get(cid, 0)
        rhs_c2 = 2
        viol = abs(deg - rhs_c2)
        if viol > TOL:
            violated_constraint_set.add(2)
            violations.append(
                f"Constraint 2 (degree): customer {cid} has degree {deg}, "
                f"expected 2"
            )
            violation_magnitudes.append(
                make_violation_entry(2, deg, rhs_c2, viol)
            )

    # Also check customers not appearing in any route
    visited_customers = set()
    for route in routes:
        for c in route:
            visited_customers.add(c)
    missing = customer_ids - visited_customers
    for cid in sorted(missing):
        violated_constraint_set.add(2)
        violations.append(
            f"Constraint 2 (degree): customer {cid} not visited (degree 0), "
            f"expected 2"
        )
        violation_magnitudes.append(
            make_violation_entry(2, 0, 2, 2.0)
        )

    # -----------------------------------------------------------------------
    # Constraint 3 (Eq 4): RCI for each route's customer set S
    # sum_{j,k in S, j<k} x_{jk} <= |S| - max(ceil(sum_a/HW), ceil(sum_q/Q))
    # -----------------------------------------------------------------------
    HW = H * W
    for route_idx, route in enumerate(routes):
        S = set(route)
        if len(S) < 2:
            continue

        # LHS: edges within S
        lhs_c3 = 0
        S_list = sorted(S)
        for i in range(len(S_list)):
            for j_idx in range(i + 1, len(S_list)):
                key = (S_list[i], S_list[j_idx])
                lhs_c3 += edges.get(key, 0)

        # RHS
        sum_area = sum(expected[cid][0] for cid in S)
        sum_weight = sum(expected[cid][1] for cid in S)
        area_vehicles = math.ceil(sum_area / HW)
        weight_vehicles = math.ceil(sum_weight / Q)
        rhs_c3 = len(S) - max(area_vehicles, weight_vehicles)

        violation_amount_c3 = lhs_c3 - rhs_c3
        if violation_amount_c3 > TOL:
            violated_constraint_set.add(3)
            reason_parts = []
            if area_vehicles > 1:
                reason_parts.append(
                    f"expected area {sum_area:.1f} exceeds vehicle area {HW}"
                )
            if weight_vehicles > 1:
                reason_parts.append(
                    f"expected weight {sum_weight:.1f} exceeds vehicle capacity {Q}"
                )
            reason = "; ".join(reason_parts) if reason_parts else "RCI violated"
            violations.append(
                f"Constraint 3 (RCI): route {route_idx + 1} with customers "
                f"{sorted(S)}: LHS={lhs_c3}, RHS={rhs_c3} ({reason})"
            )
            violation_magnitudes.append(
                make_violation_entry(3, lhs_c3, rhs_c3, violation_amount_c3)
            )

    # -----------------------------------------------------------------------
    # Constraint 4 (Eq 5): Infeasible path inequalities
    # No route should be in R^{inf} (always infeasible under all scenarios)
    # -----------------------------------------------------------------------
    for route_idx, route in enumerate(routes):
        if not route:
            continue

        if is_route_always_infeasible(route, customers_by_id, H, W, Q):
            # The route is in R^{inf}, violating constraint 5
            # LHS = sum of edges on route path, RHS = |R| - 1
            full_path = [0] + list(route) + [0]
            route_edge_set = set()
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                route_edge_set.add((min(u, v), max(u, v)))

            lhs_c4 = sum(edges.get(e, 0) for e in route_edge_set)
            rhs_c4 = len(route_edge_set) - 1
            violation_amount_c4 = lhs_c4 - rhs_c4

            if violation_amount_c4 > TOL:
                violated_constraint_set.add(4)
                violations.append(
                    f"Constraint 4 (infeasible path): route {route_idx + 1} "
                    f"with customers {sorted(route)} is always infeasible "
                    f"(in R^{{inf}})"
                )
                violation_magnitudes.append(
                    make_violation_entry(4, lhs_c4, rhs_c4, violation_amount_c4)
                )

    # -----------------------------------------------------------------------
    # Constraint 5 (Eq 6): x_{jk} in {0, 1}
    # Check that no edge has value other than 0 or 1
    # -----------------------------------------------------------------------
    for (j, k), val in edges.items():
        if val != 0 and val != 1:
            violated_constraint_set.add(5)
            viol_amount = min(abs(val - 0), abs(val - 1))
            violations.append(
                f"Constraint 5 (binary): edge ({j},{k}) has value {val}, "
                f"expected 0 or 1"
            )
            violation_magnitudes.append(
                make_violation_entry(5, val, round(val), viol_amount)
            )

    # -----------------------------------------------------------------------
    # Constraint 6 (objective envelope): defends against LLM score-gaming
    # exploits where the reported objective_value contradicts the routes.
    #
    # obj = sum_{j<k} c_{jk} x_{jk} + F(x)
    # where F(x) >= 0 (recourse penalty for unserved customers) and
    #       F(x) <= c_f * n_customers (every customer can at most be one
    #             unserved unit of recourse cost).
    # Therefore: routing_cost <= obj <= routing_cost + c_f * n_customers,
    # with routing_cost = sum of distance_matrix entries on each route's
    # depot->customer->...->depot path (recomputed exactly here).
    #
    # Reject when the reported objective lies outside that envelope by
    # more than max(1e-3, 1e-3 * envelope-width).
    # -----------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None and not (math.isnan(reported) or
                                         math.isinf(reported)):
            dm = instance.get("distance_matrix")
            params = instance.get("parameters", {})
            cf = params.get("recourse_cost_cf")
            if dm is not None and cf is not None:
                routing_cost = 0.0
                for route in routes:
                    full_path = [0] + list(route) + [0]
                    for i in range(len(full_path) - 1):
                        u, v = full_path[i], full_path[i + 1]
                        routing_cost += dm[u][v]
                obj_lower = float(routing_cost)
                obj_upper = float(routing_cost) + float(cf) * n_customers
                tol = max(1e-3, 1e-3 * max(abs(obj_lower), abs(obj_upper)))
                if reported < obj_lower - tol:
                    viol_amount = obj_lower - reported
                    violated_constraint_set.add(6)
                    violations.append(
                        f"Constraint 6 (objective envelope): reported "
                        f"objective_value={reported} is below the routing-cost "
                        f"lower bound {obj_lower} (F(x) >= 0; |diff|="
                        f"{viol_amount:.3g}, tol={tol:.3g})"
                    )
                    violation_magnitudes.append(
                        make_violation_entry(6, reported, obj_lower, viol_amount)
                    )
                elif reported > obj_upper + tol:
                    viol_amount = reported - obj_upper
                    violated_constraint_set.add(6)
                    violations.append(
                        f"Constraint 6 (objective envelope): reported "
                        f"objective_value={reported} exceeds the upper bound "
                        f"routing_cost + c_f * n = {obj_upper} (F(x) <= "
                        f"c_f * n; |diff|={viol_amount:.3g}, tol={tol:.3g})"
                    )
                    violation_magnitudes.append(
                        make_violation_entry(6, reported, obj_upper, viol_amount)
                    )

    # -----------------------------------------------------------------------
    # Build result
    # -----------------------------------------------------------------------
    feasible = len(violated_constraint_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraint_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for S2L-CVRP solutions"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to instance JSON file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to candidate solution JSON file")
    parser.add_argument("--result_path", required=True,
                        help="Path to write feasibility result JSON file")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("FEASIBLE: All hard constraints satisfied.")
    else:
        print(f"INFEASIBLE: Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
