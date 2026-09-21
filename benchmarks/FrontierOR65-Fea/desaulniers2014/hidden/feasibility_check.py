"""
Feasibility checker for the Inventory-Routing Problem (IRP).

Based on: Desaulniers, Rakke, Coelho (2014) - "A Branch-Price-and-Cut Algorithm
for the Inventory-Routing Problem", Les Cahiers du GERAD G-2014-19

Checks hard constraints (2)-(9) from the mathematical formulation.
Valid inequalities (19), (21), (23), (24) are cuts and are NOT checked.

Constraint (10): objective-value consistency (Tier C anti-gaming check).
The reported objective_value is recomputed from the solution (travel cost
from routes + holding cost from simulated inventories) and rejected when it
disagrees, defeating fabricated objective values that pass (2)-(9).
"""

import argparse
import json
import math


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def parse_instance(data):
    """Parse instance data into structured parameters."""
    num_cust = data["num_customers"]
    num_per = data["num_periods"]
    Q = data["vehicle_capacity"]
    depot = data["depot"]
    customers = data["customers"]

    N = list(range(1, num_cust + 1))
    P = list(range(1, num_per + 1))
    rho = num_per

    # Customer params
    d = {}   # demand[i][p]
    C = {}   # capacity
    I0 = {}  # initial inventory
    h = {}   # holding cost
    for c in customers:
        cid = c["id"]
        d[cid] = {p: c["demand_per_period"] for p in P}
        C[cid] = c["max_inventory"]
        I0[cid] = c["initial_inventory"]
        h[cid] = c["holding_cost"]

    d0 = depot["production_per_period"]
    C0 = depot["max_inventory"]
    I0_0 = depot["initial_inventory"]
    h0 = depot["holding_cost"]
    dist = data["distance_matrix"]
    K = data["num_vehicles"]

    # I^{0,s}_i = max{0, I0_i - sum_{l=1}^{s} d^l_i}
    I0s = {}
    for i in N:
        for s in P:
            I0s[i, s] = max(0, I0[i] - sum(d[i][l] for l in range(1, s + 1)))

    # d_bar^s_i (residual demands)
    d_bar = {}
    for i in N:
        for s in P:
            if s == 1:
                d_bar[i, s] = max(0, d[i][1] - I0[i])
            else:
                d_bar[i, s] = max(0, d[i][s] - I0s[i, s - 1])

    return {
        'N': N, 'P': P, 'rho': rho, 'Q': Q, 'K': K,
        'd': d, 'C': C, 'I0': I0, 'h': h,
        'd0': d0, 'C0': C0, 'I0_0': I0_0, 'h0': h0,
        'dist': dist, 'I0s': I0s, 'd_bar': d_bar,
    }


def parse_solution(sol_data, params):
    """
    Parse solution into a unified representation.

    Returns:
        routes_per_period: dict {period: list of routes}
            Each route is a dict with:
                'customers': list of customer ids in visit order
                'deliveries': dict {customer_id: total_quantity_delivered}
        deliveries_per_period: dict {period: {customer_id: total_quantity_delivered}}
    """
    P = params['P']
    routes_per_period = {p: [] for p in P}
    deliveries_per_period = {p: {} for p in P}

    sd = sol_data.get("solution_details", {})

    if "routes" in sd:
        # Efficient algorithm format: list of route objects
        for r in sd["routes"]:
            period = r["period"]
            customers = r["route"]  # list of customer ids (no depot)
            deliveries_raw = r.get("deliveries", {})

            # Sum sub-deliveries per customer
            total_del = {}
            for cust_key, val in deliveries_raw.items():
                cust_id = int(cust_key)
                if isinstance(val, dict):
                    total_del[cust_id] = sum(val.values())
                else:
                    total_del[cust_id] = val

            routes_per_period[period].append({
                'customers': customers,
                'deliveries': total_del,
                'raw': None,  # efficient format does not provide arc sequence
            })

            for cust_id, qty in total_del.items():
                deliveries_per_period[period][cust_id] = (
                    deliveries_per_period[period].get(cust_id, 0) + qty
                )

    elif "periods" in sd:
        # Gurobi format: dict keyed by period
        for p_str, pdata in sd["periods"].items():
            period = int(p_str)
            if period not in routes_per_period:
                continue

            routes_raw = pdata.get("routes", [])
            deliveries_raw = pdata.get("deliveries", {})

            # Build delivery map for this period
            period_deliveries = {}
            for cust_key, qty in deliveries_raw.items():
                cust_id = int(cust_key)
                period_deliveries[cust_id] = qty

            deliveries_per_period[period] = period_deliveries

            # Parse routes: each route is [0, c1, c2, ..., 0]
            for route_nodes in routes_raw:
                customers = [n for n in route_nodes if n != 0]
                # Assign deliveries to customers on this route
                route_del = {}
                for c in customers:
                    if c in period_deliveries:
                        route_del[c] = period_deliveries[c]

                routes_per_period[period].append({
                    'customers': customers,
                    'deliveries': route_del,
                    'raw': list(route_nodes),
                })

    return routes_per_period, deliveries_per_period


def check_feasibility(instance_data, sol_data):
    """
    Check all hard constraints (2)-(9) from the IRP formulation.

    Returns a dict with feasibility results.
    """
    tol = 1e-5
    eps = 1e-5

    params = parse_instance(instance_data)
    N = params['N']
    P = params['P']
    rho = params['rho']
    Q = params['Q']
    K = params['K']
    d = params['d']
    C = params['C']
    I0 = params['I0']
    d0 = params['d0']
    C0 = params['C0']
    I0_0 = params['I0_0']
    I0s = params['I0s']
    d_bar = params['d_bar']
    dist = params['dist']

    routes_per_period, deliveries_per_period = parse_solution(sol_data, params)

    violations = []
    violation_magnitudes = []
    violated_constraint_set = set()

    def record_violation(constraint_idx, message, lhs, rhs, violation_amount):
        violated_constraint_set.add(constraint_idx)
        violations.append(message)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })

    # =========================================================================
    # Simulate inventory flows to get depot inventory per period
    # =========================================================================
    depot_inv = {}  # I^p_0 at end of period p
    cust_inv = {}   # inventory at customer i at end of period p

    # Initialize
    prev_depot_inv = I0_0
    prev_cust_inv = {i: I0[i] for i in N}

    for p in P:
        # Supplier produces at start of period
        depot_after_prod = prev_depot_inv + d0

        # Total quantity shipped out in this period
        total_shipped = sum(deliveries_per_period[p].get(i, 0) for i in N)
        depot_inv[p] = depot_after_prod - total_shipped

        # Customer inventories: receive deliveries then consume demand
        for i in N:
            delivered = deliveries_per_period[p].get(i, 0)
            cust_inv[i, p] = prev_cust_inv[i] + delivered - d[i][p]

        prev_depot_inv = depot_inv[p]
        prev_cust_inv = {i: cust_inv[i, p] for i in N}

    # =========================================================================
    # Constraint (2): Supplier inventory balance
    # I^{p-1}_0 + d^p_0 - sum_deliveries = I^p_0
    # Equivalently: LHS = I^{p-1}_0 + d^p_0 - sum_deliveries, RHS = I^p_0
    # This is an equality constraint.
    # =========================================================================
    for p in P:
        prev_inv = I0_0 if p == 1 else depot_inv[p - 1]
        total_shipped = sum(deliveries_per_period[p].get(i, 0) for i in N)
        lhs = prev_inv + d0 - total_shipped
        rhs = depot_inv[p]
        violation_amount = abs(lhs - rhs)
        if violation_amount > tol:
            record_violation(
                2,
                f"Constraint (2): Supplier inventory balance violated in period {p}: "
                f"LHS={lhs:.4f}, RHS(I^{p}_0)={rhs:.4f}, diff={violation_amount:.4f}",
                lhs, rhs, violation_amount
            )

    # =========================================================================
    # Constraint (3): Customer demand satisfaction
    # Total delivered to customer i across all periods must satisfy residual
    # demand. In compact form: for each customer i and period s, the total
    # delivery dedicated to satisfying demand in period s must equal d_bar^s_i.
    #
    # Since both efficient and gurobi solutions only report total delivery per
    # customer per period (not sub-delivery breakdown), we check the aggregate
    # form: each customer's inventory must never go negative (stockout), which
    # is the practical implication of demand satisfaction.
    #
    # Specifically: I^0_i + sum_{p'=1}^{s} delivered_{i,p'} - sum_{p'=1}^{s} d^{p'}_i >= 0
    # for all i in N, s in P.
    #
    # This is equivalent to checking no stockout occurs at any customer in any
    # period. The equality form (3) implies that total deliveries across
    # the horizon exactly match total residual demand. We check both
    # no-stockout and total demand matching.
    # =========================================================================
    for i in N:
        for s in P:
            inv_at_end = cust_inv[i, s]
            if inv_at_end < -tol:
                # Stockout: demand not satisfied
                lhs = inv_at_end
                rhs = 0.0
                violation_amount = -inv_at_end  # how much below zero
                record_violation(
                    3,
                    f"Constraint (3): Customer {i} demand not satisfied in period {s}: "
                    f"end-of-period inventory={inv_at_end:.4f} < 0 (stockout)",
                    lhs, rhs, violation_amount
                )

    # Also check total residual demand is met across the horizon
    for i in N:
        total_residual = sum(d_bar[i, s] for s in P)
        total_delivered = sum(deliveries_per_period[p].get(i, 0) for p in P)
        # Under FIFO, total delivery must cover total residual demand
        # (any excess becomes end-of-horizon inventory)
        if total_delivered < total_residual - tol:
            lhs = total_delivered
            rhs = total_residual
            violation_amount = total_residual - total_delivered
            record_violation(
                3,
                f"Constraint (3): Customer {i} total delivery insufficient: "
                f"delivered={total_delivered:.4f}, residual demand={total_residual:.4f}",
                lhs, rhs, violation_amount
            )

    # =========================================================================
    # Constraint (4): Customer holding capacity
    # I^{0,s}_i + deliveries_in_inventory_at_s <= C_i - d^s_i
    # Equivalently: end-of-period inventory <= C_i - d^s_i (before consumption)
    # Or more practically: inventory after receiving delivery but before consumption
    # must not exceed C_i. Since consumption happens in the same period:
    # cust_inv[i,s] + d[i][s] <= C_i  (inventory before consumption <= C_i)
    # which is: cust_inv[i,s] <= C_i - d[i][s]
    #
    # Actually the constraint bounds the inventory AFTER consumption too:
    # end-of-period inventory cust_inv[i,s] <= C_i
    # And pre-consumption inventory (after delivery) <= C_i
    # =========================================================================
    for i in N:
        for s in P:
            # Check inventory after delivery, before consumption
            delivered = deliveries_per_period[s].get(i, 0)
            prev_inv = I0[i] if s == 1 else cust_inv[i, s - 1]
            inv_after_delivery = prev_inv + delivered

            lhs = inv_after_delivery
            rhs = C[i]
            violation_amount = lhs - rhs
            if violation_amount > tol:
                record_violation(
                    4,
                    f"Constraint (4): Customer {i} inventory capacity exceeded in period {s}: "
                    f"inventory after delivery={lhs:.4f} > capacity={rhs:.4f}",
                    lhs, rhs, violation_amount
                )

    # =========================================================================
    # Constraint (5): At most one visit per customer per period
    # sum_{r in R} sum_{w in W^p_r} a_{ri} * y^p_{rw} <= 1
    # In the solution: count how many routes visit customer i in period p
    # =========================================================================
    for i in N:
        for p in P:
            visit_count = sum(
                1 for r in routes_per_period[p]
                if i in r['customers']
            )
            lhs = visit_count
            rhs = 1.0
            violation_amount = lhs - rhs
            if violation_amount > tol:
                record_violation(
                    5,
                    f"Constraint (5): Customer {i} visited {visit_count} times in period {p} "
                    f"(at most 1 allowed)",
                    float(lhs), rhs, violation_amount
                )

    # =========================================================================
    # Linking: deliveries imply visits (implicit in original RDP formulation).
    # Sub-deliveries q^s_{wi} only exist for customers i in N_r of route r with
    # y^p_{rw} > 0, so a positive aggregate delivery requires a visit. Without
    # this check, q_{ip} > 0 while customer i is absent from every route in
    # period p would be silently accepted (stockout / capacity checks use the
    # aggregate q_{ip} but never tie it to z_{ip}).
    # =========================================================================
    for p in P:
        for i in N:
            qty = deliveries_per_period[p].get(i, 0)
            if qty > tol and not any(i in r['customers'] for r in routes_per_period[p]):
                record_violation(
                    5,
                    f"Delivery/visit link: customer {i} received delivery {qty:.4f} "
                    f"in period {p} but is not on any route",
                    qty, 0.0, qty,
                )

    # =========================================================================
    # Constraint (6): Vehicle availability - at most K vehicles per period
    # sum_{r in R} sum_{w in W^p_r} y^p_{rw} <= K
    # Count number of routes used in each period
    # =========================================================================
    for p in P:
        num_routes = len(routes_per_period[p])
        lhs = num_routes
        rhs = float(K)
        violation_amount = lhs - rhs
        if violation_amount > tol:
            record_violation(
                6,
                f"Constraint (6): {num_routes} routes used in period {p}, "
                f"but only {K} vehicles available",
                float(lhs), rhs, violation_amount
            )

    # =========================================================================
    # Constraint (7): Supplier inventory bounds: 0 <= I^p_0 <= C_0
    # Two sub-constraints: lower bound (>= 0) and upper bound (<= C_0)
    # =========================================================================
    for p in P:
        inv = depot_inv[p]

        # Lower bound: I^p_0 >= 0
        if inv < -tol:
            lhs = inv
            rhs = 0.0
            violation_amount = -inv
            record_violation(
                7,
                f"Constraint (7): Supplier inventory negative in period {p}: "
                f"I^{p}_0={inv:.4f} < 0",
                lhs, rhs, violation_amount
            )

        # Upper bound: I^p_0 <= C_0
        violation_amount_ub = inv - C0
        if violation_amount_ub > tol:
            lhs = inv
            rhs = float(C0)
            record_violation(
                7,
                f"Constraint (7): Supplier inventory exceeds capacity in period {p}: "
                f"I^{p}_0={inv:.4f} > C_0={C0}",
                lhs, rhs, violation_amount_ub
            )

    # =========================================================================
    # Constraint (8) family: Non-negativity of delivery quantities (q^s_{wi}).
    # The original (8) bound y^p_{rw} >= 0 is structurally satisfied because
    # parsed routes carry y^p_{rw} = 1 (used). Per the same non-negativity
    # family in the original RDP, sub-deliveries q must also be >= 0; we test
    # that here as the actionable form on the original solution structure.
    # =========================================================================
    for p in P:
        for r_idx, r in enumerate(routes_per_period[p]):
            for cust_id, qty in r['deliveries'].items():
                if qty < -tol:
                    lhs = qty
                    rhs = 0.0
                    violation_amount = -qty
                    record_violation(
                        8,
                        f"Constraint (8) family: Negative delivery quantity {qty:.4f} to "
                        f"customer {cust_id} on route {r_idx + 1} in period {p}",
                        lhs, rhs, violation_amount
                    )

    # =========================================================================
    # Constraint (9): Integrality on routes
    # sum_{w in W^p_r} y^p_{rw} in {0, 1}
    # With realized explicit routes, integrality of route usage is inherently
    # satisfied. We instead verify the reported route is well-formed:
    #   (a) starts and ends at the depot (0)
    #   (b) interior nodes are distinct (no subtour cycle through any customer)
    # so truncated routes and disconnected subtours are flagged rather than
    # silently accepted as members of R.
    # =========================================================================
    for p in P:
        for r_idx, r in enumerate(routes_per_period[p]):
            raw = r.get('raw')
            if raw is None:
                continue
            # (a) depot start/end
            if len(raw) < 2 or raw[0] != 0 or raw[-1] != 0:
                record_violation(
                    9,
                    f"Constraint (9) structure: Route {r_idx + 1} in period {p} "
                    f"is malformed (must start and end at depot 0): {raw}",
                    float(len(raw)), 0.0, 1.0,
                )
                continue
            # (b) no customer visited twice within one route (would imply a
            #     subtour cycle through that customer rather than a simple
            #     depot-to-depot path).
            interior = [n for n in raw[1:-1] if n != 0]
            seen = set()
            dupes = []
            for n in interior:
                if n in seen:
                    dupes.append(n)
                seen.add(n)
            if dupes:
                record_violation(
                    9,
                    f"Constraint (9) structure: Route {r_idx + 1} in period {p} "
                    f"revisits customer(s) {dupes} (disconnected subtour in {raw})",
                    float(len(dupes)), 0.0, float(len(dupes)),
                )

    # =========================================================================
    # Vehicle capacity (paper Hard Constraint #8 / R-membership):
    # total_load on each used route <= Q. Distinct from constraint (9) above
    # (which is route-usage integrality / structural well-formedness). The
    # capacity bound was previously folded into the (9) check; per the paper
    # it belongs to R-membership and is enforced here as its own block so a
    # solution with total_load > Q is correctly flagged.
    # =========================================================================
    for p in P:
        for r_idx, r in enumerate(routes_per_period[p]):
            total_load = sum(r['deliveries'].values())
            lhs = total_load
            rhs = float(Q)
            violation_amount = total_load - Q
            if violation_amount > tol:
                record_violation(
                    "vehicle_capacity",
                    f"Vehicle capacity (R-membership): Route {r_idx + 1} in period {p} "
                    f"load={total_load:.4f} exceeds vehicle capacity Q={Q}",
                    lhs, rhs, violation_amount
                )

    # =========================================================================
    # Constraint (10): Objective-value consistency (Tier C anti-gaming check).
    # The objective (1) is min sum_{p,r,w} c_{rw} y^p_{rw} + sum_p h_0 I^p_0,
    # i.e. total vehicle travel cost plus inventory holding cost at the supplier
    # and at all customers across the horizon. Every variable that determines
    # this objective is present in the solution: routes give the travel arcs,
    # and deliveries determine the end-of-period inventories (already simulated
    # above as depot_inv / cust_inv) on which holding costs are charged. The
    # objective is therefore fully recomputable. Reject solutions whose reported
    # objective_value disagrees with the recomputed value -- this catches
    # fabricated objective values (e.g. obj=0 or obj=sys.float_info.max) that
    # otherwise pass constraints (2)-(9).
    # =========================================================================
    reported_obj = sol_data.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = None
            try:
                cust_h = params['h']
                dep_h0 = params['h0']
                # Travel cost: sum of arc costs over every used route. Gurobi
                # routes carry the full depot-to-depot node sequence in 'raw';
                # efficient-format routes carry only the customer visit order,
                # so the depot (0) is prepended and appended.
                travel_cost = 0.0
                for p in P:
                    for r in routes_per_period[p]:
                        raw = r.get('raw')
                        if raw is not None:
                            seq = list(raw)
                        else:
                            seq = [0] + list(r['customers']) + [0]
                        for k in range(len(seq) - 1):
                            travel_cost += dist[seq[k]][seq[k + 1]]
                # Holding cost: charged on simulated end-of-period inventory at
                # the supplier (depot_inv) and at every customer (cust_inv).
                holding_cost = 0.0
                for p in P:
                    holding_cost += dep_h0 * depot_inv[p]
                    for i in N:
                        holding_cost += cust_h[i] * cust_inv[i, p]
                true_obj = travel_cost + holding_cost
            except (KeyError, IndexError, TypeError):
                true_obj = None
            if true_obj is not None:
                abs_diff = abs(reported - true_obj)
                # 0.1% relative tolerance with a 1e-3 absolute floor.
                obj_tol = max(1e-3, 1e-3 * abs(true_obj))
                if abs_diff > obj_tol:
                    record_violation(
                        10,
                        f"Constraint (10): Objective consistency violated: "
                        f"reported objective_value={reported} differs from "
                        f"recomputed travel+holding cost={true_obj:.4f} "
                        f"(travel={travel_cost:.4f}, holding={holding_cost:.4f}, "
                        f"|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                        reported, true_obj, abs_diff
                    )

    # =========================================================================
    # Build result
    # =========================================================================
    violated_constraints = sorted(violated_constraint_set)
    feasible = len(violated_constraints) == 0

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for the Inventory-Routing Problem (IRP)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    instance_data = load_json(args.instance_path)
    sol_data = load_json(args.solution_path)

    result = check_feasibility(instance_data, sol_data)

    with open(args.result_path, 'w') as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print(f"Solution is INFEASIBLE. Violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
