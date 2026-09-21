"""
Feasibility checker for the Split-Delivery Vehicle Routing Problem with
Time Windows (SDVRPTW).

Based on: Desaulniers (2010), "Branch-and-Price-and-Cut for the Split-Delivery
Vehicle Routing Problem with Time Windows", Operations Research 58(1):179-192.

Checks constraints (1)-(15) of the arc-flow formulation.

Tier C addition (obj-recompute defense): constraint (1) -- the objective
function itself -- is recomputed from the solution routes. The objective
minimizes the total travel cost sum_{f} sum_{(i,j)} c_{ij} x^f_{ij}; every
arc-determining variable (the routes) is present in the solution, so the
objective is fully recomputable. The reported objective_value must equal the
recomputed total arc cost (within tolerance), otherwise the solution is
rejected. This defends against candidates that pass the routing constraints
but fabricate a better-looking objective_value.
"""

import argparse
import json
import math


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_instance_info(data):
    """Build graph and precompute data structures from instance JSON."""
    n = data["num_customers"]
    Q = data["vehicle_capacity"]
    depot = data["depot"]
    customers = data["customers"]
    dist_matrix = data["distance_matrix"]

    num_nodes = n + 2  # 0..n+1

    # Service times
    service = [0] * num_nodes
    service[0] = depot["service_time"]
    for c in customers:
        service[c["id"]] = c["service_time"]
    service[n + 1] = 0

    # Time windows
    e = [0] * num_nodes
    l = [0] * num_nodes
    e[0] = depot["time_window"][0]
    l[0] = depot["time_window"][1]
    e[n + 1] = depot["time_window"][0]
    l[n + 1] = depot["time_window"][1]
    for c in customers:
        e[c["id"]] = c["time_window"][0]
        l[c["id"]] = c["time_window"][1]

    # Demands
    demand = [0] * num_nodes
    for c in customers:
        demand[c["id"]] = c["demand"]
    d_bar = [min(demand[i], Q) for i in range(num_nodes)]

    # Distance / travel-time matrices (0..n+1)
    # Node n+1 is the return depot; its distances mirror node 0.
    t_mat = [[0.0] * num_nodes for _ in range(num_nodes)]
    c_mat = [[0.0] * num_nodes for _ in range(num_nodes)]

    for i in range(n + 1):
        for j in range(n + 1):
            c_mat[i][j] = dist_matrix[i][j]
            t_mat[i][j] = dist_matrix[i][j] + service[i]

    for i in range(n + 1):
        c_mat[i][n + 1] = dist_matrix[i][0]
        t_mat[i][n + 1] = dist_matrix[i][0] + service[i]
        c_mat[n + 1][i] = dist_matrix[0][i]
        t_mat[n + 1][i] = dist_matrix[0][i] + service[n + 1]
    c_mat[n + 1][n + 1] = 0.0
    t_mat[n + 1][n + 1] = 0.0

    N = list(range(1, n + 1))
    V = list(range(0, n + 2))
    N_set = set(N)

    # Build arc set A
    arcs = set()
    for i in V:
        for j in V:
            if i == j:
                continue
            if i == n + 1 or j == 0:
                continue
            if e[i] + t_mat[i][j] <= l[j]:
                arcs.add((i, j))

    V_plus = {i: [] for i in V}
    V_minus = {i: [] for i in V}
    for (i, j) in arcs:
        V_plus[i].append(j)
        V_minus[j].append(i)

    kC = {i: math.ceil(demand[i] / Q) for i in N}
    kC_N = math.ceil(sum(demand[i] for i in N) / Q)
    F_size = sum(kC[i] for i in N)

    A_N = {(i, j) for (i, j) in arcs if i in N_set and j in N_set}

    # A*(N)
    A_star_N = set()
    A_star_ij = {}
    processed = set()
    for (i, j) in sorted(A_N):
        if (i, j) in processed:
            continue
        if (j, i) in A_N:
            rep = (min(i, j), max(i, j))
            if rep not in A_star_N:
                A_star_N.add(rep)
                A_star_ij[rep] = [(i, j), (j, i)]
                processed.add((i, j))
                processed.add((j, i))
        else:
            A_star_N.add((i, j))
            A_star_ij[(i, j)] = [(i, j)]
            processed.add((i, j))

    return {
        "n": n, "Q": Q, "num_nodes": num_nodes,
        "N": N, "N_set": N_set, "V": V,
        "e": e, "l": l, "demand": demand, "d_bar": d_bar,
        "t_mat": t_mat, "c_mat": c_mat, "service": service,
        "arcs": arcs, "V_plus": V_plus, "V_minus": V_minus,
        "kC": kC, "kC_N": kC_N, "F_size": F_size,
        "A_N": A_N, "A_star_N": A_star_N, "A_star_ij": A_star_ij,
    }


def extract_vehicles(solution):
    """
    Extract per-vehicle data from solution JSON.
    Returns list of dicts with keys: route, deliveries, num_vehicles.
    Each entry with num_vehicles > 1 is expanded into separate identical vehicles.
    """
    vehicles = []
    for r in solution["routes"]:
        route = r["route"]
        deliveries = {}
        for k, v in r["deliveries"].items():
            deliveries[int(k)] = float(v)
        nv = r.get("num_vehicles", 1)
        for _ in range(nv):
            vehicles.append({"route": list(route), "deliveries": dict(deliveries)})
    return vehicles


def record_violation(violations_list, magnitudes_list, violated_set,
                     constraint_idx, message, lhs, rhs, violation_amount,
                     tol=1e-5, eps=1e-5):
    """Record a violation if violation_amount > tol."""
    if violation_amount > tol:
        violated_set.add(constraint_idx)
        violations_list.append(message)
        normalizer = max(abs(rhs), eps)
        ratio = violation_amount / normalizer
        magnitudes_list.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": ratio,
        })


def check_feasibility(data, solution):
    """Check all hard constraints (1)-(15) of the arc-flow formulation."""
    info = build_instance_info(data)
    vehicles = extract_vehicles(solution)

    n = info["n"]
    Q = info["Q"]
    N = info["N"]
    N_set = info["N_set"]
    e = info["e"]
    l = info["l"]
    demand = info["demand"]
    d_bar = info["d_bar"]
    t_mat = info["t_mat"]
    c_mat = info["c_mat"]
    arcs = info["arcs"]
    V_plus = info["V_plus"]
    kC = info["kC"]
    kC_N = info["kC_N"]
    F_size = info["F_size"]

    tol = 1e-5
    eps = 1e-5

    violations = []
    magnitudes = []
    violated_set = set()

    num_vehicles = len(vehicles)
    depot_end = n + 1  # node n+1

    # =========================================================================
    # Constraint (2): Demand fulfillment
    # sum_f delta^f_i >= d_i for all i in N
    # =========================================================================
    for i in N:
        total_delivered = sum(v["deliveries"].get(i, 0.0) for v in vehicles)
        lhs = total_delivered
        rhs = demand[i]
        va = max(rhs - lhs, 0.0)  # >= constraint: rhs - lhs
        record_violation(violations, magnitudes, violated_set,
                         2, f"Constraint (2): Customer {i} demand not met: "
                         f"delivered={lhs:.4f}, demand={rhs}",
                         lhs, rhs, va, tol, eps)

    # =========================================================================
    # Constraint (3): Minimum visit count per customer
    # sum_f sum_{j in V+(i)} x^f_{ij} >= k^C_i for all i in N
    # =========================================================================
    for i in N:
        visit_count = 0
        for v in vehicles:
            route = v["route"]
            for idx in range(len(route)):
                if route[idx] == i:
                    visit_count += 1
        lhs = visit_count
        rhs = kC[i]
        va = max(rhs - lhs, 0.0)
        record_violation(violations, magnitudes, violated_set,
                         3, f"Constraint (3): Customer {i} visited {lhs} times, "
                         f"minimum required {rhs}",
                         lhs, rhs, va, tol, eps)

    # =========================================================================
    # Constraint (4): Fleet size computation
    # sum_f sum_{j in V+(0)\{n+1}} x^f_{0j} = H
    # H is defined as the number of vehicles used (non-empty routes).
    # In the solution, H = number of vehicles with routes visiting customers.
    # =========================================================================
    # Count vehicles leaving depot to a customer (not directly to n+1)
    departures_to_customers = 0
    for v in vehicles:
        route = v["route"]
        if len(route) >= 2 and route[0] == 0:
            next_node = route[1]
            if next_node != depot_end:
                departures_to_customers += 1

    H = sum(1 for v in vehicles if any(nd in N_set for nd in v["route"]))
    lhs = departures_to_customers
    rhs = H
    va = abs(lhs - rhs)
    record_violation(violations, magnitudes, violated_set,
                     4, f"Constraint (4): Departures from depot to customers "
                     f"({lhs}) != H ({rhs})",
                     lhs, rhs, va, tol, eps)

    # =========================================================================
    # Constraint (5): Fleet size bounds
    # H in [k^C(N), |F|], integer
    # =========================================================================
    if H < kC_N:
        va = kC_N - H
        record_violation(violations, magnitudes, violated_set,
                         5, f"Constraint (5): H={H} < k^C(N)={kC_N}",
                         H, kC_N, va, tol, eps)
    if H > F_size:
        va = H - F_size
        record_violation(violations, magnitudes, violated_set,
                         5, f"Constraint (5): H={H} > |F|={F_size}",
                         H, F_size, va, tol, eps)
    if abs(H - round(H)) > tol:
        va = abs(H - round(H))
        record_violation(violations, magnitudes, violated_set,
                         5, f"Constraint (5): H={H} is not integer",
                         H, round(H), va, tol, eps)

    # =========================================================================
    # Constraint (6): k-path inequalities (SKIPPED - cutting plane)
    # sum_f sum_{(i,j) in A^-(U)} x^f_{ij} >= k_U for all U in P(N)
    # Per math_model.txt: "initially relaxed, added as cuts" and
    # "Constraints (3)-(7) are redundant for the integer formulation but
    # strengthen the linear relaxation."
    # These are valid inequalities added as cutting planes during the
    # branch-and-price-and-cut process, not hard feasibility constraints
    # for the arc-flow formulation solved by Gurobi.
    # =========================================================================

    # Build arc usage from solution (needed for constraint 15)
    arc_usage = {}
    for v in vehicles:
        route = v["route"]
        for idx in range(len(route) - 1):
            arc = (route[idx], route[idx + 1])
            arc_usage[arc] = arc_usage.get(arc, 0) + 1

    # =========================================================================
    # Constraint (7): Arc-flow inequalities (SKIPPED - cutting plane)
    # sum_f sum_{(i,j) in A*_{i'j'}} x^f_{ij} <= 1 for all (i',j') in A*(N)
    # Per math_model.txt: "initially relaxed, added as cuts".
    # These are valid inequalities from Corollary 2, not hard feasibility
    # constraints for the arc-flow formulation.
    # =========================================================================

    # =========================================================================
    # Constraint (8): Path origin - each vehicle leaves depot exactly once
    # sum_{j in V+(0)} x^f_{0,j} = 1 for all f in F
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        route = v["route"]
        if route[0] != 0:
            record_violation(violations, magnitudes, violated_set,
                             8, f"Constraint (8): Vehicle {f_idx} route does "
                             f"not start at depot 0 (starts at {route[0]})",
                             0, 1, 1.0, tol, eps)
        # Count departures from node 0
        depart_count = sum(1 for idx in range(len(route) - 1) if route[idx] == 0)
        lhs_val = depart_count
        rhs_val = 1
        va = abs(lhs_val - rhs_val)
        record_violation(violations, magnitudes, violated_set,
                         8, f"Constraint (8): Vehicle {f_idx} departs from "
                         f"depot {lhs_val} times (must be 1)",
                         lhs_val, rhs_val, va, tol, eps)

    # =========================================================================
    # Constraint (9): Flow conservation per vehicle at each customer node
    # sum_{j in V+(i)} x^f_{ij} - sum_{j in V-(i)} x^f_{ji} = 0
    #   for all f in F, i in N
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        route = v["route"]
        for i in N:
            out_count = 0
            in_count = 0
            for idx in range(len(route) - 1):
                if route[idx] == i:
                    out_count += 1
                if route[idx + 1] == i:
                    in_count += 1
            lhs_val = out_count - in_count
            rhs_val = 0
            va = abs(lhs_val - rhs_val)
            record_violation(violations, magnitudes, violated_set,
                             9, f"Constraint (9): Vehicle {f_idx}, node {i}: "
                             f"out={out_count}, in={in_count}, balance={lhs_val}",
                             lhs_val, rhs_val, va, tol, eps)

    # =========================================================================
    # Constraint (10): Path termination
    # sum_{i in V-(n+1)} x^f_{i,n+1} = 1 for all f in F
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        route = v["route"]
        if route[-1] != depot_end:
            record_violation(violations, magnitudes, violated_set,
                             10, f"Constraint (10): Vehicle {f_idx} route does "
                             f"not end at node {depot_end} (ends at {route[-1]})",
                             0, 1, 1.0, tol, eps)
        arrive_count = sum(1 for idx in range(len(route) - 1)
                           if route[idx + 1] == depot_end)
        lhs_val = arrive_count
        rhs_val = 1
        va = abs(lhs_val - rhs_val)
        record_violation(violations, magnitudes, violated_set,
                         10, f"Constraint (10): Vehicle {f_idx} arrives at "
                         f"depot {lhs_val} times (must be 1)",
                         lhs_val, rhs_val, va, tol, eps)

    # =========================================================================
    # Constraint (11): Time consistency (linearized)
    # For each vehicle f and each arc (i,j) used: s^f_i + t_{ij} <= s^f_j
    # (when x^f_{ij}=1, i.e., on arcs actually traversed)
    #
    # Constraint (12): Time windows
    # e_i <= s^f_i <= l_i for all f, i in V
    #
    # We compute earliest feasible schedule for each vehicle and check both.
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        route = v["route"]
        if len(route) < 2:
            continue

        # Compute earliest start times along the route
        s = [0.0] * len(route)
        s[0] = e[route[0]]  # depot earliest

        for idx in range(1, len(route)):
            i_node = route[idx - 1]
            j_node = route[idx]
            arrival = s[idx - 1] + t_mat[i_node][j_node]
            s[idx] = max(e[j_node], arrival)

        # Check constraint (11): time consistency on used arcs
        for idx in range(1, len(route)):
            i_node = route[idx - 1]
            j_node = route[idx]
            lhs_val = s[idx - 1] + t_mat[i_node][j_node]
            rhs_val = s[idx]
            # Constraint: s_i + t_ij <= s_j  (i.e., lhs <= rhs)
            va = max(lhs_val - rhs_val, 0.0)
            record_violation(violations, magnitudes, violated_set,
                             11, f"Constraint (11): Vehicle {f_idx}, arc "
                             f"({i_node},{j_node}): s_i + t_ij = {lhs_val:.4f} "
                             f"> s_j = {rhs_val:.4f}",
                             lhs_val, rhs_val, va, tol, eps)

        # Check constraint (12): time windows
        for idx in range(len(route)):
            node = route[idx]
            s_val = s[idx]
            # e_node <= s_val
            va_lower = max(e[node] - s_val, 0.0)
            record_violation(violations, magnitudes, violated_set,
                             12, f"Constraint (12): Vehicle {f_idx}, node {node}: "
                             f"start time {s_val:.4f} < earliest {e[node]}",
                             s_val, e[node], va_lower, tol, eps)
            # s_val <= l_node
            va_upper = max(s_val - l[node], 0.0)
            record_violation(violations, magnitudes, violated_set,
                             12, f"Constraint (12): Vehicle {f_idx}, node {node}: "
                             f"start time {s_val:.4f} > latest {l[node]}",
                             s_val, l[node], va_upper, tol, eps)

    # =========================================================================
    # Constraint (13): Vehicle capacity
    # sum_{i in N} delta^f_i <= Q for all f in F
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        total_load = sum(v["deliveries"].values())
        lhs_val = total_load
        rhs_val = Q
        va = max(lhs_val - rhs_val, 0.0)
        record_violation(violations, magnitudes, violated_set,
                         13, f"Constraint (13): Vehicle {f_idx} load "
                         f"{lhs_val:.4f} > capacity {rhs_val}",
                         lhs_val, rhs_val, va, tol, eps)

    # =========================================================================
    # Constraint (14): Delivery quantity bounds and linking
    # 0 <= delta^f_i <= d_bar_i * sum_{j in V+(i)} x^f_{ij}
    #   for all f in F, i in N
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        route = v["route"]
        visited_in_route = set(route)
        for i in N:
            delta_fi = v["deliveries"].get(i, 0.0)
            # Check non-negativity
            if delta_fi < -tol:
                va = abs(delta_fi)
                record_violation(violations, magnitudes, violated_set,
                                 14, f"Constraint (14): Vehicle {f_idx}, "
                                 f"customer {i}: delivery {delta_fi:.4f} < 0",
                                 delta_fi, 0.0, va, tol, eps)
            # Check upper bound: d_bar_i * (1 if visited, 0 if not)
            visits_i = sum(1 for idx in range(len(route) - 1) if route[idx] == i)
            upper = d_bar[i] * visits_i
            va = max(delta_fi - upper, 0.0)
            record_violation(violations, magnitudes, violated_set,
                             14, f"Constraint (14): Vehicle {f_idx}, "
                             f"customer {i}: delivery {delta_fi:.4f} > "
                             f"d_bar*visits = {upper:.4f} "
                             f"(d_bar={d_bar[i]}, visits={visits_i})",
                             delta_fi, upper, va, tol, eps)

    # =========================================================================
    # Constraint (15): Binary requirement
    # x^f_{ij} in {0, 1} for all f in F, (i,j) in A
    # In route representation, each arc is used 0 or 1 times per vehicle.
    # Check no arc is traversed more than once by the same vehicle, and
    # that all used arcs are in A.
    # =========================================================================
    for f_idx, v in enumerate(vehicles):
        route = v["route"]
        arc_count = {}
        for idx in range(len(route) - 1):
            arc = (route[idx], route[idx + 1])
            arc_count[arc] = arc_count.get(arc, 0) + 1
        for arc, cnt in arc_count.items():
            if cnt > 1:
                va = cnt - 1
                record_violation(violations, magnitudes, violated_set,
                                 15, f"Constraint (15): Vehicle {f_idx} "
                                 f"uses arc {arc} {cnt} times (must be 0 or 1)",
                                 cnt, 1, va, tol, eps)
            if arc not in arcs:
                # Arc not in feasible arc set
                record_violation(violations, magnitudes, violated_set,
                                 15, f"Constraint (15): Vehicle {f_idx} "
                                 f"uses arc {arc} which is not in A",
                                 1, 0, 1.0, tol, eps)

    # =========================================================================
    # Constraint (1): Objective consistency  [Tier C obj-recompute defense]
    # minimize sum_{f in F} sum_{(i,j) in A} c_{ij} x^f_{ij}
    #   = total travel cost over all arcs traversed by all vehicles.
    # The routes are the only obj-determining variables and are fully present
    # in the solution, so the objective is exactly recomputable. Reject when
    # the reported objective_value disagrees with the recomputed total cost.
    # Tolerance: 0.1% relative with a 1e-3 absolute floor.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            num_nodes = info["num_nodes"]
            true_obj = 0.0
            for v in vehicles:
                route = v["route"]
                for idx in range(len(route) - 1):
                    i_node = route[idx]
                    j_node = route[idx + 1]
                    if (0 <= i_node < num_nodes) and (0 <= j_node < num_nodes):
                        true_obj += c_mat[i_node][j_node]
            # 0.1% relative tolerance with 1e-3 absolute floor.
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            abs_diff = abs(reported - true_obj)
            # A non-finite reported objective (inf / nan) can never match the
            # finite recomputed travel cost; force the violation (note nan
            # comparisons are False, so abs_diff must be set explicitly).
            if not math.isfinite(reported):
                abs_diff = float("inf")
            record_violation(violations, magnitudes, violated_set,
                             1, f"Constraint (1): Objective consistency violated: "
                             f"reported objective_value={reported} differs from "
                             f"recomputed total arc cost "
                             f"sum_f sum_(i,j) c_ij*x^f_ij={true_obj:.6f} "
                             f"(|diff|={abs_diff:.6g}, tol={obj_tol:.6g})",
                             reported, true_obj, abs_diff, obj_tol, eps)

    # =========================================================================
    # Build result
    # =========================================================================
    violated_constraints = sorted(violated_set)
    feasible = len(violated_constraints) == 0

    # Aggregate violations by constraint index for cleaner messages
    constraint_messages = {}
    for msg in violations:
        # Extract constraint index from message
        cidx = None
        for vc in violated_constraints:
            if msg.startswith(f"Constraint ({vc})"):
                cidx = vc
                break
        if cidx is not None:
            if cidx not in constraint_messages:
                constraint_messages[cidx] = []
            constraint_messages[cidx].append(msg)

    # Create summary violations list (one per violated constraint)
    summary_violations = []
    for cidx in violated_constraints:
        msgs = constraint_messages.get(cidx, [])
        if len(msgs) <= 3:
            summary_violations.extend(msgs)
        else:
            summary_violations.extend(msgs[:2])
            summary_violations.append(
                f"Constraint ({cidx}): ... and {len(msgs) - 2} more violations"
            )

    result = {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": summary_violations,
        "violation_magnitudes": magnitudes,
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for SDVRPTW (Desaulniers 2010)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    data = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(data, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE - no constraints violated")
    else:
        print(f"INFEASIBLE - violated constraints: {result['violated_constraints']}")
        for v in result["violations"]:
            print(f"  {v}")


if __name__ == "__main__":
    main()
