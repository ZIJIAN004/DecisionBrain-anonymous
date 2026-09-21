#!/usr/bin/env python3
"""
Feasibility checker for the Pickup and Delivery Problem with Time Windows (PDPTW).

Verifies a candidate solution against the three-index formulation from:
    Ropke & Cordeau (2009), "Branch and Cut and Price for the Pickup and
    Delivery Problem with Time Windows", Transportation Science 43(3):267-286.

Constraints checked (numbered as in the paper formulation):
    (2)  Each request served exactly once
    (3)  Pickup and delivery by same vehicle
    (4)  Each vehicle leaves origin depot
    (5)  Flow conservation at P ∪ D nodes
    (6)  Each vehicle returns to destination depot
    (7)  Time consistency
    (8)  Load consistency
    (9)  Precedence (pickup before delivery)
    (10) Time windows
    (11) Capacity bounds
    (12) Integrality of x
    (13) Fleet size: len(routes) <= num_vehicles
    (14) Objective consistency: reported objective_value equals the
         recomputed routing cost Σ_k Σ_{i,j} c_{ij} x^k_{ij}.
"""

import argparse
import json
import math


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_instance_data(instance):
    """Extract all needed data from the instance JSON."""
    n = instance["num_requests"]
    Q = instance["vehicle_capacity"]
    num_vehicles = instance["num_vehicles"]

    depot_origin = 0
    depot_dest = 2 * n + 1
    P = list(range(1, n + 1))
    D = list(range(n + 1, 2 * n + 1))
    N = list(range(2 * n + 2))

    # Node data indexed by node id
    nodes = {}
    for nd in instance["nodes"]:
        nodes[nd["id"]] = nd

    x_coord = {nid: nd["x"] for nid, nd in nodes.items()}
    y_coord = {nid: nd["y"] for nid, nd in nodes.items()}
    demand = {nid: nd["demand"] for nid, nd in nodes.items()}
    service_time = {nid: nd.get("service_time", 0) for nid, nd in nodes.items()}
    tw_a = {nid: nd["time_window"][0] for nid, nd in nodes.items()}
    tw_b = {nid: nd["time_window"][1] for nid, nd in nodes.items()}

    # Cost and travel time matrices
    # Per paper: t_{ij} includes service time d_i at node i
    cost = {}
    travel_time = {}
    for i in N:
        for j in N:
            if i == j:
                continue
            dist = euclidean_distance(x_coord[i], y_coord[i], x_coord[j], y_coord[j])
            cost[i, j] = dist
            travel_time[i, j] = dist + service_time[i]

    return {
        "n": n,
        "Q": Q,
        "num_vehicles": num_vehicles,
        "depot_origin": depot_origin,
        "depot_dest": depot_dest,
        "P": P,
        "D": D,
        "N": N,
        "demand": demand,
        "service_time": service_time,
        "tw_a": tw_a,
        "tw_b": tw_b,
        "cost": cost,
        "travel_time": travel_time,
    }


def extract_solution_routes(solution, n):
    """
    Extract routes from the solution JSON. Handles two formats:
      - "routes" as list of lists of node ids (efficient_algorithm)
      - "routes" as list of dicts with "nodes" key (gurobi)
    Returns list of lists of node ids.
    """
    raw_routes = solution.get("routes", [])
    routes = []
    for r in raw_routes:
        if isinstance(r, list):
            routes.append(r)
        elif isinstance(r, dict):
            routes.append(r["nodes"])
        else:
            routes.append(list(r))
    return routes


def reconstruct_variables(routes, d):
    """
    From the list of routes (each a node sequence), reconstruct:
      - x[k][(i,j)]: binary arc usage per vehicle k
      - B[k][i]: service start time at node i for vehicle k
      - Qload[k][i]: load of vehicle k upon leaving node i

    Returns (x_vars, B_vars, Qload_vars).
    """
    n = d["n"]
    x_vars = {}      # k -> dict of (i,j) -> 1
    B_vars = {}      # k -> dict of node -> time
    Qload_vars = {}  # k -> dict of node -> load

    for k, route in enumerate(routes):
        x_k = {}
        B_k = {}
        Q_k = {}

        # Build arcs
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            x_k[(i, j)] = 1

        # Simulate timing and load along route
        current_time = d["tw_a"][d["depot_origin"]]
        current_load = 0

        for idx, node in enumerate(route):
            if idx > 0:
                prev = route[idx - 1]
                tt = d["travel_time"].get((prev, node), 0.0)
                arrival = current_time + tt
                current_time = max(arrival, d["tw_a"][node])
            else:
                current_time = max(d["tw_a"][node], current_time)

            B_k[node] = current_time
            current_load += d["demand"][node]
            Q_k[node] = current_load

        x_vars[k] = x_k
        B_vars[k] = B_k
        Qload_vars[k] = Q_k

    return x_vars, B_vars, Qload_vars


def check_feasibility(instance, solution):
    """
    Check all hard constraints of the PDPTW three-index formulation.

    Returns dict with feasibility result.
    """
    tol = 1e-5
    eps = 1e-5

    d = build_instance_data(instance)
    n = d["n"]
    Q = d["Q"]
    num_vehicles_K = d["num_vehicles"]
    depot_origin = d["depot_origin"]
    depot_dest = d["depot_dest"]
    P = d["P"]
    D = d["D"]
    N = d["N"]

    routes = extract_solution_routes(solution, n)
    K_used = len(routes)

    # Pad to num_vehicles_K with empty depot-to-depot routes if fewer routes
    all_routes = list(routes)
    while len(all_routes) < num_vehicles_K:
        all_routes.append([depot_origin, depot_dest])
    K = list(range(len(all_routes)))

    x_vars, B_vars, Qload_vars = reconstruct_variables(all_routes, d)

    violations = []          # list of message strings
    violated_set = set()     # set of constraint indices
    violation_magnitudes = []  # list of dicts

    def record_violation(constraint_idx, message, lhs, rhs, operator):
        """Record a constraint violation with normalized magnitude."""
        if operator == "eq":
            violation_amount = abs(lhs - rhs)
        elif operator in ("leq", "lt"):
            violation_amount = max(0.0, lhs - rhs)
        elif operator in ("geq", "gt"):
            violation_amount = max(0.0, rhs - lhs)
        else:
            violation_amount = 0.0

        if violation_amount > tol:
            violated_set.add(constraint_idx)
            violations.append(message)
            normalizer = max(abs(rhs), eps)
            ratio = violation_amount / normalizer
            violation_magnitudes.append({
                "constraint": constraint_idx,
                "lhs": float(lhs),
                "rhs": float(rhs),
                "raw_excess": float(violation_amount),
                "normalizer": float(normalizer),
                "ratio": float(ratio),
            })

    # =========================================================================
    # Fleet size bound: len(routes) <= |K| = num_vehicles
    # (Implicit in the formulation: x^k_{ij} is indexed by k in K;
    #  the solution cannot use more vehicles than the fleet provides.)
    # =========================================================================
    record_violation(
        13,
        f"Fleet size: len(routes)={K_used} exceeds num_vehicles={num_vehicles_K}",
        float(K_used), float(num_vehicles_K), "leq"
    )

    # =========================================================================
    # Constraint (2): Each request served exactly once
    # Σ_k Σ_j x^k_{ij} = 1  ∀ i ∈ P
    # =========================================================================
    for i in P:
        total = 0
        for k in K:
            for (a, b), val in x_vars[k].items():
                if a == i:
                    total += val
        record_violation(
            2,
            f"Constraint (2): Request {i} pickup visited {total} time(s) (expected 1)",
            float(total), 1.0, "eq"
        )

    # =========================================================================
    # Constraint (3): Pickup and delivery by same vehicle
    # Σ_j x^k_{ij} - Σ_j x^k_{n+i,j} = 0  ∀ i ∈ P, k ∈ K
    # =========================================================================
    for i in P:
        for k in K:
            out_pickup = sum(
                v for (a, b), v in x_vars[k].items() if a == i
            )
            out_delivery = sum(
                v for (a, b), v in x_vars[k].items() if a == (n + i)
            )
            diff = out_pickup - out_delivery
            record_violation(
                3,
                f"Constraint (3): Vehicle {k}, request {i}: "
                f"pickup outflow={out_pickup}, delivery outflow={out_delivery} "
                f"(difference={diff}, expected 0)",
                float(diff), 0.0, "eq"
            )

    # =========================================================================
    # Constraint (4): Each vehicle leaves origin depot exactly once
    # Σ_j x^k_{0,j} = 1  ∀ k ∈ K
    # =========================================================================
    for k in K:
        out_depot = sum(
            v for (a, b), v in x_vars[k].items() if a == depot_origin
        )
        record_violation(
            4,
            f"Constraint (4): Vehicle {k} leaves depot {out_depot} time(s) "
            f"(expected 1)",
            float(out_depot), 1.0, "eq"
        )

    # =========================================================================
    # Constraint (5): Flow conservation at pickup and delivery nodes
    # Σ_j x^k_{ji} - Σ_j x^k_{ij} = 0  ∀ i ∈ P ∪ D, k ∈ K
    # =========================================================================
    for i in (P + D):
        for k in K:
            inflow = sum(
                v for (a, b), v in x_vars[k].items() if b == i
            )
            outflow = sum(
                v for (a, b), v in x_vars[k].items() if a == i
            )
            diff = inflow - outflow
            record_violation(
                5,
                f"Constraint (5): Vehicle {k}, node {i}: "
                f"inflow={inflow}, outflow={outflow} "
                f"(difference={diff}, expected 0)",
                float(diff), 0.0, "eq"
            )

    # =========================================================================
    # Constraint (6): Each vehicle returns to destination depot exactly once
    # Σ_i x^k_{i,2n+1} = 1  ∀ k ∈ K
    # =========================================================================
    for k in K:
        in_depot = sum(
            v for (a, b), v in x_vars[k].items() if b == depot_dest
        )
        record_violation(
            6,
            f"Constraint (6): Vehicle {k} enters destination depot {in_depot} "
            f"time(s) (expected 1)",
            float(in_depot), 1.0, "eq"
        )

    # =========================================================================
    # Constraint (7): Time consistency (linearized)
    # B^k_j ≥ B^k_i + t_{ij} - M(1 - x^k_{ij})  ∀ i,j ∈ N, k ∈ K
    # When x^k_{ij}=1: B^k_j ≥ B^k_i + t_{ij}
    # =========================================================================
    for k in K:
        for (i, j), val in x_vars[k].items():
            if val < 0.5:
                continue
            if i not in B_vars[k] or j not in B_vars[k]:
                continue
            tt = d["travel_time"].get((i, j), 0.0)
            lhs = B_vars[k][j]
            rhs_val = B_vars[k][i] + tt
            record_violation(
                7,
                f"Constraint (7): Vehicle {k}, arc ({i},{j}): "
                f"B[{j}]={lhs:.4f} < B[{i}]+t_{{{i},{j}}}={rhs_val:.4f}",
                rhs_val, lhs, "leq"
            )

    # =========================================================================
    # Constraint (8): Load consistency (linearized)
    # Q^k_j ≥ Q^k_i + q_j - M(1 - x^k_{ij})  ∀ i,j ∈ N, k ∈ K
    # When x^k_{ij}=1: Q^k_j ≥ Q^k_i + q_j
    # =========================================================================
    for k in K:
        for (i, j), val in x_vars[k].items():
            if val < 0.5:
                continue
            if i not in Qload_vars[k] or j not in Qload_vars[k]:
                continue
            q_j = d["demand"][j]
            lhs = Qload_vars[k][j]
            rhs_val = Qload_vars[k][i] + q_j
            record_violation(
                8,
                f"Constraint (8): Vehicle {k}, arc ({i},{j}): "
                f"Q[{j}]={lhs:.4f} < Q[{i}]+q_{j}={rhs_val:.4f}",
                rhs_val, lhs, "leq"
            )

    # =========================================================================
    # Constraint (9): Precedence – pickup before delivery
    # B^k_i + t_{i,n+i} ≤ B^k_{n+i}  ∀ i ∈ P, k ∈ K
    # =========================================================================
    for i in P:
        delivery = n + i
        for k in K:
            if i not in B_vars[k] or delivery not in B_vars[k]:
                continue
            tt = d["travel_time"].get((i, delivery), 0.0)
            lhs = B_vars[k][i] + tt
            rhs_val = B_vars[k][delivery]
            record_violation(
                9,
                f"Constraint (9): Vehicle {k}, request {i}: "
                f"B[{i}]+t_{{{i},{delivery}}}={lhs:.4f} > B[{delivery}]={rhs_val:.4f}",
                lhs, rhs_val, "leq"
            )

    # =========================================================================
    # Constraint (10): Time windows
    # a_i ≤ B^k_i ≤ b_i  ∀ i ∈ N, k ∈ K
    # =========================================================================
    for k in K:
        for node, B_val in B_vars[k].items():
            a_i = d["tw_a"][node]
            b_i = d["tw_b"][node]
            # Check a_i ≤ B^k_i
            record_violation(
                10,
                f"Constraint (10): Vehicle {k}, node {node}: "
                f"B={B_val:.4f} < a={a_i:.4f} (early arrival)",
                a_i, B_val, "leq"
            )
            # Check B^k_i ≤ b_i
            record_violation(
                10,
                f"Constraint (10): Vehicle {k}, node {node}: "
                f"B={B_val:.4f} > b={b_i:.4f} (late arrival)",
                B_val, b_i, "leq"
            )

    # =========================================================================
    # Constraint (11): Capacity bounds
    # max{0, q_i} ≤ Q^k_i ≤ min{Q, Q+q_i}  ∀ i ∈ N, k ∈ K
    # =========================================================================
    for k in K:
        for node, Q_val in Qload_vars[k].items():
            q_i = d["demand"][node]
            lb = max(0, q_i)
            ub = min(Q, Q + q_i)
            # Check lb ≤ Q^k_i
            record_violation(
                11,
                f"Constraint (11): Vehicle {k}, node {node}: "
                f"Q={Q_val:.4f} < lower bound {lb:.4f}",
                float(lb), float(Q_val), "leq"
            )
            # Check Q^k_i ≤ ub
            record_violation(
                11,
                f"Constraint (11): Vehicle {k}, node {node}: "
                f"Q={Q_val:.4f} > upper bound {ub:.4f}",
                float(Q_val), float(ub), "leq"
            )

    # =========================================================================
    # Constraint (12): Integrality of x
    # x^k_{ij} ∈ {0, 1}  ∀ i,j ∈ N, k ∈ K
    # (By construction from routes, x values are 0 or 1, but check anyway)
    # =========================================================================
    for k in K:
        for (i, j), val in x_vars[k].items():
            frac = abs(val - round(val))
            record_violation(
                12,
                f"Constraint (12): Vehicle {k}, arc ({i},{j}): "
                f"x={val} not binary (fractional part={frac:.6f})",
                float(frac), 0.0, "leq"
            )

    # =========================================================================
    # Constraint (14): Objective consistency
    # The reported objective_value must match the recomputed routing cost
    #     Σ_k Σ_{i,j ∈ route_k} c_{ij}
    # where c_{ij} is the Euclidean distance between nodes i and j (per the
    # paper's three-index objective (1) and the solution_schema description
    # "Total routing distance traveled by all vehicles").
    # Sum only over user-provided routes; padded depot→depot routes contribute
    # zero cost (depot_origin and depot_dest share coordinates) but are an
    # artifact of constraint checking, not part of the reported solution.
    # =========================================================================
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            for route in routes:
                for idx in range(len(route) - 1):
                    i, j = route[idx], route[idx + 1]
                    true_obj += d["cost"].get((i, j), 0.0)
            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_set.add(14)
                violations.append(
                    f"Constraint (14): Objective consistency violated: "
                    f"reported objective_value={reported} differs from "
                    f"recomputed Σ_k Σ_{{i,j}} c_{{ij}} x^k_{{ij}}={true_obj} "
                    f"(|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 14,
                    "lhs": float(reported),
                    "rhs": float(true_obj),
                    "raw_excess": float(abs_diff),
                    "normalizer": float(normalizer),
                    "ratio": float(abs_diff / normalizer),
                })

    # =========================================================================
    # Compile results
    # =========================================================================
    violated_constraints = sorted(violated_set)
    feasible = len(violated_constraints) == 0

    # Deduplicate violation messages per constraint index
    seen_constraints_msg = {}
    deduped_violations = []
    for msg in violations:
        # Use the message directly; already unique per check
        deduped_violations.append(msg)

    return {
        "feasible": feasible,
        "violated_constraints": violated_constraints,
        "violations": deduped_violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for PDPTW (Ropke & Cordeau 2009)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON solution file")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON feasibility result")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)
    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    status = "FEASIBLE" if result["feasible"] else "INFEASIBLE"
    n_violated = len(result["violated_constraints"])
    n_magnitudes = len(result["violation_magnitudes"])
    print(f"[{status}] violated_constraints={result['violated_constraints']} "
          f"({n_magnitudes} violation instances)")


if __name__ == "__main__":
    main()
