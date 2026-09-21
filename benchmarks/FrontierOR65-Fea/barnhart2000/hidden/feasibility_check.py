"""
Feasibility Checker for the Origin-Destination Integer Multicommodity Flow (ODIMCF) Problem.

Checks candidate solutions against the mathematical formulation from:
Barnhart, Hane, and Vance (2000), Operations Research 48(2), 318-326.

Hard constraints checked (Formulation 1 — Node-Arc):
  Constraint (2): Capacity constraints on arcs
  Constraint (3): Flow conservation at each node for each commodity
  Constraint (4): Binary (integrality) domain for arc-flow variables
  Constraint (5): Objective consistency — reported objective_value must equal the
                  value recomputed from the routing decisions present in the
                  solution: sum over commodities of either (artificial_arc_cost *
                  demand) when rejected, else sum_{ij in path_arcs}(cost[ij] * demand).
"""

import argparse
import json
from collections import defaultdict


def check_feasibility(instance_data, solution_data):
    """
    Check all hard constraints of the ODIMCF formulation.

    Returns a dict with feasibility results.
    """
    tol = 1e-5
    eps = 1e-5

    violations = []
    violation_magnitudes = []
    violated_constraints_set = set()

    # ---- Parse instance ----
    nodes = instance_data["network"]["nodes"]
    num_nodes = instance_data["network"]["num_nodes"]
    arcs = instance_data["network"]["arcs"]
    num_arcs = instance_data["network"]["num_arcs"]
    commodities = instance_data["commodities"]["commodity_list"]
    num_commodities = instance_data["commodities"]["num_commodities"]

    # Build arc lookup: arc_id -> arc info
    arc_by_id = {}
    for arc in arcs:
        arc_by_id[arc["arc_id"]] = arc

    # Build adjacency info for arc validation
    arc_set = set()
    for arc in arcs:
        arc_set.add((arc["from_node"], arc["to_node"], arc["arc_id"]))

    # ---- Parse solution ----
    sol_commodities = solution_data.get("commodities", [])

    # Handle infeasible/empty solutions
    if not sol_commodities:
        # No commodities in solution — check if this is a "no solution" marker
        status = solution_data.get("status", "")
        if status == "no_feasible_solution":
            return {
                "feasible": False,
                "violated_constraints": [],
                "violations": ["Solution status is 'no_feasible_solution': no candidate to check."],
                "violation_magnitudes": []
            }
        # Empty commodities but not explicitly marked — treat as missing all assignments
        # This violates constraint (3) for every commodity
        for comm in commodities:
            k = comm["commodity_id"]
            origin = comm["origin"]
            violated_constraints_set.add(3)
            violations.append(
                f"Commodity {k}: no assignment found in solution (flow conservation violated at origin {origin})"
            )
            rhs_val = 1.0
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": 0.0,
                "rhs": rhs_val,
                "raw_excess": abs(rhs_val),
                "normalizer": normalizer,
                "ratio": abs(rhs_val) / normalizer
            })
        return {
            "feasible": False,
            "violated_constraints": sorted(violated_constraints_set),
            "violations": violations,
            "violation_magnitudes": violation_magnitudes
        }

    # Build commodity lookup from solution
    sol_comm_by_id = {}
    for sc in sol_commodities:
        sol_comm_by_id[sc["commodity_id"]] = sc

    # ---- Reconstruct arc-flow variables x^k_{ij} ----
    # x^k_{ij} = 1 if commodity k uses arc ij (not rejected), 0 otherwise
    # For the path-based solution, commodity k uses the arcs in its path_arcs list
    x = defaultdict(lambda: defaultdict(float))  # x[k][arc_id] = 0 or 1

    for comm in commodities:
        k = comm["commodity_id"]
        sc = sol_comm_by_id.get(k)
        if sc is None:
            continue
        if sc.get("rejected", False):
            # Commodity rejected (uses artificial arc) — no real arcs used
            continue
        for arc_entry in sc.get("path_arcs", []):
            aid = arc_entry["arc_id"]
            x[k][aid] = 1.0

    # ====================================================================
    # Constraint (4): Binary variable domain
    #   x^k_{ij} in {0, 1} for all ij in A, for all k in K
    #
    # In the path-based solution encoding, each commodity is either rejected
    # or assigned exactly one path with binary arc usage. We check:
    #   (a) Each x^k_{ij} value is 0 or 1
    #   (b) Each commodity has exactly one assignment (one path or rejected)
    #   (c) Arc IDs referenced in the solution are valid arcs in the instance
    # ====================================================================
    for comm in commodities:
        k = comm["commodity_id"]
        sc = sol_comm_by_id.get(k)
        if sc is None:
            # Commodity missing from solution entirely — this is an assignment issue
            # Treated under constraint (3) flow conservation
            continue

        rejected = sc.get("rejected", False)
        path_arcs = sc.get("path_arcs", [])

        # Check: commodity must be either rejected or have a non-empty path
        if not rejected and len(path_arcs) == 0:
            violated_constraints_set.add(4)
            violations.append(
                f"Commodity {k}: neither rejected nor assigned a path "
                f"(binary/assignment constraint violated)"
            )
            # violation_amount: should be 1 path assigned, have 0
            rhs_val = 1.0
            violation_magnitudes.append({
                "constraint": 4,
                "lhs": 0.0,
                "rhs": rhs_val,
                "raw_excess": 1.0,
                "normalizer": max(abs(rhs_val), eps),
                "ratio": 1.0 / max(abs(rhs_val), eps)
            })

        # Check arc validity: every arc referenced must exist in the instance
        for arc_entry in path_arcs:
            aid = arc_entry["arc_id"]
            if aid not in arc_by_id:
                violated_constraints_set.add(4)
                violations.append(
                    f"Commodity {k}: references non-existent arc_id {aid}"
                )
                rhs_val = 1.0
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": 0.0,
                    "rhs": rhs_val,
                    "raw_excess": 1.0,
                    "normalizer": max(abs(rhs_val), eps),
                    "ratio": 1.0 / max(abs(rhs_val), eps)
                })

    # ====================================================================
    # Constraint (2): Capacity constraints
    #   sum_{k in K} q^k * x^k_{ij} <= d_{ij}, for all ij in A
    # ====================================================================
    for arc in arcs:
        aid = arc["arc_id"]
        capacity = arc["capacity"]
        from_node = arc["from_node"]
        to_node = arc["to_node"]

        # Compute LHS: sum of demands of commodities using this arc
        lhs = 0.0
        for comm in commodities:
            k = comm["commodity_id"]
            demand = comm["demand"]
            lhs += demand * x[k][aid]

        rhs_val = float(capacity)
        violation_amount = lhs - rhs_val  # For <= constraint

        if violation_amount > tol:
            violated_constraints_set.add(2)
            violations.append(
                f"Arc {aid} ({from_node}->{to_node}): capacity exceeded, "
                f"flow={lhs:.4f} > capacity={rhs_val:.4f}"
            )
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 2,
                "lhs": lhs,
                "rhs": rhs_val,
                "raw_excess": violation_amount,
                "normalizer": normalizer,
                "ratio": violation_amount / normalizer
            })

    # ====================================================================
    # Constraint (3): Flow conservation
    #   sum_{ij in A} x^k_{ij} - sum_{ji in A} x^k_{ji} = b^k_i,
    #       for all i in N, for all k in K
    #   where b^k_i = 1 if i is origin of k, -1 if destination, 0 otherwise
    #
    # For rejected commodities (using the artificial arc from origin to
    # destination), the flow on real arcs is 0. The artificial arc provides
    # the unit flow, so net flow at origin = 1 (via artificial), at
    # destination = -1. Since artificial arcs are not in the network arc set,
    # for rejected commodities: all b^k_i checks with real arcs yield
    # net flow = 0 at every node. This is consistent because the artificial
    # arc is outside the real network — the model accounts for it separately.
    #
    # We check flow conservation on real arcs only for non-rejected
    # commodities. For rejected commodities, real-arc flow must be 0 at all
    # nodes (which is trivially satisfied since they have no path_arcs).
    # ====================================================================
    # Build outgoing and incoming arc lookups
    outgoing_arcs = defaultdict(list)  # node -> [arc_id, ...]
    incoming_arcs = defaultdict(list)  # node -> [arc_id, ...]
    for arc in arcs:
        outgoing_arcs[arc["from_node"]].append(arc["arc_id"])
        incoming_arcs[arc["to_node"]].append(arc["arc_id"])

    for comm in commodities:
        k = comm["commodity_id"]
        origin = comm["origin"]
        destination = comm["destination"]

        sc = sol_comm_by_id.get(k)
        if sc is None:
            # Commodity missing — violated at origin node
            violated_constraints_set.add(3)
            violations.append(
                f"Commodity {k}: missing from solution "
                f"(flow conservation violated)"
            )
            rhs_val = 1.0
            normalizer = max(abs(rhs_val), eps)
            violation_magnitudes.append({
                "constraint": 3,
                "lhs": 0.0,
                "rhs": rhs_val,
                "raw_excess": 1.0,
                "normalizer": normalizer,
                "ratio": 1.0 / normalizer
            })
            continue

        rejected = sc.get("rejected", False)

        if rejected:
            # For rejected commodity, all real arc flows should be 0.
            # This is trivially satisfied if path_arcs is empty.
            # But check anyway in case solution has path_arcs AND rejected=true
            if sc.get("path_arcs", []):
                violated_constraints_set.add(3)
                violations.append(
                    f"Commodity {k}: marked as rejected but has path_arcs "
                    f"(flow conservation inconsistency)"
                )
                rhs_val = 0.0
                flow_sum = float(len(sc["path_arcs"]))
                violation_amount = abs(flow_sum)
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": flow_sum,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })
            continue

        # Non-rejected commodity: check flow conservation at every node
        for node in nodes:
            # b^k_i
            if node == origin:
                b_ki = 1.0
            elif node == destination:
                b_ki = -1.0
            else:
                b_ki = 0.0

            # sum of x^k_{ij} for arcs leaving node i
            out_flow = 0.0
            for aid in outgoing_arcs[node]:
                out_flow += x[k][aid]

            # sum of x^k_{ji} for arcs entering node i
            in_flow = 0.0
            for aid in incoming_arcs[node]:
                in_flow += x[k][aid]

            lhs = out_flow - in_flow
            rhs_val = b_ki
            violation_amount = abs(lhs - rhs_val)

            if violation_amount > tol:
                violated_constraints_set.add(3)
                violations.append(
                    f"Commodity {k} at node {node}: flow conservation violated, "
                    f"net_flow={lhs:.4f}, expected={rhs_val:.4f}"
                )
                normalizer = max(abs(rhs_val), eps)
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs_val,
                    "raw_excess": violation_amount,
                    "normalizer": normalizer,
                    "ratio": violation_amount / normalizer
                })

    # ====================================================================
    # Constraint (5): Objective consistency (Tier C defense vs score-gaming)
    #   reported objective_value must equal the value recomputed directly
    #   from the routing variables in the solution:
    #     true_obj = sum_{rejected k} (artificial_arc_cost[k] * demand[k])
    #              + sum_{non-rejected k} sum_{ij in path_arcs[k]} (cost[ij] * demand[k])
    #   All obj-determining variables (rejected flag + path_arcs) are in the
    #   solution, so a full recompute is exact.
    # ====================================================================
    reported_obj_raw = solution_data.get("objective_value")
    if reported_obj_raw is not None:
        try:
            reported = float(reported_obj_raw)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = 0.0
            comm_by_id = {c["commodity_id"]: c for c in commodities}
            for comm in commodities:
                k = comm["commodity_id"]
                demand = comm["demand"]
                sc = sol_comm_by_id.get(k)
                if sc is None:
                    continue
                if sc.get("rejected", False):
                    true_obj += float(comm["artificial_arc_cost"]) * float(demand)
                else:
                    for arc_entry in sc.get("path_arcs", []):
                        aid = arc_entry["arc_id"]
                        arc = arc_by_id.get(aid)
                        if arc is None:
                            # already flagged under constraint (4); skip
                            continue
                        true_obj += float(arc["cost"]) * float(demand)

            abs_diff = abs(reported - true_obj)
            # 0.1% relative tolerance with 1e-3 absolute floor
            obj_tol = max(1e-3, 1e-3 * abs(true_obj))
            if abs_diff > obj_tol:
                violated_constraints_set.add(5)
                violations.append(
                    f"Objective consistency violated: reported objective_value="
                    f"{reported} differs from recomputed total cost="
                    f"{true_obj} (|diff|={abs_diff:.3g}, tol={obj_tol:.3g})"
                )
                normalizer = max(abs(true_obj), eps)
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": reported,
                    "rhs": true_obj,
                    "raw_excess": abs_diff,
                    "normalizer": normalizer,
                    "ratio": abs_diff / normalizer
                })

    # ---- Assemble result ----
    feasible = len(violated_constraints_set) == 0
    result = {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for ODIMCF (Barnhart et al. 2000)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON file containing the data instance")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path to the JSON file containing the candidate solution")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Path to write the JSON file containing the feasibility result")
    args = parser.parse_args()

    with open(args.instance_path, "r") as f:
        instance_data = json.load(f)

    with open(args.solution_path, "r") as f:
        solution_data = json.load(f)

    result = check_feasibility(instance_data, solution_data)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print(f"FEASIBLE: No constraint violations found.")
    else:
        print(f"INFEASIBLE: {len(result['violated_constraints'])} constraint(s) violated.")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
