"""
Feasibility checker for the Capped-Fleet Vehicle Routing Problem with Time Windows.

Instances are Gehring & Homberger VRPTW benchmarks (Homberger & Gehring, 2005),
restricted to a hard fleet size taken from the published best-known solution.

All arithmetic is integral. Distance and travel time between two nodes are
floor(10 * euclidean(coordinates)), and every time in the instance is already
expressed on that same x10 scale, so feasibility never turns on a tolerance.

Constraint indices follow hidden/mathematical_formulation.md:

    (1)  objective consistency: the reported objective_value equals the
         recomputed total distance
    (2)  every customer is served exactly once
    (5)  at most max_vehicles routes are used
    (7)  service at each customer begins within its time window
    (8)  every route returns to the depot before the depot closes
    (10) the load of each route does not exceed vehicle capacity

Index 0 is reserved for a solution that cannot be parsed as a schedule at all.

CLI contract:
    python feasibility_check.py --instance_path I --solution_path S --result_path R
Writes to R: {"feasible": bool|None, "violated_constraints": [...],
              "violations": [...], "violation_magnitudes": [...]}
"""

import argparse
import json
import math

SCALE = 10
EPS = 1e-9


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def distance(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return math.floor(SCALE * math.sqrt(dx * dx + dy * dy))


def write_result(result_path, result):
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def normalise_routes(raw):
    """Routes as {"0": [...], "1": [...]}, the shape solution_schema.json asks for.

    A bare list of lists is also accepted. That is laxer than the schema, but it
    can only ever turn a well-formed schedule written in the wrong container into
    a graded solution; it cannot make an infeasible schedule pass, since every
    constraint below is checked either way.
    """
    if isinstance(raw, dict):
        return [(str(k), v) for k, v in sorted(raw.items(), key=lambda kv: str(kv[0]))]
    if isinstance(raw, list):
        return [(str(i), v) for i, v in enumerate(raw)]
    return None


def check_feasibility(instance_path, solution_path, result_path):
    inst = load_json(instance_path)
    sol = load_json(solution_path)

    coords = inst["coordinates"]
    demand = inst["demand"]
    tw = inst["time_window"]
    service = inst["service_time"]
    capacity = inst["capacity"]
    max_vehicles = inst["max_vehicles"]
    num_nodes = len(coords)

    # ---- solution must be parseable as a set of routes --------------------
    if "routes" not in sol or sol["routes"] is None:
        result = write_result(result_path, {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["No solution data in solution file"],
            "violation_magnitudes": [],
        })
        print("No solution data in solution file")
        print(f"Result written to {result_path}")
        return result

    routes = normalise_routes(sol["routes"])
    if routes is None or any(not isinstance(seq, list) for _, seq in routes):
        result = write_result(result_path, {
            "feasible": None,
            "violated_constraints": [],
            "violations": ["solution.routes must map each vehicle to a list of "
                           "customer indices"],
            "violation_magnitudes": [],
        })
        print("solution.routes is not a mapping of vehicles to customer lists")
        print(f"Result written to {result_path}")
        return result

    violations = []
    violated_constraints = set()
    violation_magnitudes = []

    def add_violation(constraint_idx, message, lhs, rhs, violation_amount):
        violated_constraints.add(constraint_idx)
        violations.append(message)
        normalizer = max(abs(rhs), EPS) if isinstance(rhs, (int, float)) else 1.0
        violation_magnitudes.append({
            "constraint": constraint_idx,
            "lhs": lhs,
            "rhs": rhs,
            "raw_excess": violation_amount,
            "normalizer": normalizer,
            "ratio": (violation_amount / normalizer
                      if isinstance(violation_amount, (int, float)) else None),
        })

    # ---- node identifiers -------------------------------------------------
    for vid, seq in routes:
        for c in seq:
            if isinstance(c, bool) or not isinstance(c, int) or c <= 0 or c >= num_nodes:
                add_violation(0, f"route {vid} contains {c!r}, which is not a customer "
                                 f"index in 1..{num_nodes - 1} (the depot is implicit "
                                 f"and must not be listed)", c, None, None)

    routes = [(vid, [c for c in seq
                     if isinstance(c, int) and not isinstance(c, bool)
                     and 0 < c < num_nodes])
              for vid, seq in routes]
    nonempty = [(vid, seq) for vid, seq in routes if seq]

    # ---- (2) every customer served exactly once ---------------------------
    seen = {}
    for vid, seq in nonempty:
        for c in seq:
            if c in seen:
                add_violation(2, f"customer {c} is served twice (routes {seen[c]} "
                                 f"and {vid})", 2, 1, 1)
            seen[c] = vid
    missing = [c for c in range(1, num_nodes) if c not in seen]
    if missing:
        add_violation(2, f"{len(missing)} customers are never served, e.g. "
                         f"{missing[:10]}", len(missing), 0, len(missing))

    # ---- (5) fleet cap ----------------------------------------------------
    if len(nonempty) > max_vehicles:
        add_violation(5, f"{len(nonempty)} routes are used but at most "
                         f"{max_vehicles} are allowed",
                      len(nonempty), max_vehicles, len(nonempty) - max_vehicles)

    total = 0
    for vid, seq in nonempty:
        # ---- (10) capacity ------------------------------------------------
        load = sum(demand[c] for c in seq)
        if load > capacity:
            add_violation(10, f"route {vid} carries {load} which exceeds capacity "
                              f"{capacity}", load, capacity, load - capacity)

        # ---- (7) time windows along the route -----------------------------
        t = 0
        prev = 0
        for c in seq:
            d = distance(coords, prev, c)
            t += d
            total += d
            if t > tw[c][1]:
                add_violation(7, f"route {vid} reaches customer {c} at {t} but it is "
                                 f"due at {tw[c][1]}", t, tw[c][1], t - tw[c][1])
            t = max(t, tw[c][0]) + service
            prev = c
        d = distance(coords, prev, 0)
        t += d
        total += d
        # ---- (8) depot return ---------------------------------------------
        if t > tw[0][1]:
            add_violation(8, f"route {vid} returns to the depot at {t}, after it "
                             f"closes at {tw[0][1]}", t, tw[0][1], t - tw[0][1])

    # ---- (1) objective consistency ----------------------------------------
    # total is an exact integer sum of floor(10 * euclid); the reported objective
    # is that sum divided by 10, so the comparison is made on the integer scale
    # and admits no tolerance.
    objective = total / SCALE
    reported = sol.get("objective_value")
    if reported is None:
        add_violation(1, "solution.objective_value is missing; the reported "
                         "objective is part of the submission contract",
                      None, objective, None)
    else:
        try:
            scaled = float(reported) * SCALE
        except (TypeError, ValueError):
            add_violation(1, f"objective_value {reported!r} is not a number",
                          reported, objective, None)
        else:
            if round(scaled) != total:
                add_violation(1, f"objective_value {reported} does not match the "
                                 f"recomputed distance {objective:.1f}",
                              float(reported), objective,
                              abs(float(reported) - objective))

    feasible = not violations
    result = write_result(result_path, {
        "feasible": feasible,
        "violated_constraints": sorted(violated_constraints),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes if not feasible else [],
    })
    print("Feasibility: {}".format("FEASIBLE" if feasible else "INFEASIBLE"))
    print(f"Routes used: {len(nonempty)} / {max_vehicles}   objective: {objective:.1f}")
    for v in violations[:20]:
        print("  - " + v)
    print(f"Result written to {result_path}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for capped-fleet VRPTW"
    )
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the JSON file containing the data instance."
    )
    parser.add_argument(
        "--solution_path", type=str, required=True,
        help="Path to the JSON file containing the candidate solution."
    )
    parser.add_argument(
        "--result_path", type=str, required=True,
        help="Path to write the JSON file containing the feasibility result."
    )
    args = parser.parse_args()
    check_feasibility(args.instance_path, args.solution_path, args.result_path)


if __name__ == "__main__":
    main()
