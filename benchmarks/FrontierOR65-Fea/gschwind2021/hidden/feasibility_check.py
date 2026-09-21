#!/usr/bin/env python3
"""
Feasibility checker for PGMRC/CGMRC solutions.
Paper: "A Branch-and-Price Framework for Decomposing Graphs into Relaxed Cliques"
Authors: Gschwind, Irnich, Furini, Wolfler Calvo (2017)

Hard constraints checked (numbered top-to-bottom from the formulation):
  Constraint 1 — (1b)/(1b'): vertex coverage
      Partitioning: each vertex in exactly one selected RC.
      Covering: each vertex in at least one selected RC.
  Constraint 2 — (1c): binary domain (lambda_S in {0,1}).
      Automatically satisfied by the solution format.
  Constraint 3 — S-plex validity: each selected S must satisfy
      deg_{G[S]}(i) >= |S| - s  for every i in S.
  Constraint 4 — Connectivity: if connectivity_required, the subgraph
      G[S] must be connected for every selected S.
  Constraint 5 — Objective consistency: the reported objective_value must
      equal the recomputed objective (1a) min sum_S lambda_S, i.e. the
      number of relaxed cliques in the decomposition = len(solution).
      This is a Tier C defense: full recompute (every variable that
      determines the objective — the list of selected RCs — is present in
      the solution), so the reported value is checked against the exact
      recomputed count with an integer tolerance of 0.5.
"""

import argparse
import json
from collections import deque


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_adj(vertices, edges):
    adj = {v: set() for v in vertices}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def is_connected(S, adj):
    """Check if the induced subgraph G[S] is connected."""
    if len(S) <= 1:
        return True
    S_set = set(S)
    visited = set()
    queue = deque([S[0]])
    visited.add(S[0])
    while queue:
        u = queue.popleft()
        for w in adj.get(u, set()):
            if w in S_set and w not in visited:
                visited.add(w)
                queue.append(w)
    return len(visited) == len(S_set)


def check_feasibility(instance, solution):
    tol = 1e-5
    eps = 1e-5

    vertices = instance["graph"]["vertices"]
    edges = instance["graph"]["edges"]
    adj = build_adj(vertices, edges)
    V_set = set(vertices)
    n = len(vertices)

    settings = instance["problem_settings"]
    rc_type = settings.get("relaxed_clique_type", "s-plex")
    if rc_type != "s-plex":
        raise ValueError(
            f"This checker only supports relaxed_clique_type='s-plex' "
            f"(got {rc_type!r}). The dataset has been narrowed to the "
            f"s-plex variant; other RC types are out of scope."
        )
    partitioning = settings["decomposition_type"] == "partitioning"
    s_param = settings.get("parameter_s", 2)
    connectivity_required = settings.get("connectivity_required", False)

    rcs = solution["solution"]  # list of lists of vertex indices

    violations = []
    violation_magnitudes = []
    violated_set = set()

    # ------------------------------------------------------------------
    # Constraint 1 — (1b)/(1b'): vertex coverage
    # ------------------------------------------------------------------
    # Count how many selected RCs contain each vertex
    vertex_count = {v: 0 for v in vertices}
    for rc in rcs:
        for v in rc:
            if v in vertex_count:
                vertex_count[v] += 1

    if partitioning:
        # Each vertex must appear exactly once: sum = 1
        # Check vertices appearing 0 times (under-covered)
        uncovered = [v for v in vertices if vertex_count[v] == 0]
        if uncovered:
            violated_set.add(1)
            violations.append(
                f"Constraint 1 (partitioning): vertices not covered: {uncovered}"
            )
            for v in uncovered:
                lhs = 0.0
                rhs = 1.0
                raw_excess = rhs - lhs  # >= constraint sense: rhs - lhs
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

        # Check vertices appearing more than once (over-covered)
        multi = {v: c for v, c in vertex_count.items() if c > 1}
        if multi:
            violated_set.add(1)
            violations.append(
                f"Constraint 1 (partitioning): vertices covered multiple times: "
                f"{list(multi.keys())} (counts: {list(multi.values())})"
            )
            for v, cnt in multi.items():
                lhs = float(cnt)
                rhs = 1.0
                raw_excess = abs(lhs - rhs)  # equality: |lhs - rhs|
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })
    else:
        # Covering: each vertex at least once: sum >= 1
        uncovered = [v for v in vertices if vertex_count[v] < 1]
        if uncovered:
            violated_set.add(1)
            violations.append(
                f"Constraint 1 (covering): vertices not covered: {uncovered}"
            )
            for v in uncovered:
                lhs = 0.0
                rhs = 1.0
                raw_excess = rhs - lhs
                normalizer = max(abs(rhs), eps)
                violation_magnitudes.append({
                    "constraint": 1,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint 2 — (1c): binary domain
    # Automatically satisfied: the solution is a list of selected RCs,
    # so lambda_S = 1 for each listed RC and 0 for all others.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Constraint 3 — S-plex validity
    # Each selected S must satisfy: deg_{G[S]}(i) >= |S| - s for all i in S
    # Equivalently: each vertex is non-adjacent to at most s-1 others in S.
    # ------------------------------------------------------------------
    for rc_idx, rc in enumerate(rcs):
        S_set = set(rc)
        size_S = len(S_set)
        threshold = size_S - s_param  # minimum required degree in G[S]
        for v in rc:
            deg_v = sum(1 for u in adj.get(v, set()) if u in S_set and u != v)
            if deg_v < threshold - tol:
                violated_set.add(3)
                lhs = float(deg_v)
                rhs = float(threshold)
                raw_excess = rhs - lhs  # >= constraint: rhs - lhs
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Constraint 3 (s-plex): RC {rc_idx} {rc}: vertex {v} has "
                    f"degree {deg_v} in G[S] but needs >= {threshold}"
                )
                violation_magnitudes.append({
                    "constraint": 3,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint 4 — Connectivity
    # If connectivity_required, G[S] must be connected for each selected S.
    # ------------------------------------------------------------------
    if connectivity_required:
        for rc_idx, rc in enumerate(rcs):
            if len(rc) <= 1:
                continue
            if not is_connected(rc, adj):
                violated_set.add(4)
                # For connectivity, violation is binary (connected or not).
                # Use LHS=0 (not connected) vs RHS=1 (must be connected).
                lhs = 0.0
                rhs = 1.0
                raw_excess = 1.0
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Constraint 4 (connectivity): RC {rc_idx} {rc} is not connected"
                )
                violation_magnitudes.append({
                    "constraint": 4,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    # ------------------------------------------------------------------
    # Constraint 5 — Objective consistency (Tier C defense)
    # The objective (1a) is  min sum_{S} lambda_S, i.e. the number of
    # relaxed cliques selected in the decomposition. Every variable that
    # determines the objective (the list of selected RCs) is present in
    # the solution, so the true objective is recomputed EXACTLY as
    # len(solution["solution"]) and compared to the reported value.
    # Tolerance is 0.5 because the objective is an integer count: any
    # genuine mismatch is at least 1, so an exploit that lies (e.g.
    # objective_value=0 or sys.float_info.max) is always caught while
    # honest float-encoded integers (182.0 == 182) pass.
    # ------------------------------------------------------------------
    reported_obj = solution.get("objective_value")
    if reported_obj is not None:
        try:
            reported = float(reported_obj)
        except (TypeError, ValueError):
            reported = None
        if reported is not None:
            true_obj = float(len(rcs))
            abs_diff = abs(reported - true_obj)
            obj_tol = 0.5  # integer-count objective: any real mismatch is >= 1
            if abs_diff > obj_tol:
                violated_set.add(5)
                lhs = reported
                rhs = true_obj
                raw_excess = abs_diff  # equality: |lhs - rhs|
                normalizer = max(abs(rhs), eps)
                violations.append(
                    f"Constraint 5 (objective consistency): reported "
                    f"objective_value={reported} differs from recomputed "
                    f"objective sum_S(lambda_S)=len(solution)={true_obj} "
                    f"(|diff|={abs_diff:.6g}, tol={obj_tol})"
                )
                violation_magnitudes.append({
                    "constraint": 5,
                    "lhs": lhs,
                    "rhs": rhs,
                    "raw_excess": raw_excess,
                    "normalizer": normalizer,
                    "ratio": raw_excess / normalizer,
                })

    feasible = len(violated_set) == 0
    return {
        "feasible": feasible,
        "violated_constraints": sorted(violated_set),
        "violations": violations,
        "violation_magnitudes": violation_magnitudes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Feasibility checker for PGMRC/CGMRC solutions (Gschwind et al. 2017)"
    )
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file.")
    parser.add_argument("--solution_path", required=True,
                        help="Path to the JSON solution file.")
    parser.add_argument("--result_path", required=True,
                        help="Path to write the JSON feasibility result.")
    args = parser.parse_args()

    instance = load_json(args.instance_path)
    solution = load_json(args.solution_path)

    result = check_feasibility(instance, solution)

    with open(args.result_path, "w") as f:
        json.dump(result, f, indent=2)

    if result["feasible"]:
        print("Solution is FEASIBLE.")
    else:
        print("Solution is INFEASIBLE.")
        for v in result["violations"]:
            print(f"  - {v}")


if __name__ == "__main__":
    main()
