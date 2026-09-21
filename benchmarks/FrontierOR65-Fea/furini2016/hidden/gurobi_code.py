#!/usr/bin/env python3
"""
PP-G2KP Model for the Guillotine Two-Dimensional Knapsack Problem (G2KP).

Based on: Furini, Malaguti, Thomopulos (2016)
"Modeling Two-Dimensional Guillotine Cutting Problems via Integer Programming"
INFORMS Journal on Computing 28(4): 736-751.

Implements the complete PP-G2KP MIP model (constraints 1-7) with Gurobi.
"""

import json
import argparse
import sys
from collections import deque

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
# ---------------------------------------------------------------------------
# Bounded Subset Sum (DP for complete cut-position set)
# ---------------------------------------------------------------------------

def bounded_subset_sum(sizes, copies, max_val):
    """
    Compute all achievable positive sums < max_val using at most copies[i]
    copies of sizes[i], for each i.

    Uses a bitset (Python big-integer) approach for efficiency:
    - reachable is an integer whose k-th bit is set iff sum k is achievable.
    - For each item i with size s and count c, we accumulate shifts of
      the current reachable set by s, s+s, ..., c*s (bounded copies).

    Returns a set of integers.
    """
    if not sizes or max_val <= 1:
        return set()

    mask = (1 << max_val) - 1
    reachable = 1  # bit 0 set: sum 0 is achievable

    for size, count in zip(sizes, copies):
        if size <= 0 or size >= max_val:
            continue
        new_reachable = reachable
        temp = reachable
        for _ in range(count):
            temp = (temp << size) & mask
            new_reachable |= temp
            if not temp:
                break
        reachable = new_reachable

    return {s for s in range(1, max_val) if (reachable >> s) & 1}


# ---------------------------------------------------------------------------
# Cut-position computation
# ---------------------------------------------------------------------------

def compute_complete_positions(plate_l, plate_w, items):
    """
    Compute the complete position sets Q(j,h) and Q(j,v) for a plate of
    dimensions plate_l x plate_w.

    Q(j,h) = achievable sums of item widths for fitting items, filtered to
             q < plate_w / 2  (symmetric positions removed).
    Q(j,v) = achievable sums of item lengths for fitting items, filtered to
             q < plate_l / 2.

    Returns (h_positions, v_positions) as sorted lists.
    """
    fitting = [(it['length'], it['width'], it['copies'])
               for it in items
               if it['length'] <= plate_l and it['width'] <= plate_w]

    if not fitting:
        return [], []

    h_sizes  = [w for (l, w, c) in fitting]
    h_copies = [c for (l, w, c) in fitting]
    v_sizes  = [l for (l, w, c) in fitting]
    v_copies = [c for (l, w, c) in fitting]

    h_all = bounded_subset_sum(h_sizes, h_copies, plate_w)
    v_all = bounded_subset_sum(v_sizes, v_copies, plate_l)

    # Remove symmetric positions: keep only q < dim/2
    h_pos = sorted(q for q in h_all if 2 * q < plate_w)
    v_pos = sorted(q for q in v_all if 2 * q < plate_l)

    return h_pos, v_pos


def can_fit_any_item(plate_l, plate_w, items):
    return any(it['length'] <= plate_l and it['width'] <= plate_w
               for it in items)


# ---------------------------------------------------------------------------
# Plate and variable enumeration (Procedure 1 from the paper)
# ---------------------------------------------------------------------------

def enumerate_plates_and_cuts(L, W, items):
    """
    Procedure 1: Plate-and-variable enumeration.

    Starting from plate 0 (L x W), generate all reachable sub-plates through
    guillotine cuts at complete-position-set positions. A new plate is added
    to J only if it can fit at least one item.

    Returns
    -------
    plates    : list of (l, w) tuples indexed by plate id
    plate_map : dict (l, w) -> plate_id
    cuts      : list of (parent_id, orientation, position, child1_id, child2_id)
                child_id may be None if that child cannot fit any item (waste).
    J_bar     : dict plate_id -> item dict  (plates matching an item's dims)
    """
    # Build item dimension map: (l, w) -> item  (highest profit wins ties)
    item_map = {}
    for it in items:
        key = (it['length'], it['width'])
        if key not in item_map or it['profit'] > item_map[key]['profit']:
            item_map[key] = it

    plates    = [(L, W)]
    plate_map = {(L, W): 0}
    cuts      = []

    queue     = deque([0])
    processed = set()

    while queue:
        pid = queue.popleft()
        if pid in processed:
            continue
        processed.add(pid)

        pl, pw = plates[pid]
        h_pos, v_pos = compute_complete_positions(pl, pw, items)

        # Horizontal cuts: split width into q and pw-q
        for q in h_pos:
            c1_dims = (pl, q)
            c2_dims = (pl, pw - q)

            c1_id = _get_or_add_plate(c1_dims, plates, plate_map, queue, items)
            c2_id = _get_or_add_plate(c2_dims, plates, plate_map, queue, items)

            cuts.append((pid, 'h', q, c1_id, c2_id))

        # Vertical cuts: split length into q and pl-q
        for q in v_pos:
            c1_dims = (q,      pw)
            c2_dims = (pl - q, pw)

            c1_id = _get_or_add_plate(c1_dims, plates, plate_map, queue, items)
            c2_id = _get_or_add_plate(c2_dims, plates, plate_map, queue, items)

            cuts.append((pid, 'v', q, c1_id, c2_id))

    # Determine J_bar (item plates)
    J_bar = {pid: item_map[dims]
             for pid, dims in enumerate(plates)
             if dims in item_map}

    return plates, plate_map, cuts, J_bar


def _get_or_add_plate(dims, plates, plate_map, queue, items):
    """
    If dims can fit some item and is not yet in J, add it.
    Returns plate_id or None (if the plate is waste).
    """
    pl, pw = dims
    if pl <= 0 or pw <= 0:
        return None
    if not can_fit_any_item(pl, pw, items):
        return None
    if dims not in plate_map:
        new_id = len(plates)
        plate_map[dims] = new_id
        plates.append(dims)
        queue.append(new_id)
    return plate_map[dims]


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_g2kp(instance_path, solution_path, time_limit):
    # ------------------------------------------------------------------
    # Load instance
    # ------------------------------------------------------------------
    with open(instance_path) as f:
        inst = json.load(f)

    L     = inst['panel']['length']
    W     = inst['panel']['width']
    items = inst['items']

    print(f"Instance: {inst.get('instance_name', instance_path)}")
    print(f"Panel: {L} x {W},  n={len(items)} item types")

    # ------------------------------------------------------------------
    # Enumerate plates and cuts
    # ------------------------------------------------------------------
    plates, plate_map, cuts, J_bar = enumerate_plates_and_cuts(L, W, items)

    n_plates = len(plates)
    n_cuts   = len(cuts)
    print(f"Plates: {n_plates},  Cuts (x-variables): {n_cuts},  "
          f"Item plates (y-variables): {len(J_bar)}")

    # ------------------------------------------------------------------
    # Build Gurobi model
    # ------------------------------------------------------------------
    model = gp.Model("PP-G2KP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # y variables: for each item plate j in J_bar
    y = {}
    for pid, item in J_bar.items():
        y[pid] = model.addVar(
            vtype=GRB.INTEGER, lb=0,
            name=f"y[{pid}]"
        )

    # x variables: for each cut
    x = {}
    for idx, (pid, orient, q, c1_id, c2_id) in enumerate(cuts):
        x[idx] = model.addVar(
            vtype=GRB.INTEGER, lb=0,
            name=f"x[{pid},{orient},{q}]"
        )

    # Objective (1): maximize sum of p_j * y_j
    model.setObjective(
        gp.quicksum(item['profit'] * y[pid] for pid, item in J_bar.items()),
        GRB.MAXIMIZE
    )

    # ------------------------------------------------------------------
    # Build flow-balance structures
    # ------------------------------------------------------------------
    # flow_in[j]  = LinExpr  (sum of x variables whose cuts PRODUCE plate j)
    # flow_out[j] = LinExpr  (sum of x variables that CUT plate j)
    flow_in  = {pid: [] for pid in range(n_plates)}
    flow_out = {pid: [] for pid in range(n_plates)}

    for idx, (pid, orient, q, c1_id, c2_id) in enumerate(cuts):
        flow_out[pid].append(x[idx])
        if c1_id is not None:
            flow_in[c1_id].append(x[idx])
        if c2_id is not None:
            flow_in[c2_id].append(x[idx])

    # ------------------------------------------------------------------
    # Add constraints
    # ------------------------------------------------------------------

    # Constraint (4): original panel used at most once
    #   sum_{o,q} x^o_{q,0}  (+y_0 if 0 in J_bar)  <= 1
    lhs_panel = gp.quicksum(flow_out[0])
    if 0 in J_bar:
        lhs_panel = lhs_panel + y[0]
    model.addConstr(lhs_panel <= 1, name="panel_use")

    # Constraints (2) and (3): flow balance for all plates j != 0
    for pid in range(1, n_plates):
        in_expr  = gp.quicksum(flow_in[pid])  if flow_in[pid]  else gp.LinExpr()
        out_expr = gp.quicksum(flow_out[pid]) if flow_out[pid] else gp.LinExpr()
        lhs = in_expr - out_expr

        if pid in J_bar:
            # Constraint (2): for item plates
            model.addConstr(lhs - y[pid] >= 0, name=f"flow[{pid}]")
        else:
            # Constraint (3): for non-item plates
            if flow_in[pid] or flow_out[pid]:
                model.addConstr(lhs >= 0, name=f"flow[{pid}]")

    # Constraint (5): capacity  y_j <= u_j
    for pid, item in J_bar.items():
        model.addConstr(y[pid] <= item['copies'], name=f"cap[{pid}]")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    # ------------------------------------------------------------------
    # Extract and write solution
    # ------------------------------------------------------------------
    status = model.Status

    if model.SolCount == 0:
        obj_val = 0.0
        items_selected = []
        cuts_used = []
        sol_status = "infeasible_or_no_solution"
    else:
        obj_val = model.ObjVal
        items_selected = []
        for pid, item in J_bar.items():
            val = y[pid].X
            if val > 0.5:
                items_selected.append({
                    "item_id":    item['id'],
                    "plate_id":   pid,
                    "plate_dims": list(plates[pid]),
                    "copies":     int(round(val))
                })
        cuts_used = []
        for idx, (pid, orient, q, c1_id, c2_id) in enumerate(cuts):
            val = x[idx].X
            if val > 0.5:
                cuts_used.append({
                    "parent_plate_id": pid,
                    "orientation":     orient,
                    "position":        q,
                    "child1_plate_id": c1_id,
                    "child2_plate_id": c2_id,
                    "count":           int(round(val))
                })
        sol_status = "optimal" if status == GRB.OPTIMAL else "time_limit"

    solution = {
        "objective_value": obj_val,
        "status":          sol_status,
        "items_selected":  items_selected,
        "cuts_used":       cuts_used,
        "plates":          [{"id": pid, "length": pl, "width": pw,
                             "in_J_bar": pid in J_bar}
                            for pid, (pl, pw) in enumerate(plates)],
        "num_plates":      n_plates,
        "num_cuts":        n_cuts,
    }

    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Status: {sol_status}")
    print(f"Objective value: {obj_val}")
    print(f"Solution written to: {solution_path}")
    return obj_val


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Solve G2KP via the PP-G2KP MIP model using Gurobi."
    )
    parser.add_argument(
        "--instance_path", required=True,
        help="Path to the JSON instance file."
    )
    parser.add_argument(
        "--solution_path", required=True,
        help="Path where the solution JSON will be written."
    )
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="Maximum solver runtime in seconds (default: 3600)."
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    solve_g2kp(args.instance_path, args.solution_path, args.time_limit)


if __name__ == "__main__":
    main()
