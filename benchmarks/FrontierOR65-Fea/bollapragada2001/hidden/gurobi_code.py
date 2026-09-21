"""
Gurobi MILP implementation for the Discrete Truss Structure Design Problem.

Based on Formulation (5) from:
  Bollapragada, Ghattas, and Hooker (2001)
  "Optimal Design of Truss Structures by Logic-Based Branch and Cut"
  Operations Research, 49(1):42-51

The MILP uses binary variables y_{ik} to select discrete cross-sectional areas,
disaggregated elongation variables v_{ikl}, and linearized Hooke's law.
"""

import argparse
import json
import math
import sys

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
    """Load a problem instance from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def build_b_matrix(instance):
    """
    Build the direction cosine matrix b[i][j] where:
      i = bar index (0-based)
      j = DOF index (0-based)

    b[i][j] is the cosine of the angle between bar i and DOF j.
    For a 2D problem, each bar connects two nodes. Each free node contributes
    2 DOFs (x, y). The direction cosine for a bar at one of its endpoints
    equals the component of the bar's unit direction vector along that DOF's
    direction, with sign depending on whether the node is the start or end node.
    """
    bars = instance["bars"]
    dofs = instance["degrees_of_freedom"]
    nodes = {n["node_id"]: n for n in instance["nodes"]}

    num_bars = len(bars)
    num_dofs = len(dofs)

    # Build a map from (node_id, direction) -> dof_index (0-based)
    dof_map = {}
    for dof in dofs:
        dof_map[(dof["node"], dof["direction"])] = dof["dof_id"] - 1

    b = [[0.0] * num_dofs for _ in range(num_bars)]

    dim = instance.get("dimension", 2)
    directions = ["x", "y"] if dim == 2 else ["x", "y", "z"]

    for bar_idx, bar in enumerate(bars):
        ni = bar["node_i"]
        nj = bar["node_j"]
        node_i = nodes[ni]
        node_j = nodes[nj]

        # Compute unit direction vector from node_i to node_j
        dx = node_j["x"] - node_i["x"]
        dy = node_j["y"] - node_i["y"]
        dz = 0.0
        if dim == 3:
            dz = node_j.get("z", 0.0) - node_i.get("z", 0.0)

        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            continue

        # Unit direction vector components
        cos_vals = [dx / length, dy / length]
        if dim == 3:
            cos_vals.append(dz / length)

        # For node_i (start node): contribution is +cos along each direction
        # For node_j (end node): contribution is -cos along each direction
        # Convention: bar force s_i positive = tension.
        # The equilibrium eq is: sum_i b[i][j] * s[i][l] = p[j][l]
        # For a bar going from node_i to node_j with unit vector e:
        #   At node_j: +e contributes to equilibrium
        #   At node_i: -e contributes to equilibrium
        # The compatibility eq is: sum_j b[i][j] * d[j][l] = v[i][l]
        # For consistency, b[i][j] for DOFs at node_j = +cos, at node_i = -cos
        # This follows standard structural analysis sign conventions.

        for d_idx, direction in enumerate(directions):
            # node_i DOFs (if free)
            if (ni, direction) in dof_map:
                j = dof_map[(ni, direction)]
                b[bar_idx][j] = -cos_vals[d_idx]
            # node_j DOFs (if free)
            if (nj, direction) in dof_map:
                j = dof_map[(nj, direction)]
                b[bar_idx][j] = cos_vals[d_idx]

    return b


def solve(instance, time_limit):
    """Build and solve the MILP formulation (5)."""
    num_bars = instance["num_bars"]
    num_dofs = instance["num_free_dofs"]
    num_loads = instance["num_loading_conditions"]

    bars = instance["bars"]
    dofs = instance["degrees_of_freedom"]
    loads = instance["loading_conditions"]

    E = instance["material_properties"]["modulus_of_elasticity"]
    c = instance["material_properties"]["cost_density"]

    areas = instance["discrete_areas"]
    K = len(areas)

    # Build stress bounds per bar
    stress_bounds = {}
    if "bar_specific_stress_bounds" in instance:
        for sb in instance["bar_specific_stress_bounds"]:
            stress_bounds[sb["bar_id"]] = (sb["lower"], sb["upper"])
    else:
        sl = instance["stress_bounds"]["lower"]
        su = instance["stress_bounds"]["upper"]
        for bar in bars:
            stress_bounds[bar["bar_id"]] = (sl, su)

    # Displacement bounds
    d_lb = instance["displacement_bounds"]["lower"]
    d_ub = instance["displacement_bounds"]["upper"]

    # Build b matrix
    b = build_b_matrix(instance)

    # Pre-compute elongation bounds per bar (incorporating stress bounds)
    # v_i^L = max(v_i^L, (h_i / E_i) * sigma_i^L)
    # v_i^U = min(v_i^U, (h_i / E_i) * sigma_i^U)
    # Paper assumes no explicit elongation bounds are given beyond stress bounds,
    # so we derive them from stress bounds.
    v_lb = []
    v_ub = []
    for bar in bars:
        h_i = bar["length"]
        E_i = E  # uniform modulus
        sigma_L, sigma_U = stress_bounds[bar["bar_id"]]
        # Elongation bounds from stress: v = (h/E) * sigma
        vL = (h_i / E_i) * sigma_L
        vU = (h_i / E_i) * sigma_U
        v_lb.append(vL)
        v_ub.append(vU)

    # Build load vectors p[j][l]
    p = [[0.0] * num_loads for _ in range(num_dofs)]
    for load_idx, lc in enumerate(loads):
        for ld in lc["loads"]:
            dof_idx = ld["dof_id"] - 1
            p[dof_idx][load_idx] = ld["force"]

    # ---- Build Gurobi Model ----
    model = gp.Model("TrussDesign_MILP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Decision variables
    # y[i][k] binary: 1 if bar i uses area k
    y = {}
    for i in range(num_bars):
        for k in range(K):
            y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")

    # s[i][l]: force in bar i under load l (unrestricted)
    s = {}
    for i in range(num_bars):
        for l in range(num_loads):
            s[i, l] = model.addVar(lb=-GRB.INFINITY, name=f"s_{i}_{l}")

    # d[j][l]: displacement at DOF j under load l
    d = {}
    for j in range(num_dofs):
        for l in range(num_loads):
            lb_val = d_lb if d_lb is not None else -GRB.INFINITY
            ub_val = d_ub if d_ub is not None else GRB.INFINITY
            d[j, l] = model.addVar(lb=lb_val, ub=ub_val, name=f"d_{j}_{l}")

    # v[i][k][l]: disaggregated elongation variable
    v = {}
    for i in range(num_bars):
        for k in range(K):
            for l in range(num_loads):
                v[i, k, l] = model.addVar(lb=-GRB.INFINITY, name=f"v_{i}_{k}_{l}")

    model.update()

    # ---- Objective: min sum_i c_i * h_i * sum_k A_{ik} * y_{ik} ----
    obj = gp.LinExpr()
    for i in range(num_bars):
        h_i = bars[i]["length"]
        for k in range(K):
            obj += c * h_i * areas[k] * y[i, k]
    model.setObjective(obj, GRB.MINIMIZE)

    # ---- Constraints ----

    # 1. Exactly one size per bar: sum_k y[i][k] = 1
    for i in range(num_bars):
        model.addConstr(
            gp.quicksum(y[i, k] for k in range(K)) == 1,
            name=f"one_size_{i}"
        )

    # 2. Equilibrium: sum_i b[i][j] * s[i][l] = p[j][l]
    for j in range(num_dofs):
        for l in range(num_loads):
            model.addConstr(
                gp.quicksum(b[i][j] * s[i, l] for i in range(num_bars)) == p[j][l],
                name=f"equil_{j}_{l}"
            )

    # 3. Compatibility: sum_j b[i][j] * d[j][l] = sum_k v[i][k][l]
    for i in range(num_bars):
        for l in range(num_loads):
            model.addConstr(
                gp.quicksum(b[i][j] * d[j, l] for j in range(num_dofs))
                == gp.quicksum(v[i, k, l] for k in range(K)),
                name=f"compat_{i}_{l}"
            )

    # 4. Hooke's law (linearized): (E_i/h_i) * sum_k A_{ik} * v[i][k][l] = s[i][l]
    for i in range(num_bars):
        h_i = bars[i]["length"]
        E_i = E
        for l in range(num_loads):
            model.addConstr(
                (E_i / h_i) * gp.quicksum(areas[k] * v[i, k, l] for k in range(K))
                == s[i, l],
                name=f"hooke_{i}_{l}"
            )

    # 5. Elongation bounds: v_i^L * y[i][k] <= v[i][k][l] <= v_i^U * y[i][k]
    for i in range(num_bars):
        for k in range(K):
            for l in range(num_loads):
                model.addConstr(
                    v[i, k, l] >= v_lb[i] * y[i, k],
                    name=f"vlo_{i}_{k}_{l}"
                )
                model.addConstr(
                    v[i, k, l] <= v_ub[i] * y[i, k],
                    name=f"vhi_{i}_{k}_{l}"
                )

    # 6. Linking groups: bars in the same group must have the same y variables
    if instance.get("linking_groups"):
        for group in instance["linking_groups"]:
            bar_ids = group["bar_ids"]
            ref_bar = bar_ids[0] - 1  # 0-based
            for bid in bar_ids[1:]:
                bi = bid - 1  # 0-based
                for k in range(K):
                    model.addConstr(
                        y[bi, k] == y[ref_bar, k],
                        name=f"link_{ref_bar}_{bi}_{k}"
                    )

    # Solve
    model.optimize()

    # Extract solution
    result = {"solver": "Gurobi_MILP", "status": "unknown"}

    if model.SolCount > 0:
        result["objective_value"] = model.ObjVal
        result["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"
        result["mip_gap"] = model.MIPGap

        # Extract bar areas
        bar_areas = []
        for i in range(num_bars):
            for k in range(K):
                if y[i, k].X > 0.5:
                    bar_areas.append({
                        "bar_id": bars[i]["bar_id"],
                        "area": areas[k],
                        "area_index": k
                    })
                    break
        result["bar_areas"] = bar_areas

        # Extract displacements
        displacements = []
        for j in range(num_dofs):
            for l in range(num_loads):
                displacements.append({
                    "dof_id": dofs[j]["dof_id"],
                    "load": l + 1,
                    "value": d[j, l].X
                })
        result["displacements"] = displacements

        # Extract bar forces
        bar_forces = []
        for i in range(num_bars):
            for l in range(num_loads):
                bar_forces.append({
                    "bar_id": bars[i]["bar_id"],
                    "load": l + 1,
                    "force": s[i, l].X
                })
        result["bar_forces"] = bar_forces
    else:
        result["objective_value"] = None
        result["status"] = "infeasible"

    result["solve_time"] = model.Runtime
    result["num_vars"] = model.NumVars
    result["num_constrs"] = model.NumConstrs

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Gurobi MILP solver for discrete truss design (Bollapragada et al. 2001)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    print(f"Loaded instance: {instance['problem_name']}")
    print(f"  Bars: {instance['num_bars']}, DOFs: {instance['num_free_dofs']}, "
          f"Loads: {instance['num_loading_conditions']}")
    print(f"  Discrete areas: {instance['discrete_areas']}")
    print(f"  Time limit: {args.time_limit}s")

    result = solve(instance, args.time_limit)

    print(f"\nResult: status={result['status']}, objective={result['objective_value']}")
    if "mip_gap" in result:
        print(f"  MIP gap: {result['mip_gap']:.6f}")
    print(f"  Solve time: {result['solve_time']:.2f}s")

    if result.get("bar_areas"):
        print("\nBar areas:")
        for ba in result["bar_areas"]:
            print(f"  Bar {ba['bar_id']}: area = {ba['area']}")

    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSolution written to {args.solution_path}")


if __name__ == "__main__":
    main()
