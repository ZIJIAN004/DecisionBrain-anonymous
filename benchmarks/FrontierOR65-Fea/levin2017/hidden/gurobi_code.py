"""
Gurobi LP implementation of the SAV routing problem from Levin (2017).

Implements the linear program defined by constraints (33)-(55) for
shared autonomous vehicle (SAV) fleet routing on a cell-transmission-
model-based network.
"""

import json
import argparse
import os
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


def main():
    parser = argparse.ArgumentParser(
        description="Solve SAV routing LP via Gurobi (Levin 2017 formulation)"
    )
    parser.add_argument("--instance_path", type=str, required=True)
    parser.add_argument("--solution_path", type=str, required=True)
    parser.add_argument("--time_limit", type=int, required=True)
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ----------------------------------------------------------------
    # Load instance data
    # ----------------------------------------------------------------
    with open(args.instance_path) as f:
        data = json.load(f)

    nodes = data["network"]["nodes"]
    links = data["network"]["links"]
    T = data["time_parameters"]["time_horizon_T"]
    dt_sec = data["time_parameters"]["time_step_duration_seconds"]
    od_pairs = data["demand"]["od_pairs"]
    fleet = data["fleet"]["initial_distribution"]  # dict: centroid_id -> count

    # ----------------------------------------------------------------
    # Build network sets
    # ----------------------------------------------------------------
    # Centroid set Z
    Z = set()
    for node in nodes:
        if node["type"] == "centroid":
            Z.add(node["id"])

    # All node ids
    all_node_ids = set(n["id"] for n in nodes)

    # Link sets with parameters
    A_o = set()        # road links
    A_z_plus = set()   # centroid connectors outgoing from centroid
    A_z_minus = set()  # centroid connectors incoming to centroid

    # Link parameters indexed by (from, to) tuple
    fftt = {}   # free-flow travel time in time steps
    cwtt = {}   # congested wave travel time in time steps
    Q = {}      # capacity per time step
    KL = {}     # jam density * length (max vehicles on link)

    for link in links:
        i = link["from"]
        j = link["to"]
        arc = (i, j)

        fftt[arc] = link["free_flow_travel_time_steps"]
        cwtt[arc] = link["congested_travel_time_steps"]
        Q[arc] = link["capacity_vph"] * (dt_sec / 3600.0)
        KL[arc] = link["jam_density_vehicles"]

        if link["type"] == "road":
            A_o.add(arc)
        elif link["type"] == "centroid_connector":
            if i in Z:
                A_z_plus.add(arc)
            if j in Z:
                A_z_minus.add(arc)

    # Combined sets
    A_o_plus = A_o | A_z_plus          # A_o ∪ A_z_plus
    A_o_minus = A_o | A_z_minus        # A_o ∪ A_z_minus
    A_all = A_o | A_z_plus | A_z_minus  # all links

    # Adjacency: gamma_plus[j] = outgoing links from j, gamma_minus[j] = incoming links to j
    gamma_plus = {n["id"]: [] for n in nodes}
    gamma_minus = {n["id"]: [] for n in nodes}

    for arc in A_all:
        i, j = arc
        gamma_plus[i].append(arc)
        gamma_minus[j].append(arc)

    # ----------------------------------------------------------------
    # Build demand dictionary d[r, s, t]
    # ----------------------------------------------------------------
    d = {}
    for od in od_pairs:
        r = od["origin"]
        s = od["destination"]
        for t_dep in od["departure_times"]:
            key = (r, s, t_dep)
            d[key] = d.get(key, 0) + 1

    # Collect active OD pairs (r, s) that have nonzero demand
    od_set = set()
    for od in od_pairs:
        od_set.add((od["origin"], od["destination"]))
    # We need all (r, s) in Z x Z for completeness of omega / e variables
    ZxZ = [(r, s) for r in Z for s in Z]

    # ----------------------------------------------------------------
    # Pre-compute valid index sets for variables
    # ----------------------------------------------------------------
    Z_list = sorted(Z)

    # y_turn indices: (i, j, k, s, t) where (i,j) in A_o_plus, (j,k) in gamma_plus[j]
    # For efficiency, build a list of (i,j,k) triples
    turn_triples = []  # list of (i, j, k)
    for (i, j) in A_o_plus:
        for (j2, k) in gamma_plus[j]:
            assert j2 == j
            turn_triples.append((i, j, k))

    # For constraint (44): identify which y_turn vars must be zero
    # For (j,k) in A_z_minus, the centroid is k. y_turn[i,j,k,s,t] = 0 if s != k.
    # We implement this by not creating those variables (or fixing them).
    # Build set of A_z_minus destination nodes for quick lookup.
    az_minus_dest = {}  # arc (j,k) -> centroid k
    for (j, k) in A_z_minus:
        az_minus_dest[(j, k)] = k

    # ----------------------------------------------------------------
    # Create Gurobi model
    # ----------------------------------------------------------------
    model = gp.Model("SAV_routing_LP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("Method", 2)  # barrier often good for large LPs

    # ----------------------------------------------------------------
    # Decision variables
    # ----------------------------------------------------------------
    time_range = range(T + 1)  # 0..T

    # y_turn[i,j,k,s,t] -- turning flow
    # For constraint (44) efficiency: skip s != k when (j,k) in A_z_minus
    y_turn_keys = []
    for (i, j, k) in turn_triples:
        if (j, k) in az_minus_dest:
            # Only destination = k allowed
            dest_k = az_minus_dest[(j, k)]
            for t in time_range:
                y_turn_keys.append((i, j, k, dest_k, t))
        else:
            for s in Z_list:
                for t in time_range:
                    y_turn_keys.append((i, j, k, s, t))

    y_turn = model.addVars(y_turn_keys, lb=0.0, name="y_turn")

    # y_cent[i,j,s,t] -- centroid departure flow, (i,j) in A_z_plus
    y_cent_keys = [
        (i, j, s, t)
        for (i, j) in A_z_plus
        for s in Z_list
        for t in time_range
    ]
    y_cent = model.addVars(y_cent_keys, lb=0.0, name="y_cent")

    # N_U[i,j,s,t] and N_D[i,j,s,t] -- cumulative counts on all links
    NU_keys = [
        (i, j, s, t)
        for (i, j) in A_all
        for s in Z_list
        for t in time_range
    ]
    N_U = model.addVars(NU_keys, lb=0.0, name="N_U")
    N_D = model.addVars(NU_keys, lb=0.0, name="N_D")

    # p[j,t] -- parked vehicles at centroid j
    p_keys = [(j, t) for j in Z_list for t in time_range]
    p = model.addVars(p_keys, lb=0.0, name="p")

    # e[r,s,t] -- departing travelers
    e_keys = [(r, s, t) for (r, s) in ZxZ for t in time_range]
    e = model.addVars(e_keys, lb=0.0, name="e")

    # omega[r,s,t] -- waiting demand
    omega_keys = [(r, s, t) for (r, s) in ZxZ for t in time_range]
    omega = model.addVars(omega_keys, lb=0.0, name="omega")

    model.update()

    # ----------------------------------------------------------------
    # Helper: get y_turn variable, returning 0 if key doesn't exist
    # (handles constraint 44 implicitly -- variables not created are 0)
    # ----------------------------------------------------------------
    def get_y_turn(i, j, k, s, t):
        key = (i, j, k, s, t)
        if key in y_turn:
            return y_turn[key]
        return 0.0

    # ----------------------------------------------------------------
    # Fix initial conditions
    # ----------------------------------------------------------------
    # (41) N_U[i,j,s,0] = 0
    # (42) N_D[i,j,s,0] = 0
    for (i, j) in A_all:
        for s in Z_list:
            model.addConstr(N_U[i, j, s, 0] == 0, name=f"init_NU_{i}_{j}_{s}")
            model.addConstr(N_D[i, j, s, 0] == 0, name=f"init_ND_{i}_{j}_{s}")

    # Fix p[j,0] = fleet[j]
    for j in Z_list:
        fleet_j = fleet.get(str(j), fleet.get(j, 0))
        model.addConstr(p[j, 0] == fleet_j, name=f"init_p_{j}")

    # Fix omega[r,s,0] = 0
    for (r, s) in ZxZ:
        model.addConstr(omega[r, s, 0] == 0, name=f"init_omega_{r}_{s}")

    # ----------------------------------------------------------------
    # Constraint (34): N_U evolution for A_o only
    # N_U[i,j,s,t+1] = N_U[i,j,s,t] + sum_{(j,k) in gamma_plus[j]} y_turn[i,j,k,s,t]
    # Note: A_z_plus links are handled by constraint (46) using y_cent instead.
    # ----------------------------------------------------------------
    for (i, j) in A_o:
        for s in Z_list:
            for t in range(T):
                outflow = gp.quicksum(
                    get_y_turn(i, j, k, s, t)
                    for (j2, k) in gamma_plus[j]
                )
                model.addConstr(
                    N_U[i, j, s, t + 1] == N_U[i, j, s, t] + outflow,
                    name=f"c34_{i}_{j}_{s}_{t}"
                )

    # ----------------------------------------------------------------
    # Constraint (35): N_D evolution for A_o only
    # N_D[j,k,s,t+1] = N_D[j,k,s,t] + sum_{(i,j) in gamma_minus[j] ∩ A_o_plus} y_turn[i,j,k,s,t]
    # Note: A_z_minus is handled by constraint (47), A_z_plus by constraint (46b).
    # ----------------------------------------------------------------
    for (j, k) in A_o:
        # Incoming links to j that are in A_o_plus (since y_turn only defined there)
        incoming = [(i2, j2) for (i2, j2) in gamma_minus[j] if (i2, j2) in A_o_plus]
        for s in Z_list:
            for t in range(T):
                inflow = gp.quicksum(
                    get_y_turn(i2, j, k, s, t)
                    for (i2, j2) in incoming
                )
                model.addConstr(
                    N_D[j, k, s, t + 1] == N_D[j, k, s, t] + inflow,
                    name=f"c35_{j}_{k}_{s}_{t}"
                )

    # ----------------------------------------------------------------
    # Constraint (36): Sending flow constraint for A_o only
    # sum_{(j,k)} y_turn[i,j,k,s,t] <= N_U[i,j,s,t - fftt + 1] - N_D[i,j,s,t]
    # for t in [fftt-1, T]
    # Note: A_z_plus links use centroid departure flow (constraint 45/46).
    # ----------------------------------------------------------------
    for (i, j) in A_o:
        tau = fftt[(i, j)]
        for s in Z_list:
            for t in range(tau - 1, T + 1):
                lhs = gp.quicksum(
                    get_y_turn(i, j, k, s, t)
                    for (j2, k) in gamma_plus[j]
                )
                model.addConstr(
                    lhs <= N_U[i, j, s, t - tau + 1] - N_D[i, j, s, t],
                    name=f"c36_{i}_{j}_{s}_{t}"
                )

    # ----------------------------------------------------------------
    # Constraint (37): Zero flow before free-flow time
    # y_turn[i,j,k,s,t] = 0 for t < fftt[(i,j)] - 1
    # ----------------------------------------------------------------
    for (i, j) in A_o_plus:
        tau = fftt[(i, j)]
        if tau <= 1:
            continue  # range(0, 0) is empty
        for (j2, k) in gamma_plus[j]:
            for s in Z_list:
                for t in range(0, tau - 1):
                    var = get_y_turn(i, j, k, s, t)
                    if isinstance(var, gp.Var):
                        model.addConstr(
                            var == 0,
                            name=f"c37_{i}_{j}_{k}_{s}_{t}"
                        )

    # ----------------------------------------------------------------
    # Constraint (38): Sending capacity for A_o
    # sum_{s, (j,k)} y_turn[i,j,k,s,t] <= Q[(i,j)]
    # ----------------------------------------------------------------
    for (i, j) in A_o:
        cap = Q[(i, j)]
        for t in time_range:
            lhs = gp.quicksum(
                get_y_turn(i, j, k, s, t)
                for (j2, k) in gamma_plus[j]
                for s in Z_list
            )
            model.addConstr(lhs <= cap, name=f"c38_{i}_{j}_{t}")

    # ----------------------------------------------------------------
    # Constraint (39): Receiving capacity for A_o
    # sum_{(i,j) in gamma_minus[j] ∩ A_o_plus, s} y_turn[i,j,k,s,t] <= Q[(j,k)]
    # ----------------------------------------------------------------
    for (j, k) in A_o:
        cap = Q[(j, k)]
        incoming = [(i2, j2) for (i2, j2) in gamma_minus[j] if (i2, j2) in A_o_plus]
        for t in time_range:
            lhs = gp.quicksum(
                get_y_turn(i2, j, k, s, t)
                for (i2, j2) in incoming
                for s in Z_list
            )
            model.addConstr(lhs <= cap, name=f"c39_{j}_{k}_{t}")

    # ----------------------------------------------------------------
    # Constraint (40): Receiving congested wave constraint for A_o
    # sum_{(i,j), s} y_turn[i,j,k,s,t] <= sum_s (N_U[j,k,s,t-cwtt+1] - N_D[j,k,s,t]) + KL[(j,k)]
    # for t >= cwtt - 1
    # ----------------------------------------------------------------
    for (j, k) in A_o:
        w = cwtt[(j, k)]
        kl = KL[(j, k)]
        incoming = [(i2, j2) for (i2, j2) in gamma_minus[j] if (i2, j2) in A_o_plus]
        for t in range(w - 1, T + 1):
            lhs = gp.quicksum(
                get_y_turn(i2, j, k, s, t)
                for (i2, j2) in incoming
                for s in Z_list
            )
            rhs = gp.quicksum(
                N_U[j, k, s, t - w + 1] - N_D[j, k, s, t]
                for s in Z_list
            ) + kl
            model.addConstr(lhs <= rhs, name=f"c40_{j}_{k}_{t}")

    # ----------------------------------------------------------------
    # Constraint (43): Parking evolution
    # p[j,t+1] = p[j,t]
    #   + sum_{(i,j) in gamma_minus[j]} (N_U[i,j,j,t] - N_D[i,j,j,t])   (arrivals)
    #   - sum_{(j,k) in gamma_plus[j], s} y_cent[j,k,s,t]                (departures)
    # ----------------------------------------------------------------
    for j in Z_list:
        for t in range(T):
            # Arrivals: vehicles with destination j arriving at centroid j
            arrivals = gp.quicksum(
                N_U[i2, j, j, t] - N_D[i2, j, j, t]
                for (i2, j2) in gamma_minus[j]
            )
            # Departures: vehicles leaving centroid j
            departures = gp.quicksum(
                y_cent[j, k, s, t]
                for (j2, k) in gamma_plus[j]
                for s in Z_list
            )
            model.addConstr(
                p[j, t + 1] == p[j, t] + arrivals - departures,
                name=f"c43_{j}_{t}"
            )

    # ----------------------------------------------------------------
    # Constraint (44): No through-flow on A_z_minus for wrong destinations
    # Already handled by not creating y_turn variables for s != k
    # when (j,k) in A_z_minus. No additional constraints needed.
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # Constraint (45): Outgoing flow bounded by parking
    # sum_{(i,j) in gamma_plus[i], s} y_cent[i,j,s,t] <= p[i,t]
    # ----------------------------------------------------------------
    for i in Z_list:
        for t in time_range:
            lhs = gp.quicksum(
                y_cent[i, j, s, t]
                for (i2, j) in gamma_plus[i]
                for s in Z_list
            )
            model.addConstr(lhs <= p[i, t], name=f"c45_{i}_{t}")

    # ----------------------------------------------------------------
    # Constraint (46): N_U evolution for A_z_plus using centroid departure flow
    # N_U[i,j,s,t+1] = N_U[i,j,s,t] + y_cent[i,j,s,t]
    # ----------------------------------------------------------------
    for (i, j) in A_z_plus:
        for s in Z_list:
            for t in range(T):
                model.addConstr(
                    N_U[i, j, s, t + 1] == N_U[i, j, s, t] + y_cent[i, j, s, t],
                    name=f"c46_{i}_{j}_{s}_{t}"
                )

    # ----------------------------------------------------------------
    # Constraint (46b): N_D evolution for A_z_plus (fftt = 1 for centroid connectors)
    # N_D[i,j,s,t+1] = N_U[i,j,s,t]
    # ----------------------------------------------------------------
    for (i, j) in A_z_plus:
        for s in Z_list:
            for t in range(T):
                model.addConstr(
                    N_D[i, j, s, t + 1] == N_U[i, j, s, t],
                    name=f"c46b_{i}_{j}_{s}_{t}"
                )

    # ----------------------------------------------------------------
    # Constraint (47): N_D evolution for A_z_minus
    # N_D[i,j,s,t+1] = N_U[i,j,s,t]
    # ----------------------------------------------------------------
    for (i, j) in A_z_minus:
        for s in Z_list:
            for t in range(T):
                model.addConstr(
                    N_D[i, j, s, t + 1] == N_U[i, j, s, t],
                    name=f"c47_{i}_{j}_{s}_{t}"
                )

    # ----------------------------------------------------------------
    # Constraint (48): Fleet conservation
    # sum_i p[i, T] == total_fleet
    # ----------------------------------------------------------------
    total_fleet = sum(fleet.get(str(j), fleet.get(j, 0)) for j in Z_list)
    model.addConstr(
        gp.quicksum(p[j, T] for j in Z_list) == total_fleet,
        name="c48_fleet_conservation"
    )

    # ----------------------------------------------------------------
    # Constraint (49): Departing travelers bounded by waiting demand
    # e[r,s,t] <= omega[r,s,t]
    # ----------------------------------------------------------------
    for (r, s) in ZxZ:
        for t in time_range:
            model.addConstr(
                e[r, s, t] <= omega[r, s, t],
                name=f"c49_{r}_{s}_{t}"
            )

    # ----------------------------------------------------------------
    # Constraint (50): Departing travelers bounded by departing vehicles
    # e[r,s,t] <= sum_{(r,j) in gamma_plus[r]} y_cent[r,j,s,t]
    # ----------------------------------------------------------------
    for (r, s) in ZxZ:
        for t in time_range:
            rhs = gp.quicksum(
                y_cent[r, j, s, t]
                for (r2, j) in gamma_plus[r]
            )
            model.addConstr(
                e[r, s, t] <= rhs,
                name=f"c50_{r}_{s}_{t}"
            )

    # ----------------------------------------------------------------
    # Constraint (51): Waiting demand evolution
    # omega[r,s,t+1] = omega[r,s,t] + d[r,s,t] - e[r,s,t]
    # ----------------------------------------------------------------
    for (r, s) in ZxZ:
        for t in range(T):
            demand_rst = d.get((r, s, t), 0)
            model.addConstr(
                omega[r, s, t + 1] == omega[r, s, t] + demand_rst - e[r, s, t],
                name=f"c51_{r}_{s}_{t}"
            )

    # ----------------------------------------------------------------
    # Constraint (52): All demand served by T
    # omega[r,s,T] = 0
    # ----------------------------------------------------------------
    for (r, s) in ZxZ:
        model.addConstr(omega[r, s, T] == 0, name=f"c52_{r}_{s}")

    # ----------------------------------------------------------------
    # Objective (33): Minimize total system time
    # sum_{(i,j), s, t} (N_U[i,j,s,t] - N_D[i,j,s,t])
    # + sum_{(r,s), t} omega[r,s,t]
    # ----------------------------------------------------------------
    obj_network = gp.quicksum(
        N_U[i, j, s, t] - N_D[i, j, s, t]
        for (i, j) in A_all
        for s in Z_list
        for t in time_range
    )
    obj_waiting = gp.quicksum(
        omega[r, s, t]
        for (r, s) in ZxZ
        for t in time_range
    )
    model.setObjective(obj_network + obj_waiting, GRB.MINIMIZE)

    # ----------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------
    model.optimize()

    # ----------------------------------------------------------------
    # Output solution
    # ----------------------------------------------------------------
    result = {}

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        try:
            result["objective_value"] = model.ObjVal
        except Exception:
            result["objective_value"] = None
    else:
        result["objective_value"] = None

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.LOADED: "LOADED",
        GRB.CUTOFF: "CUTOFF",
    }
    result["status"] = status_map.get(model.status, f"UNKNOWN_{model.status}")

    result["variables"] = {
        "num_y_turn": len(y_turn_keys),
        "num_y_cent": len(y_cent_keys),
        "num_N_U": len(NU_keys),
        "num_N_D": len(NU_keys),
        "num_p": len(p_keys),
        "num_e": len(e_keys),
        "num_omega": len(omega_keys),
        "total_variables": model.NumVars,
        "total_constraints": model.NumConstrs,
    }

    # Export variable values for feasibility checking
    if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
        # y_turn: key "i|j|k|s|t"
        y_dict = {}
        for key, var in y_turn.items():
            i, j, k, s, t = key
            val = var.X
            if abs(val) > 1e-9:
                y_dict[f"{i}|{j}|{k}|{s}|{t}"] = val
        result["y"] = y_dict

        # y_centroid: key "i|j|s|t"
        yc_dict = {}
        for key, var in y_cent.items():
            i, j, s, t = key
            val = var.X
            if abs(val) > 1e-9:
                yc_dict[f"{i}|{j}|{s}|{t}"] = val
        result["y_centroid"] = yc_dict

        # N_U: key "i|j|s|t"
        nu_dict = {}
        for key, var in N_U.items():
            i, j, s, t = key
            val = var.X
            if abs(val) > 1e-9:
                nu_dict[f"{i}|{j}|{s}|{t}"] = val
        result["N_U"] = nu_dict

        # N_D: key "i|j|s|t"
        nd_dict = {}
        for key, var in N_D.items():
            i, j, s, t = key
            val = var.X
            if abs(val) > 1e-9:
                nd_dict[f"{i}|{j}|{s}|{t}"] = val
        result["N_D"] = nd_dict

        # p: key "j|t"
        p_dict = {}
        for key, var in p.items():
            j, t = key
            val = var.X
            if abs(val) > 1e-9:
                p_dict[f"{j}|{t}"] = val
        result["p"] = p_dict

        # e: key "r|s|t"
        e_dict = {}
        for key, var in e.items():
            r, s, t = key
            val = var.X
            if abs(val) > 1e-9:
                e_dict[f"{r}|{s}|{t}"] = val
        result["e"] = e_dict

        # omega: key "r|s|t"
        omega_dict = {}
        for key, var in omega.items():
            r, s, t = key
            val = var.X
            if abs(val) > 1e-9:
                omega_dict[f"{r}|{s}|{t}"] = val
        result["omega"] = omega_dict

    # Write solution
    sol_dir = os.path.dirname(args.solution_path)
    if sol_dir:
        os.makedirs(sol_dir, exist_ok=True)
    with open(args.solution_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Status: {result['status']}")
    print(f"Objective: {result['objective_value']}")
    print(f"Variables: {model.NumVars}, Constraints: {model.NumConstrs}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
