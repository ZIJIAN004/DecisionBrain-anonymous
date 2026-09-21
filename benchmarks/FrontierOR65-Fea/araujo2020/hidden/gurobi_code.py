#!/usr/bin/env python3
"""
Gurobi implementation of the MMRCPSP / MMRCMPSP formulation from:
  Araujo et al. (2019): "Strong Bounds for Resource Constrained Project
  Scheduling: Preprocessing and Cutting Planes"

Implements the PDT (Pulse Discrete Time) MILP formulation (Eqs. 7-16)
with preprocessing for time window reduction (Eqs. 4-6).

NOT SPECIFIED IN PAPER:
  - epsilon value in Eq. (7): inferred as 1e-6 (small tiebreaker coefficient)
  - beta_p upper bound source: inferred as sum of max-mode durations (serial schedule)
"""

import argparse
import json
from collections import defaultdict, deque

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
    with open(path) as f:
        return json.load(f)


def topological_sort(job_ids, successors_map):
    """Return topological order of job_ids given successors_map."""
    predecessors = defaultdict(list)
    for jid in job_ids:
        for s in successors_map.get(jid, []):
            predecessors[s].append(jid)

    in_degree = {jid: len(predecessors[jid]) for jid in job_ids}
    queue = deque([jid for jid in job_ids if in_degree[jid] == 0])
    order = []
    while queue:
        jid = queue.popleft()
        order.append(jid)
        for s in successors_map.get(jid, []):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    return order


def compute_cpm_forward(jobs_data, project):
    """
    Forward CPM pass: compute earliest start times using minimum (fastest) mode durations.
    Returns dict {job_id: earliest_start_time}.
    """
    sigma = project['release_date']
    pid = project['project_id']
    job_ids = [j['job_id'] for j in jobs_data if j['project_id'] == pid]
    successors_map = {j['job_id']: list(j['successors'])
                      for j in jobs_data if j['project_id'] == pid}
    min_dur = {j['job_id']: min(m['duration'] for m in j['modes'])
               for j in jobs_data if j['project_id'] == pid}

    topo = topological_sort(job_ids, successors_map)
    est = {jid: sigma for jid in job_ids}
    for jid in topo:
        for s in successors_map.get(jid, []):
            est[s] = max(est[s], est[jid] + min_dur[jid])
    return est


def compute_back_cpm(jobs_data, project):
    """
    Backward pass: compute longest path from each job to sink using minimum durations.
    back[j] = total min-duration path length from j to project sink.
    Used to compute L_{jm} and latest start times.
    """
    pid = project['project_id']
    job_ids = [j['job_id'] for j in jobs_data if j['project_id'] == pid]
    successors_map = {j['job_id']: list(j['successors'])
                      for j in jobs_data if j['project_id'] == pid}
    min_dur = {j['job_id']: min(m['duration'] for m in j['modes'])
               for j in jobs_data if j['project_id'] == pid}

    topo = topological_sort(job_ids, successors_map)
    back = {jid: 0 for jid in job_ids}
    for jid in reversed(topo):
        succs = successors_map.get(jid, [])
        if succs:
            back[jid] = min_dur[jid] + max(back[s] for s in succs)
        else:
            back[jid] = min_dur[jid]
    return back


def compute_beta_ub(jobs_data, project):
    """
    Compute upper bound beta_p for project completion time.
    INFERRED ASSUMPTION: beta_p = sigma_p + sum of max-mode durations (worst-case
    serial schedule). This is always a valid upper bound since jobs can always
    be scheduled sequentially.
    """
    sigma = project['release_date']
    total = sum(
        max(m['duration'] for m in j['modes'])
        for j in jobs_data if j['project_id'] == project['project_id']
    )
    return sigma + total


def build_time_windows(jobs_data, project, est, back, sigma_p, lambda_p, alpha):
    """
    Compute T_{jm} = (earliest_start, latest_start) for each (job, mode).
    l_{jm} = sigma_p + lambda_p - L_{jm} + alpha   [Eq. 6]
    L_{jm} = d_{jm} + max_{s in S_j} back[s]       (d_{jm} + longest path from successors)
    """
    pid = project['project_id']
    successors_map = {j['job_id']: list(j['successors'])
                      for j in jobs_data if j['project_id'] == pid}

    T_jm = {}
    for j in jobs_data:
        if j['project_id'] != pid:
            continue
        jid = j['job_id']
        e_j = est[jid]
        succs = successors_map.get(jid, [])

        for m_data in j['modes']:
            mid = m_data['mode_id']
            d_jm = m_data['duration']
            # L_{jm}: d_{jm} + longest min-duration path from successors to sink
            if succs:
                L_jm = d_jm + max(back[s] for s in succs)
            else:
                L_jm = d_jm
            l_jm = sigma_p + lambda_p - L_jm + alpha
            # Ensure valid window; if l_jm < e_j, window is empty (mode infeasible)
            l_jm_int = int(max(e_j, l_jm))
            T_jm[(jid, mid)] = (int(e_j), l_jm_int)

    return T_jm


def build_and_solve(instance, time_limit):
    jobs_data = instance['jobs']
    projects = instance['projects']
    resources = instance['resources']
    renewable_res = resources.get('renewable', [])
    nonrenewable_res = resources.get('nonrenewable', [])

    # Project-level data
    sigma_map = {p['project_id']: p['release_date'] for p in projects}
    sink_map = {p['project_id']: p['artificial_sink_job_id'] for p in projects}

    est_all = {}
    back_all = {}
    lambda_map = {}
    beta_map = {}

    for p in projects:
        pid = p['project_id']
        est = compute_cpm_forward(jobs_data, p)
        back = compute_back_cpm(jobs_data, p)
        est_all.update(est)
        back_all.update(back)
        a_p = sink_map[pid]
        lambda_map[pid] = est[a_p] - sigma_map[pid]   # CPD [Eq. 3]
        beta_map[pid] = compute_beta_ub(jobs_data, p)

    # Compute alpha [Eq. 4]
    alpha = sum(
        beta_map[p['project_id']] - sigma_map[p['project_id']] - lambda_map[p['project_id']]
        for p in projects
    )
    alpha = max(0, alpha)

    # Compute t_check and T [Eq. 5]
    t_check = int(max(sigma_map[p['project_id']] + lambda_map[p['project_id']] + alpha
                       for p in projects))
    T_set = range(t_check + 1)

    # Build time windows T_{jm} [Eq. 6]
    T_jm = {}
    for p in projects:
        pid = p['project_id']
        tw = build_time_windows(
            jobs_data, p,
            {j['job_id']: est_all[j['job_id']] for j in jobs_data if j['project_id'] == pid},
            {j['job_id']: back_all[j['job_id']] for j in jobs_data if j['project_id'] == pid},
            sigma_map[pid], lambda_map[pid], alpha
        )
        T_jm.update(tw)

    # Build index structures
    job_mode_data = {j['job_id']: {m['mode_id']: m for m in j['modes']} for j in jobs_data}
    successors_map = {j['job_id']: list(j['successors']) for j in jobs_data}

    # -------------------------------------------------------------------
    # Build Gurobi model
    # -------------------------------------------------------------------
    model = gp.Model("MMRCPSP")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)

    # EPSILON: NOT EXPLICITLY SPECIFIED IN PAPER
    # Inferred assumption: epsilon = 1e-6 (small enough to be pure tiebreaker)
    epsilon = 1e-6

    # Decision variables x_{jmt} [Eq. 14]: binary, start-time indexed
    x = {}
    for j in jobs_data:
        jid = j['job_id']
        for m_data in j['modes']:
            mid = m_data['mode_id']
            e_j, l_jm = T_jm[(jid, mid)]
            for t in range(e_j, l_jm + 1):
                x[(jid, mid, t)] = model.addVar(vtype=GRB.BINARY, name=f"x_{jid}_{mid}_{t}")

    # Decision variables z_{jmt} [Eq. 15]: binary, processing-time indexed
    # z_{jmt}=1 if job j is being processed at time t in mode m
    # Domain: t in [e_j, l_jm + d_jm - 1] (processing window)
    z = {}
    for j in jobs_data:
        jid = j['job_id']
        for m_data in j['modes']:
            mid = m_data['mode_id']
            d_jm = m_data['duration']
            if d_jm == 0:
                continue  # Zero-duration jobs are never "being processed"
            e_j, l_jm = T_jm[(jid, mid)]
            for t in range(e_j, min(l_jm + d_jm, t_check + 1)):
                z[(jid, mid, t)] = model.addVar(vtype=GRB.BINARY, name=f"z_{jid}_{mid}_{t}")

    # h: integer makespan tiebreaker variable [Eq. 16]
    h = model.addVar(vtype=GRB.INTEGER, lb=0, name="h")
    model.update()

    # -------------------------------------------------------------------
    # Objective [Eq. 7]: minimize TPD + epsilon * h
    # -------------------------------------------------------------------
    obj = gp.LinExpr()
    for p in projects:
        pid = p['project_id']
        a_p = sink_map[pid]
        sigma_p = sigma_map[pid]
        lambda_p = lambda_map[pid]
        for m_data in job_mode_data[a_p].values():
            mid = m_data['mode_id']
            e_j, l_jm = T_jm[(a_p, mid)]
            for t in range(e_j, l_jm + 1):
                if (a_p, mid, t) in x:
                    obj += (t - (sigma_p + lambda_p)) * x[(a_p, mid, t)]
    obj += epsilon * h
    model.setObjective(obj, GRB.MINIMIZE)

    # -------------------------------------------------------------------
    # Constraint (8): Each job assigned to exactly one mode and start time
    # -------------------------------------------------------------------
    for j in jobs_data:
        jid = j['job_id']
        lhs = gp.LinExpr()
        for m_data in j['modes']:
            mid = m_data['mode_id']
            e_j, l_jm = T_jm[(jid, mid)]
            for t in range(e_j, l_jm + 1):
                if (jid, mid, t) in x:
                    lhs += x[(jid, mid, t)]
        model.addConstr(lhs == 1, name=f"assign_{jid}")

    # -------------------------------------------------------------------
    # Constraint (9): Non-renewable resource capacity
    # -------------------------------------------------------------------
    for k_data in nonrenewable_res:
        kid = k_data['resource_id']
        cap = k_data['capacity']
        lhs = gp.LinExpr()
        for j in jobs_data:
            jid = j['job_id']
            for m_data in j['modes']:
                mid = m_data['mode_id']
                q = m_data['nonrenewable_consumption'][kid]
                if q == 0:
                    continue
                e_j, l_jm = T_jm[(jid, mid)]
                for t in range(e_j, l_jm + 1):
                    if (jid, mid, t) in x:
                        lhs += q * x[(jid, mid, t)]
        model.addConstr(lhs <= cap, name=f"nonrenew_{kid}")

    # -------------------------------------------------------------------
    # Constraint (10): Renewable resource capacity for each time period
    # -------------------------------------------------------------------
    # Build index: for each t, which (j,m) z-variables are defined there?
    z_by_t = defaultdict(list)
    for (jid, mid, t) in z:
        z_by_t[t].append((jid, mid))

    for r_data in renewable_res:
        rid = r_data['resource_id']
        cap = r_data['capacity']
        for t in T_set:
            entries = z_by_t.get(t, [])
            if not entries:
                continue
            lhs = gp.LinExpr()
            for (jid, mid) in entries:
                q = job_mode_data[jid][mid]['renewable_consumption'][rid]
                lhs += q * z[(jid, mid, t)]
            if lhs.size() > 0:
                model.addConstr(lhs <= cap, name=f"renew_{rid}_{t}")

    # -------------------------------------------------------------------
    # Constraint (11): Aggregated precedence constraints
    # -------------------------------------------------------------------
    for j in jobs_data:
        jid = j['job_id']
        for s_id in successors_map.get(jid, []):
            s_modes = job_mode_data[s_id]
            lhs = gp.LinExpr()
            # (t + d_{jm}) * x_{jmt} for job j
            for m_data in j['modes']:
                mid = m_data['mode_id']
                d_jm = m_data['duration']
                e_j, l_jm = T_jm[(jid, mid)]
                for t in range(e_j, l_jm + 1):
                    if (jid, mid, t) in x:
                        lhs += (t + d_jm) * x[(jid, mid, t)]
            # - i * x_{szi} for successor s
            for m_data in s_modes.values():
                smid = m_data['mode_id']
                e_s, l_sm = T_jm[(s_id, smid)]
                for i in range(e_s, l_sm + 1):
                    if (s_id, smid, i) in x:
                        lhs -= i * x[(s_id, smid, i)]
            model.addConstr(lhs <= 0, name=f"prec_{jid}_{s_id}")

    # -------------------------------------------------------------------
    # Constraint (12): Linking z and x variables
    # z_{jmt} = sum_{t'=max(e_j, t-d_jm+1)}^{min(l_jm, t)} x_{jmt'}
    # -------------------------------------------------------------------
    for j in jobs_data:
        jid = j['job_id']
        for m_data in j['modes']:
            mid = m_data['mode_id']
            d_jm = m_data['duration']
            if d_jm == 0:
                continue  # No linking needed for zero-duration jobs
            e_j, l_jm = T_jm[(jid, mid)]
            for t in range(e_j, min(l_jm + d_jm, t_check + 1)):
                if (jid, mid, t) not in z:
                    continue
                rhs = gp.LinExpr()
                t_lo = max(e_j, t - d_jm + 1)
                t_hi = min(l_jm, t)
                for tp in range(t_lo, t_hi + 1):
                    if (jid, mid, tp) in x:
                        rhs += x[(jid, mid, tp)]
                model.addConstr(z[(jid, mid, t)] - rhs == 0, name=f"link_{jid}_{mid}_{t}")

    # -------------------------------------------------------------------
    # Constraint (13): Makespan computation (tiebreaker)
    # h >= t * x_{a_p, m, t} for all projects p, modes m, times t
    # -------------------------------------------------------------------
    for p in projects:
        pid = p['project_id']
        a_p = sink_map[pid]
        lhs = gp.LinExpr()
        lhs += h
        for m_data in job_mode_data[a_p].values():
            mid = m_data['mode_id']
            e_j, l_jm = T_jm[(a_p, mid)]
            for t in range(e_j, l_jm + 1):
                if (a_p, mid, t) in x:
                    lhs -= t * x[(a_p, mid, t)]
        model.addConstr(lhs >= 0, name=f"makespan_{pid}")

    # -------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------
    model.optimize()

    # -------------------------------------------------------------------
    # Extract solution
    # -------------------------------------------------------------------
    solution = {}
    status = model.Status

    if status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:
        # Compute TPD (objective without epsilon*h term)
        tpd_val = 0.0
        schedule = []
        for p in projects:
            pid = p['project_id']
            a_p = sink_map[pid]
            sigma_p = sigma_map[pid]
            lambda_p = lambda_map[pid]
            for m_data in job_mode_data[a_p].values():
                mid = m_data['mode_id']
                e_j, l_jm = T_jm[(a_p, mid)]
                for t in range(e_j, l_jm + 1):
                    if (a_p, mid, t) in x and x[(a_p, mid, t)].X > 0.5:
                        tpd_val += t - (sigma_p + lambda_p)

        for j in jobs_data:
            jid = j['job_id']
            for m_data in j['modes']:
                mid = m_data['mode_id']
                e_j, l_jm = T_jm[(jid, mid)]
                for t in range(e_j, l_jm + 1):
                    if (jid, mid, t) in x and x[(jid, mid, t)].X > 0.5:
                        schedule.append({'job_id': jid, 'mode_id': mid, 'start_time': t})

        solution['objective_value'] = tpd_val
        solution['status'] = 'optimal' if status == GRB.OPTIMAL else 'feasible'
        solution['schedule'] = schedule
        solution['makespan'] = int(round(h.X))
        solution['solver_runtime_seconds'] = model.Runtime
    else:
        solution['objective_value'] = None
        solution['status'] = 'infeasible_or_no_solution'
        solution['solver_runtime_seconds'] = model.Runtime

    return solution


def main():
    parser = argparse.ArgumentParser(description="Gurobi MILP solver for MMRCPSP (Araujo et al. 2019)")
    parser.add_argument("--instance_path", required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", required=True,
                        help="Path to write the solution JSON file")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    instance = load_instance(args.instance_path)
    solution = build_and_solve(instance, args.time_limit)

    with open(args.solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"Solution written to {args.solution_path}")
    print(f"Objective value (TPD): {solution.get('objective_value')}")
    print(f"Status: {solution.get('status')}")


if __name__ == "__main__":
    main()
