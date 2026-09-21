"""
Feasibility checker for the job-shop scheduling problem with a common deadline
(JSSP_DEADLINE).

A proposed solution is FEASIBLE iff it is a machine-feasible, precedence-valid,
complete schedule (every job exactly once, operations in instance order, each
machine processes at most one operation at a time) whose makespan does not exceed
the instance deadline, and whose reported objective_value equals the recomputed
makespan (anti-cheat, mirroring the objective-consistency check used by FrontierOR
checkers).

CLI contract (FrontierOR):
    python feasibility_check.py --instance_path I --solution_path S --result_path R
Writes to R: {"feasible": bool, "violated_constraints": [...], "violations": [...],
             "violation_magnitudes": [...]}
"""
import argparse
import json

EPS = 1e-5


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_feasibility(instance_path, solution_path, result_path):
    data = load_json(instance_path)
    sol = load_json(solution_path)

    violations = []
    magnitudes = []

    def bad(name, msg, mag=0.0):
        violations.append("[{}] {}".format(name, msg))
        magnitudes.append(float(mag))

    # --- instance side -----------------------------------------------------
    num_jobs = data["num_jobs"]
    deadline = data["deadline"]
    inst_jobs = {j["job_id"]: j["operations"] for j in data["jobs"]}
    if len(inst_jobs) != num_jobs:
        bad("INSTANCE", "jobs[] does not match num_jobs")

    # --- solution side -----------------------------------------------------
    if not isinstance(sol.get("schedule"), list):
        bad("C1_STRUCTURE", "solution.schedule must be a list of per-job schedules")
        sol_sched = []
    else:
        sol_sched = sol["schedule"]

    if len(sol_sched) != num_jobs:
        bad("C1_STRUCTURE", "schedule covers {} jobs but instance has {}".format(
            len(sol_sched), num_jobs))

    seen_jobs = set()
    comp = {}   # job_id -> completion time
    machine_intervals = {}  # machine -> list of (start, end)

    for entry in sol_sched:
        jid = entry.get("job_id")
        if jid in seen_jobs:
            bad("C1_STRUCTURE", "job {} scheduled more than once".format(jid))
            continue
        if jid not in inst_jobs:
            bad("C1_STRUCTURE", "job {} not present in instance".format(jid))
            continue
        seen_jobs.add(jid)
        ops = entry.get("operations") or []
        inst_ops = inst_jobs[jid]
        if len(ops) != len(inst_ops):
            bad("C2_JOB_SEQUENCE", "job {} has {} scheduled operations, instance expects {}".format(
                jid, len(ops), len(inst_ops)))
            continue

        prev_end = None
        for k, op in enumerate(ops):
            machine = op.get("machine")
            start = op.get("start_time")
            if not isinstance(start, (int, float)):
                bad("C2_JOB_SEQUENCE", "job {} op {} start_time missing".format(jid, k))
                continue
            dur = inst_ops[k]["processing_time"]
            if inst_ops[k]["machine"] != machine:
                bad("C2_JOB_SEQUENCE", "job {} op {} on machine {}, instance requires machine {}".format(
                    jid, k, machine, inst_ops[k]["machine"]))
            if start < -EPS:
                bad("C2_JOB_SEQUENCE", "job {} op {} start_time {} < 0".format(jid, k, start))
            end = start + dur
            if prev_end is not None and start < prev_end - EPS:
                mag = prev_end - start
                bad("C3_PRECEDENCE", "job {} op {} starts {} before previous op finishes {}".format(
                    jid, k, start, prev_end), mag)
            prev_end = end
            machine_intervals.setdefault(machine, []).append((start, end, jid, k))

        last = ops[-1]
        cj = last["start_time"] + inst_ops[-1]["processing_time"]
        comp[jid] = cj
        comp_field = entry.get("completion_time")
        if comp_field is None or abs(comp_field - cj) > EPS:
            bad("C4_COMPLETION", "job {} completion_time {} != recomputed {}".format(
                jid, comp_field, cj))

    # --- machine capacity --------------------------------------------------
    for machine, intervals in machine_intervals.items():
        intervals.sort(key=lambda iv: (iv[0], iv[1]))
        for i in range(1, len(intervals)):
            p_start, p_end, _, _ = intervals[i - 1]
            c_start, _, jid, k = intervals[i]
            if c_start < p_end - EPS:
                mag = p_end - c_start
                bad("C5_CAPACITY", "machine {} overlap: job {} op {} starts {} while machine busy until {}".format(
                    machine, jid, k, c_start, p_end), mag)

    # --- deadline + objective ----------------------------------------------
    makespan = max(comp.values()) if comp else 0.0
    if makespan > deadline + EPS:
        bad("C6_DEADLINE", "makespan {} exceeds deadline {}".format(makespan, deadline),
            makespan - deadline)
    obj = sol.get("objective_value")
    if obj is None:
        bad("C7_OBJECTIVE", "solution.objective_value missing")
    else:
        try:
            obj_f = float(obj)
        except (TypeError, ValueError):
            bad("C7_OBJECTIVE", "objective_value {!r} is not numeric".format(obj))
        else:
            if abs(obj_f - makespan) > EPS:
                bad("C7_OBJECTIVE", "objective_value {} != recomputed makespan {}".format(obj, makespan),
                    abs(obj_f - makespan))

    feasible = not violations
    unique_violated = []
    for v in violations:
        name = v.split("]")[0][1:].strip() if "]" in v else "UNKNOWN"
        if name not in unique_violated:
            unique_violated.append(name)

    result = {
        "feasible": feasible,
        "violated_constraints": unique_violated,
        "violations": violations,
        "violation_magnitudes": magnitudes if not feasible else [],
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Feasibility: {}".format("FEASIBLE" if feasible else "INFEASIBLE"))
    for v in violations:
        print("  - " + v)
    return result


def main():
    ap = argparse.ArgumentParser(description="Feasibility checker for JSSP with deadline")
    ap.add_argument("--instance_path", required=True, help="Path to instance JSON")
    ap.add_argument("--solution_path", required=True, help="Path to candidate solution JSON")
    ap.add_argument("--result_path", required=True, help="Path to write feasibility result JSON")
    args = ap.parse_args()
    check_feasibility(args.instance_path, args.solution_path, args.result_path)


if __name__ == "__main__":
    main()
