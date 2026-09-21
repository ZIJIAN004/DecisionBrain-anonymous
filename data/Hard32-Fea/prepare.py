#!/usr/bin/env python3
"""Download pinned sources and reconstruct the 32 Hard32-Fea instances."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "instances"
SELECTION = ROOT / "benchmarks" / "Hard32-Fea" / "selection.json"
JOBSHOP_COMMIT = "460510f197744eed1cbcbbdfd6ec3252252f412c"
PYVRP_COMMIT = "7474b068a8effa7f39448b241314b615c9271525"
JOBSHOP_URL = (
    "https://raw.githubusercontent.com/Pabloo22/job_shop_lib/"
    f"{JOBSHOP_COMMIT}/job_shop_lib/benchmarking/benchmark_instances.json"
)
PYVRP_BASE = (
    "https://raw.githubusercontent.com/PyVRP/Instances/"
    f"{PYVRP_COMMIT}"
)


def download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "DecisionBrain-data-preparer"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(  # noqa: S310 - pinned HTTPS sources
                request, timeout=30
            ) as response:
                return response.read().decode("utf-8")
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_vrplib(text: str) -> tuple[dict[str, str], dict[str, list[list[int]]]]:
    headers: dict[str, str] = {}
    sections: dict[str, list[list[int]]] = {}
    current: list[list[int]] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "EOF":
            continue
        if line.endswith("_SECTION"):
            current = sections.setdefault(line, [])
        elif current is not None:
            values = [int(value) for value in line.split()]
            if values == [-1]:
                current = None
            else:
                current.append(values)
        elif ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    return headers, sections


def indexed_values(rows: list[list[int]], width: int) -> list[list[int]]:
    ordered = sorted(rows, key=lambda row: row[0])
    if any(len(row) != width + 1 for row in ordered):
        raise ValueError("unexpected VRPLIB section width")
    if [row[0] for row in ordered] != list(range(1, len(ordered) + 1)):
        raise ValueError("VRPLIB node identifiers are not contiguous from one")
    return [row[1:] for row in ordered]


def build_jssp(source: dict, row: dict) -> dict:
    jobs = []
    for job_id, (machines, durations) in enumerate(
        zip(source["machines_matrix"], source["duration_matrix"], strict=True)
    ):
        jobs.append(
            {
                "job_id": job_id,
                "operations": [
                    {"machine": machine, "processing_time": duration}
                    for machine, duration in zip(machines, durations, strict=True)
                ],
            }
        )
    return {
        "problem_type": "JSSP_DEADLINE",
        "num_jobs": len(jobs),
        "num_machines": len(source["machines_matrix"][0]),
        "num_operations": sum(len(job["operations"]) for job in jobs),
        "deadline": int(row["limit"]),
        "jobs": jobs,
    }


def build_vrptw(text: str, row: dict) -> dict:
    headers, sections = parse_vrplib(text)
    coordinates = indexed_values(sections["NODE_COORD_SECTION"], 2)
    demands = [values[0] for values in indexed_values(sections["DEMAND_SECTION"], 1)]
    windows = indexed_values(sections["TIME_WINDOW_SECTION"], 2)
    return {
        "name": row["base_instance"],
        "num_customers": len(coordinates) - 1,
        "capacity": int(headers["CAPACITY"]),
        "max_vehicles": int(row["limit"]),
        "service_time": int(headers["SERVICE_TIME"]) * 10,
        "coordinates": coordinates,
        "demand": demands,
        "time_window": [[10 * start, 10 * end] for start, end in windows],
        "fleet_lower_bound": int(row["fleet_lower_bound"]),
    }


def build_pdptw(text: str, row: dict) -> dict:
    headers, sections = parse_vrplib(text)
    coordinates = indexed_values(sections["NODE_COORD_SECTION"], 2)
    records = indexed_values(sections["PICKUP_AND_DELIVERY_SECTION"], 6)
    demands = [record[0] for record in records]
    windows = [[record[1], record[2]] for record in records]
    service = [record[3] for record in records]
    requests = [
        [node_id, record[5] - 1]
        for node_id, record in enumerate(records)
        if record[5] != 0
    ]
    return {
        "num_nodes": len(coordinates),
        "num_requests": len(requests),
        "capacity": int(headers["CAPACITY"]),
        "coordinates": coordinates,
        "demand": demands,
        "time_window": windows,
        "service_time": service,
        "requests": requests,
        "name": row.get("base_instance", row["instance_name"]),
        "max_vehicles": int(row["limit"]),
    }


def pyvrp_path(row: dict) -> str:
    name = row.get("base_instance", row["instance_name"])
    if row["task"] == "vrptw_minfleet":
        customers = int(row["num_customers"])
        return f"VRPTW/GH{customers}/{name}.vrp"
    return f"PDPTW/{row['tier']}/{name}.vrp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = json.loads(SELECTION.read_text(encoding="utf-8"))
    if len(rows) != 32:
        raise SystemExit(f"expected 32 selected cases, found {len(rows)}")
    jobshop = json.loads(download_text(JOBSHOP_URL))
    source_cache: dict[str, str] = {}
    written = 0
    for row in rows:
        task = row["task"]
        if task == "jssp_deadline":
            payload = build_jssp(jobshop[row["base_instance"]], row)
        else:
            source_path = pyvrp_path(row)
            if source_path not in source_cache:
                source_cache[source_path] = download_text(f"{PYVRP_BASE}/{source_path}")
            if task == "vrptw_minfleet":
                payload = build_vrptw(source_cache[source_path], row)
            elif task == "pdptw_minfleet":
                payload = build_pdptw(source_cache[source_path], row)
            else:
                raise SystemExit(f"unsupported task: {task}")
        destination = (
            args.output
            / task
            / "instance"
            / f"large_instance_{int(row['instance_index'])}.json"
        )
        if destination.exists() and not args.force:
            raise SystemExit(f"refusing to overwrite {destination}; pass --force")
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        expected_bytes = int(row["instance_bytes"])
        if len(encoded) != expected_bytes:
            raise SystemExit(
                f"generated size mismatch for {row['instance_name']}: "
                f"expected {expected_bytes}, got {len(encoded)}"
            )
        destination.write_bytes(encoded)
        written += 1
    print(f"prepared and verified {written} Hard32-Fea instances in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
