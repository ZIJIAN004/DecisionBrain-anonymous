#!/usr/bin/env python3
"""Verify or download the pinned FrontierOR65-Fea data files."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = ROOT / "benchmarks" / "FrontierOR65-Fea" / "index.json"
LOCK = HERE / "sources.lock.json"
DEFAULT_OUTPUT = HERE / "dataset"
DEFAULT_MANIFEST = HERE / "remote_manifest.json"
API_ROOT = "https://huggingface.co/api/datasets/SmartOR/FrontierOR"
RESOLVE_ROOT = "https://huggingface.co/datasets/SmartOR/FrontierOR/resolve"


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DecisionBrain-data-fetcher"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.read()
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def directory_tree(revision: str, path: str) -> list[dict]:
    revision_part = urllib.parse.quote(revision, safe="")
    path_part = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    url = f"{API_ROOT}/tree/{revision_part}/{path_part}?expand=true&limit=1000"
    payload = json.loads(request_bytes(url))
    if not isinstance(payload, list):
        raise ValueError(f"unexpected tree response for {path}")
    return payload


def file_record(entry: dict) -> dict:
    record = {
        "path": entry["path"],
        "size": int(entry["size"]),
        "git_oid": entry.get("oid"),
    }
    if entry.get("lfs"):
        record["sha256"] = entry["lfs"]["oid"]
    if entry.get("xetHash"):
        record["xet_hash"] = entry["xetHash"]
    return record


def inspect_case(revision: str, paper_id: str, case: dict) -> tuple[str, dict]:
    index = int(case["instance_index"])
    instance_path = f"{paper_id}/instance/large_instance_{index}.json"
    solution_path = f"{paper_id}/gurobi_solution/large_solution_{index}.json"
    try:
        entries = directory_tree(revision, f"{paper_id}/instance")
        entries += directory_tree(revision, f"{paper_id}/gurobi_solution")
    except Exception as error:
        raise RuntimeError(f"failed to inspect remote case {paper_id}: {error}") from error
    files = {entry["path"]: entry for entry in entries if entry.get("type") == "file"}
    if instance_path not in files:
        raise FileNotFoundError(f"remote instance missing: {instance_path}")
    if solution_path not in files:
        raise FileNotFoundError(f"remote reference solution missing: {solution_path}")
    instance = file_record(files[instance_path])
    expected_size = int(case["instance_bytes"])
    if instance["size"] != expected_size:
        raise ValueError(
            f"remote size mismatch for {instance_path}: "
            f"index={expected_size}, remote={instance['size']}"
        )
    return paper_id, {
        "instance_index": index,
        "instance": instance,
        "reference_solution": file_record(files[solution_path]),
    }


def remote_manifest(workers: int) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    suite = json.loads(INDEX.read_text(encoding="utf-8"))
    revision = lock["source"]["revision"]
    cases = suite["cases"]
    if suite["case_count"] != 65 or len(cases) != 65:
        raise ValueError("FrontierOR65-Fea index must contain exactly 65 cases")
    inspected: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(inspect_case, revision, paper_id, case): paper_id
            for paper_id, case in cases.items()
        }
        for future in concurrent.futures.as_completed(futures):
            paper_id, record = future.result()
            inspected[paper_id] = record
    ordered = {paper_id: inspected[paper_id] for paper_id in cases}
    return {
        "schema_version": 1,
        "dataset": "FrontierOR65-Fea",
        "source": lock["source"]["dataset"],
        "revision": revision,
        "case_count": len(ordered),
        "total_instance_bytes": sum(item["instance"]["size"] for item in ordered.values()),
        "cases": ordered,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(revision: str, record: dict, output: Path, force: bool) -> None:
    destination = output / record["path"]
    expected_hash = record.get("sha256")
    if destination.is_file() and destination.stat().st_size == record["size"]:
        if not expected_hash or sha256(destination) == expected_hash:
            return
    if destination.exists() and not force:
        raise FileExistsError(f"invalid existing file; pass --force: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in record["path"].split("/"))
    url = f"{RESOLVE_ROOT}/{urllib.parse.quote(revision, safe='')}/{encoded_path}?download=true"
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(  # noqa: S310 - pinned HTTPS source
        urllib.request.Request(url, headers={"User-Agent": "DecisionBrain-data-fetcher"}),
        timeout=120,
    ) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)
    if temporary.stat().st_size != record["size"]:
        raise ValueError(f"downloaded size mismatch: {record['path']}")
    if expected_hash and sha256(temporary) != expected_hash:
        raise ValueError(f"downloaded SHA-256 mismatch: {record['path']}")
    temporary.replace(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the verified files; without this flag only remote metadata is checked",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least one")
    manifest = remote_manifest(args.workers)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"verified remote metadata for {manifest['case_count']} cases at "
        f"revision {manifest['revision']}"
    )
    print(f"instance bytes: {manifest['total_instance_bytes']}")
    print(f"manifest: {args.manifest}")
    if args.download:
        for case in manifest["cases"].values():
            download_file(manifest["revision"], case["instance"], args.output, args.force)
            download_file(
                manifest["revision"], case["reference_solution"], args.output, args.force
            )
        print(f"downloaded and checksum-verified data in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
