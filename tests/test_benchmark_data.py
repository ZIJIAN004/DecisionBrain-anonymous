from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data_script(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_suite_indexes_have_expected_counts():
    expected = {"FrontierOR65-Fea": 65, "FrontierOR10-Inf": 10}
    for suite, count in expected.items():
        payload = json.loads(
            (PROJECT_ROOT / "benchmarks" / suite / "index.json").read_text(encoding="utf-8")
        )
        assert payload["suite"] == suite
        assert payload["case_count"] == count
        assert len(payload["cases"]) == count

    hard32 = json.loads(
        (PROJECT_ROOT / "benchmarks" / "Hard32-Fea" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(hard32) == 32
    assert Counter(row["task"] for row in hard32) == {
        "jssp_deadline": 6,
        "vrptw_minfleet": 6,
        "pdptw_minfleet": 20,
    }


def test_external_data_sources_are_pinned():
    frontieror = json.loads(
        (PROJECT_ROOT / "data" / "FrontierOR65-Fea" / "sources.lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert frontieror["source"]["revision"] == (
        "37ccd8b6dca3bf7f4e0c58941a6ed156832a6d9e"
    )
    assert frontieror["source"]["license"] == "CC-BY-4.0"

    hard32 = json.loads(
        (PROJECT_ROOT / "data" / "Hard32-Fea" / "sources.lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert hard32["distribution"] == "download-and-reconstruct-locally"
    assert hard32["sources"]["jssp_deadline"]["retrieval"]["commit"] == (
        "460510f197744eed1cbcbbdfd6ec3252252f412c"
    )
    assert hard32["sources"]["vrptw_minfleet"]["retrieval"]["commit"] == (
        "7474b068a8effa7f39448b241314b615c9271525"
    )


def test_frontieror10_inf_files_match_index():
    suite = "FrontierOR10-Inf"
    index = json.loads(
        (PROJECT_ROOT / "benchmarks" / suite / "index.json").read_text(encoding="utf-8")
    )
    data_root = PROJECT_ROOT / "data" / suite
    found = []
    for case_id, case in index["cases"].items():
        instance = (
            data_root
            / case_id
            / "instance"
            / f"large_instance_{int(case['instance_index'])}.json"
        )
        assert instance.is_file(), instance
        assert instance.stat().st_size == int(case["instance_bytes"]), instance
        assert isinstance(json.loads(instance.read_text(encoding="utf-8")), dict)
        found.append(instance)
    assert len(found) == index["case_count"] == 10


def test_hard32_vrptw_transformation_is_deterministic():
    preparer = load_data_script("data/Hard32-Fea/prepare.py", "hard32_prepare")
    source = """NAME: sample
TYPE: CVRPTW
DIMENSION: 2
CAPACITY: 10
SERVICE_TIME: 3
NODE_COORD_SECTION
1 0 0
2 3 4
DEMAND_SECTION
1 0
2 5
TIME_WINDOW_SECTION
1 0 20
2 2 8
DEPOT_SECTION
1
-1
EOF
"""
    result = preparer.build_vrptw(
        source,
        {"base_instance": "sample", "limit": 1, "fleet_lower_bound": 1},
    )
    assert result == {
        "name": "sample",
        "num_customers": 1,
        "capacity": 10,
        "max_vehicles": 1,
        "service_time": 30,
        "coordinates": [[0, 0], [3, 4]],
        "demand": [0, 5],
        "time_window": [[0, 200], [20, 80]],
        "fleet_lower_bound": 1,
    }


def test_hard32_pdptw_transformation_converts_request_ids_to_zero_based():
    preparer = load_data_script("data/Hard32-Fea/prepare.py", "hard32_prepare_pdptw")
    source = """NAME: sample
TYPE: PDPTW
DIMENSION: 3
CAPACITY: 10
NODE_COORD_SECTION
1 0 0
2 1 1
3 2 2
PICKUP_AND_DELIVERY_SECTION
1 0 0 100 0 0 0
2 5 10 50 2 0 3
3 -5 20 80 2 2 0
DEPOT_SECTION
1
-1
EOF
"""
    result = preparer.build_pdptw(
        source,
        {"instance_name": "sample", "limit": 1},
    )
    assert result["requests"] == [[1, 2]]
    assert result["num_requests"] == 1
    assert result["max_vehicles"] == 1


def test_frontieror65_default_mode_never_downloads_payload(monkeypatch, tmp_path):
    fetcher = load_data_script("data/FrontierOR65-Fea/fetch.py", "frontieror65_fetch")
    manifest = {
        "revision": "pinned-revision",
        "case_count": 65,
        "total_instance_bytes": 123,
        "cases": {},
    }
    monkeypatch.setattr(fetcher, "remote_manifest", lambda workers: manifest)
    monkeypatch.setattr(
        fetcher,
        "parse_args",
        lambda: SimpleNamespace(
            download=False,
            output=tmp_path / "dataset",
            manifest=tmp_path / "remote_manifest.json",
            workers=1,
            force=False,
        ),
    )
    monkeypatch.setattr(
        fetcher,
        "download_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download attempted")),
    )

    assert fetcher.main() == 0
    written = json.loads((tmp_path / "remote_manifest.json").read_text(encoding="utf-8"))
    assert written == manifest
