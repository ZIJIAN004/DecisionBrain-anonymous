from __future__ import annotations

import asyncio
import json

from decisionbrain.benchmark.intake_data_tools import build_intake_data_tools


def tools_for(tmp_path, payload: dict):
    data = tmp_path / "data"
    data.mkdir()
    (data / "instance.json").write_text(json.dumps(payload), encoding="utf-8")
    return {tool.name: tool for tool in build_intake_data_tools(tmp_path)}


def invoke(tool, **arguments):
    return json.loads(asyncio.run(tool.handler(**arguments)))


def test_stream_get_handles_array_indices_numeric_keys_and_escaped_pointer_tokens(tmp_path) -> None:
    tools = tools_for(
        tmp_path,
        {
            "items": [{"id": 1}, {"id": 2}],
            "numeric": {"0": "object-key"},
            "a/b": {"~key": "escaped"},
            "cost": 1.25,
        },
    )

    assert invoke(tools["data_get"], path="instance.json", pointer="/items/1/id")["value"] == 2
    assert (
        invoke(tools["data_get"], path="instance.json", pointer="/numeric/0")["value"]
        == "object-key"
    )
    assert (
        invoke(tools["data_get"], path="instance.json", pointer="/a~1b/~0key")["value"] == "escaped"
    )
    assert invoke(tools["data_get"], path="instance.json", pointer="/cost")["value"] == 1.25


def test_stream_keys_and_slice_return_stable_pagination(tmp_path) -> None:
    tools = tools_for(
        tmp_path,
        {"mapping": {"a": [1, 2], "b": {"nested": True}}, "items": [10, 20, 30]},
    )

    keys = invoke(tools["data_keys"], path="instance.json", pointer="/mapping", start=0, count=1)
    assert keys["entries"] == [{"key": "a", "kind": "array"}]
    assert keys["total_keys"] == 2
    assert keys["next_start"] == 1

    sliced = invoke(tools["data_slice"], path="instance.json", pointer="/items", start=1, count=1)
    assert sliced["items"] == [20]
    assert sliced["total_items"] == 3
    assert sliced["next_start"] == 2


def test_stream_get_reconstructs_nested_object_and_array(tmp_path) -> None:
    tools = tools_for(
        tmp_path,
        {"selected": {"name": "alpha", "details": {"active": True}, "values": [1, 2]}},
    )

    result = invoke(tools["data_get"], path="instance.json", pointer="/selected")

    assert result["kind"] == "object"
    assert result["value"] == {
        "name": "alpha",
        "details": {"active": True},
        "values": [1, 2],
    }


def test_stream_slice_reconstructs_object_elements(tmp_path) -> None:
    tools = tools_for(
        tmp_path,
        {"items": [{"id": 1, "nested": {"values": [10, 20]}}, {"id": 2}]},
    )

    sliced = invoke(tools["data_slice"], path="instance.json", pointer="/items", count=1)

    assert sliced["items"] == [{"id": 1, "nested": {"values": [10, 20]}}]
    assert sliced["total_items"] == 2
    assert sliced["next_start"] == 1


def test_stream_tools_reject_path_escape_and_oversized_subtree(tmp_path) -> None:
    tools = tools_for(tmp_path, {"huge": list(range(20_000))})
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    escaped = asyncio.run(tools["data_get"].handler(path="../outside.json", pointer=""))
    assert escaped.startswith("data_get error:")

    oversized = asyncio.run(tools["data_get"].handler(path="instance.json", pointer="/huge"))
    assert oversized.startswith("data_get error:")
    assert "too large" in oversized
