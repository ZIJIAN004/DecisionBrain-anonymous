import asyncio

import pytest

from decisionbrain.config import Settings
from decisionbrain.core.models import ChatRequest, StreamChannel
from decisionbrain.exceptions import LLMResponseError
from decisionbrain.infrastructure import llm_client
from decisionbrain.infrastructure.llm_client import LLMClient


def run(coroutine):
    return asyncio.run(coroutine)


def test_complete_forwards_tools_to_non_streaming_call(monkeypatch):
    captured = {}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "lookup data",
                "parameters": {"type": "object"},
            },
        }
    ]

    async def fake_call_llm(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return '{"ok": true}', "", None

    monkeypatch.setattr(llm_client, "_call_llm", fake_call_llm)

    service = LLMClient(Settings(debug=False))
    response = run(
        service.complete(
            ChatRequest(
                messages=[{"role": "system", "content": "system"}],
                tools=tools,
            )
        )
    )

    assert response.content == '{"ok": true}'
    assert captured["args"][0] == ChatRequest(
        messages=[{"role": "system", "content": "system"}],
        tools=tools,
    )
    assert "tools" not in captured["kwargs"]
    assert captured["kwargs"]["temperature"] == 0.2


def test_complete_reports_each_network_retry(monkeypatch):
    calls = 0
    retries = []

    async def flaky_call_llm(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError(f"temporary failure {calls}")
        return '{"ok": true}', "", None

    async def record_retry(event):
        retries.append(dict(event))

    monkeypatch.setattr(llm_client, "_call_llm", flaky_call_llm)
    settings = Settings(debug=False).model_copy(
        update={"opt_llm_max_attempts": 4, "opt_llm_retry_base_delay_seconds": 0.001}
    )
    service = LLMClient(settings)
    service.set_retry_event_handler(record_retry)

    response = run(service.complete(ChatRequest(messages=[{"role": "user", "content": "x"}])))

    assert response.content == '{"ok": true}'
    assert calls == 3
    assert [event["next_attempt"] for event in retries] == [2, 3]
    assert all(event["max_attempts"] == 4 for event in retries)
    assert [event["backoff_seconds"] for event in retries] == [0.001, 0.002]
    assert service.last_call_metrics == {"retry_count": 2, "retry_backoff_ms": 3}


def test_provider_http_error_is_raised_instead_of_becoming_empty_response(monkeypatch):
    class FakeResponse:
        status_code = 400
        text = '{"error":{"code":"invalid_request_error"}}'

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(llm_client.httpx, "AsyncClient", lambda **kwargs: FakeClient())

    with pytest.raises(LLMResponseError) as exc_info:
        run(
            llm_client._call_llm(
                ChatRequest(messages=[{"role": "user", "content": "x"}]),
                target_url="https://example.invalid/chat",
                api_key="secret",
                timeout=10,
                model="test-model",
                reasoning_effort="max",
                temperature=0.2,
            )
        )

    assert exc_info.value.provider_status == 400
    assert "invalid_request_error" in str(exc_info.value)


def test_debug_complete_uses_stream_and_emits_reasoning_and_content(monkeypatch):
    async def fail_call_llm(*args, **kwargs):
        raise AssertionError("debug complete must not use non-streaming _call_llm")

    async def fake_call_llm_stream(*args, **kwargs):
        yield {"reasoning_content": "先判断约束。"}
        yield {"content": '{"decision":'}
        yield {"content": '"proceed"}'}

    monkeypatch.setattr(llm_client, "_call_llm", fail_call_llm)
    monkeypatch.setattr(llm_client, "_call_llm_stream", fake_call_llm_stream)

    events = []

    async def record_event(event):
        events.append(dict(event))

    service = LLMClient(Settings(debug=True))
    service.set_debug_stream_handler(record_event)

    response = run(
        service.complete(
            ChatRequest(
                messages=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "problem"},
                ]
            )
        )
    )

    assert response.content == '{"decision":"proceed"}'
    assert response.reasoning == "先判断约束。"
    assert [event["kind"] for event in events] == [
        "started",
        "chunk",
        "chunk",
        "chunk",
        "finished",
    ]
    assert events[0]["model"] == "test-model"
    assert events[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "problem"},
    ]
    assert [event.get("channel") for event in events if event["kind"] == "chunk"] == [
        StreamChannel.REASONING.value,
        StreamChannel.CODE.value,
        StreamChannel.CODE.value,
    ]
    assert (
        "".join(
            event["text"] for event in events if event.get("channel") == StreamChannel.CODE.value
        )
        == '{"decision":"proceed"}'
    )


def test_debug_complete_forwards_tools_to_streaming_call(monkeypatch):
    captured = {}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "lookup data",
                "parameters": {"type": "object"},
            },
        }
    ]

    async def fake_call_llm_stream(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        yield {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ]
        }

    monkeypatch.setattr(llm_client, "_call_llm_stream", fake_call_llm_stream)

    service = LLMClient(Settings(debug=True))
    response = run(
        service.complete(
            ChatRequest(
                messages=[{"role": "system", "content": "system"}],
                tools=tools,
            )
        )
    )

    assert captured["args"][0] == ChatRequest(
        messages=[{"role": "system", "content": "system"}],
        tools=tools,
    )
    assert "tools" not in captured["kwargs"]
    assert captured["kwargs"]["temperature"] == 0.2
    assert response.tool_calls == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
