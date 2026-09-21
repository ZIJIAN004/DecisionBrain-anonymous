"""OpenAI-compatible LLM client using native OpenAI message formats."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

import httpx

from ..config import Settings
from ..core.models import ChatRequest, ChatResponse
from ..exceptions import ConfigurationError, LLMResponseError

_T = TypeVar("_T")
DebugStreamEventHandler = Callable[[dict[str, Any]], Awaitable[None]]
RetryEventHandler = Callable[[dict[str, Any]], Awaitable[None]]

logger = logging.getLogger(__name__)


def _validate_configuration(target_url: str | None, api_key: str) -> None:
    if not target_url:
        raise ConfigurationError("LLM_MODEL_URL is not configured")
    if not api_key or api_key == "xxx":
        raise ConfigurationError("LLM_API_KEY is not configured")


async def _call_llm(
    request: ChatRequest,
    *,
    target_url: str | None,
    api_key: str,
    timeout: int,
    model: str,
    reasoning_effort: str,
    temperature: float,
) -> tuple[str | None, str, list[dict] | None, dict[str, int]]:
    """Run a non-streaming request and return content, reasoning, and tool calls."""
    _validate_configuration(target_url, api_key)
    payload = _build_chat_payload(
        request,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        stream=False,
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(target_url, json=payload, headers=headers)
        if r.status_code != 200:
            raise LLMResponseError(r.status_code, r.text[:1000])
        payload_json = r.json()
        msg = (payload_json.get("choices") or [{}])[0].get("message", {})
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        tool_calls = msg.get("tool_calls")
        if not content and not tool_calls:
            logger.warning("Optimization LLM returned empty content and no tool calls")
        return (content or None), reasoning, tool_calls, _normalize_usage(payload_json.get("usage"))


async def _call_llm_stream(
    request: ChatRequest,
    *,
    target_url: str | None,
    api_key: str,
    timeout: int,
    model: str,
    reasoning_effort: str,
    temperature: float,
):
    """Yield raw delta dictionaries from a streaming request.

    Each choices[0].delta may contain role, content, reasoning_content, or fragmented
    tool_calls that the caller must merge.
    """
    _validate_configuration(target_url, api_key)
    payload = _build_chat_payload(
        request,
        model=model,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        stream=True,
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", target_url, json=payload, headers=headers) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode(errors="replace")
                raise LLMResponseError(r.status_code, body[:1000])
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage_delta = {"__usage__": chunk["usage"]}
                    else:
                        usage_delta = {}
                    choices = chunk.get("choices") or []
                    if not choices:
                        if usage_delta:
                            yield usage_delta
                        continue
                    delta = {**(choices[0].get("delta") or {}), **usage_delta}
                except Exception:
                    continue
                if delta:
                    yield delta


def _build_chat_payload(
    request: ChatRequest,
    *,
    model: str,
    reasoning_effort: str,
    temperature: float,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": request.messages,
        "temperature": temperature,
        "stream": stream,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if request.tools:
        payload["tools"] = request.tools
        payload["tool_choice"] = request.tool_choice
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


# ------------------------------------------------------------------
# Retry helpers
# ------------------------------------------------------------------


def _is_network_error(exc: Exception) -> bool:
    module = type(exc).__module__
    name = type(exc).__name__
    if "httpx" in module or "httpcore" in module:
        return "StatusError" not in name
    return isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError))


async def _retry_llm(
    call: Callable[[], AsyncIterator[_T] | _T],
    *,
    streaming: bool = False,
    max_attempts: int,
    retry_base_delay_seconds: float,
    retry_event_handler: RetryEventHandler | None = None,
) -> AsyncIterator[_T]:
    """Apply shared backoff and retry behavior to complete and stream calls."""
    for attempt in range(max_attempts):
        emitted = False
        try:
            if streaming:
                async for chunk in call():
                    emitted = True
                    yield chunk
                return
            else:
                result = await call()
                yield result
                return
        except Exception as exc:
            if emitted or attempt >= max_attempts - 1 or not _is_network_error(exc):
                raise
            backoff_seconds = retry_base_delay_seconds * (2**attempt)
            if retry_event_handler is not None:
                await retry_event_handler(
                    {
                        "attempt": attempt + 1,
                        "next_attempt": attempt + 2,
                        "max_attempts": max_attempts,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "backoff_seconds": backoff_seconds,
                    }
                )
            await asyncio.sleep(backoff_seconds)


def _merge_stream_tool_calls(chunks: list[dict]) -> list[dict]:
    """Merge incremental SSE tool-call fragments into complete tool calls."""
    merged: dict[int, dict[str, Any]] = {}
    for tc in chunks:
        idx = tc.get("index", 0)
        if idx not in merged:
            merged[idx] = {
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
                "function": {"name": "", "arguments": ""},
            }
        entry = merged[idx]
        if tc.get("id"):
            entry["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            entry["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            entry["function"]["arguments"] += fn["arguments"]
    return [merged[i] for i in sorted(merged)]


def _normalize_usage(value: Any) -> dict[str, int]:
    """Normalize OpenAI-compatible usage, including cached input tokens."""
    if not isinstance(value, dict):
        return {}
    details = value.get("prompt_tokens_details") or value.get("input_tokens_details") or {}
    cached = value.get(
        "cached_tokens",
        value.get(
            "cache_read_input_tokens",
            details.get("cached_tokens", details.get("cache_read_input_tokens", 0)),
        ),
    )
    input_tokens = int(value.get("prompt_tokens", value.get("input_tokens", 0)) or 0)
    output_tokens = int(value.get("completion_tokens", value.get("output_tokens", 0)) or 0)
    cached_tokens = int(cached or 0)
    total_tokens = int(value.get("total_tokens", 0) or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


# ------------------------------------------------------------------
# LLMClient
# ------------------------------------------------------------------


class LLMClient:
    """OpenAI-compatible LLM client using native dictionary messages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._debug_stream_handler: DebugStreamEventHandler | None = None
        self._retry_event_handler: RetryEventHandler | None = None
        self._last_call_metrics: dict[str, int] = {}

    def set_debug_stream_handler(self, handler: DebugStreamEventHandler | None) -> None:
        self._debug_stream_handler = handler

    def set_retry_event_handler(self, handler: RetryEventHandler | None) -> None:
        self._retry_event_handler = handler

    @property
    def last_call_metrics(self) -> dict[str, int]:
        return dict(self._last_call_metrics)

    async def _record_call_retry(self, event: dict[str, Any]) -> None:
        self._last_call_metrics["retry_count"] += 1
        self._last_call_metrics["retry_backoff_ms"] += int(
            float(event.get("backoff_seconds") or 0) * 1000
        )
        if self._retry_event_handler is not None:
            await self._retry_event_handler(event)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Return a ChatResponse for a non-streaming request."""
        self._last_call_metrics = {"retry_count": 0, "retry_backoff_ms": 0}
        self.validate_configuration()
        if self.settings.debug:
            return await self._complete_via_stream_debug(request)

        async for result in _retry_llm(
            lambda: _call_llm(
                request,
                model=self.settings.llm_chat_model,
                reasoning_effort=self.settings.opt_reasoning_effort.strip(),
                temperature=self.settings.opt_llm_temperature,
                target_url=self.settings.llm_model_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.opt_llm_timeout,
            ),
            max_attempts=self.settings.opt_llm_max_attempts,
            retry_base_delay_seconds=self.settings.opt_llm_retry_base_delay_seconds,
            retry_event_handler=self._record_call_retry,
        ):
            if len(result) == 3:
                content, reasoning, tool_calls = result
                usage = {}
            else:
                content, reasoning, tool_calls, usage = result
            self._last_call_metrics.update(usage)
            return ChatResponse(
                content=content or "", reasoning=reasoning, tool_calls=tool_calls, raw={"usage": usage}
            )

    async def _complete_via_stream_debug(self, request: ChatRequest) -> ChatResponse:
        """Stream debug deltas and tool calls, then merge them into a ChatResponse."""
        request_id = uuid.uuid4().hex
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_deltas: list[dict[str, Any]] = []
        usage: dict[str, int] = {}
        content_chars = 0
        reasoning_chars = 0
        model = self.settings.llm_chat_model
        effort = self.settings.opt_reasoning_effort.strip()

        await self._emit_debug_stream_event(
            {
                "kind": "started",
                "request_id": request_id,
                "message_count": len(request.messages),
                "model": model,
                "reasoning_effort": effort,
                "messages": request.messages,
            }
        )

        async def _stream_raw():
            async for delta in _call_llm_stream(
                request,
                model=model,
                reasoning_effort=effort,
                temperature=self.settings.opt_llm_temperature,
                target_url=self.settings.llm_model_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.opt_llm_timeout,
            ):
                yield delta

        try:
            async for delta in _retry_llm(
                _stream_raw,
                streaming=True,
                max_attempts=self.settings.opt_llm_max_attempts,
                retry_base_delay_seconds=self.settings.opt_llm_retry_base_delay_seconds,
                retry_event_handler=self._record_call_retry,
            ):
                if "__usage__" in delta:
                    usage = _normalize_usage(delta.get("__usage__"))
                    continue
                rc = delta.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    reasoning_parts.append(rc)
                    reasoning_chars += len(rc)
                    await self._emit_debug_stream_event(
                        {
                            "kind": "chunk",
                            "request_id": request_id,
                            "channel": "reasoning",
                            "text": rc,
                        }
                    )
                c = delta.get("content")
                if isinstance(c, str) and c:
                    content_parts.append(c)
                    content_chars += len(c)
                    await self._emit_debug_stream_event(
                        {
                            "kind": "chunk",
                            "request_id": request_id,
                            "channel": "code",
                            "text": c,
                        }
                    )
                tcs = delta.get("tool_calls")
                if tcs:
                    tool_call_deltas.extend(tcs)
                    await self._emit_debug_stream_event(
                        {
                            "kind": "tool_call_delta",
                            "request_id": request_id,
                            "tool_calls": tcs,
                        }
                    )
        except Exception as exc:
            await self._emit_debug_stream_event(
                {
                    "kind": "finished",
                    "request_id": request_id,
                    "status": "error",
                    "error": str(exc),
                    "content_chars": content_chars,
                    "reasoning_chars": reasoning_chars,
                }
            )
            raise
        else:
            await self._emit_debug_stream_event(
                {
                    "kind": "finished",
                    "request_id": request_id,
                    "status": "ok",
                    "content_chars": content_chars,
                    "reasoning_chars": reasoning_chars,
                }
            )

        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        tool_calls = _merge_stream_tool_calls(tool_call_deltas) if tool_call_deltas else None
        if not content and not tool_calls:
            logger.warning("Optimization LLM returned empty content and no tool calls")
        self._last_call_metrics.update(usage)
        return ChatResponse(
            content=content or None, reasoning=reasoning, tool_calls=tool_calls, raw={"usage": usage}
        )

    def validate_configuration(self) -> None:
        if not self.settings.llm_api_key or self.settings.llm_api_key == "xxx":
            raise ConfigurationError(
                "LLM_API_KEY is not configured. Set LLM_API_KEY and LLM_MODEL_URL."
            )
        if not self.settings.llm_model_url:
            raise ConfigurationError("LLM_MODEL_URL is not configured. Set LLM_MODEL_URL.")

    async def _emit_debug_stream_event(self, event: dict[str, Any]) -> None:
        if not self.settings.debug or self._debug_stream_handler is None:
            return
        await self._debug_stream_handler(event)
