import asyncio
import io
from datetime import datetime, timezone

import pytest
from rich.console import Console

from helpers import MemoryEventSink

from decisionbrain.cli.rich_events import RichEventSink, RichRenderer
from decisionbrain.config import Settings
from decisionbrain.core.models import AgentStage
from decisionbrain.events import (
    RunEvent,
    RunEventDraft,
    CompositeEventSink,
    EVENT_PROGRESS_REPORTED,
    EventLevel,
    EventPublisher,
    JsonlEventSink,
)
from decisionbrain.exceptions import EventPublishingError
from decisionbrain.run_storage import RunRepository
from decisionbrain.runtime.agent_runtime import generate_run_id


RUN_ID = "20260703-073045Z-a1b2c3d4"
FIXED_TIME = datetime(2026, 7, 3, 7, 30, 45, tzinfo=timezone.utc)


def run(coroutine):
    return asyncio.run(coroutine)


def event(
    event_type: str,
    message: str,
    *,
    sequence: int = 1,
    stage: AgentStage | None = None,
    level: EventLevel = EventLevel.INFO,
    payload=None,
    duration_ms: int | None = None,
) -> RunEvent:
    return RunEvent(
        event_id=f"event-{sequence}",
        run_id=RUN_ID,
        sequence=sequence,
        timestamp=FIXED_TIME,
        type=event_type,
        stage=stage,
        level=level,
        message=message,
        payload=payload or {},
        duration_ms=duration_ms,
    )


def render(events, *, debug: bool = False) -> str:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    sink = RichEventSink(console, RichRenderer(debug=debug))

    async def render_all():
        for item in events:
            await sink.handle(item)

    run(render_all())
    return output.getvalue()


def test_memory_and_composite_sinks_preserve_order():
    calls = []

    class RecordingSink:
        def __init__(self, name):
            self.name = name

        async def handle(self, item):
            calls.append((self.name, item.sequence))

    memory = MemoryEventSink()
    composite = CompositeEventSink(RecordingSink("first"), RecordingSink("second"), memory)
    item = event("run_created", "created")

    run(composite.handle(item))

    assert calls == [("first", 1), ("second", 1)]
    assert memory.events == [item]


def test_composite_fails_fast_and_publisher_refuses_to_continue():
    calls = []

    class FailingSink:
        async def handle(self, item):
            calls.append("failing")
            raise RuntimeError("sink failed")

    class NeverReachedSink:
        async def handle(self, item):
            calls.append("unexpected")

    publisher = EventPublisher(
        CompositeEventSink(FailingSink(), NeverReachedSink()),
        run_id=RUN_ID,
        clock=lambda: FIXED_TIME,
    )

    with pytest.raises(RuntimeError, match="sink failed"):
        run(publisher.publish(RunEventDraft(type="run_started", message="start")))
    with pytest.raises(EventPublishingError, match="cannot continue"):
        run(publisher.publish(RunEventDraft(type="error", message="again")))
    assert calls == ["failing"]


def test_publisher_assigns_strict_sequences_under_concurrency():
    memory = MemoryEventSink()
    publisher = EventPublisher(memory, run_id=RUN_ID, clock=lambda: FIXED_TIME)

    async def publish_all():
        return await asyncio.gather(
            *(
                publisher.publish(RunEventDraft(type="assistant_message", message=f"消息 {index}"))
                for index in range(20)
            )
        )

    published = run(publish_all())

    assert [item.sequence for item in published] == list(range(1, 21))
    assert [item.sequence for item in memory.events] == list(range(1, 21))
    assert all(item.run_id == RUN_ID for item in published)


def test_jsonl_sink_persists_the_exact_published_events(tmp_path):
    repository = RunRepository(Settings(runs_dir=tmp_path / "runs"))
    run_record = repository.create_run_record(
        run_id=generate_run_id(), workspace=tmp_path, command="test events"
    )
    memory = MemoryEventSink()
    publisher = EventPublisher(
        CompositeEventSink(JsonlEventSink(repository), memory),
        run_id=run_record.run_id,
        clock=lambda: FIXED_TIME,
        event_id_factory=lambda: "fixed-event-id",
    )

    published = run(
        publisher.publish(
            RunEventDraft(
                type="user_message",
                message="请优化中文排程",
                payload={"语言": "中文"},
            )
        )
    )

    assert repository.read_events(run_record.run_id) == [published]
    assert memory.events == [published]


def test_publisher_requires_bound_run_id():
    publisher = EventPublisher(MemoryEventSink(), clock=lambda: FIXED_TIME)
    with pytest.raises(EventPublishingError, match="not bound to a run_id"):
        run(publisher.publish(RunEventDraft(type="run_started", message="start")))


def test_concise_normal_and_trace_filter_and_detail_levels():
    events = [
        event("run_started", "run internal", sequence=1),
        event("user_message", "用户问题", sequence=2),
        event(
            "stage_started",
            "开始验证",
            sequence=3,
            stage=AgentStage.SOLVING,
        ),
        event("warning", "模拟警告", sequence=4, level=EventLevel.WARNING),
        event(
            "assistant_message",
            "调用求解器完成",
            sequence=5,
        ),
        event(
            "assistant_message",
            "**中文 Markdown** 结果",
            sequence=6,
        ),
        event(
            "run_started",
            "debug only",
            sequence=7,
            level=EventLevel.DEBUG,
        ),
    ]

    normal = render(events)
    debug = render(events, debug=True)

    assert "求解" in normal and "模拟警告" in normal and "求解器" in normal
    assert "arguments" not in normal and "debug only" not in normal
    assert "debug only" in debug
    # Event trace details are hidden from debug output.


def test_debug_payload_renders_as_readable_tree_instead_of_raw_json():
    output = render(
        [
            event(
                "diagnostic",
                "Runtime 输入与配置诊断",
                sequence=1,
                level=EventLevel.DEBUG,
                payload={
                    "agent_version": "4.0.0",
                    "snapshot": {
                        "copied_files": 5,
                        "ignored": [
                            {
                                "relative_path": "runs",
                                "reason": "ignored_directory",
                                "size_bytes": None,
                            }
                        ],
                    },
                },
            )
        ],
        debug=True,
    )

    # Diagnostic messages remain visible while event trace details are hidden.
    assert "Runtime 输入与配置诊断" in output


def test_debug_renderer_shows_llm_stream_chunks_inline():
    output = render(
        [
            event(
                "llm_stream_started",
                "LLM intake 开始流式输出",
                sequence=1,
                level=EventLevel.DEBUG,
                payload={
                    "kind": "started",
                    "request_id": "request-1",
                    "role": "intake",
                    "message_count": 1,
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": "系统提示"},
                        {"role": "user", "content": "用户问题"},
                    ],
                },
            ),
            event(
                "llm_stream_chunk",
                "先判断。",
                sequence=2,
                level=EventLevel.DEBUG,
                payload={
                    "kind": "chunk",
                    "request_id": "request-1",
                    "role": "intake",
                    "channel": "reasoning",
                },
            ),
            event(
                "llm_stream_chunk",
                '{"decision":"proceed"}',
                sequence=3,
                level=EventLevel.DEBUG,
                payload={
                    "kind": "chunk",
                    "request_id": "request-1",
                    "role": "intake",
                    "channel": "code",
                },
            ),
            event(
                "llm_stream_finished",
                "LLM intake 流式输出完成 (ok)",
                sequence=4,
                level=EventLevel.DEBUG,
                payload={
                    "kind": "finished",
                    "request_id": "request-1",
                    "role": "intake",
                    "status": "ok",
                    "content_chars": 22,
                    "reasoning_chars": 4,
                },
            ),
        ],
        debug=True,
    )

    assert "LLM intake stream" in output
    assert "Actual LLM input" in output
    assert "model=test-model" in output
    assert "messages=2" in output
    assert "系统提示" in output
    assert "用户问题" in output
    assert "LLM intake · Reasoning" in output
    assert "先判断。" in output
    assert "LLM intake · Final output" in output
    assert '{"decision":"proceed"}' in output
    assert "LLM intake completed" in output
    assert "type=llm_stream_started" not in output
    assert "type=llm_stream_chunk" not in output


def test_debug_renderer_suppresses_accumulated_stream_previews():
    events = [
        event(
            "assistant_message",
            "累计思考预览第一版",
            sequence=1,
            payload={"stream": "reasoning"},
        ),
        event(
            "assistant_message",
            "普通进度消息",
            sequence=2,
        ),
    ]

    normal = render(events)
    debug = render(events, debug=True)

    assert "累计思考预览第一版" in normal
    assert "累计思考预览第一版" not in debug
    assert "普通进度消息" in debug


def test_debug_renderer_shows_tool_interaction_before_next_llm_reasoning():
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": '{"path":"algorithm_design.json"}',
        },
    }
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "设计算法"},
    ]
    output = render(
        [
            event(
                "llm_stream_started",
                "LLM algorithm_design 开始流式输出",
                sequence=1,
                level=EventLevel.DEBUG,
                payload={
                    "request_id": "request-1",
                    "role": "algorithm_design",
                    "messages": messages,
                },
            ),
            event(
                "llm_stream_chunk",
                "现在写入 JSON。",
                sequence=2,
                level=EventLevel.DEBUG,
                payload={
                    "request_id": "request-1",
                    "role": "algorithm_design",
                    "channel": "reasoning",
                },
            ),
            event(
                "llm_stream_finished",
                "LLM algorithm_design 流式输出完成 (ok)",
                sequence=3,
                level=EventLevel.DEBUG,
                payload={"request_id": "request-1", "role": "algorithm_design"},
            ),
            event(
                "tool_call",
                "tool call: write_file",
                sequence=4,
                level=EventLevel.DEBUG,
                payload={
                    "tool_call_id": "call-1",
                    "tool_name": "write_file",
                    "arguments": {"path": "algorithm_design.json"},
                },
            ),
            event(
                "tool_result",
                "wrote algorithm_design.json",
                sequence=5,
                level=EventLevel.DEBUG,
                payload={"tool_call_id": "call-1", "tool_name": "write_file"},
            ),
            event(
                "llm_stream_started",
                "LLM algorithm_design 开始流式输出",
                sequence=6,
                level=EventLevel.DEBUG,
                payload={
                    "request_id": "request-2",
                    "role": "algorithm_design",
                    "messages": messages
                    + [
                        {"role": "assistant", "tool_calls": [tool_call]},
                        {
                            "role": "tool",
                            "tool_call_id": "call-1",
                            "content": "wrote algorithm_design.json",
                        },
                    ],
                },
            ),
            event(
                "llm_stream_chunk",
                "继续检查结果。",
                sequence=7,
                level=EventLevel.DEBUG,
                payload={
                    "request_id": "request-2",
                    "role": "algorithm_design",
                    "channel": "reasoning",
                },
            ),
        ],
        debug=True,
    )

    call_position = output.index("tool_call: write_file")
    result_position = output.index("wrote algorithm_design.json")
    next_reasoning_position = output.index("继续检查结果。")
    assert call_position < result_position < next_reasoning_position
    assert output.count("tool_call: write_file") == 1
    assert output.count("wrote algorithm_design.json") == 1


def test_renderer_shows_progress_report_in_normal_mode():
    output = render(
        [
            event(
                EVENT_PROGRESS_REPORTED,
                "已经识别求解结构",
                sequence=1,
                stage=AgentStage.ALGORITHM_DESIGN,
                payload={
                    "summary": "已经识别求解结构",
                    "completed": ["读取订单"],
                    "key_findings": ["存在容量约束"],
                    "assumptions": ["订单不可拆分"],
                    "next_step": "比较候选算法",
                },
            )
        ]
    )

    assert "Progress Update" in output
    assert "已经识别求解结构" in output
    assert "比较候选算法" in output


def test_renderer_handles_error_and_non_tty_console():
    output = render(
        [
            event("error", "求解器调用失败", sequence=1, level=EventLevel.ERROR),
            event("error", "无法生成最终结果", sequence=2, level=EventLevel.ERROR),
        ],
    )

    assert "求解器调用失败" in output
    assert "无法生成最终结果" in output
    assert "\x1b[" not in output
