import asyncio
from datetime import datetime, timezone

from decisionbrain.core.models import AgentStage
from decisionbrain.events import EventLevel, RunEvent
from decisionbrain.run_storage import AgentHtmlReport, RunRecord, RunStatus


NOW = datetime(2026, 7, 15, 8, 33, 19, tzinfo=timezone.utc)


def test_html_report_preserves_complete_multiline_tool_content():
    report = AgentHtmlReport()
    event = RunEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        timestamp=NOW,
        type="stage_started",
        stage=AgentStage.SOLVING,
        level=EventLevel.INFO,
        message="开始求解",
    )
    asyncio.run(report.handle(event))
    asyncio.run(
        report.record_turn(
            {
                "stage": "solving",
                "duration_ms": 123,
                "request": {
                    "messages": [
                        {"role": "system", "content": "system\nline two"},
                        {"role": "user", "content": "solve everything"},
                    ],
                    "tools": [{"function": {"name": "shell"}}],
                    "tool_choice": "auto",
                },
                "response": {
                    "reasoning": "first thought\nsecond thought",
                    "content": None,
                    "tool_calls": [],
                },
                "tool_executions": [
                    {
                        "tool_call_id": "call-write",
                        "name": "write_file",
                        "arguments": {
                            "path": "solver.py",
                            "content": "print('one')\nprint('two')\n",
                        },
                        "result": "wrote solver.py\n27 bytes",
                    },
                    {
                        "tool_call_id": "call-shell",
                        "name": "shell",
                        "arguments": {"command": "python solver.py\necho finished"},
                        "result": "one\ntwo\nfinished\n",
                    },
                ],
            }
        )
    )
    record = RunRecord(
        run_id="run-1",
        created_at=NOW,
        updated_at=NOW,
        status=RunStatus.SUCCEEDED,
        workspace="/workspace",
        command="dbn run demo",
        config={"debug": True},
    )

    output = report.render(record)

    assert "system\nline two" in output
    assert "first thought\nsecond thought" in output
    assert "print(&#x27;one&#x27;)\nprint(&#x27;two&#x27;)\n" in output
    assert "python solver.py\necho finished" in output
    assert "one\ntwo\nfinished\n" in output
    assert "...[truncated]" not in output
    assert "event-1" not in output
    assert "run_id" not in output
    assert ".tool-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:16px}" in output
