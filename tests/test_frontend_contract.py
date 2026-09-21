import asyncio
from pathlib import Path
from types import SimpleNamespace

from decisionbrain.api.app import app, root


ROOT = Path(__file__).resolve().parents[1]


def test_required_dom_controls_exist():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for element_id in (
        "chat-attachment-file",
        "upload-attachment-btn",
        "user-input",
        "send-btn",
        "language-select",
    ):
        assert f'id="{element_id}"' in html


def test_frontend_and_openapi_use_only_run_endpoints():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    paths = set(app.openapi()["paths"])
    for path in (
        "/api/runs",
        "/api/runs/{run_id}/continue",
        "/api/runs/{run_id}/cancel",
        "/api/runs/{run_id}/events",
        "/api/runs/{run_id}/artifacts",
    ):
        assert path in paths
    assert "/api/runs" in html
    assert "/api/opt" not in html
    assert "session_id" not in html


def test_frontend_consumes_agent_event_contract():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for event_type in (
        "stage_started",
        "stage_finished",
        "assistant_message",
        "artifact_created",
        "clarification_requested",
        "user_message",
        "error",
        "run_finished",
    ):
        assert f"event.type === '{event_type}'" in html
    for legacy_type in ("reasoning" + "_delta", "code" + "_delta"):
        assert legacy_type not in html


def test_frontend_uses_display_view_without_debug_panels():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "const DISPLAY_VIEW = 'display';" in html
    assert "'/events?view=' + DISPLAY_VIEW" in html
    assert "'/continue?view=' + DISPLAY_VIEW" in html
    assert "推理过程" not in html
    assert "生成代码" not in html
    assert "runtime-reason" not in html


def test_frontend_clarification_submit_has_busy_state():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert 'id="clarification-submit"' in html
    assert "const activeControllers = new Map();" in html
    assert "let activeController" not in html
    assert "isRunBusy(selectedRunId)" in html
    assert "control.disabled = true;" in html
    assert "status: 'running'" in html


def test_frontend_supports_multiple_active_runs():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "const pendingRuns = new Map();" in html
    assert "function makePendingRunId()" in html
    assert "activeControllers.set(pendingRunId, controller);" in html
    assert "activeControllers.set(runId, controller);" in html
    assert "await handleRunEvent(JSON.parse(line.slice(5).trim()), pendingRunId);" in html
    assert "if (!message || isStateBusy(state)) return;" in html


def test_frontend_does_not_mark_clarifying_stage_done():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "clarificationStageFinishSequences(events)" in html
    assert "latestPendingClarificationStage(events, status)" in html
    assert "blockedFinishSequences.has(event.sequence)" in html
    assert "note: t('needs_clarification')" in html


def test_frontend_syncs_sse_status_back_to_run_list():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "function syncRunListRecord(record)" in html
    assert "runs[index] = record;" in html
    assert "syncRunListRecord(state.record);" in html


def test_frontend_visualizes_feasibility_feedback_loop():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    for stage in (
        "intake",
        "problem_contract",
        "algorithm_design",
        "solving",
        "feasibility_review",
        "explanation",
    ):
        assert f"'{stage}'" in html
    assert "function renderFeasibilityFeedback(state)" in html
    assert "accepted ? t('accepted') : t('rejected')" in html
    assert "t('responsibility') + ' → '" in html
    assert "config.feasibility_review_enabled === false" in html


def test_frontend_supports_configurable_english_and_chinese_ui():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "const DEFAULT_UI_LANGUAGE = '__DBN_UI_LANGUAGE__';" in html
    assert "const LANGUAGE_RESOURCES =" in html
    assert "en: {" in html
    assert "zh: {" in html
    assert "localStorage.setItem(UI_LANGUAGE_KEY, uiLanguage);" in html
    assert "languageSelect.addEventListener('change'" in html


def test_frontend_injects_server_default_language(monkeypatch):
    monkeypatch.setattr("decisionbrain.api.app.Settings", lambda: SimpleNamespace(ui_language="zh"))

    response = asyncio.run(root())

    assert "__DBN_UI_LANGUAGE__" not in response.body.decode("utf-8")
    assert "const DEFAULT_UI_LANGUAGE = 'zh';" in response.body.decode("utf-8")
