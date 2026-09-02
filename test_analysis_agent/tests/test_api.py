"""
API & Bridge Endpoint Tests
===========================
Tests for FastAPI endpoints using TestClient, validating request
payloads, status codes, and response JSON structure.

Run:
    cd ulak-analyst-assist/test_analysis_agent
    pytest tests/test_api.py -v
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure the package root is importable
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MOCK_LLM_RESPONSE_TEXT = (
    "Test Case ID: TC-001\n"
    "Requirement: REQ-101\n"
    "Test Type: Boundary\n"
    "Preconditions: System is in idle state.\n"
    "Steps:\n1. Send input at boundary value.\n2. Observe output.\n"
    "Expected Result: System accepts the boundary input without error."
)


def _make_mock_agent(response_text: str = MOCK_LLM_RESPONSE_TEXT) -> MagicMock:
    """Create a mock agent returning a canned response."""
    agent_mock = MagicMock()
    final_msg = MagicMock()
    final_msg.content = response_text
    final_msg.type = "ai"
    agent_mock.invoke.return_value = {"messages": [final_msg]}
    return agent_mock


@pytest.fixture
def client():
    """Yield a TestClient with a fully mocked agent in app_state.

    The patch on agent.build_agent stays ACTIVE for the entire fixture
    lifetime: bridge.get_thread_agent calls agent.build_agent() for every
    per-thread agent, and without the patch it would construct a real
    real ChatGoogleGenerativeAI agent and try to reach the Gemini API (hanging the test).
    """
    import bridge
    mock_base = _make_mock_agent()

    with patch("agent.build_agent", return_value=mock_base):
        tc = TestClient(bridge.app, raise_server_exceptions=False)
        bridge.app_state["base_agent"] = mock_base
        bridge.app_state["thread_agents"] = {}
        bridge.app_state["startup_error"] = None
        yield tc


@pytest.fixture
def client_no_agent():
    """TestClient with no agent loaded (simulates startup-in-progress)."""
    import bridge

    with patch("agent.build_agent", return_value=_make_mock_agent()):
        tc = TestClient(bridge.app, raise_server_exceptions=False)
        bridge.app_state["base_agent"] = None
        bridge.app_state["thread_agents"] = {}
        bridge.app_state["startup_error"] = None
        yield tc


@pytest.fixture
def client_error_agent():
    """TestClient whose agent raises on invoke."""
    import bridge

    error_agent = MagicMock()
    error_agent.invoke.side_effect = RuntimeError("LLM connection lost")

    with patch("agent.build_agent", return_value=error_agent):
        tc = TestClient(bridge.app, raise_server_exceptions=False)
        bridge.app_state["base_agent"] = error_agent
        bridge.app_state["thread_agents"] = {}
        bridge.app_state["startup_error"] = None
        yield tc


# ===================================================================
# 1. Health Endpoint
# ===================================================================
class TestHealthEndpoint:
    """GET /health"""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_agent_loaded(self, client):
        data = client.get("/health").json()
        assert data["agent_loaded"] is True

    def test_health_no_agent(self, client_no_agent):
        data = client_no_agent.get("/health").json()
        assert data["agent_loaded"] is False

    def test_health_startup_error_field(self, client_no_agent):
        data = client_no_agent.get("/health").json()
        assert "startup_error" in data


# ===================================================================
# 2. Chat Endpoint — Happy Path
# ===================================================================
class TestChatEndpoint:
    """POST /chat"""

    def test_chat_returns_200(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "What is ISO 29119?"}],
            "thread_id": "test-thread-1",
        })
        assert resp.status_code == 200

    def test_chat_returns_sse_stream(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "thread_id": "t1",
        })
        assert resp.headers.get("content-type") == "text/event-stream"

    def test_chat_stream_contains_text_delta(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "t2",
        })
        assert "text-delta" in resp.text

    def test_chat_stream_contains_finish(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "t3",
        })
        assert "[DONE]" in resp.text

    def test_chat_stream_contains_agent_response(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "t4",
        })
        assert "TC-001" in resp.text or "Boundary" in resp.text

    @pytest.mark.parametrize("role", ["user", "assistant", "system"])
    def test_chat_accepts_valid_roles(self, client, role):
        resp = client.post("/chat", json={
            "messages": [{"role": role, "content": f"message as {role}"}],
            "thread_id": f"role-test-{role}",
        })
        assert resp.status_code == 200


# ===================================================================
# 3. Chat Endpoint — Payload Validation
# ===================================================================
class TestChatPayloadValidation:
    """Validate request payload constraints."""

    def test_chat_empty_messages_returns_400(self, client):
        resp = client.post("/chat", json={
            "messages": [],
            "thread_id": "empty",
        })
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_chat_no_messages_field(self, client):
        resp = client.post("/chat", json={
            "thread_id": "no-msg",
        })
        assert resp.status_code in (400, 422)

    def test_chat_invalid_role_filtered(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "invalid_role", "content": "test"}],
            "thread_id": "bad-role",
        })
        assert resp.status_code == 400

    def test_chat_empty_content_filtered(self, client):
        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": ""},
                {"role": "user", "content": "actual message"},
            ],
            "thread_id": "empty-content",
        })
        assert resp.status_code == 200

    def test_chat_context_appended(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "Analyze this"}],
            "context": "Some PDF context text here",
            "thread_id": "ctx-test",
        })
        assert resp.status_code == 200

    def test_chat_thread_id_default(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200

    def test_chat_files_list_accepted(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "files-test",
            "files": [{"id": "f1", "name": "doc.pdf", "url": "http://example.com/doc.pdf"}],
        })
        assert resp.status_code == 200


# ===================================================================
# 4. Chat Endpoint — Agent Not Ready (503)
# ===================================================================
class TestChatAgentNotReady:
    """Verify 503 when agent hasn't loaded yet."""

    def test_chat_returns_503(self, client_no_agent):
        resp = client_no_agent.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 503

    def test_chat_503_body(self, client_no_agent):
        data = client_no_agent.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
        }).json()
        assert "error" in data
        assert "starting up" in data["error"].lower()


# ===================================================================
# 5. Trace Endpoint
# ===================================================================
class TestTraceEndpoint:
    """POST /trace"""

    def test_trace_returns_200(self, client):
        resp = client.post("/trace", json={
            "message": "What does MIL-STD-882E say?",
            "thread_id": "trace-1",
        })
        assert resp.status_code == 200

    def test_trace_returns_json(self, client):
        resp = client.post("/trace", json={
            "message": "test",
            "thread_id": "trace-json",
        })
        assert resp.headers["content-type"].startswith("application/json")

    def test_trace_has_required_fields(self, client):
        data = client.post("/trace", json={
            "message": "test query",
            "thread_id": "trace-fields",
        }).json()
        for field in ["thread_id", "messages", "kb_called", "kb_returned_content", "answer"]:
            assert field in data, f"Missing field: {field}"

    def test_trace_answer_is_string(self, client):
        data = client.post("/trace", json={
            "message": "test",
            "thread_id": "trace-answer",
        }).json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_trace_messages_is_list(self, client):
        data = client.post("/trace", json={
            "message": "test",
            "thread_id": "trace-msgs",
        }).json()
        assert isinstance(data["messages"], list)

    def test_trace_empty_message_returns_400(self, client):
        resp = client.post("/trace", json={
            "message": "",
            "thread_id": "trace-empty",
        })
        assert resp.status_code == 400

    def test_trace_whitespace_message_returns_400(self, client):
        resp = client.post("/trace", json={
            "message": "   ",
            "thread_id": "trace-ws",
        })
        assert resp.status_code == 400

    def test_trace_no_agent_returns_503(self, client_no_agent):
        resp = client_no_agent.post("/trace", json={
            "message": "test",
            "thread_id": "trace-503",
        })
        assert resp.status_code == 503

    def test_trace_agent_error_returns_500(self, client_error_agent):
        resp = client_error_agent.post("/trace", json={
            "message": "test",
            "thread_id": "trace-err",
        })
        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_trace_kb_called_field(self, client):
        data = client.post("/trace", json={
            "message": "test",
            "thread_id": "trace-kb",
        }).json()
        assert isinstance(data["kb_called"], bool)

    def test_trace_thread_id_echoed(self, client):
        data = client.post("/trace", json={
            "message": "test",
            "thread_id": "echo-thread",
        }).json()
        assert data["thread_id"] == "echo-thread"


# ===================================================================
# 6. Streaming Response Structure
# ===================================================================
class TestStreamingStructure:
    """Verify SSE stream format from /chat."""

    def test_sse_data_prefix(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "sse-prefix",
        })
        lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
        assert len(lines) > 0

    def test_sse_text_start_event(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "sse-start",
        })
        assert "text-start" in resp.text

    def test_sse_text_end_event(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "sse-end",
        })
        assert "text-end" in resp.text

    def test_sse_finish_event(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "sse-finish",
        })
        assert "finishReason" in resp.text

    def test_sse_done_marker(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}],
            "thread_id": "sse-done",
        })
        assert "[DONE]" in resp.text


# ===================================================================
# 7. Pydantic Model Validation
# ===================================================================
class TestPydanticModels:
    """Verify request/response models accept valid shapes."""

    def test_chat_request_minimal(self, client):
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200

    def test_chat_request_all_fields(self, client):
        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "What is testing?"},
            ],
            "context": "Additional context",
            "thread_id": "full-req",
            "files": [{"id": "f1", "name": "doc.pdf", "url": "http://example.com/doc.pdf"}],
        })
        assert resp.status_code == 200

    def test_trace_request_minimal(self, client):
        resp = client.post("/trace", json={
            "message": "test",
        })
        assert resp.status_code == 200
