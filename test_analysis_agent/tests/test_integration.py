"""
End-to-End Integration Tests for ULAK Test Analysis Agent
=========================================================
Tests real interactions between ChromaDB, Ollama Embeddings/LLM,
Docling/Text loaders, dynamic session indexing, and the FastAPI bridge.

Prerequisites:
- Ollama running locally (ollama serve) with models:
  - nomic-embed-text
  - gemma4:12b (or the model configured in agent.py)

Run:
    cd ulak-analyst-assist/test_analysis_agent
    uv run pytest tests/test_integration.py -v -m integration
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import agent
import bridge
import vector_embed

# ===================================================================
# Helper Fixtures & Health Checks
# ===================================================================

def is_ollama_online() -> bool:
    """Check if the local Ollama instance is responsive."""
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False

# Skip marker for tests that require a live Ollama daemon
requires_ollama = pytest.mark.skipif(
    not is_ollama_online(),
    reason="Ollama server is not running on http://localhost:11434"
)

# Default per-test wall-clock cap for direct agent.invoke() calls that have
# no built-in HTTP timeout. Prevents a slow/stuck tool call (e.g. a lazily
# built ChromaDB collection) from hanging the whole suite indefinitely.
AGENT_INVOKE_TIMEOUT_SECONDS = 90.0

pytestmark = pytest.mark.integration


def invoke_with_timeout(qa_agent, payload, config, context, timeout=AGENT_INVOKE_TIMEOUT_SECONDS):
    """Run agent.invoke() in a worker thread and enforce a hard timeout.

    qa_agent.invoke() has no native timeout, so a stuck tool call (e.g. a
    standards collection that doesn't exist yet and has to be built/embedded
    on first use) can hang pytest forever. This wraps the call so it fails
    loudly instead.
    """
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(qa_agent.invoke, payload, config=config, context=context)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            pytest.fail(
                f"agent.invoke() did not return within {timeout}s. "
                "Likely a slow/stuck tool call (e.g. building or querying a "
                "ChromaDB collection) or a slow local model response."
            )


@pytest.fixture(scope="module")
def temp_workspace():
    """Create an isolated workspace directory for uploaded files and collections."""
    workspace = tempfile.mkdtemp(prefix="agent_integration_test_")
    uploads_dir = os.path.join(workspace, "uploads")
    collections_dir = os.path.join(workspace, "chromadb_user")
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(collections_dir, exist_ok=True)

    orig_uploads = vector_embed.USER_UPLOADS_DIR
    orig_collections = vector_embed.USER_COLLECTIONS_DIR

    vector_embed.USER_UPLOADS_DIR = uploads_dir
    vector_embed.USER_COLLECTIONS_DIR = collections_dir

    yield {
        "workspace": workspace,
        "uploads": uploads_dir,
        "collections": collections_dir,
    }

    # Teardown
    vector_embed.USER_UPLOADS_DIR = orig_uploads
    vector_embed.USER_COLLECTIONS_DIR = orig_collections
    shutil.rmtree(workspace, ignore_errors=True)


# ===================================================================
# 1. RAG & User Document Ingestion Integration
# ===================================================================
class TestUserDocumentRAGIntegration:
    """Verify loading real documents into session collections and retrieving them."""

    @requires_ollama
    def test_build_session_retriever_and_query(self, temp_workspace):
        """Verify that a newly uploaded file is embedded and retrievable via MMR."""
        doc_path = os.path.join(temp_workspace["uploads"], "SRS_Security.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "# Security Specifications\n\n"
                "## REQ-SEC-001: Password Lockout Policy\n"
                "The system shall lock the user account after 5 consecutive failed login attempts "
                "within a 15-minute window. The unlock duration must be 30 minutes.\n\n"
                "## REQ-SEC-002: Session Expiration\n"
                "Inactivity timeout for administrative sessions shall be 10 minutes."
            )

        session_id = "test-session-rag-01"
        tool, report = vector_embed.build_session_retriever_tool(
            file_paths=[doc_path],
            session_id=session_id,
            k=2,
        )

        assert report["chunk_count"] > 0
        assert "SRS_Security.md" in report["files_indexed"]
        assert len(report["failed_files"]) == 0

        # Execute retrieval via the LangChain tool directly
        retrieval_result = tool.invoke("What is the lockout policy for failed logins?")

        assert "REQ-SEC-001" in str(retrieval_result)
        assert "5 consecutive failed" in str(retrieval_result)

    def test_get_document_structure_tool(self, temp_workspace):
        """Verify get_document_structure correctly extracts header text without LLM."""
        doc_name = "SRS_Architecture.md"
        doc_path = Path(temp_workspace["uploads"]) / f"att123__{doc_name}"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "# System Architecture\n"
                "## 1. Introduction\nThis is the high level design.\n"
                "## 2. Component Diagram\nDetailed microservice overview."
            )

        vector_embed.register_upload(doc_name, doc_path)

        structure = vector_embed.get_document_structure.invoke(doc_name)
        assert "System Architecture" in structure
        assert "Component Diagram" in structure


# ===================================================================
# 2. Multi-Tenant Session Isolation Integration
# ===================================================================
class TestSessionIsolationIntegration:
    """Verify that thread/session ChromaDB collections do not leak data across tenants."""

    @requires_ollama
    def test_session_data_isolation(self, temp_workspace):
        # Session A: Ingests Project Alpha document
        doc_a = os.path.join(temp_workspace["uploads"], "Project_Alpha.txt")
        with open(doc_a, "w", encoding="utf-8") as f:
            f.write("CONFIDENTIAL_ALPHA_SECRET_KEY = 99887711\nSystem shall boot in 2 seconds.")

        tool_a, _ = vector_embed.build_session_retriever_tool(
            file_paths=[doc_a],
            session_id="session_alpha",
            collection_suffix="alpha",
        )

        # Session B: Ingests Project Beta document
        doc_b = os.path.join(temp_workspace["uploads"], "Project_Beta.txt")
        with open(doc_b, "w", encoding="utf-8") as f:
            f.write("CONFIDENTIAL_BETA_TOKEN = 44556622\nSystem shall operate at 24V DC.")

        tool_b, _ = vector_embed.build_session_retriever_tool(
            file_paths=[doc_b],
            session_id="session_beta",
            collection_suffix="beta",
        )

        # Query Session A for Beta's secret
        result_a = tool_a.invoke("CONFIDENTIAL_BETA_TOKEN")
        # Query Session B for Alpha's secret
        result_b = tool_b.invoke("CONFIDENTIAL_ALPHA_SECRET_KEY")

        # Session A must NOT know Beta's content, and Session B must NOT know Alpha's content
        assert "44556622" not in str(result_a)
        assert "99887711" not in str(result_b)


# ===================================================================
# 3. Model Cognition & ISO/IEC/IEEE 29119 Compliance Integration
# ===================================================================
class TestModelBehaviorAndISO29119Compliance:
    """Verify that the real model outputs structured test cases and detects ambiguity."""

    @requires_ollama
    def test_agent_generates_compliant_test_plan(self, temp_workspace):
        """Verify requirement traceability, structured fields, and boundary tests."""
        doc_path = os.path.join(temp_workspace["uploads"], "SRS_Motor.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(
                "REQ-DRV-042: The motor controller shall limit rotational speed between "
                "100 RPM and 3500 RPM. Any speed command outside this range must return ERR_INVALID_RANGE."
            )

        session_tool, _ = vector_embed.build_session_retriever_tool(
            file_paths=[doc_path],
            session_id="session_motor_test",
        )

        thread_tools = list(vector_embed.tools) + [session_tool]
        qa_agent = agent.build_agent(tools=thread_tools, has_user_document=True)

        config = {"configurable": {"thread_id": "thread-test-iso"}}
        ctx = agent.Context(user_id="qa_tester")

        response = invoke_with_timeout(
            qa_agent,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Please analyze REQ-DRV-042 from the uploaded document "
                            "and generate a test case according to ISO/IEC/IEEE 29119."
                        ),
                    }
                ]
            },
            config=config,
            context=ctx,
        )

        output_text = response["messages"][-1].content

        # 1. Traceability check
        assert "REQ-DRV-042" in output_text

        # 2. ISO/IEC/IEEE 29119 required structural sections
        for section_header in ["Preconditions", "Expected Result"]:
            assert section_header.lower() in output_text.lower()

        # 3. Test steps / boundary analysis presence
        assert any(term in output_text for term in ["100", "3500", "ERR_INVALID_RANGE", "Boundary", "Steps"])

    @requires_ollama
    def test_agent_detects_ambiguous_requirement(self):
        """Verify the agent flags missing parameters (thresholds/timing/messaging).

        Uses an explicitly empty tool list so this test exercises the model's
        own reasoning about ambiguity rather than any retrieval tool. Passing
        `tools=None` here would silently fall back to the full default
        toolset (see agent.build_agent), which forces the mandatory
        `search_testing_standards` call in the system prompt's grounding
        rules -- and since no standards collection is seeded in this test,
        that call can hang or fail unpredictably.
        """
        qa_agent = agent.build_agent(tools=[], has_user_document=False)

        config = {"configurable": {"thread_id": "thread-ambiguity-check"}}
        ctx = agent.Context(user_id="qa_tester")

        ambiguous_req = "Requirement REQ-PERF-99: The system shall respond quickly under high load."

        response = invoke_with_timeout(
            qa_agent,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Generate a test plan for this requirement: {ambiguous_req}",
                    }
                ]
            },
            config=config,
            context=ctx,
        )

        output_text = response["messages"][-1].content.lower()

        # Agent should identify ambiguity (e.g., missing threshold/limit or duration/timing)
        assert any(term in output_text for term in ["ambiguous", "untestable", "threshold", "limit", "missing"])


# ===================================================================
# 4. End-to-End FastAPI Lifespan & Streaming Integration
# ===================================================================
class TestFastAPIBridgeE2E:
    """Test full HTTP endpoints with active lifespan startup."""

    def test_health_with_lifespan(self):
        """Verify that app startup background thread initializes base_agent."""
        with TestClient(bridge.app) as client:
            # Poll briefly for the background thread to finish building the agent
            for _ in range(20):
                resp = client.get("/health")
                data = resp.json()
                if data.get("agent_loaded"):
                    break
                time.sleep(0.3)

            assert resp.status_code == 200
            assert data["status"] == "ok"

    @requires_ollama
    def test_trace_endpoint_live(self):
        """Verify /trace triggers agent execution and records run metadata."""
        with TestClient(bridge.app) as client:
            # Wait for agent readiness
            time.sleep(1.0)

            resp = client.post(
                "/trace",
                json={
                    "message": "What is boundary value analysis in software testing?",
                    "thread_id": "trace-live-integration",
                },
                timeout=60.0,
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["thread_id"] == "trace-live-integration"
            assert "answer" in data
            assert len(data["answer"]) > 20
            assert "trace_id" in data

    @requires_ollama
    def test_chat_sse_stream_live(self):
        """Verify /chat SSE streaming delivers real text-delta events from the model."""
        with TestClient(bridge.app) as client:
            time.sleep(1.0)

            resp = client.post(
                "/chat",
                json={
                    "messages": [{"role": "user", "content": "Reply with only the word: READY"}],
                    "thread_id": "chat-live-integration",
                },
                timeout=60.0,
            )

            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream"

            raw_sse = resp.text
            assert "text-start" in raw_sse
            assert "text-delta" in raw_sse
            assert "[DONE]" in raw_sse