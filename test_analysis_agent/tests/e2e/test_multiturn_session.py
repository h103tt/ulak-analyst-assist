"""
Multi-turn Conversation & Session Tests
=========================================
Covers thread-agent caching correctness/isolation in bridge.get_thread_agent
(mocked, fast), plus real query-reformulation and checkpointer-continuity
behavior across turns of the same thread_id via the live /trace endpoint.

Run:
    cd test_analysis_agent
    pytest tests/e2e/test_multiturn_session.py -v                  # fast subset
    pytest tests/e2e/test_multiturn_session.py -v -m integration    # + live turns
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tests.e2e._helpers import requires_gemini  # noqa: E402

pytestmark = pytest.mark.e2e


# ===================================================================
# 1. Thread-agent cache correctness & isolation (mocked, fast)
# ===================================================================
class TestThreadAgentCaching:
    @pytest.fixture(autouse=True)
    def _reset_thread_agents(self):
        import bridge

        bridge.app_state["thread_agents"] = {}
        yield
        bridge.app_state["thread_agents"] = {}

    def test_same_thread_same_files_reuses_cached_agent(self):
        import bridge

        with patch("agent.build_agent") as mock_build:
            mock_build.side_effect = lambda *a, **k: MagicMock()

            agent1, _ = bridge.get_thread_agent("thread-a", [], base_agent=MagicMock())
            agent2, _ = bridge.get_thread_agent("thread-a", [], base_agent=MagicMock())

        assert mock_build.call_count == 1, "Second call with an unchanged file set should hit the cache"
        assert agent1 is agent2

    def test_different_threads_get_isolated_agents(self):
        import bridge

        with patch("agent.build_agent") as mock_build:
            mock_build.side_effect = lambda *a, **k: MagicMock()

            agent_a, _ = bridge.get_thread_agent("thread-a", [], base_agent=MagicMock())
            agent_b, _ = bridge.get_thread_agent("thread-b", [], base_agent=MagicMock())

        assert mock_build.call_count == 2, "Each new thread_id should build its own agent"
        assert agent_a is not agent_b

    def test_changed_file_signature_triggers_rebuild(self):
        import bridge
        from bridge import UploadedFile

        files_v1 = [UploadedFile(id="f1", name="doc.pdf", url="http://x/1")]
        files_v2 = [UploadedFile(id="f2", name="doc2.pdf", url="http://x/2")]

        with patch("agent.build_agent") as mock_build, \
             patch("bridge.download_file", return_value=Path("dummy.pdf")), \
             patch("vector_embed.build_session_retriever_tool") as mock_session_tool:
            mock_build.side_effect = lambda *a, **k: MagicMock()
            mock_session_tool.return_value = (
                MagicMock(),
                {"collection": "c1", "files_indexed": ["doc.pdf"], "failed_files": [], "chunk_count": 1},
            )

            _, report1 = bridge.get_thread_agent("thread-c", files_v1, base_agent=MagicMock())
            build_calls_after_v1 = mock_build.call_count

            mock_session_tool.return_value = (
                MagicMock(),
                {"collection": "c2", "files_indexed": ["doc2.pdf"], "failed_files": [], "chunk_count": 1},
            )
            _, report2 = bridge.get_thread_agent("thread-c", files_v2, base_agent=MagicMock())

        assert mock_build.call_count > build_calls_after_v1, (
            "A changed file set (different files_signature) must rebuild the thread agent, "
            "not silently reuse the previous one -- otherwise a user's new attachment "
            "would never actually get indexed/searchable."
        )
        assert report1["files_indexed"] == ["doc.pdf"]
        assert report2["files_indexed"] == ["doc2.pdf"]

    def test_same_thread_same_uploaded_files_reuses_cache(self):
        """Re-sending the identical attachment set (e.g. a client retry) should
        NOT re-download/re-index -- files_signature is stable across retries."""
        import bridge
        from bridge import UploadedFile

        files = [UploadedFile(id="f1", name="doc.pdf", url="http://x/1")]

        with patch("agent.build_agent") as mock_build, \
             patch("bridge.download_file", return_value=Path("dummy.pdf")) as mock_download, \
             patch("vector_embed.build_session_retriever_tool") as mock_session_tool:
            mock_build.side_effect = lambda *a, **k: MagicMock()
            mock_session_tool.return_value = (
                MagicMock(),
                {"collection": "c1", "files_indexed": ["doc.pdf"], "failed_files": [], "chunk_count": 1},
            )

            bridge.get_thread_agent("thread-d", files, base_agent=MagicMock())
            bridge.get_thread_agent("thread-d", files, base_agent=MagicMock())

        assert mock_download.call_count == 1, "Identical file set on a retry should not re-download"
        assert mock_session_tool.call_count == 1, "Identical file set on a retry should not re-index"


# ===================================================================
# 2. Thread-agent cache under concurrency (mocked, threading)
# ===================================================================
class TestThreadAgentConcurrency:
    """bridge.get_thread_agent mutates app_state["thread_agents"] (a plain
    dict) with a read-then-write check-then-act pattern and no lock. This
    test documents the real behavior under a burst of concurrent first
    requests for a brand-new thread_id, rather than assuming it's safe."""

    def test_concurrent_first_requests_for_new_thread(self):
        import bridge

        bridge.app_state["thread_agents"] = {}
        build_count = 0
        lock_for_counter = threading.Lock()

        def _fake_build(*_a, **_k):
            nonlocal build_count
            with lock_for_counter:
                build_count += 1
            return MagicMock()

        with patch("agent.build_agent", side_effect=_fake_build):
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                futures = [
                    pool.submit(bridge.get_thread_agent, "thread-race", [], MagicMock())
                    for _ in range(10)
                ]
                results = [f.result() for f in futures]

        distinct_agents = {id(agent) for agent, _report in results}

        # Documented, not asserted-safe: without a lock, a burst of concurrent
        # first requests for the same brand-new thread can build more than
        # one agent (each losing thread's result is discarded, but the extra
        # ChatGoogleGenerativeAI/session setup work is wasted, and mid-race callers can
        # briefly see different agent instances -- i.e. different
        # conversation memories -- for what should be one thread).
        assert build_count >= 1
        if build_count > 1 or len(distinct_agents) > 1:
            pytest.xfail(
                f"bridge.get_thread_agent is not race-safe: {build_count} build_agent() "
                f"calls and {len(distinct_agents)} distinct agent instances for 10 "
                f"concurrent first-requests on the same new thread_id. Consider a "
                f"per-thread_id lock (e.g. a dict of threading.Lock) around the "
                f"check-then-build-then-store section in bridge.get_thread_agent."
            )


# ===================================================================
# 3. Query reformulation across turns (live)
# ===================================================================
@pytest.mark.integration
@requires_gemini
class TestQueryReformulationLive:
    @pytest.fixture(scope="class")
    def live_client(self):
        import bridge

        with TestClient(bridge.app) as client:
            for _ in range(30):
                if client.get("/health").json().get("agent_loaded"):
                    break
                time.sleep(0.5)
            yield client

    @staticmethod
    def _search_tool_queries(messages: list[dict]) -> list[str]:
        queries = []
        for m in messages:
            if m["type"] != "AIMessage":
                continue
            for call in m.get("tool_calls", []):
                if call.get("name") == "search_testing_standards":
                    query = (call.get("args") or {}).get("query") or ""
                    queries.append(query)
        return queries

    def test_followup_query_is_reformulated_not_raw(self, live_client):
        thread_id = "e2e-multiturn-reformulation"

        first = live_client.post(
            "/trace",
            json={"message": "What does MIL-STD-461 cover?", "thread_id": thread_id},
            timeout=90.0,
        )
        assert first.status_code == 200

        elliptical_followup = "what about the timing requirements?"
        second = live_client.post(
            "/trace",
            json={"message": elliptical_followup, "thread_id": thread_id},
            timeout=90.0,
        )
        assert second.status_code == 200
        data = second.json()

        followup_queries = self._search_tool_queries(data["messages"])
        assert followup_queries, "Follow-up turn never called search_testing_standards"

        for q in followup_queries:
            assert q.strip().lower() != elliptical_followup.strip().lower(), (
                "Tool was called with the raw elliptical follow-up text instead "
                "of a reformulated, entity-complete query."
            )
        assert any("MIL-STD-461" in q or "461" in q for q in followup_queries), (
            f"Reformulated quer{'y' if len(followup_queries) == 1 else 'ies'} "
            f"{followup_queries!r} do not fold in the standard named in turn 1."
        )

    def test_checkpointer_retains_prior_turn_context(self, live_client):
        thread_id = "e2e-multiturn-memory"

        live_client.post(
            "/trace",
            json={"message": "Let's talk about MIL-STD-882E today.", "thread_id": thread_id},
            timeout=90.0,
        )
        second = live_client.post(
            "/trace",
            json={"message": "Which standard did I just mention?", "thread_id": thread_id},
            timeout=90.0,
        )
        assert second.status_code == 200
        assert "882" in second.json()["answer"], (
            "Agent did not recall the standard named in the previous turn of the "
            "same thread -- checkpointer/conversation memory may not be wired "
            "correctly for this thread_id."
        )
