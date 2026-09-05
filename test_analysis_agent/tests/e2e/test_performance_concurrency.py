"""
Performance & Concurrency Tests (mocked, CI-safe)
====================================================
Exercises the real FastAPI ASGI app under concurrent load with the agent
mocked out at the agent.build_agent boundary -- no Gemini/ChromaDB needed,
so these run in every CI invocation. Focused on: cross-thread response
isolation under concurrency, SSE heartbeat behavior on slow turns, and
sequential-throughput sanity.

Run:
    cd test_analysis_agent
    pytest tests/e2e/test_performance_concurrency.py -v
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

pytestmark = pytest.mark.e2e

CONCURRENT_REQUEST_COUNT = 20
P95_LATENCY_BUDGET_SECONDS = 3.0


def _mock_agent_for_thread(thread_id: str) -> MagicMock:
    """A distinct mock agent whose canned answer embeds its own thread_id,
    so cross-thread response leakage would be immediately detectable."""
    agent_mock = MagicMock()
    final = MagicMock(content=f"ANSWER-FOR-{thread_id}", type="ai")
    agent_mock.invoke.return_value = {"messages": [final]}
    return agent_mock


@pytest.fixture
def app_ready():
    import bridge

    mock_base = MagicMock()
    final = MagicMock(content="base-ready", type="ai")
    mock_base.invoke.return_value = {"messages": [final]}
    bridge.app_state["base_agent"] = mock_base
    bridge.app_state["thread_agents"] = {}
    bridge.app_state["startup_error"] = None
    return bridge


class TestConcurrentChatRequests:
    @pytest.mark.asyncio
    async def test_concurrent_requests_do_not_cross_thread_leak(self, app_ready):
        import bridge

        thread_ids = [f"perf-thread-{i}" for i in range(CONCURRENT_REQUEST_COUNT)]

        # get_thread_agent normally builds/caches one agent per thread_id;
        # patch it directly so every thread deterministically gets its own
        # distinguishable mock without touching vector_embed/download logic.
        def _fake_get_thread_agent(thread_id, _files, _base_agent, **_kwargs):
            return _mock_agent_for_thread(thread_id), {"files_indexed": [], "failed_files": [], "chunk_count": 0}

        transport = httpx.ASGITransport(app=bridge.app)
        with patch("bridge.get_thread_agent", side_effect=_fake_get_thread_agent):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async def _one_request(tid: str):
                    started = time.perf_counter()
                    resp = await client.post(
                        "/chat",
                        json={"messages": [{"role": "user", "content": "test"}], "thread_id": tid},
                    )
                    return tid, resp, time.perf_counter() - started

                results = await asyncio.gather(*[_one_request(tid) for tid in thread_ids])

        latencies = []
        for tid, resp, elapsed in results:
            assert resp.status_code == 200
            assert f"ANSWER-FOR-{tid}" in resp.text, (
                f"Thread {tid} did not receive its own mocked answer -- "
                f"possible cross-thread response leakage under concurrency."
            )
            latencies.append(elapsed)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95) - 1]
        assert p95 < P95_LATENCY_BUDGET_SECONDS, (
            f"p95 latency across {CONCURRENT_REQUEST_COUNT} concurrent mocked "
            f"/chat requests was {p95:.2f}s (budget {P95_LATENCY_BUDGET_SECONDS}s)"
        )

    @pytest.mark.asyncio
    async def test_concurrent_requests_all_succeed(self, app_ready):
        import bridge

        def _fake_get_thread_agent(thread_id, _files, _base_agent, **_kwargs):
            return _mock_agent_for_thread(thread_id), {"files_indexed": [], "failed_files": [], "chunk_count": 0}

        transport = httpx.ASGITransport(app=bridge.app)
        with patch("bridge.get_thread_agent", side_effect=_fake_get_thread_agent):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                responses = await asyncio.gather(*[
                    client.post(
                        "/chat",
                        json={"messages": [{"role": "user", "content": "test"}], "thread_id": f"burst-{i}"},
                    )
                    for i in range(CONCURRENT_REQUEST_COUNT)
                ])

        statuses = [r.status_code for r in responses]
        assert statuses == [200] * CONCURRENT_REQUEST_COUNT, statuses


class TestHeartbeatBehavior:
    @pytest.mark.asyncio
    async def test_slow_turn_emits_sse_keepalive_comments(self, app_ready, monkeypatch):
        import bridge

        # Shrink the heartbeat interval so a short artificial delay in the
        # mocked agent.invoke is enough to trigger at least one keep-alive
        # without making this test slow.
        monkeypatch.setattr(bridge, "HEARTBEAT_INTERVAL", 0.05)

        slow_agent = MagicMock()

        def _slow_invoke(*_a, **_k):
            time.sleep(0.3)
            final = MagicMock(content="slow answer", type="ai")
            return {"messages": [final]}

        slow_agent.invoke.side_effect = _slow_invoke

        def _fake_get_thread_agent(thread_id, _files, _base_agent, **_kwargs):
            return slow_agent, {"files_indexed": [], "failed_files": [], "chunk_count": 0}

        transport = httpx.ASGITransport(app=bridge.app)
        with patch("bridge.get_thread_agent", side_effect=_fake_get_thread_agent):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/chat",
                    json={"messages": [{"role": "user", "content": "test"}], "thread_id": "heartbeat-test"},
                )

        assert resp.status_code == 200
        assert ": keep-alive" in resp.text, (
            "Expected at least one SSE keep-alive comment line during a turn "
            "slower than HEARTBEAT_INTERVAL"
        )
        assert "slow answer" in resp.text


class TestSequentialThroughput:
    def test_sequential_trace_calls_stay_within_linear_budget(self, app_ready):
        from fastapi.testclient import TestClient

        n = 15
        per_call_budget_s = 0.5  # generous for a fully mocked call

        bridge = app_ready

        def _fake_get_thread_agent(thread_id, _files, _base_agent, **_kwargs):
            return _mock_agent_for_thread(thread_id), {"files_indexed": [], "failed_files": [], "chunk_count": 0}

        client = TestClient(bridge.app, raise_server_exceptions=False)
        durations = []
        with patch("bridge.get_thread_agent", side_effect=_fake_get_thread_agent):
            for i in range(n):
                started = time.perf_counter()
                resp = client.post("/trace", json={"message": "test", "thread_id": f"seq-{i}"})
                durations.append(time.perf_counter() - started)
                assert resp.status_code == 200

        total = sum(durations)
        assert total < n * per_call_budget_s, (
            f"{n} sequential mocked /trace calls took {total:.2f}s total "
            f"(budget {n * per_call_budget_s:.2f}s) -- avg {statistics.mean(durations):.3f}s/call"
        )
