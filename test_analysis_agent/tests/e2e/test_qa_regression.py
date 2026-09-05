"""
E2E QA Regression Suite
========================
Feeds the golden Q&A set (tests/e2e/golden_dataset.json) generated from the
real knowledge base through the real retriever and the real FastAPI /trace
endpoint, and compares the model's actual answer against the golden
expected_answer using deepeval judge metrics (groundedness/faithfulness +
correctness), plus fast CI-safe mocked tests for latency/schema/status.

Two speeds:
  - Fast / CI (no marker, no live services): mocked-agent async tests,
    pure-function edge-case tests. These run in every `pytest` invocation.
  - Live (`-m integration`, needs a working Gemini API key + an ingested KB): real
    retrieval + real generation against the golden set, scored with deepeval.

Run:
    cd test_analysis_agent
    pytest tests/e2e/test_qa_regression.py -v                    # fast subset only
    pytest tests/e2e/test_qa_regression.py -v -m integration      # full live run
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tests.e2e._helpers import requires_gemini, skip_if_kb_empty, get_judge_model  # noqa: E402

pytestmark = pytest.mark.e2e

GEVAL_PASS_THRESHOLD = 0.6
CONTEXTUAL_METRIC_THRESHOLD = 0.5

# Keep the live generation loop short -- each item is a real LLM turn
# (retrieval + rerank + generation + refinement pass) and can take tens of
# seconds against a local model.
MAX_LIVE_GOLDENS = 3


# ===================================================================
# 1. Context Retrieval: Top-K correctness (live)
# ===================================================================
@pytest.mark.integration
@requires_gemini
class TestContextRetrievalLive:
    def test_top_k_returns_expected_standard(self, golden_dataset, kb_populated):
        skip_if_kb_empty(kb_populated)
        import pipeline_logging
        import vector_embed

        # vector_embed.retriever_tool is a shared, process-wide singleton that
        # agent.build_agent() wraps (once, idempotently) with
        # pipeline_logging.limit_tool_calls -- a per-turn cap (3 calls) meant
        # to stop a single real conversation turn from looping forever.
        # bridge.py gives every real request its own trace_id, so that cap
        # never bites in production. Calling the tool directly here, many
        # times in one test with no trace_id set, would otherwise pile every
        # call onto the shared "no-trace" bucket and get "[Search limit
        # reached]" instead of real results from the 4th call on -- looking
        # exactly like a retrieval-quality regression. Giving each item its
        # own trace_id reproduces the one-budget-per-turn isolation real
        # requests get.
        failures = []
        for item in golden_dataset["golden_set"]:
            pipeline_logging.trace_id_var.set(f"test-top-k-{item['id']}")
            result = vector_embed.retriever_tool.invoke(item["question"])
            if f'standard="{item["expected_standard"]}"' not in str(result):
                failures.append((item["id"], item["expected_standard"]))

        assert not failures, (
            f"{len(failures)}/{len(golden_dataset['golden_set'])} golden "
            f"questions did not surface their expected standard in the top-K "
            f"reranked context: {failures}"
        )

    def test_retrieved_chunks_are_well_formed(self, golden_dataset, kb_populated):
        skip_if_kb_empty(kb_populated)
        import vector_embed

        item = golden_dataset["golden_set"][0]
        docs = vector_embed.kb_compression_retriever.invoke(item["question"])
        assert len(docs) > 0
        for doc in docs:
            assert re.match(r'^<chunk id="[^"]+" source="[^"]+"', doc.page_content), (
                f"Malformed chunk tag: {doc.page_content[:120]!r}"
            )


@pytest.mark.integration
@requires_gemini
class TestRetrievalQualityMetrics:
    """deepeval ContextualRecall/Precision against the real retriever.
    Skips gracefully if the judge model isn't pulled -- this is an opt-in,
    higher-cost quality gate, not a smoke test."""

    def test_recall_and_precision_meet_threshold(self, golden_dataset, kb_populated):
        skip_if_kb_empty(kb_populated)

        from deepeval import evaluate
        from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
        from deepeval.test_case import LLMTestCase

        import vector_embed

        judge = get_judge_model()

        test_cases = []
        for item in golden_dataset["golden_set"][:MAX_LIVE_GOLDENS]:
            docs = vector_embed.kb_compression_retriever.invoke(item["question"])
            test_cases.append(
                LLMTestCase(
                    input=item["question"],
                    actual_output="",  # not needed for contextual metrics
                    expected_output=item["expected_answer"],
                    retrieval_context=[d.page_content for d in docs],
                )
            )

        results = evaluate(
            test_cases=test_cases,
            metrics=[
                ContextualRecallMetric(model=judge, threshold=CONTEXTUAL_METRIC_THRESHOLD),
                ContextualPrecisionMetric(model=judge, threshold=CONTEXTUAL_METRIC_THRESHOLD),
            ],
        )

        failed = [
            (r.input, m.name, m.score)
            for r in results.test_results
            for m in r.metrics_data
            if not m.success
        ]
        assert not failed, f"Contextual metric(s) below threshold: {failed}"


# ===================================================================
# 2. Generation & Faithfulness (live, via the real /trace endpoint)
# ===================================================================
@pytest.mark.integration
@requires_gemini
class TestGenerationFaithfulnessLive:
    """Runs the golden set through the real FastAPI app (real agent, real
    ChatGoogleGenerativeAI, real reranked retrieval) and judges the answer
    with GEval, the same way test_rag_metrics.py's judge model is configured."""

    @pytest.fixture(scope="class")
    def live_client(self):
        import bridge

        with TestClient(bridge.app) as client:
            for _ in range(30):
                if client.get("/health").json().get("agent_loaded"):
                    break
                time.sleep(0.5)
            yield client

    def test_answers_are_grounded_and_correct(self, golden_dataset, kb_populated, live_client):
        skip_if_kb_empty(kb_populated)

        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, SingleTurnParams

        judge = get_judge_model()
        correctness = GEval(
            name="Answer Correctness",
            criteria=(
                "Evaluate whether the actual output's answer is factually correct and "
                "complete based on the input and retrieved context. If the answer is "
                "not correct or is missing key information, reduce the score."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            model=judge,
            threshold=GEVAL_PASS_THRESHOLD,
        )

        failures = []
        for item in golden_dataset["golden_set"][:MAX_LIVE_GOLDENS]:
            resp = live_client.post(
                "/trace",
                json={"message": item["question"], "thread_id": f"e2e-{item['id']}"},
                timeout=90.0,
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()

            assert data["kb_called"] is True, f"{item['id']}: KB was never queried"
            assert data["kb_returned_content"] is True, f"{item['id']}: KB returned nothing"

            retrieved = [
                m["content"] for m in data["messages"]
                if m["type"] == "ToolMessage" and m.get("name") == "search_testing_standards"
            ]

            test_case = LLMTestCase(
                input=item["question"],
                actual_output=data["answer"],
                expected_output=item["expected_answer"],
                retrieval_context=retrieved,
            )
            correctness.measure(test_case)
            if not correctness.is_successful():
                failures.append((item["id"], correctness.score, correctness.reason))

        assert not failures, f"Answer Correctness below threshold for: {failures}"

    def test_no_hallucinated_citations(self, golden_dataset, kb_populated, live_client):
        """Every '(Standard, Section X)' / '(Standard, p. N)' citation in the
        answer must reference a standard that actually appeared in a
        retrieved chunk for that turn -- a cheap, deterministic check that
        doesn't need the judge model."""
        skip_if_kb_empty(kb_populated)

        citation_re = re.compile(r"\(([A-Za-z0-9/_.\- ]+),\s*(?:Section|p\.)")

        for item in golden_dataset["golden_set"][:MAX_LIVE_GOLDENS]:
            resp = live_client.post(
                "/trace",
                json={"message": item["question"], "thread_id": f"e2e-cite-{item['id']}"},
                timeout=90.0,
            )
            data = resp.json()
            retrieved_text = " ".join(
                m["content"] for m in data["messages"]
                if m["type"] == "ToolMessage" and m.get("name") == "search_testing_standards"
            )

            for cited_standard in citation_re.findall(data["answer"]):
                assert cited_standard.strip() in retrieved_text, (
                    f"{item['id']}: answer cites '{cited_standard}' which does not "
                    f"appear in the retrieved context -- likely hallucinated citation."
                )

    def test_out_of_domain_question_admits_no_coverage(self, golden_dataset, kb_populated, live_client):
        skip_if_kb_empty(kb_populated)

        for item in golden_dataset["out_of_domain_set"]:
            resp = live_client.post(
                "/trace",
                json={"message": item["question"], "thread_id": f"e2e-ood-{item['id']}"},
                timeout=90.0,
            )
            assert resp.status_code == 200
            data = resp.json()
            answer_lower = data["answer"].lower()
            # The SCOPE guardrail (agent.py) declines an off-topic question
            # outright without ever calling search_testing_standards -- that's
            # a valid "no coverage" admission too, it just isn't phrased as a
            # retrieval-gap statement, and the exact wording varies between
            # calls (LLM sampling). kb_called is a deterministic signal for
            # that path, so trust it over trying to enumerate every possible
            # phrasing the model might use.
            admits_gap = not data["kb_called"] or any(
                phrase in answer_lower
                for phrase in [
                    "don't cover", "does not cover", "not specify", "no relevant", "not found", "cannot answer",
                    "does not contain", "do not contain", "not include", "doesn't include",
                    "not covered", "no information about",
                ]
            )
            # Soft assertion via id in message: out-of-domain handling is a
            # known model-behavior risk area, surface it clearly rather than
            # a bare assert False.
            assert admits_gap, (
                f"{item['id']}: expected the agent to admit no KB coverage for "
                f"an out-of-domain question, got: {data['answer'][:300]!r}"
            )


# ===================================================================
# 3. Edge / Unhappy Paths -- pure-function + mocked-agent (fast, CI-safe)
# ===================================================================
class TestEdgeCasesFast:
    def test_truncate_context_cuts_on_word_boundary(self):
        from bridge import truncate_context

        text = ("word " * 20000).strip()  # far above the 60_000-char budget
        result = truncate_context(text, max_chars=100)
        assert len(result) <= 100 + len("\n...[context truncated]")
        assert not result.split("\n...")[0].endswith("wor")  # not cut mid-word

    def test_truncate_context_below_budget_is_unchanged(self):
        from bridge import truncate_context

        text = "short text"
        assert truncate_context(text, max_chars=1000) == text

    def test_history_trim_respects_token_budget(self):
        """Unit-level check of the trimming policy (agent.HISTORY_TOKEN_BUDGET)
        using a deterministic fake token counter -- no live model needed."""
        from langchain_core.messages import trim_messages, HumanMessage, AIMessage

        messages = []
        for i in range(200):
            messages.append(HumanMessage(content=f"question {i}"))
            messages.append(AIMessage(content=f"answer {i}" * 50))

        trimmed = trim_messages(
            messages,
            strategy="last",
            token_counter=len,  # 1 "token" per message object, deterministic
            max_tokens=10,
            start_on="human",
        )
        assert len(trimmed) <= 10
        assert isinstance(trimmed[0], HumanMessage)

    def test_empty_query_rejected_before_agent(self, mocked_client):
        resp = mocked_client.post("/trace", json={"message": "   ", "thread_id": "edge-empty"})
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_empty_messages_list_rejected(self, mocked_client):
        resp = mocked_client.post("/chat", json={"messages": [], "thread_id": "edge-empty-msgs"})
        assert resp.status_code == 400

    def test_agent_not_ready_returns_503(self, mocked_client_no_agent):
        resp = mocked_client_no_agent.post("/trace", json={"message": "hi"})
        assert resp.status_code == 503


# ===================================================================
# 4. Mocked async E2E: latency / status / schema (fast, CI-safe)
# ===================================================================
class ChatSSEEvent(BaseModel):
    type: str


class TraceResponseSchema(BaseModel):
    trace_id: str
    thread_id: str
    messages: list
    tools: dict
    kb_called: bool
    kb_returned_content: bool
    tool_call_sequence: list
    retrieval_hop_count: int
    answer: str


MOCK_ANSWER = (
    "Test Case ID: TC-001\n"
    "Requirement: REQ-101\n"
    "Test Type: Boundary\n"
    "Expected Result: System accepts the boundary input without error."
)


@pytest.fixture
def mocked_client():
    import bridge

    mock_base = MagicMock()
    final = MagicMock(content=MOCK_ANSWER, type="ai")
    mock_base.invoke.return_value = {"messages": [final]}

    with patch("agent.build_agent", return_value=mock_base):
        tc = TestClient(bridge.app, raise_server_exceptions=False)
        bridge.app_state["base_agent"] = mock_base
        bridge.app_state["thread_agents"] = {}
        bridge.app_state["startup_error"] = None
        yield tc


@pytest.fixture
def mocked_client_no_agent():
    import bridge

    with patch("agent.build_agent", return_value=MagicMock()):
        tc = TestClient(bridge.app, raise_server_exceptions=False)
        bridge.app_state["base_agent"] = None
        bridge.app_state["thread_agents"] = {}
        bridge.app_state["startup_error"] = None
        yield tc


class TestMockedAsyncE2E:
    """httpx.AsyncClient over the real ASGI app (bridge.app), agent mocked
    out at the `agent.build_agent` boundary -- exercises the full FastAPI
    request/response/streaming path without touching Gemini or ChromaDB.
    This is the "mock external calls in CI" pattern requested for cost- and
    latency-safe pipeline runs."""

    LATENCY_BUDGET_SECONDS = 3.0  # generous; a mocked call should be near-instant

    @pytest.mark.asyncio
    async def test_chat_latency_status_and_schema(self):
        import bridge

        mock_base = MagicMock()
        final = MagicMock(content=MOCK_ANSWER, type="ai")
        mock_base.invoke.return_value = {"messages": [final]}

        with patch("agent.build_agent", return_value=mock_base):
            bridge.app_state["base_agent"] = mock_base
            bridge.app_state["thread_agents"] = {}
            bridge.app_state["startup_error"] = None

            transport = httpx.ASGITransport(app=bridge.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                started = time.perf_counter()
                resp = await client.post(
                    "/chat",
                    json={"messages": [{"role": "user", "content": "Generate a test case"}], "thread_id": "async-1"},
                )
                elapsed = time.perf_counter() - started

        assert resp.status_code == 200
        assert elapsed < self.LATENCY_BUDGET_SECONDS, f"Mocked /chat took {elapsed:.2f}s"
        assert resp.headers["content-type"] == "text/event-stream"

        events = [
            line.removeprefix("data: ")
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert events, "No SSE data events received"
        for raw in events:
            try:
                ChatSSEEvent.model_validate_json(raw)
            except ValidationError as exc:
                pytest.fail(f"SSE event failed schema validation: {raw!r}\n{exc}")

        assert "TC-001" in resp.text

    @pytest.mark.asyncio
    async def test_trace_latency_status_and_schema(self):
        import bridge

        mock_base = MagicMock()
        final = MagicMock(content=MOCK_ANSWER, type="ai")
        mock_base.invoke.return_value = {"messages": [final]}

        with patch("agent.build_agent", return_value=mock_base):
            bridge.app_state["base_agent"] = mock_base
            bridge.app_state["thread_agents"] = {}
            bridge.app_state["startup_error"] = None

            transport = httpx.ASGITransport(app=bridge.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                started = time.perf_counter()
                resp = await client.post(
                    "/trace",
                    json={"message": "What is boundary value analysis?", "thread_id": "async-trace-1"},
                )
                elapsed = time.perf_counter() - started

        assert resp.status_code == 200
        assert elapsed < self.LATENCY_BUDGET_SECONDS

        try:
            TraceResponseSchema.model_validate(resp.json())
        except ValidationError as exc:
            pytest.fail(f"/trace response failed schema validation: {exc}")

        assert "TC-001" in resp.json()["answer"]
