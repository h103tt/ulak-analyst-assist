"""
Refine (Second-Pass) Logic Tests
===================================
Isolated tests for refine.py -- the citation/contradiction-checking pass
that runs after the agent's draft answer, before it reaches the user -- plus
its integration point in bridge._compute_answer_sync (the try/except
fallback that must never let a refine failure lose the turn).

Run:
    cd test_analysis_agent
    pytest tests/e2e/test_refine_logic.py -v
    pytest tests/e2e/test_refine_logic.py -v -m integration
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tests.e2e._helpers import requires_ollama, is_model_pulled  # noqa: E402

pytestmark = pytest.mark.e2e


# ===================================================================
# 1. aggregate_tool_context (fast, pure function)
# ===================================================================
class TestAggregateToolContext:
    def test_collects_only_tool_messages(self):
        from refine import aggregate_tool_context

        messages = [
            HumanMessage(content="question"),
            AIMessage(content="thinking..."),
            ToolMessage(content="Chunk A content", name="search_testing_standards", tool_call_id="1"),
            AIMessage(content="final answer"),
        ]
        result = aggregate_tool_context(messages)
        assert "Chunk A content" in result
        assert "thinking..." not in result
        assert "final answer" not in result

    def test_deduplicates_identical_tool_content(self):
        from refine import aggregate_tool_context

        messages = [
            ToolMessage(content="Same chunk", name="search_testing_standards", tool_call_id="1"),
            ToolMessage(content="Same chunk", name="search_testing_standards", tool_call_id="2"),
            ToolMessage(content="Different chunk", name="search_testing_standards", tool_call_id="3"),
        ]
        result = aggregate_tool_context(messages)
        assert result.count("Same chunk") == 1
        assert "Different chunk" in result

    def test_skips_empty_and_whitespace_only_content(self):
        from refine import aggregate_tool_context

        messages = [
            ToolMessage(content="", name="search_testing_standards", tool_call_id="1"),
            ToolMessage(content="   ", name="search_testing_standards", tool_call_id="2"),
            ToolMessage(content="Real content", name="search_testing_standards", tool_call_id="3"),
        ]
        result = aggregate_tool_context(messages)
        assert result.strip() == "Real content"

    def test_no_tool_messages_returns_empty_string(self):
        from refine import aggregate_tool_context

        messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert aggregate_tool_context(messages) == ""

    def test_preserves_call_order(self):
        from refine import aggregate_tool_context

        messages = [
            ToolMessage(content="First", name="t", tool_call_id="1"),
            ToolMessage(content="Second", name="t", tool_call_id="2"),
            ToolMessage(content="Third", name="t", tool_call_id="3"),
        ]
        result = aggregate_tool_context(messages)
        assert result.index("First") < result.index("Second") < result.index("Third")


# ===================================================================
# 2. refine_answer short-circuit and correction flow (mocked)
# ===================================================================
class TestRefineAnswerMocked:
    def test_empty_context_short_circuits_without_calling_llm(self):
        from refine import refine_answer

        with patch("refine.get_llm") as mock_get_llm:
            result = refine_answer("What is X?", "", "draft answer")

        mock_get_llm.assert_not_called()
        assert result == "draft answer"

    def test_whitespace_only_context_short_circuits(self):
        from refine import refine_answer

        with patch("refine.get_llm") as mock_get_llm:
            result = refine_answer("What is X?", "   \n  ", "draft answer")

        mock_get_llm.assert_not_called()
        assert result == "draft answer"

    def test_non_empty_context_invokes_model_and_returns_its_output(self):
        from refine import refine_answer

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="corrected answer")

        with patch("refine.get_llm", return_value=mock_model):
            result = refine_answer("What is X?", "some retrieved context", "draft answer")

        assert result == "corrected answer"
        mock_model.invoke.assert_called_once()

    def test_model_invoked_with_system_and_human_message(self):
        from refine import refine_answer, REFINE_SYSTEM_PROMPT

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="corrected")

        with patch("refine.get_llm", return_value=mock_model):
            refine_answer("What is X?", "the context", "the draft")

        call_args = mock_model.invoke.call_args
        messages_arg = call_args[0][0]
        assert isinstance(messages_arg[0], SystemMessage)
        assert messages_arg[0].content == REFINE_SYSTEM_PROMPT
        assert isinstance(messages_arg[1], HumanMessage)
        assert "What is X?" in messages_arg[1].content
        assert "the context" in messages_arg[1].content
        assert "the draft" in messages_arg[1].content

    def test_callbacks_passed_through_to_model_config(self):
        from refine import refine_answer

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="corrected")
        fake_callback = MagicMock()

        with patch("refine.get_llm", return_value=mock_model):
            refine_answer("q", "ctx", "draft", callbacks=[fake_callback])

        call_kwargs = mock_model.invoke.call_args[1]
        assert call_kwargs["config"]["callbacks"] == [fake_callback]

    def test_no_callbacks_omits_config(self):
        from refine import refine_answer

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content="corrected")

        with patch("refine.get_llm", return_value=mock_model):
            refine_answer("q", "ctx", "draft")

        call_kwargs = mock_model.invoke.call_args[1]
        assert call_kwargs.get("config") is None

    def test_non_string_model_content_is_coerced_to_string(self):
        from refine import refine_answer

        mock_model = MagicMock()
        mock_model.invoke.return_value = MagicMock(content=["list", "content"])

        with patch("refine.get_llm", return_value=mock_model):
            result = refine_answer("q", "ctx", "draft")

        assert isinstance(result, str)


# ===================================================================
# 3. bridge._compute_answer_sync <-> refine integration (mocked)
# ===================================================================
class TestBridgeRefineIntegration:
    def _agent_with_tool_context(self, draft_text: str):
        agent_mock = MagicMock()
        tool_msg = ToolMessage(content="retrieved context chunk", name="search_testing_standards", tool_call_id="1")
        final = AIMessage(content=draft_text)
        agent_mock.invoke.return_value = {"messages": [tool_msg, final]}
        return agent_mock

    def test_refine_success_replaces_raw_answer(self):
        import bridge

        agent_mock = self._agent_with_tool_context("raw draft")

        with patch("refine.refine_answer", return_value="refined answer") as mock_refine:
            answer = bridge._compute_answer_sync(
                agent_mock, "thread-1", [{"role": "user", "content": "hi"}], time.perf_counter()
            )

        mock_refine.assert_called_once()
        assert answer == "refined answer"

    def test_refine_exception_falls_back_to_raw_answer(self):
        """The comment in bridge.py is explicit: refinement is a
        quality-only step -- if it fails, the turn must still return the
        original draft rather than propagating the exception and losing
        the answer entirely."""
        import bridge

        agent_mock = self._agent_with_tool_context("raw draft that must survive")

        with patch("refine.refine_answer", side_effect=RuntimeError("judge model unreachable")):
            answer = bridge._compute_answer_sync(
                agent_mock, "thread-1", [{"role": "user", "content": "hi"}], time.perf_counter()
            )

        assert answer == "raw draft that must survive"

    def test_no_tool_context_skips_refine_llm_call(self):
        """No ToolMessages in the result -> aggregate_tool_context returns
        "" -> refine_answer's own short-circuit should avoid the LLM call,
        end to end through the real (non-mocked) refine module."""
        import bridge

        agent_mock = MagicMock()
        final = AIMessage(content="answer with no retrieval")
        agent_mock.invoke.return_value = {"messages": [final]}

        with patch("refine.get_llm") as mock_get_llm:
            answer = bridge._compute_answer_sync(
                agent_mock, "thread-1", [{"role": "user", "content": "hi"}], time.perf_counter()
            )

        mock_get_llm.assert_not_called()
        assert answer == "answer with no retrieval"


# ===================================================================
# 4. Live correction of a fabricated citation (requires_ollama)
# ===================================================================
@pytest.mark.integration
@requires_ollama
class TestRefineAnswerLive:
    def test_refine_removes_citation_not_present_in_context(self):
        import agent

        if not is_model_pulled(agent.MODEL_NAME):
            pytest.skip(f"Configured generation model '{agent.MODEL_NAME}' is not pulled in Ollama")

        from refine import refine_answer

        real_context = (
            '<chunk id="mil-461-001" source="MIL-STD-461.pdf" standard="MIL-STD-461" '
            'category="Environmental_and_hardware" section="1. Scope" page="1">'
            "This standard establishes electromagnetic interference requirements "
            "for the control of electromagnetic emissions and susceptibility "
            "characteristics of electronic equipment.</chunk>"
        )
        fabricated_draft = (
            "MIL-STD-461 defines EMI requirements (MIL-STD-461, Section 9.9.9), "
            "which does not appear anywhere in the retrieved text above."
        )

        result = refine_answer(
            "What does MIL-STD-461 cover?",
            real_context,
            fabricated_draft,
        )

        assert "9.9.9" not in result, (
            f"refine_answer did not remove/correct a citation absent from the "
            f"aggregated context. Result: {result!r}"
        )
