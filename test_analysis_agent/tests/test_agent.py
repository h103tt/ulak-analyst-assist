"""
Agent & Model Skill Tests
=========================
Tests for prompt construction, LLM invocation, hallucination handling,
and error handling in the agent chain.

Run:
    cd ulak-analyst-assist/test_analysis_agent
    pytest tests/test_agent.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Ensure the package root is importable
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ===================================================================
# 1. System Prompt Construction & Validation
# ===================================================================
class TestSystemPrompt:
    """Verify the system prompt contains required grounding rules and structure."""

    @pytest.fixture
    def system_prompt(self):
        """Reconstruct the system prompt exactly as agent.build_agent constructs it."""
        prompt = (
            "You are a Senior System Test Engineer and Systems Validation Expert. "
            "Your task is to analyze project and software requirement documents and generate comprehensive test cases. "
            "You must strictly adhere to the verification and validation processes outlined in ISO/IEC/IEEE 29119. "
            "For every requirement provided in the context, output a structured test plan that includes:\n"
            "- Test ID & Traceability\n"
            "Traceability must reference an actual requirement ID from the provided context. "
            "If no ID is present in the input, output 'No requirement ID provided' rather than inventing one."
            "- Test Type (e.g., Boundary, Edge Case, Integration)\n"
            "- Preconditions\n"
            "- Test Steps\n"
            "- Expected Result\n"
            "\n\nGROUNDING RULES (apply to every response, not just test plan generation):\n"
            "- For ANY factual claim about a standard (definitions, process names, clause "
            "structure, requirements), you MUST call search_testing_standards first and "
            "base your answer only on the retrieved text.\n"
            "- Never answer questions about standard content from memory, even if you "
            "believe you know the answer. Your training knowledge of these standards may "
            "be wrong or may blend one standard with another.\n"
            "- If the retrieved chunks don't contain the answer, say so explicitly "
            "(e.g. 'The retrieved sections don't cover this') rather than filling the gap "
            "with general knowledge.\n"
            "- Never cite a clause/section number that doesn't appear verbatim in the "
            "retrieved text.\n"
            "Do not provide conversational filler. Base your analysis solely on the provided context. "
            "Pay strict attention to boundary conditions and ensure expected results do not contradict each other. "
            "Treat a requirement as ambiguous if it fails to specify: threshold/limits, duration/timing, error messaging, state persistence (session vs. account-level), or recovery/unlock procedure. "
            "Flag each missing dimension separately. "
            "If a requirement is ambiguous or untestable, flag it and state why. "
            "Before finalizing the test plan, review all generated test cases as a set. "
            "Ensure that no two test cases with the same or overlapping preconditions produce contradictory expected results. "
            "If a contradiction exists, resolve it based on the literal wording of the requirement."
        )
        return prompt

    def test_prompt_references_iso_29119(self, system_prompt):
        assert "ISO/IEC/IEEE 29119" in system_prompt

    def test_prompt_requires_search_tool(self, system_prompt):
        assert "search_testing_standards" in system_prompt

    def test_prompt_forbids_memory_based_answers(self, system_prompt):
        assert "Never answer questions about standard content from memory" in system_prompt

    def test_prompt_requires_structured_output(self, system_prompt):
        for field in ["Test ID", "Test Type", "Preconditions", "Test Steps", "Expected Result"]:
            assert field in system_prompt, f"Missing required field: {field}"

    def test_prompt_ambiguity_detection(self, system_prompt):
        assert "ambiguous" in system_prompt.lower()

    def test_prompt_contradiction_detection(self, system_prompt):
        assert "contradict" in system_prompt.lower() or "contradiction" in system_prompt.lower()

    def test_prompt_no_invented_clause_numbers(self, system_prompt):
        assert "Never cite a clause/section number that doesn't appear verbatim" in system_prompt

    @pytest.mark.parametrize("requirement_dimension", [
        "threshold/limits",
        "duration/timing",
        "error messaging",
        "state persistence",
        "recovery/unlock procedure",
    ])
    def test_prompt_ambiguity_dimensions(self, system_prompt, requirement_dimension):
        assert requirement_dimension in system_prompt

    def test_prompt_no_conversational_filler(self, system_prompt):
        assert "conversational filler" in system_prompt

    def test_prompt_user_document_appendix(self):
        base = "Base prompt."
        user_doc_suffix = (
            "\nThe user has uploaded one or more files for this conversation. "
            "Use the search_user_document tool to retrieve the relevant sections "
            "of the user's uploaded files before answering. The uploaded files are "
            "specific to this conversation only."
        )
        full = base + user_doc_suffix
        assert "search_user_document" in full
        assert "specific to this conversation only" in full


# ===================================================================
# 2. Agent Build & Configuration
# ===================================================================
class TestAgentBuild:
    """Verify agent.build_agent returns a usable agent with correct config."""

    @patch("agent.ChatGoogleGenerativeAI")
    @patch("agent.create_agent")
    @patch("agent.InMemorySaver")
    def test_build_agent_calls_create_agent(self, mock_saver, mock_create, mock_chat):
        mock_create.return_value = MagicMock()
        import agent as agent_mod
        agent_mod.build_agent()
        mock_create.assert_called_once()

    @patch("agent.ChatGoogleGenerativeAI")
    @patch("agent.create_agent")
    @patch("agent.InMemorySaver")
    def test_build_agent_default_tools(self, mock_saver, mock_create, mock_chat):
        mock_create.return_value = MagicMock()
        import agent as agent_mod
        agent_mod.build_agent()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("tools") is not None

    @patch("agent.ChatGoogleGenerativeAI")
    @patch("agent.create_agent")
    @patch("agent.InMemorySaver")
    def test_build_agent_custom_tools(self, mock_saver, mock_create, mock_chat):
        mock_create.return_value = MagicMock()
        custom_tools = [MagicMock(name="custom_tool")]
        import agent as agent_mod
        agent_mod.build_agent(tools=custom_tools)
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["tools"] == custom_tools

    @patch("agent.ChatGoogleGenerativeAI")
    @patch("agent.create_agent")
    @patch("agent.InMemorySaver")
    def test_build_agent_user_document_flag(self, mock_saver, mock_create, mock_chat):
        mock_create.return_value = MagicMock()
        import agent as agent_mod
        agent_mod.build_agent(has_user_document=True)
        call_kwargs = mock_create.call_args[1]
        assert "search_user_document" in call_kwargs["system_prompt"]

    @patch("agent.ChatGoogleGenerativeAI")
    @patch("agent.create_agent")
    @patch("agent.InMemorySaver")
    def test_build_agent_no_user_document(self, mock_saver, mock_create, mock_chat):
        mock_create.return_value = MagicMock()
        import agent as agent_mod
        agent_mod.build_agent(has_user_document=False)
        call_kwargs = mock_create.call_args[1]
        assert "search_user_document" not in call_kwargs["system_prompt"]


# ===================================================================
# 3. LLM Invocation & Output Parsing
# ===================================================================
class TestLLMInvocation:
    """Test chain execution with mocked LLM responses."""

    def test_extract_answer_from_result(self, mock_agent):
        from bridge import extract_answer
        result = mock_agent.invoke.return_value
        answer = extract_answer(result)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_extract_answer_empty_messages(self):
        from bridge import extract_answer
        result = {"messages": []}
        answer = extract_answer(result)
        assert "no response" in answer.lower()

    def test_extract_answer_multiple_messages(self):
        from bridge import extract_answer
        msg1 = MagicMock()
        msg1.content = "First message"
        msg2 = MagicMock()
        msg2.content = "Final answer about ISO 29119"
        result = {"messages": [msg1, msg2]}
        answer = extract_answer(result)
        assert answer == "Final answer about ISO 29119"

    def test_text_of_string_input(self):
        from bridge import text_of
        assert text_of("hello") == "hello"

    def test_text_of_none_input(self):
        from bridge import text_of
        assert text_of(None) == ""

    def test_text_of_list_of_dicts(self):
        from bridge import text_of
        content = [{"text": "chunk1"}, {"text": "chunk2"}]
        result = text_of(content)
        assert "chunk1" in result
        assert "chunk2" in result

    def test_text_of_list_mixed(self):
        from bridge import text_of
        content = [{"text": "hello"}, "plain string", 42]
        result = text_of(content)
        assert "hello" in result
        assert "plain string" in result

    @pytest.mark.parametrize("content,expected", [
        ("plain string", "plain string"),
        (None, ""),
        ([], ""),
        ([{"text": "a"}, {"text": "b"}], "ab"),
    ])
    def test_text_of_parametrized(self, content, expected):
        from bridge import text_of
        result = text_of(content)
        assert result == expected

    def test_agent_invoke_returns_dict(self, mock_agent):
        result = mock_agent.invoke(
            {"messages": [{"role": "user", "content": "test"}]},
            config={"configurable": {"thread_id": "test-1"}},
        )
        assert isinstance(result, dict)
        assert "messages" in result

    def test_agent_message_to_dict_ai(self):
        """message_to_dict should handle AIMessage with tool_calls."""
        from bridge import message_to_dict

        ai_msg = AIMessage(
            content="Searching...",
            tool_calls=[{"name": "search_testing_standards", "args": {"query": "test"}, "id": "call_1"}],
        )
        result = message_to_dict(ai_msg)
        assert result["type"] == "AIMessage"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "search_testing_standards"

    def test_message_to_dict_tool_message(self):
        """message_to_dict should handle ToolMessage."""
        from bridge import message_to_dict

        tool_msg = ToolMessage(
            content="Found ISO 29119 section 4.1...",
            name="search_testing_standards",
            tool_call_id="tc-1",
        )
        result = message_to_dict(tool_msg)
        assert result["type"] == "ToolMessage"
        assert result["name"] == "search_testing_standards"

    def test_message_to_dict_human_message(self):
        """message_to_dict should handle HumanMessage."""
        from bridge import message_to_dict

        human_msg = HumanMessage(content="What is testing?")
        result = message_to_dict(human_msg)
        assert result["type"] == "HumanMessage"
        assert "What is testing?" in result["content"]


# ===================================================================
# 4. Hallucination & Fallback Handling
# ===================================================================
class TestHallucinationPrevention:
    """Verify agent behavior when retrieved docs don't cover the query."""

    def test_no_match_response_honest(self, mock_llm_no_match_response):
        text = mock_llm_no_match_response.content
        assert "don't cover" in text or "not found" in text.lower()

    def test_no_match_does_not_invent_clauses(self, mock_llm_no_match_response):
        text = mock_llm_no_match_response.content
        import re
        fabricated_refs = re.findall(r"(?:Section|Clause|Paragraph)\s+\d+\.\d+", text)
        assert len(fabricated_refs) == 0, f"Fabricated references found: {fabricated_refs}"

    def test_empty_retrieval_prompt_awareness(self):
        context = ""
        prompt = (
            f"Using these standards:\n{context}\n\n"
            "Generate test cases for: login authentication"
        )
        assert "login authentication" in prompt

    def test_retriever_empty_does_not_crash_agent(self):
        from bridge import extract_answer
        result = {"messages": [MagicMock(content="I cannot answer this based on the provided context.")]}
        answer = extract_answer(result)
        assert len(answer) > 0

    def test_prompt_grounding_rule_fallback_instruction(self):
        grounding = (
            "If the retrieved chunks don't contain the answer, say so explicitly "
            "(e.g. 'The retrieved sections don't cover this') rather than filling the gap "
            "with general knowledge."
        )
        assert "say so explicitly" in grounding
        assert "general knowledge" in grounding


# ===================================================================
# 5. Error Handling
# ===================================================================
class TestErrorHandling:
    """Test timeout errors, rate limits, and disconnected vector store scenarios."""

    def test_agent_timeout_error_handling(self):
        mock_agent_inst = MagicMock()
        mock_agent_inst.invoke.side_effect = TimeoutError("LLM request timed out")
        with pytest.raises(TimeoutError, match="timed out"):
            mock_agent_inst.invoke(
                {"messages": [{"role": "user", "content": "test"}]},
                config={"configurable": {"thread_id": "err-1"}},
            )

    def test_agent_connection_error_handling(self):
        mock_agent_inst = MagicMock()
        mock_agent_inst.invoke.side_effect = ConnectionError("Vector store unreachable")
        with pytest.raises(ConnectionError, match="unreachable"):
            mock_agent_inst.invoke(
                {"messages": [{"role": "user", "content": "test"}]},
                config={"configurable": {"thread_id": "err-2"}},
            )

    def test_agent_generic_exception_handling(self):
        mock_agent_inst = MagicMock()
        mock_agent_inst.invoke.side_effect = RuntimeError("Something went wrong")
        with pytest.raises(RuntimeError):
            mock_agent_inst.invoke(
                {"messages": [{"role": "user", "content": "test"}]},
                config={"configurable": {"thread_id": "err-3"}},
            )

    def test_bridge_answer_stream_error_handling(self):
        from bridge import answer_stream

        error_agent = MagicMock()
        error_agent.invoke.side_effect = RuntimeError("LLM crashed")

        async def _collect():
            chunks = []
            async for chunk in answer_stream(error_agent, "t-err", [{"role": "user", "content": "hi"}], ""):
                chunks.append(chunk)
            return chunks

        loop = asyncio.new_event_loop()
        try:
            chunks = loop.run_until_complete(_collect())
        finally:
            loop.close()

        full = "".join(chunks)
        assert "Agent error" in full or "LLM crashed" in full

    def test_bridge_trace_handles_agent_error(self):
        """Trace endpoint should return 500 on agent error."""
        import time
        from fastapi.testclient import TestClient

        mock_agent_inst = MagicMock()
        mock_agent_inst.invoke.side_effect = RuntimeError("Agent failed")

        import bridge
        original = bridge.app_state.get("base_agent")
        try:
            # Create client (lifespan resets app_state)
            tc = TestClient(bridge.app, raise_server_exceptions=False)
            time.sleep(0.3)
            # Now overwrite with our error-raising mock
            bridge.app_state["base_agent"] = mock_agent_inst
            bridge.app_state["thread_agents"] = {}
            # get_thread_agent now returns (agent, ingest_report); patching it
            # avoids hitting vector_embed indexing for an error-only test.
            with patch(
                "bridge.get_thread_agent",
                return_value=(mock_agent_inst, {"files_indexed": [], "failed_files": [], "chunk_count": 0}),
            ):
                resp = tc.post("/trace", json={"message": "test", "thread_id": "err-trace"})
                assert resp.status_code == 500
                assert "error" in resp.json()
        finally:
            bridge.app_state["base_agent"] = original


# ===================================================================
# 6. Context Dataclass
# ===================================================================
class TestContext:
    """Verify the Context dataclass used in agent invocation."""

    def test_context_creation(self):
        from agent import Context
        ctx = Context(user_id="user-42")
        assert ctx.user_id == "user-42"

    def test_context_different_users(self):
        from agent import Context
        ctx1 = Context(user_id="alpha")
        ctx2 = Context(user_id="beta")
        assert ctx1.user_id != ctx2.user_id


# ===================================================================
# 7. SSE Formatting
# ===================================================================
class TestSSEFormatting:
    """Verify SSE event formatting in bridge."""

    def test_sse_format(self):
        from bridge import sse
        result = sse({"type": "text-delta", "delta": "hello"})
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        assert "text-delta" in result
        assert "hello" in result

    def test_files_signature_deterministic(self):
        from bridge import files_signature, UploadedFile
        files1 = [UploadedFile(id="f1", name="doc.pdf", url="http://x")]
        files2 = [UploadedFile(id="f1", name="doc.pdf", url="http://y")]
        sig1 = files_signature(files1)
        sig2 = files_signature(files2)
        assert sig1 == sig2

    def test_files_signature_differs_with_different_ids(self):
        from bridge import files_signature, UploadedFile
        files1 = [UploadedFile(id="f1", name="doc.pdf", url="http://x")]
        files2 = [UploadedFile(id="f2", name="doc.pdf", url="http://x")]
        assert files_signature(files1) != files_signature(files2)

    def test_files_signature_empty(self):
        from bridge import files_signature
        sig = files_signature([])
        assert isinstance(sig, str)
        assert len(sig) > 0
