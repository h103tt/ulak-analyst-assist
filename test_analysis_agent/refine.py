"""Answer aggregation + refinement: a second pass over an agent turn's draft
answer, run after the retrieved tool results have been combined into one
context block, to catch citation/contradiction issues before the answer
reaches the user."""

from langchain_core.messages import SystemMessage, HumanMessage
from agent import get_llm


def _text_of(content) -> str:
    """Flatten LangChain message content to plain text. Some models (e.g.
    gemini-3.6-flash) return content as a list of blocks (``[{"type": "text",
    "text": "..."}, ...]``) instead of a plain string -- naive ``str()`` on
    that would show the user Python's repr of the block list instead of the
    text. Mirrors bridge.py's text_of(); duplicated rather than imported
    because bridge.py imports this module."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                parts.append(text if isinstance(text, str) else str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return "" if content is None else str(content)

REFINE_SYSTEM_PROMPT = (
    "You are a meticulous reviewer for a Senior System Test Engineer's answers. "
    "You are given the original question, the aggregated context retrieved from "
    "the standards knowledge base and/or the user's uploaded documents, and a "
    "draft answer already produced in a previous pass.\n\n"
    "Check the draft against the aggregated context:\n"
    "- Every citation (standard name, clause/section) must appear verbatim in "
    "the aggregated context. If a citation doesn't appear there, remove or "
    "correct it -- never invent one.\n"
    "- Test cases with the same or overlapping preconditions must not produce "
    "contradictory expected results. Resolve any contradiction you find using "
    "the literal wording of the requirement.\n"
    "- Requirements flagged as ambiguous must name the missing dimension "
    "(threshold/limits, duration/timing, error messaging, state persistence, "
    "recovery/unlock procedure).\n\n"
    "Only change what is actually wrong. If the draft already satisfies all of "
    "the above, return it unchanged. Output only the final answer text -- no "
    "commentary about what you checked or changed."
)


def aggregate_tool_context(messages) -> str:
    """Concatenate and de-duplicate ToolMessage content from an agent run
    (search_testing_standards / search_user_document results, possibly from
    more than one call) into one context block to check citations against."""
    seen = set()
    blocks = []
    for m in messages:
        if type(m).__name__ != "ToolMessage":
            continue
        content = _text_of(m.content).strip()
        if not content or content in seen:
            continue
        seen.add(content)
        blocks.append(content)
    return "\n\n---\n\n".join(blocks)


def refine_answer(
    question: str,
    aggregated_context: str,
    draft_answer: str,
    callbacks: list | None = None,
) -> str:
    """Run the refinement pass. Falls back to the draft untouched if nothing
    was retrieved -- there's no context to check citations against, so a
    second pass would only add latency with no grounding benefit.

    ``callbacks`` (optional) lets the caller attach a tracing handler (see
    pipeline_logging.HopTracingHandler) to this LLM call the same way it's
    attached to the main agent invoke -- without it, this second pass is
    invisible to that handler."""
    if not aggregated_context.strip():
        return draft_answer

    model = get_llm()
    response = model.invoke(
        [
            SystemMessage(content=REFINE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Original question:\n{question}\n\n"
                    f"Aggregated retrieved context:\n{aggregated_context}\n\n"
                    f"Draft answer:\n{draft_answer}"
                )
            ),
        ],
        config={"callbacks": callbacks} if callbacks else None,
    )
    return _text_of(response.content)
