"""Inspect what the agent actually did during a turn — every tool call, tool
result, and message — instead of just the final answer.

Usage:
    uv run --directory test_analysis_agent python inspect_trace.py "your question here"

If no question is given, one is read from stdin.
"""

import sys

import agent as agent_mod
import vector_embed


def message_label(message) -> str:
    kind = type(message).__name__
    if kind == "HumanMessage":
        return "user"
    if kind == "AIMessage":
        return "assistant"
    if kind == "SystemMessage":
        return "system"
    if kind == "ToolMessage":
        return f"tool::{getattr(message, 'name', '?')}"
    return kind


def summarize(message) -> None:
    kind = type(message).__name__
    print(f"[{message_label(message)}]")

    if hasattr(message, "tool_calls") and message.tool_calls:
        for call in message.tool_calls:
            args = call.get("args") or call.get("tool_input") or {}
            print(f"  -> CALL {call.get('name')}({args})")

    content = str(message.content).strip()
    if content:
        print(f"     {content[:500]}")
    print()


def run_trace(prompt: str) -> None:
    print(f"QUESTION: {prompt}\n")

    agent = agent_mod.build_agent()
    thread_id = "trace-inspect"
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 50},
        context=agent_mod.Context(user_id=thread_id),
    )

    for message in result["messages"]:
        summarize(message)

    kb_called = any(
        (hasattr(m, "tool_calls") and any(tc.get("name") == "search_testing_standards" for tc in m.tool_calls))
        or (type(m).__name__ == "ToolMessage" and getattr(m, "name", None) == "search_testing_standards")
        for m in result["messages"]
    )
    kb_returned_content = any(
        type(m).__name__ == "ToolMessage"
        and getattr(m, "name", None) == "search_testing_standards"
        and len(str(m.content).strip()) > 0
        for m in result["messages"]
    )

    print("== VERDICT ==")
    print(f"searched knowledge_base?        {kb_called}")
    print(f"knowledge_base returned content? {kb_returned_content}")
    print(f"tools available:                {[t.name for t in vector_embed.tools]}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ").strip()
    run_trace(question)
