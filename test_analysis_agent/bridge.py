import asyncio
import hashlib
import json
import os
import sys
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import uuid
import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)

import agent
import refine
import vector_embed


class ChatMessage(BaseModel):
    role: str
    content: str


class UploadedFile(BaseModel):
    id: str
    name: str
    url: str


class ChatRequestBody(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    context: str = ""
    thread_id: str = "default"
    files: list[UploadedFile] = Field(default_factory=list)


@dataclass
class ThreadAgentEntry:
    agent: object
    signature: str
    has_user_document: bool


app_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state["base_agent"] = None
    app_state["thread_agents"] = {}
    app_state["startup_error"] = None

    def _build_base_agent() -> None:
        try:
            app_state["base_agent"] = agent.build_agent()
        except Exception as exc:
            traceback.print_exc()
            app_state["startup_error"] = str(exc)

    # Build the (potentially slow) base agent off the event loop so the
    # ASGI app can start accepting connections right away. /health and the
    # /chat and /trace 503 checks below cover the window before it's ready.
    threading.Thread(target=_build_base_agent, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                out.append(text if isinstance(text, str) else str(text))
            else:
                out.append(str(item))
        return "".join(out)
    return "" if content is None else str(content)


def extract_answer(result) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "The agent returned no response."
    return text_of(messages[-1].content)


def extract_question(user_messages: list) -> str:
    """The actual user question for this turn, skipping the synthetic
    'Attached file context:' message chat() appends when files are attached."""
    for m in reversed(user_messages):
        content = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(content, str) and not content.startswith("Attached file context:"):
            return content
    return user_messages[-1].get("content", "") if user_messages else ""


def message_to_dict(message) -> dict:
    """Convert a LangChain message into a plain JSON-serializable dict so a
    caller can inspect tool calls and tool results, not just the final text."""
    kind = type(message).__name__
    item: dict = {"type": kind}

    name = getattr(message, "name", None)
    if name:
        item["name"] = name

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        item["tool_calls"] = [
            {
                "name": call.get("name"),
                "args": call.get("args") or call.get("tool_input"),
            }
            for call in tool_calls
        ]

    content = getattr(message, "content", None)
    item["content"] = text_of(content)[:4000]
    return item


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


CONTEXT_CHAR_BUDGET = 60000

# LangGraph's default recursion_limit (25 graph steps) can be too tight for
# multi-hop questions that need several sequential tool calls (e.g. comparing
# two standards) -- raised so those don't get cut off mid-reasoning.
AGENT_RECURSION_LIMIT = 50


def truncate_context(text: str, max_chars: int = CONTEXT_CHAR_BUDGET) -> str:
    """Cap attached-file context at max_chars, cutting on a whitespace boundary
    instead of mid-word/mid-clause so the tail isn't a broken fragment that
    contradicts what the KB retriever returns."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind(" ", 0, max_chars)
    if cut <= 0:
        cut = max_chars
    return text[:cut] + "\n...[context truncated]"


def files_signature(files: list[UploadedFile]) -> str:
    """Hash of stable attachment ids+names — rebuilt when the file set
    changes. Signed URLs are intentionally excluded: they change on every
    request and would otherwise defeat the per-thread agent cache."""
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda item: item.id):
        digest.update(f.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f.name.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def download_file(url: str, file_name: str) -> Path:
    """Download a signed storage URL into a temp file, preserving extension
    so the loader can detect the document type."""
    suffix = Path(file_name).suffix.lower() or ".txt"
    fd, path = tempfile.mkstemp(prefix="ulak_upload_", suffix=suffix)
    os.close(fd)
    try:
        with httpx.Client(timeout=60) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(path, "wb") as out_file:
                    for chunk in response.iter_bytes():
                        out_file.write(chunk)
        return Path(path)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        raise


def get_thread_agent(thread_id: str, files: list[UploadedFile], base_agent) -> object:
    """Return the cached per-thread agent, rebuilding it when the uploaded
    file set changes. Each thread gets its own InMemorySaver (conversation
    memory) and its own ephemeral Chroma collection (file memory)."""
    thread_agents: dict[str, ThreadAgentEntry] = app_state["thread_agents"]
    signature = files_signature(files)
    entry = thread_agents.get(thread_id)

    if entry is not None and entry.signature == signature:
        return entry.agent

    if not files:
        # No uploaded files -> plain base agent with thread-scoped checkpointer.
        no_files_agent = agent.build_agent()
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=no_files_agent, signature=signature, has_user_document=False
        )
        return no_files_agent

    downloaded: list[Path] = []
    try:
        for f in files:
            downloaded.append(download_file(f.url, f.name))

        session_tool = vector_embed.build_session_retriever_tool(
            [str(p) for p in downloaded],
            session_id=thread_id,
            collection_suffix=uuid.uuid4().hex[:8],
        )
        thread_tools = list(vector_embed.tools) + [session_tool]
        thread_agent = agent.build_agent(tools=thread_tools, has_user_document=True)
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=thread_agent, signature=signature, has_user_document=True
        )
        return thread_agent
    except Exception as exc:
        traceback.print_exc()
        print(f"[bridge] session indexing failed for thread {thread_id}: {exc}")
        # Fall back to a fresh per-thread agent (own checkpointer) so the
        # conversation still works without leaking chat memory across threads.
        # The cached signature is poisoned ("fallback:...") so a retry with the
        # same files re-attempts indexing instead of silently reusing this
        # tool-less fallback forever.
        fallback_agent = agent.build_agent()
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=fallback_agent, signature=f"fallback:{signature}", has_user_document=False
        )
        return fallback_agent
    finally:
        for path in downloaded:
            try:
                path.unlink()
            except OSError:
                pass


async def answer_stream(agent_instance, thread_id: str, user_messages: list, context_text: str):
    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }
        user_context = agent.Context(user_id=thread_id)

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: agent_instance.invoke(
                {"messages": user_messages},
                config=config,
                context=user_context,
            ),
        )
        answer = extract_answer(result)

        try:
            aggregated_context = refine.aggregate_tool_context(result.get("messages", []))
            question = extract_question(user_messages)
            answer = refine.refine_answer(question, aggregated_context, answer)
        except Exception:
            traceback.print_exc()
            # Refinement is a quality pass on top of an already-valid draft --
            # if it fails, keep streaming the draft rather than losing the turn.
    except Exception as exc:
        traceback.print_exc()
        answer = f"Agent error: {exc}"

    yield sse({"type": "text-start", "id": "text-1"})
    for i in range(0, len(answer), 48):
        yield sse({"type": "text-delta", "id": "text-1", "delta": answer[i : i + 48]})
        await asyncio.sleep(0.015)
    yield sse({"type": "text-end", "id": "text-1"})
    yield sse({"type": "finish", "finishReason": "stop"})
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_loaded": app_state.get("base_agent") is not None,
        "startup_error": app_state.get("startup_error"),
    }


class TraceRequestBody(BaseModel):
    message: str
    thread_id: str = "default"


@app.post("/trace")
async def trace(body: TraceRequestBody):
    """Run one turn and return the full message trace (tool calls + tool
    results), so callers can see whether the knowledge base was actually
    searched and what it returned.

    NOTE: this consumes the thread's conversation state (it invokes the same
    per-thread agent as /chat). Inspect traces with a throwaway thread_id or
    accept that the turn becomes part of the conversation.
    """
    base_agent = app_state.get("base_agent")
    if base_agent is None:
        return JSONResponse(status_code=503, content={"error": "Agent is still starting up"})

    body.message = body.message.strip()
    if not body.message:
        return JSONResponse(status_code=400, content={"error": "Message is required"})

    agent_instance = get_thread_agent(body.thread_id, [], base_agent)

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: agent_instance.invoke(
                {"messages": [{"role": "user", "content": body.message}]},
                config={
                    "configurable": {"thread_id": body.thread_id},
                    "recursion_limit": AGENT_RECURSION_LIMIT,
                },
                context=agent.Context(user_id=body.thread_id),
            ),
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})

    messages = [message_to_dict(m) for m in result.get("messages", [])]
    trace_data = {
        "thread_id": body.thread_id,
        "messages": messages,
        "kb_called": any(
            m["type"] == "AIMessage"
            and any(tc["name"] == "search_testing_standards" for tc in m.get("tool_calls", []))
            for m in messages
        )
        or any(
            m["type"] == "ToolMessage" and m.get("name") == "search_testing_standards"
            for m in messages
        ),
        "kb_returned_content": any(
            m["type"] == "ToolMessage"
            and m.get("name") == "search_testing_standards"
            and bool(str(m.get("content", "")).strip())
            for m in messages
        ),
        "tool_call_sequence": [
            tc["name"]
            for m in messages
            if m["type"] == "AIMessage"
            for tc in m.get("tool_calls", [])
        ],
        "retrieval_hop_count": sum(
            1
            for m in messages
            if m["type"] == "ToolMessage"
            and m.get("name") in ("search_testing_standards", "search_user_document")
        ),
        "answer": extract_answer(result),
    }
    return trace_data


@app.post("/chat")
async def chat(body: ChatRequestBody):
    base_agent = app_state.get("base_agent")
    if base_agent is None:
        return JSONResponse(status_code=503, content={"error": "Agent is still starting up"})

    if not body.messages:
        return JSONResponse(status_code=400, content={"error": "Messages are required"})

    user_messages = [
        {"role": m.role, "content": m.content}
        for m in body.messages
        if m.content and m.role in ("user", "assistant", "system")
    ]
    if not user_messages:
        return JSONResponse(status_code=400, content={"error": "No usable messages"})

    if body.context:
        user_messages.append(
            {"role": "user", "content": f"Attached file context:\n{truncate_context(body.context)}"}
        )

    agent_instance = get_thread_agent(body.thread_id, body.files, base_agent)

    headers = {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        "connection": "keep-alive",
    }

    return StreamingResponse(
        answer_stream(agent_instance, body.thread_id, user_messages, body.context),
        headers=headers,
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    import sys
    import uvicorn

    def run_server() -> None:
        # uvicorn.run with reload=True cannot work here: reload_dirs only
        # accepts directories, and the .venv lives inside AGENT_DIR, so any
        # directory watch restarts on every site-packages import. Run
        # reload-free by default.
        uvicorn.run(app, host="127.0.0.1", port=8010)

    # `python bridge.py --watch` restarts the server when agent source files
    # change, using watchfiles instead of uvicorn's reload so .venv and
    # knowledge_base can be excluded.
    if "--watch" in sys.argv:
        from watchfiles import run_process

        def watch_filter(change: int, path: str) -> bool:
            path = str(path)
            return path.endswith(".py") and f"{os.sep}.venv{os.sep}" not in path

        run_process(AGENT_DIR, target=run_server, watch_filter=watch_filter)
    else:
        run_server()