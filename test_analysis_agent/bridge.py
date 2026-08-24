import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
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
import rag_debug
import vector_embed
from pipeline_logging import (
    add_stage,
    add_tool_summaries,
    create_run,
    get_run,
    list_runs,
    setup_logging,
    trace_id_var,
)


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
    ingest_report: dict


app_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state["base_agent"] = None
    app_state["thread_agents"] = {}
    app_state["startup_error"] = None

    # Ensure the persistent upload + per-session vector dirs exist.
    os.makedirs(vector_embed.USER_UPLOADS_DIR, exist_ok=True)
    os.makedirs(vector_embed.USER_COLLECTIONS_DIR, exist_ok=True)
    setup_logging()

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


def download_file(attachment_id: str, url: str, file_name: str) -> Path:
    """Download a signed storage URL into the persistent uploads directory,
    preserving the extension so the loader can detect the document type.

    Files persist on disk (uploads/<attachment_id>__<safe_name>) instead of
    being deleted after each request, so tools such as
    ``get_document_structure`` can still read them later and re-parses are
    cheap. The file name is sanitized to prevent path traversal.
    """
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", file_name) or "upload"
    dest_dir = Path(vector_embed.USER_UPLOADS_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{attachment_id}__{safe_name}"
    try:
        with httpx.Client(timeout=60) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(dest, "wb") as out_file:
                    for chunk in response.iter_bytes():
                        out_file.write(chunk)
        vector_embed.register_upload(file_name, dest)
        return dest
    except Exception:
        # Remove a partially written file, then surface the error so the
        # caller can report it instead of silently dropping the attachment.
        try:
            dest.unlink()
        except OSError:
            pass
        raise


def get_thread_agent(
    thread_id: str, files: list[UploadedFile], base_agent
) -> tuple[object, dict]:
    """Return the cached per-thread agent and its ingest report, rebuilding
    when the uploaded file set changes. Each thread gets its own
    InMemorySaver (conversation memory) and its own persistent Chroma
    collection (file memory, named after the thread id).

    Partial failures are NOT swallowed silently anymore: the returned ingest
    report carries ``failed_files`` / ``error`` so /chat can stream a warning
    to the client while still answering with whatever loaded successfully.
    """
    thread_agents: dict[str, ThreadAgentEntry] = app_state["thread_agents"]
    signature = files_signature(files)
    entry = thread_agents.get(thread_id)

    if entry is not None and entry.signature == signature:
        return entry.agent, entry.ingest_report

    if not files:
        # No uploaded files -> plain base agent with thread-scoped checkpointer.
        no_files_agent = agent.build_agent()
        report = {"files_indexed": [], "failed_files": [], "chunk_count": 0}
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=no_files_agent,
            signature=signature,
            has_user_document=False,
            ingest_report=report,
        )
        return no_files_agent, report

    downloaded: list[Path] = []
    failures: list[tuple[str, str]] = []
    for f in files:
        try:
            downloaded.append(download_file(f.id, f.url, f.name))
        except Exception as exc:
            traceback.print_exc()
            failures.append((f.name, str(exc)))
            add_stage(
                trace_id_var.get(),
                "file_download",
                status="error",
                file=f.name,
                error=str(exc),
            )
    if not downloaded:
        # Nothing could even be fetched. Fall back to a tool-less agent with a
        # poisoned signature so a retry re-attempts the download.
        fallback_agent = agent.build_agent()
        report = {
            "files_indexed": [],
            "failed_files": failures,
            "chunk_count": 0,
            "error": "All attachments failed to download",
        }
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=fallback_agent,
            signature=f"fallback:{signature}",
            has_user_document=False,
            ingest_report=report,
        )
        return fallback_agent, report

    try:
        session_tool, ingest_report = vector_embed.build_session_retriever_tool(
            [str(p) for p in downloaded],
            session_id=thread_id,
            collection_suffix=uuid.uuid4().hex[:8],
        )
        ingest_report["failed_files"] = ingest_report.get("failed_files", []) + failures
        thread_tools = list(vector_embed.tools) + [session_tool]
        thread_agent = agent.build_agent(tools=thread_tools, has_user_document=True)
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=thread_agent,
            signature=signature,
            has_user_document=True,
            ingest_report=ingest_report,
        )
        return thread_agent, ingest_report
    except Exception as exc:
        traceback.print_exc()
        print(f"[bridge] session indexing failed for thread {thread_id}: {exc}")
        # Fall back to a fresh per-thread agent (own checkpointer) so the
        # conversation still works without leaking chat memory across threads.
        # The cached signature is poisoned ("fallback:...") so a retry with the
        # same files re-attempts indexing instead of silently reusing this
        # tool-less fallback forever.
        fallback_agent = agent.build_agent()
        report = {
            "files_indexed": [],
            "failed_files": failures
            + [(f.name, "indexing failed") for f in files],
            "chunk_count": 0,
            "error": str(exc),
        }
        thread_agents[thread_id] = ThreadAgentEntry(
            agent=fallback_agent,
            signature=f"fallback:{signature}",
            has_user_document=False,
            ingest_report=report,
        )
        return fallback_agent, report


async def answer_stream(
    agent_instance,
    thread_id: str,
    user_messages: list,
    context_text: str,
    trace_id: str = "",
    ingest_report: dict | None = None,
):
    answer = ""
    start = time.perf_counter()
    try:
        config = {"configurable": {"thread_id": thread_id}}
        user_context = agent.Context(user_id=thread_id)
        last_user_text = next(
            (m["content"] for m in reversed(user_messages) if m["role"] == "user"),
            "",
        )
        rag_debug.log_query(thread_id, last_user_text)

        def _invoke():
            # contextvars do NOT propagate into the executor thread, so set the
            # trace_id there so logs emitted during the agent run correlate.
            if trace_id:
                trace_id_var.set(trace_id)
            return agent_instance.invoke(
                {"messages": user_messages},
                config=config,
                context=user_context,
            )

        result = await asyncio.get_running_loop().run_in_executor(None, _invoke)
        answer = extract_answer(result)
        rag_debug.log_generation(
            agent.MODEL_NAME,
            time.perf_counter() - start,
            answer,
            usage=rag_debug.extract_usage(result.get("messages", [])),
        )
        rag_debug.log_prompt_assembled(
            thread_id,
            agent.get_system_prompt(has_user_document=ingest_report is not None),
            [
                (
                    getattr(m, "name", "") or "tool",
                    text_of(getattr(m, "content", ""))[:2000],
                )
                for m in result.get("messages", [])
                if type(m).__name__ == "ToolMessage"
            ],
            last_user_text,
        )

        if trace_id:
            add_stage(
                trace_id,
                "agent_invoke_done",
                status="ok",
                answer_chars=len(answer),
            )
            summaries = [
                {
                    "name": getattr(m, "name", "") or "",
                    "content_preview": text_of(getattr(m, "content", ""))[:400],
                }
                for m in result.get("messages", [])
                if type(m).__name__ == "ToolMessage"
            ]
            add_tool_summaries(trace_id, summaries)
    except Exception as exc:
        traceback.print_exc()
        answer = f"Agent error: {exc}"
        if trace_id:
            add_stage(trace_id, "agent_invoke_done", status="error", error=str(exc))

    start_payload = {"type": "text-start", "id": "text-1"}
    if ingest_report and ingest_report.get("failed_files"):
        start_payload["file_indexing"] = {
            "warnings": [
                f"{name}: {err}" for name, err in ingest_report["failed_files"]
            ],
            "files_indexed": ingest_report.get("files_indexed", []),
        }
    yield sse(start_payload)
    for i in range(0, len(answer), 48):
        yield sse({"type": "text-delta", "id": "text-1", "delta": answer[i : i + 48]})
        await asyncio.sleep(0.015)
    yield sse({"type": "text-end", "id": "text-1"})
    yield sse({"type": "finish", "finishReason": "stop"})
    yield "data: [DONE]\n\n"


def _tool_summary_intermediate(messages: list[dict]) -> dict:
    """Aggregate tool-call info from a message trace (used by /trace)."""
    tool_messages = [m for m in messages if m["type"] == "ToolMessage"]
    return {
        "tools_called": [
            {
                "name": m.get("name"),
                "content_preview": str(m.get("content", ""))[:400],
            }
            for m in tool_messages
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_loaded": app_state.get("base_agent") is not None,
        "startup_error": app_state.get("startup_error"),
    }


@app.get("/debug/runs")
async def debug_runs(limit: int = 50):
    """List recent pipeline runs (newest first) with stage + tool summaries."""
    return {"runs": list_runs(limit=limit)}


@app.get("/debug/runs/{trace_id}")
async def debug_run(trace_id: str):
    run = get_run(trace_id)
    if run is None:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return run


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

    trace_id = create_run(body.thread_id, "trace")
    agent_instance, ingest_report = get_thread_agent(body.thread_id, [], base_agent)

    try:
        def _invoke():
            trace_id_var.set(trace_id)
            return agent_instance.invoke(
                {"messages": [{"role": "user", "content": body.message}]},
                config={"configurable": {"thread_id": body.thread_id}},
                context=agent.Context(user_id=body.thread_id),
            )

        result = await asyncio.get_running_loop().run_in_executor(None, _invoke)
        add_stage(trace_id, "trace_invoke_done", status="ok")
    except Exception as exc:
        traceback.print_exc()
        add_stage(trace_id, "trace_invoke_done", status="error", error=str(exc))
        return JSONResponse(status_code=500, content={"error": str(exc)})

    messages = [message_to_dict(m) for m in result.get("messages", [])]
    trace_data = {
        "trace_id": trace_id,
        "thread_id": body.thread_id,
        "messages": messages,
        "tools": _tool_summary_intermediate(messages),
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
            {"role": "user", "content": f"Attached file context:\n{body.context[:60000]}"}
        )

    trace_id = create_run(body.thread_id, "chat")
    agent_instance, ingest_report = get_thread_agent(body.thread_id, body.files, base_agent)
    add_stage(
        trace_id,
        "agent_prepared",
        status="ok",
        meta={
            "has_user_document": any(files := body.files),
            "failed_files": len(ingest_report.get("failed_files", [])),
        },
    )

    headers = {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        "connection": "keep-alive",
    }

    return StreamingResponse(
        answer_stream(
            agent_instance,
            body.thread_id,
            user_messages,
            body.context,
            trace_id=trace_id,
            ingest_report=ingest_report,
        ),
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
