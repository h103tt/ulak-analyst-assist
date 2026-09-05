"""
Structured JSON pipeline logging + in-memory run registry.

Usage (bridge.py):
    from pipeline_logging import setup_logging, create_run, add_stage, get_run
    setup_logging()
    trace_id = create_run(thread_id=..., source="chat")
    add_stage(trace_id, "agent_invoke_done", status="ok", answer_chars=123, tools=[...])
    run = get_run(trace_id)   # -> {trace_id, thread_id, source, stages, created_at}
"""
from __future__ import annotations

import contextvars
import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone

from collections import defaultdict

_tool_call_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

DEFAULT_MAX_TOOL_CALLS_PER_TURN = 3
# Per-request correlation id, propagated through contextvars so any logger
# emitting inside the request carries the same trace_id automatically.
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)

_MAX_RUNS = 200
_run_lock = threading.Lock()
RUNS: deque[dict] = deque(maxlen=_MAX_RUNS)


class JsonFormatter(logging.Formatter):
    """RFC3339-timestamped, single-line JSON records with optional extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id_var.get() or getattr(record, "trace_id", ""),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("stage", "meta"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    """Attach the JSON formatter to the root logger. Idempotent."""
    root = logging.getLogger()
    for handler in root.handlers:
        if getattr(handler, "_ulak_json", False):
            return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler._ulak_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)


def create_run(thread_id: str, source: str) -> str:
    """Create a run record and return its trace_id."""
    import uuid

    trace_id = uuid.uuid4().hex
    run = {
        "trace_id": trace_id,
        "thread_id": thread_id,
        "source": source,
        "stages": [],
        "tools": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _run_lock:
        RUNS.append(run)
    trace_id_var.set(trace_id)
    return trace_id


def add_stage(
    trace_id: str,
    stage: str,
    *,
    status: str = "ok",
    **meta,
) -> None:
    """Append one stage entry to the matching run record (no-op if missing)."""
    entry = {
        "stage": stage,
        "status": status,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    if meta:
        entry["meta"] = meta
    with _run_lock:
        for run in RUNS:
            if run["trace_id"] == trace_id:
                run["stages"].append(entry)
                return


def add_tool_summaries(trace_id: str, summaries: list[dict]) -> None:
    """Record retrieved-context summaries (tool name, chars, preview)."""
    with _run_lock:
        for run in RUNS:
            if run["trace_id"] == trace_id:
                run["tools"].extend(summaries)
                return


def get_run(trace_id: str) -> dict | None:
    with _run_lock:
        for run in RUNS:
            if run["trace_id"] == trace_id:
                # Return a shallow copy so callers can't mutate the registry.
                return {**run, "stages": list(run["stages"]), "tools": list(run["tools"])}
    return None


def list_runs(limit: int = 50) -> list[dict]:
    """Return the most recent runs (newest first), each with a shallow copy of
    its stages/tools so callers can't mutate the registry."""
    with _run_lock:
        recent = list(RUNS)[-limit:]
    return [
        {**run, "stages": list(run["stages"]), "tools": list(run["tools"])}
        for run in reversed(recent)
    ]


# --- Hop-level tracing --------------------------------------------------
#
# add_stage() above is called by hand, once per coarse checkpoint
# (agent_prepared, agent_invoke_done, stream_complete...). It doesn't see
# what happens *inside* one agent_invoke: how many reasoning/tool-call
# hops the ReAct loop took, how long each query-expansion variant's LLM
# call took, or where in a multi-hop turn things are slow or erroring.
#
# HopTracingHandler is a LangChain callback handler that fills that gap:
# attach one instance (per turn) to agent_instance.invoke(config=...) and
# to refine.refine_answer(callbacks=...), and every nested LLM call,
# retriever call, and tool call automatically gets its own add_stage()
# entry (so it shows up in /debug/runs/{trace_id} alongside the coarse
# checkpoints) plus a live JSON log line, as it happens -- not just once
# the whole turn finishes.

import time
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler

_hop_log = logging.getLogger("hop_tracing")


def _preview(text: Any, limit: int = 200) -> str:
    text = text if isinstance(text, str) else str(text)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "..."


class HopTracingHandler(BaseCallbackHandler):
    """Per-turn callback handler. Create one instance per agent turn (do not
    reuse across turns/threads -- run_id timing state isn't cleared).

    Also accumulates token usage across every LLM call it sees for the
    turn -- attach the SAME instance to both the main agent.invoke() and
    refine.refine_answer() (as bridge.py already does) and usage_summary()
    then reflects the true end-to-end cost of one user request: every
    ReAct-loop hop (a multi-hop turn makes several separate model calls,
    each with its own token usage) plus the refinement pass, none of which
    rag_debug.extract_usage() captured on its own -- it only looked at the
    last message in the agent's final result, i.e. one hop out of however
    many actually ran, and never saw refine's call at all."""

    def __init__(self) -> None:
        self._starts: dict[Any, float] = {}
        self.llm_call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    def _elapsed(self, run_id: Any) -> float | None:
        start = self._starts.pop(run_id, None)
        return round(time.perf_counter() - start, 2) if start is not None else None

    def _emit(self, stage: str, status: str = "ok", **meta) -> None:
        trace_id = trace_id_var.get()
        if trace_id:
            add_stage(trace_id, stage, status=status, **meta)
        level = logging.ERROR if status == "error" else logging.INFO
        _hop_log.log(level, stage, extra={"stage": stage, "meta": meta})

    @staticmethod
    def _extract_usage(response: Any) -> dict | None:
        """Pull token usage off an LLMResult. For chat models (this
        project's only path) it lives on each generation's AIMessage, not
        on response.llm_output -- Gemini leaves that dict empty."""
        try:
            for generation_list in response.generations:
                for generation in generation_list:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
                    if usage:
                        return {
                            "input_tokens": usage.get("input_tokens") or 0,
                            "output_tokens": usage.get("output_tokens") or 0,
                            "total_tokens": usage.get("total_tokens") or 0,
                        }
        except Exception:  # noqa: BLE001 - usage tracking must never break the actual turn
            pass
        return None

    def usage_summary(self) -> dict:
        """Aggregate token usage across every LLM call seen so far this
        turn (every ReAct hop + the refine pass)."""
        return {
            "llm_calls": self.llm_call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }

    # LLM calls: agent reasoning steps, each query-expansion variant, and
    # the refinement pass (when refine.refine_answer is given this handler).
    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()
        self._emit("llm_call_start", prompt_count=len(prompts))

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()
        self._emit("llm_call_start")

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = self._extract_usage(response)
        if usage:
            self.llm_call_count += 1
            self.total_input_tokens += usage["input_tokens"]
            self.total_output_tokens += usage["output_tokens"]
            self.total_tokens += usage["total_tokens"]
        self._emit("llm_call_end", elapsed_s=self._elapsed(run_id), **(usage or {}))

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._emit(
            "llm_call_error", status="error", elapsed_s=self._elapsed(run_id), error=str(error)
        )

    # Retriever calls: one per query-expansion variant, so a single tool
    # call can log several of these.
    def on_retriever_start(self, serialized, query, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()
        self._emit("retrieval_start", query=_preview(query))

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self._emit(
            "retrieval_end", elapsed_s=self._elapsed(run_id), chunk_count=len(documents)
        )

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._emit(
            "retrieval_error", status="error", elapsed_s=self._elapsed(run_id), error=str(error)
        )

    # Tool calls: search_testing_standards / search_user_document /
    # get_document_structure -- one per retrieval "hop" in a multi-hop turn.
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
        self._starts[run_id] = time.perf_counter()
        self._emit("tool_call_start", tool=name, input=_preview(input_str))

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._emit("tool_call_end", elapsed_s=self._elapsed(run_id))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._emit(
            "tool_call_error", status="error", elapsed_s=self._elapsed(run_id), error=str(error)
        )


# --- Per-turn tool call limiting ----------------------------------------
#
# The ReAct loop has no built-in sense of "I've searched enough" — a small
# model can keep reformulating the same query indefinitely (see hop_tracing
# logs: 5+ near-identical search_user_document calls in one turn). This caps
# how many times a given tool can actually execute within a single turn;
# once the cap is hit, the tool returns a synthetic result instead of
# running again, which forces the model to answer with what it already has
# instead of looping.


def limit_tool_calls(tool, max_calls: int = DEFAULT_MAX_TOOL_CALLS_PER_TURN):
    """Wrap a LangChain tool so it refuses to execute more than `max_calls`
    times per turn (keyed by trace_id_var, which bridge.py sets once per
    request). Past the cap, returns a canned message instead of invoking
    the tool, so the model is forced to stop retrieving and answer.

    Idempotent: `tool` objects (e.g. ``vector_embed.tools``) are shared
    module-level singletons reused across every ``build_agent()`` call, so
    without this guard each call stacked another wrapper around the same
    tool -- every real invocation then tripped N nested counters at once,
    making the cap trigger sooner (and eventually immediately) the longer
    a process ran and called ``build_agent()`` repeatedly (e.g. one call
    per bilingual_eval.py question, or one per bridge.py chat thread)."""
    if getattr(tool, "_ulak_call_limited", False):
        return tool

    limit_message = (
        "[Search limit reached: this tool has already been called {n} times "
        "this turn. Do not call it again. Answer using only the results "
        "already retrieved above, and if something isn't covered by them, "
        "state explicitly that the retrieved sections don't cover it.]"
    )

    def _check_and_count() -> str | None:
        trace_id = trace_id_var.get() or "no-trace"
        counts = _tool_call_counts[trace_id]
        counts[tool.name] += 1
        if counts[tool.name] > max_calls:
            return limit_message.format(n=max_calls)
        return None

    if getattr(tool, "func", None) is not None:
        original_func = tool.func

        def limited_func(*args, **kwargs):
            blocked = _check_and_count()
            if blocked is not None:
                return blocked
            return original_func(*args, **kwargs)

        tool.func = limited_func

    if getattr(tool, "coroutine", None) is not None:
        original_coroutine = tool.coroutine

        async def limited_coroutine(*args, **kwargs):
            blocked = _check_and_count()
            if blocked is not None:
                return blocked
            return await original_coroutine(*args, **kwargs)

        tool.coroutine = limited_coroutine

    tool._ulak_call_limited = True
    return tool


def clear_tool_call_counts(trace_id: str) -> None:
    """Free the per-turn counters once a turn is done, so the dict doesn't
    grow unbounded over a long-running process."""
    _tool_call_counts.pop(trace_id, None)