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
