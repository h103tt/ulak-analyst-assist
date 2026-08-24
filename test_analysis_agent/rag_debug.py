"""Console-side RAG observability: config flags, colored section headers,
timers, and lifecycle loggers for the query / prompt / generation stages.

Environment variables:
    DEBUG_MODE=1                   master switch for all debug output
    VERBOSE_CHUNKS=1               verbose chunk previews during ingestion
    LOG_PROMPTS=1                  dump assembled prompts to debug_output/prompts/
    CHUNK_PREVIEW_COUNT=5          number of chunks previewed per document
    PREVIEW_HEAD_CHARS=160         head characters shown in chunk previews
    PREVIEW_TAIL_CHARS=80          tail characters shown in chunk previews
    RETRIEVAL_SCORE_THRESHOLD=1.2  max cosine distance for a chunk to pass
                                   (unset/0 = no threshold filtering)
    NO_COLOR=1                     disable ANSI colors
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator


def _flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
        return value or None
    except ValueError:
        return None


DEBUG_MODE = _flag("DEBUG_MODE")
VERBOSE_CHUNKS = _flag("VERBOSE_CHUNKS", DEBUG_MODE)
LOG_PROMPTS = _flag("LOG_PROMPTS", DEBUG_MODE)
CHUNK_PREVIEW_COUNT = _int_env("CHUNK_PREVIEW_COUNT", 5)
PREVIEW_HEAD_CHARS = _int_env("PREVIEW_HEAD_CHARS", 160)
PREVIEW_TAIL_CHARS = _int_env("PREVIEW_TAIL_CHARS", 80)
SCORE_THRESHOLD = _float_env("RETRIEVAL_SCORE_THRESHOLD")


class C:
    RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
    INGESTION = "\033[36m"   # cyan
    CHUNKING = "\033[35m"    # magenta
    STORAGE = "\033[34m"     # blue
    RETRIEVAL = "\033[33m"   # yellow
    GENERATION = "\033[32m"  # green
    QUERY = "\033[95m"       # bright magenta
    OK, WARN, ERR = "\033[92m", "\033[93m", "\033[91m"


_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str) -> str:
    return code if _USE_COLOR else ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def section(stage: str, title: str, color: str = "") -> None:
    """Print a stage header like:  [RETRIEVAL] ── Top-k candidates ─────────"""
    if not DEBUG_MODE:
        return
    col = _c(color or C.BOLD)
    reset = _c(C.RESET)
    bar = "─" * max(10, 62 - len(stage) - len(title))
    print(f"\n{col}[{stage}]{reset} {title} {col}{bar}{reset}")


def field(key: str, value: Any, color: str = "") -> None:
    """Print one indented key/value line under a section header."""
    if not DEBUG_MODE:
        return
    print(f"  {_c(color or C.DIM)}|{reset_c()} {key:<30} {value}")


def reset_c() -> str:
    return _c(C.RESET)


def status(level: str, stage: str, message: str) -> None:
    """Print an OK/WARN/ERR status line, independent of DEBUG_MODE."""
    icon, col = {"ok": ("+", C.OK), "warn": ("!", C.WARN), "err": ("x", C.ERR)}[level]
    prefix = f"[{stage}] " if stage else ""
    print(f"{_c(col)}{prefix}{icon} {message}{_c(C.RESET)}")


@contextmanager
def timed(label: str) -> Iterator[dict]:
    """Time a block; on exit prints duration and fills info['duration_s']."""
    info: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield info
    finally:
        info["duration_s"] = round(time.perf_counter() - start, 3)
        if DEBUG_MODE:
            print(f"  {_c(C.OK)}+{_c(C.RESET)} {label}: {info['duration_s']}s")


def _doc_meta(doc) -> dict:
    return {k: v for k, v in (doc.metadata or {}).items() if k != "embedding"}


def _doc_source(doc) -> str:
    meta = _doc_meta(doc)
    return str(meta.get("source_file") or meta.get("source") or "?")


def _preview(text: str) -> str:
    text = (text or "").replace("\n", " ")
    head = text[:PREVIEW_HEAD_CHARS]
    tail = text[-PREVIEW_TAIL_CHARS:] if len(text) > PREVIEW_HEAD_CHARS else ""
    return f"{head}...{tail}" if tail else head


def _doc_hash(doc) -> str:
    return hashlib.sha1((doc.page_content or "").encode()).hexdigest()


# ---------------------------------------------------------------- queries --
def log_query(session_id: str, query: str) -> None:
    section("QUERY", "User input received", C.QUERY)
    field("timestamp", now_iso())
    field("session_id", session_id)
    field("query_chars", len(query))
    field("raw_prompt", repr(query[:500]))


# --------------------------------------------------------------- prompts --
def log_prompt_assembled(
    session_id: str,
    system_prompt: str,
    context_blocks: list[tuple[str, str]],
    user_message: str,
) -> None:
    """Log the assembled prompt pieces; optionally persist the full text."""
    section("GENERATION", "Prompt assembly", C.GENERATION)
    field("system_prompt_chars", len(system_prompt))
    field("context_blocks", len(context_blocks))
    for name, text in context_blocks:
        field(f"context[{name}]", f"{len(text)} chars: {text[:120]!r}")
    field("user_message_chars", len(user_message))
    if LOG_PROMPTS:
        from debug_artifacts import dump_text

        parts = [f"=== SYSTEM ===\n{system_prompt}", "=== CONTEXT ==="]
        parts += [f"--- {name} ---\n{text}" for name, text in context_blocks]
        parts.append(f"=== USER ===\n{user_message}")
        path = dump_text("prompts", session_id, "\n\n".join(parts))
        field("prompt_dump", path)


# ------------------------------------------------------------ generation --
def extract_usage(messages: list) -> dict | None:
    """Pull token usage off the last message that carries usage_metadata."""
    for message in reversed(messages):
        usage = getattr(message, "usage_metadata", None)
        if usage:
            return {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
    return None


def log_generation(
    model: str,
    latency_s: float,
    answer: str,
    usage: dict | None = None,
    ttft_s: float | None = None,
) -> None:
    section("GENERATION", "Model execution complete", C.GENERATION)
    field("model", model)
    field("total_latency_s", round(latency_s, 3))
    field("time_to_first_token_s", ttft_s if ttft_s is not None else "n/a (non-streaming)")
    if usage:
        field("input_tokens", usage.get("input_tokens"))
        field("output_tokens", usage.get("output_tokens"))
        field("total_tokens", usage.get("total_tokens"))
    field("answer_chars", len(answer))
    field("answer_preview", repr(answer[:300]))


# ------------------------------------------------------------- retrieval --
_LAST_CANDIDATES: dict[str, list[str]] = {}


def register_candidates(tag: str, docs: list) -> None:
    """Remember the pre-rerank candidate set so log_final_results can mark
    which chunks survived the compressor/reranker."""
    _LAST_CANDIDATES[tag] = [_doc_hash(doc) for doc in docs]


def log_retrieval_candidate(tag: str, rank: int, score: float, doc, passed: bool) -> None:
    """Log one vector-search hit with its distance and pass/filter verdict."""
    if not DEBUG_MODE:
        return
    flag = f"{_c(C.OK)}PASS{_c(C.RESET)}" if passed else f"{_c(C.ERR)}FILTERED{_c(C.RESET)}"
    meta = _doc_meta(doc)
    print(
        f"  {_c(C.RETRIEVAL)}#{rank:<3}{_c(C.RESET)} "
        f"distance={score:.4f} {flag} "
        f"source={_doc_source(doc)} "
        f"meta={ {k: meta[k] for k in list(meta)[:4]} }"
    )
    print(f"      head: {_preview(doc.page_content)!r}")


def log_final_results(tag: str, docs: list) -> None:
    """Log post-rerank results, marking which candidates were filtered out."""
    if not DEBUG_MODE:
        return
    section("RETRIEVAL", f"Final context for '{tag}' (post-rerank)", C.RETRIEVAL)
    previous = _LAST_CANDIDATES.pop(tag, [])
    surviving = {_doc_hash(doc) for doc in docs}
    filtered = len(previous) - len(surviving & set(previous))
    field("chunks_passed_to_llm", len(docs))
    field("candidates_filtered_out", max(0, filtered))
    for rank, doc in enumerate(docs, 1):
        print(
            f"  {_c(C.OK)}>{_c(C.RESET)} rank={rank} "
            f"source={_doc_source(doc)} "
            f"meta={ {k: _doc_meta(doc)[k] for k in list(_doc_meta(doc))[:4]} }"
        )
        print(f"      text: {_preview(doc.page_content)!r}")
