"""Artifact persistence for RAG debugging: converted markdown, chunk dumps,
ChromaDB snapshots, and prompt dumps under ./debug_output/.

Directory layout:
    debug_output/
        markdown/   raw Docling-converted .md files
        chunks/     chunk metadata + preview JSON per document
        chroma/     storage-integrity snapshots (ids/metas/documents)
        prompts/    assembled prompt text (when LOG_PROMPTS=1)

All functions are safe to call with DEBUG_MODE off — they no-op cheaply.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_ROOT = os.path.join(AGENT_DIR, "debug_output")

SUBDIRS = ("markdown", "chunks", "chroma", "prompts")


def ensure_dirs() -> None:
    for sub in SUBDIRS:
        os.makedirs(os.path.join(DEBUG_ROOT, sub), exist_ok=True)


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "doc"
    return stem[:120]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def dump_text(subdir: str, name: str, content: str) -> str:
    """Write ``content`` to debug_output/<subdir>/<name>.txt|md and return path."""
    ensure_dirs()
    path = os.path.join(DEBUG_ROOT, subdir, f"{_safe_stem(name)}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def save_markdown_export(file_name: str, markdown: str) -> dict:
    """Save a Docling markdown export; returns {path, chars}."""
    started = time.perf_counter()
    path = dump_text("markdown", file_name, markdown)
    info = {
        "file": file_name,
        "path": path,
        "chars": len(markdown),
        "save_duration_s": round(time.perf_counter() - started, 4),
    }
    from rag_debug import field, section, C

    if os.environ.get("DEBUG_MODE", "").lower() in ("1", "true"):
        section("INGESTION", "Docling markdown export saved", C.INGESTION)
        field("saved_path", info["path"])
        field("char_count", info["chars"])
    return info


def save_chunk_dump(source_name: str, docs: list) -> str:
    """Persist chunk metadata + head/tail previews as JSON; returns path.

    Each entry: index, char_count, token_count (if tokenizer available),
    metadata, head/tail preview.
    """
    if not docs:
        return ""
    ensure_dirs()
    entries = []
    for idx, doc in enumerate(docs):
        text = doc.page_content or ""
        entry: dict[str, Any] = {
            "index": idx,
            "char_count": len(text),
            "metadata": doc.metadata,
            "head": text[:200],
            "tail": text[-100:] if len(text) > 200 else "",
        }
        entries.append(entry)

    payload = {
        "source": source_name,
        "generated_at": now_iso(),
        "total_chunks": len(docs),
        "avg_chars": round(sum(e["char_count"] for e in entries) / len(entries), 1),
        "chunks": entries,
    }
    path = os.path.join(
        DEBUG_ROOT, "chunks", f"{_safe_stem(source_name)}_{_timestamp()}.json"
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def save_chroma_snapshot(
    collection_name: str,
    ids: list,
    metadatas: list,
    documents: list,
    count_before: int | None = None,
    count_after: int | None = None,
) -> str:
    """Save a storage-integrity snapshot of an ingested batch to JSON."""
    ensure_dirs()
    payload = {
        "collection": collection_name,
        "generated_at": now_iso(),
        "count_before": count_before,
        "count_after": count_after,
        "inserted": len(ids),
        "records": [
            {"id": i, "metadata": m, "document_preview": (d or "")[:300]}
            for i, m, d in zip(ids, metadatas, documents)
        ],
    }
    path = os.path.join(
        DEBUG_ROOT, "chroma", f"{_safe_stem(collection_name)}_{_timestamp()}.json"
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def inspect_chromadb_collection(collection_name: str, limit: int = 5) -> dict:
    """Standalone helper: inspect what currently lives in a Chroma collection.

    Works for both the knowledge-base store (``iso_files``) and per-session
    user collections under ``chromadb/user_collections``. Returns a dict with
    ids / metadatas / document previews and prints a readable summary.
    """
    import chromadb
    from rag_debug import C, _c, field, section, status

    # Try the KB persist dir first, then user collections.
    last_error: Exception | None = None
    client = None
    for persist_dir in (
        os.path.join(AGENT_DIR, "chromadb"),
        os.path.join(AGENT_DIR, "chromadb", "user_collections"),
    ):
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_collection(collection_name)
            break
        except Exception as exc:  # noqa: BLE001 - try next location
            last_error = exc
            client = None
    if client is None:
        status("err", "CHROMA", f"collection '{collection_name}' not found ({last_error})")
        return {"error": str(last_error), "collection": collection_name}

    total = collection.count()
    fetched = min(max(limit, 0), total)
    result = collection.get(include=["metadatas", "documents"], limit=fetched) if fetched else {}

    summary = {
        "collection": collection_name,
        "total_documents": total,
        "sampled": fetched,
        "ids": result.get("ids", []),
        "metadatas": result.get("metadatas", []),
        "document_previews": [(d or "")[:300] for d in result.get("documents", [])],
    }

    section("STORAGE", f"inspect_chromadb_collection('{collection_name}')", C.STORAGE)
    field("total_documents", total)
    field("sampled", fetched)
    for i, (cid, meta, preview) in enumerate(
        zip(summary["ids"], summary["metadatas"], summary["document_previews"]), 1
    ):
        print(f"  {_c(C.STORAGE)}[{i}]{_c(C.RESET)} id={cid}")
        print(f"      metadata: {meta}")
        print(f"      preview:  {(preview or '')[:160]!r}")
    status("ok", "CHROMA", f"{total} documents in '{collection_name}'")
    return summary


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "iso_files"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    inspect_chromadb_collection(name, limit=limit)
