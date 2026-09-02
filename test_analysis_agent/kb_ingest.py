"""Resumable knowledge-base -> Gemini embedding pipeline.

Splits ingestion into two independent stages so a Gemini quota failure never
throws away local docling/OCR work, and re-running never re-embeds a chunk
that is already stored:

    parse   local only, no API calls -- docling-parses every KB file not
            already cached under chunk_cache/, in the same parallel
            (fast files) / sequential (OCR-heavy files) split as before.
    embed   quota-bound -- embeds whatever chunk_cache/ has that Chroma
            doesn't yet, rotating through every Gemini key in .env
            (GEMINI_API_KEY / GEMINI_API_KEY1..N) as each hits its daily
            per-project quota.
    status  read-only -- compares chunk_cache/ against the iso_files
            collection and prints exactly what's missing, per file.

Usage:
    uv run python kb_ingest.py            # parse, then embed
    uv run python kb_ingest.py parse
    uv run python kb_ingest.py embed
    uv run python kb_ingest.py status

Why this exists: two earlier one-off scripts considered a file "done" the
moment it had *any* chunk in Chroma. A 429 mid-embed silently left files
truncated (one had 96 of 300 chunks) and permanently skipped on every later
run. This tool checks exact chunk counts, not presence.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

import gemini_keys
import vector_embed as ve

CHUNK_CACHE_DIR = os.path.join(ve.AGENT_DIR, "chunk_cache")
SLOW_FILES = {"MIL-STD-1586A.pdf", "MIL-STD-882E.pdf"}  # OCR-heavy: process one at a time
KB_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".xls")


def _kb_files() -> list[Path]:
    return sorted(p for p in Path(ve.KB_DIR).rglob("*") if p.suffix.lower() in KB_EXTENSIONS)


def _cache_path(file_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", file_name)
    return os.path.join(CHUNK_CACHE_DIR, f"{stem}.json")


def _write_cache(file_name: str, docs: list[Document]) -> None:
    """Write atomically (temp file + replace) so a crash mid-write never
    leaves a corrupt cache that a later run would trust."""
    os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)
    payload = {
        "source_file": file_name,
        "parsed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_chunks": len(docs),
        "chunks": [
            {"id": d.id, "page_content": d.page_content, "metadata": d.metadata}
            for d in docs
        ],
    }
    path = _cache_path(file_name)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, path)


def _read_cache(file_name: str) -> dict | None:
    path = _cache_path(file_name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- parse
def parse_one(path: Path) -> str:
    if _read_cache(path.name) is not None:
        return f"{path.name}: cached, skipped"
    docs = ve.process_single_file(path)
    if not docs:
        return f"{path.name}: FAILED, no chunks produced"
    _write_cache(path.name, docs)
    return f"{path.name}: {len(docs)} chunks parsed and cached"


def cmd_parse() -> None:
    files = _kb_files()
    fast = [p for p in files if p.name not in SLOW_FILES]
    slow = [p for p in files if p.name in SLOW_FILES]

    print(f"Parsing {len(fast)} file(s) in parallel, {len(slow)} sequentially...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(parse_one, p): p for p in fast}
        for future in as_completed(futures):
            path = futures[future]
            try:
                print(f"[{path.name}] {future.result()}", flush=True)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"[{path.name}] EXCEPTION: {exc}", flush=True)

    for path in slow:
        print(f"\n--- {path.name} ---", flush=True)
        try:
            print(parse_one(path), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{path.name}] EXCEPTION: {exc}", flush=True)

    print("\nParsing done.", flush=True)


# --------------------------------------------------------------------------- shared
def _collection_state() -> dict[str, set[str]]:
    """source_file -> set of chunk ids currently stored in the collection."""
    res = ve.vector_store.get(limit=20000, include=["metadatas"])
    per: dict[str, set[str]] = {}
    for doc_id, meta in zip(res["ids"], res["metadatas"]):
        per.setdefault(meta.get("source_file"), set()).add(doc_id)
    return per


# --------------------------------------------------------------------------- status
def cmd_status() -> None:
    files = _kb_files()
    in_chroma = _collection_state()
    total_expected = total_have = total_missing = 0

    print(f"{'file':<32} {'in_chroma':>9} {'expected':>8} {'missing':>8}")
    for path in files:
        cache = _read_cache(path.name)
        have = len(in_chroma.get(path.name, set()))
        if cache is None:
            print(f"{path.name:<32} {have:>9} {'?':>8} {'?':>8}  (not parsed yet)")
            continue
        expected = cache["total_chunks"]
        missing = expected - have
        total_expected += expected
        total_have += have
        total_missing += max(0, missing)
        print(f"{path.name:<32} {have:>9} {expected:>8} {missing:>8}")

    print()
    print(f"TOTAL chunks in chroma: {total_have}")
    print(f"TOTAL expected        : {total_expected}")
    print(f"TOTAL still to embed  : {total_missing}")


# --------------------------------------------------------------------------- embed
def _missing_documents(file_name: str, cache: dict, have_ids: set[str]) -> list[Document]:
    """Cached chunks not yet in Chroma -- after confirming the overlap is
    byte-identical text. A re-parse can renumber chunks; without this check
    a stale id could silently pair with different text and corrupt retrieval."""
    overlap_ids = [c["id"] for c in cache["chunks"] if c["id"] in have_ids]
    if overlap_ids:
        stored = ve.vector_store.get(ids=overlap_ids, include=["documents"])
        stored_text = dict(zip(stored["ids"], stored["documents"]))
        cached_text = {c["id"]: c["page_content"] for c in cache["chunks"]}
        mismatched = [i for i in overlap_ids if stored_text.get(i) != cached_text.get(i)]
        if mismatched:
            print(
                f"  [{file_name}] {len(mismatched)} stored chunk(s) don't match the "
                f"cache text (re-parsed differently) -- re-embedding the whole file.",
                flush=True,
            )
            ve.vector_store._collection.delete(ids=overlap_ids)
            have_ids = set()

    return [
        Document(page_content=c["page_content"], metadata=c["metadata"], id=c["id"])
        for c in cache["chunks"]
        if c["id"] not in have_ids
    ]


def _probe_keys(keys: list[str]) -> list[str]:
    """Validate every key with one 1-text embed call before the real run --
    a dead or malformed key raises INVALID_ARGUMENT, not RESOURCE_EXHAUSTED,
    which _add_batch_with_backoff does not retry. Hitting that mid-rotation
    would crash the whole embed pass instead of just skipping the bad key."""
    valid = []
    for i, key in enumerate(keys, 1):
        try:
            ve.use_api_key(key)
            ve.embeddings.embed_query("probe")
            valid.append(key)
        except Exception as exc:  # noqa: BLE001 - any failure disqualifies the key
            reason = str(exc).splitlines()[0][:100]
            print(f"  key {i}/{len(keys)}: unusable, excluding from rotation ({reason})", flush=True)
    print(f"{len(valid)}/{len(keys)} key(s) usable, ~{len(valid) * 1000} quota units available today", flush=True)
    return valid


def cmd_embed() -> None:
    print("Probing configured Gemini keys...", flush=True)
    keys = _probe_keys(gemini_keys.all_keys())
    if not keys:
        raise EnvironmentError("No usable Gemini API keys (see gemini_keys.py).")

    in_chroma = _collection_state()
    plan: list[tuple[str, dict, set[str]]] = []
    for path in _kb_files():
        cache = _read_cache(path.name)
        if cache is None:
            print(f"[{path.name}] not parsed yet, skipping (run `parse` first)", flush=True)
            continue
        have_ids = in_chroma.get(path.name, set())
        if cache["total_chunks"] - len(have_ids) > 0:
            plan.append((path.name, cache, have_ids))
    # Bank whole files early under a hard quota ceiling.
    plan.sort(key=lambda item: item[1]["total_chunks"] - len(item[2]))

    if not plan:
        print("Nothing to embed -- every parsed file is already fully in Chroma.", flush=True)
        return

    key_idx = 0
    ve.use_api_key(keys[key_idx])
    run_total = 0

    for file_name, cache, have_ids in plan:
        docs = _missing_documents(file_name, cache, have_ids)
        if not docs:
            continue
        print(f"\n--- {file_name}: embedding {len(docs)} chunk(s) ---", flush=True)
        file_done = 0
        for chunk_batch in ve._batches_by_char_budget(docs, ve.EMBED_BATCH_SIZE, ve.EMBED_BATCH_MAX_CHARS):
            batch = filter_complex_metadata(chunk_batch)
            while True:
                try:
                    ve._add_batch_with_backoff(ve.vector_store, batch)
                    break
                except ve.QuotaExhausted:
                    key_idx += 1
                    if key_idx >= len(keys):
                        print(
                            f"\nAll {len(keys)} key(s) exhausted for today. Run "
                            f"`uv run python kb_ingest.py embed` again later to "
                            f"resume -- nothing done so far is lost.",
                            flush=True,
                        )
                        return
                    print(f"  key {key_idx} exhausted, rotating to key {key_idx + 1}/{len(keys)}", flush=True)
                    ve.use_api_key(keys[key_idx])
            file_done += len(batch)
            run_total += len(batch)
            print(
                f"[{file_name}] {file_done}/{len(docs)} "
                f"· run total {run_total} · key {key_idx + 1}/{len(keys)}",
                flush=True,
            )

    print("\nEmbedding pass complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=["status", "parse", "embed", "all"])
    args = parser.parse_args()

    if args.command in ("parse", "all"):
        cmd_parse()
    if args.command in ("embed", "all"):
        cmd_embed()
    if args.command == "status":
        cmd_status()
    if args.command == "all":
        print()
        cmd_status()
