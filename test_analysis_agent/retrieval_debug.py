"""Debug wrappers around the RAG retrievers: log embedding timing, vector
search hits (rank / distance / metadata / preview / pass-filter verdict),
and the final post-rerank context handed to the LLM.

The wrappers are transparent: they call the wrapped retriever unchanged and
only add observability. They are installed in vector_embed.py.
"""
from __future__ import annotations

import time
from typing import Any

import rag_debug
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rag_debug import C, field, section, status


def _threshold() -> float | None:
    # Read lazily so tests can set the env var after import.
    return rag_debug.SCORE_THRESHOLD


def log_embedding(query: str, vector: list[float] | None) -> None:
    """Log embedding generation timing + dimension for a query."""
    if not rag_debug.DEBUG_MODE:
        return
    section("RETRIEVAL", "Query embedding", C.RETRIEVAL)
    field("query_preview", repr(query[:120]))
    field("vector_dimension", len(vector) if vector else "n/a")


def timed_embed_query(store, query: str) -> tuple[list[float], float]:
    """Embed ``query`` via ``store.embeddings``, returning (vector, ms)."""
    start = time.perf_counter()
    vector = store.embeddings.embed_query(query)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return vector, elapsed_ms


def debug_similarity_search_with_score(store, tag: str, query: str, k: int):
    """Run a scored similarity search on ``store`` with full hit logging.

    Returns (docs, scores) exactly as the underlying search would; logging
    is additive. Applies SCORE_THRESHOLD only to the PASS/FILTERED verdict
    display — filtering itself stays with the reranker unless you wire
    ``filter_by_threshold`` into a pipeline.
    """
    vector, embed_ms = timed_embed_query(store, query)
    if rag_debug.DEBUG_MODE:
        field("embedding_time_ms", embed_ms)
        field("vector_dimension", len(vector))

    start = time.perf_counter()
    pairs = store.similarity_search_with_score(query, k=k)
    search_ms = round((time.perf_counter() - start) * 1000, 2)

    if rag_debug.DEBUG_MODE:
        section("RETRIEVAL", f"Vector search hits for '{tag}' (top-{k})", C.RETRIEVAL)
        field("search_time_ms", search_ms)
        threshold = _threshold()
        field("score_threshold", threshold if threshold else "none")
        rag_debug.register_candidates(tag, [doc for doc, _ in pairs])
        for rank, (doc, score) in enumerate(pairs, 1):
            passed = True if threshold is None else score <= threshold
            rag_debug.log_retrieval_candidate(tag, rank, score, doc, passed)
    return [doc for doc, _ in pairs], [score for _, score in pairs]


class LoggedCompressionRetriever(BaseRetriever):
    """Drop-in replacement for ContextualCompressionRetriever that logs the
    pre-rerank candidate set and the post-rerank context per query."""

    base_retriever: Any
    compressor: Any
    tag: str = "retrieval"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        start = time.perf_counter()
        docs = self.base_retriever.invoke(query)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        if rag_debug.DEBUG_MODE:
            section("RETRIEVAL", f"Vector search for '{self.tag}'", C.RETRIEVAL)
            field("query_preview", repr(str(query)[:120]))
            field("candidates_returned", len(docs))
            field("retrieval_time_ms", elapsed_ms)
        rag_debug.register_candidates(self.tag, docs)

        start = time.perf_counter()
        compressed = list(self.compressor.compress_documents(docs, str(query)))
        rerank_ms = round((time.perf_counter() - start) * 1000, 2)
        if rag_debug.DEBUG_MODE:
            field("reranker_time_ms", rerank_ms)
        rag_debug.log_final_results(self.tag, compressed)
        return compressed


def logged_compression_retriever(base_retriever, compressor, tag: str):
    """Build a LoggedCompressionRetriever (use INSTEAD of
    ContextualCompressionRetriever for per-query visibility)."""
    return LoggedCompressionRetriever(
        base_retriever=base_retriever, compressor=compressor, tag=tag
    )


def describe_store(store, name: str) -> None:
    """Log collection name + document count for a Chroma-backed store."""
    try:
        count = store._collection.count()
    except Exception:  # noqa: BLE001 - best-effort introspection
        count = "?"
    status("ok", "STORAGE", f"{name}: {count} documents")
