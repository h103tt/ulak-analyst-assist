"""
Hybrid search (dense + sparse) unit tests for vector_embed.py
==============================================================
Tests the ``HybridRetriever`` (reciprocal rank fusion) and
``build_hybrid_retriever`` construction, including graceful degradation to
dense-only retrieval when the corpus is empty or ``rank_bm25`` is unavailable.

Run:
    cd ulak-analyst-assist/test_analysis_agent
    pytest tests/test_hybrid_retrieval.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Ensure the package root is importable (conftest installs module mocks first).
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import vector_embed  # noqa: E402


def _doc(citation_tag: str, text: str = "content") -> Document:
    """Small helper: build a Document with the citation metadata used for dedupe."""
    return Document(
        page_content=text,
        metadata={"citation_tag": citation_tag, "source_file": f"{citation_tag}.pdf"},
    )


def _hr(dense, sparse=None, k=10, weights=(0.5, 0.5)):
    return vector_embed.HybridRetriever(
        dense_retriever=dense, sparse_retriever=sparse, k=k, weights=weights
    )


# ===================================================================
# 1. HybridRetriever - reciprocal rank fusion logic
# ===================================================================
class TestHybridRetrieverRRF:
    def test_merges_and_deduplicates_both_channels(self):
        """Docs from both channels are combined, duplicates are removed."""
        dense_docs = [_doc("d1", "alpha beta gamma"), _doc("d2", "delta epsilon")]
        sparse_docs = [_doc("s1", "alpha delta"), _doc("d1", "alpha beta gamma")]

        dense = MagicMock()
        dense.invoke.return_value = dense_docs
        sparse = MagicMock()
        sparse.invoke.return_value = sparse_docs

        hr = _hr(dense, sparse)
        results = hr.invoke("alpha")

        tags = [d.metadata["citation_tag"] for d in results]
        assert len(tags) == len(set(tags)), "duplicates were not de-duplicated"
        assert set(tags) == {"d1", "d2", "s1"}
        # d1 appears in both channels, so it should be ranked first.
        assert tags[0] == "d1"

    def test_doc_ranked_by_both_channels_outranks_single_hit(self):
        """A doc present in both lists gets a higher fused score than a doc in
        only one, even when the dual-hit doc is lower in each individual list."""
        # d2 is in both lists but low; d1 is top of dense only.
        dense_docs = [_doc("d1", "first"), _doc("d2", "second")]
        sparse_docs = [_doc("d2", "second")]

        dense = MagicMock()
        dense.invoke.return_value = dense_docs
        sparse = MagicMock()
        sparse.invoke.return_value = sparse_docs

        hr = _hr(dense, sparse)
        results = hr.invoke("query")
        assert results[0].metadata["citation_tag"] == "d2"

    def test_respects_k_limit(self):
        """The fused list is truncated to k results."""
        dense_docs = [_doc(f"d{i}") for i in range(5)]
        sparse_docs = [_doc(f"s{i}") for i in range(5)]

        dense = MagicMock()
        dense.invoke.return_value = dense_docs
        sparse = MagicMock()
        sparse.invoke.return_value = sparse_docs

        hr = _hr(dense, sparse, k=3)
        assert len(hr.invoke("q")) == 3

    def test_dense_only_when_sparse_is_none(self):
        """sparse_retriever=None => return the dense results unchanged."""
        dense_docs = [_doc("a"), _doc("b")]
        dense = MagicMock()
        dense.invoke.return_value = dense_docs

        hr = _hr(dense, sparse=None, k=2)
        assert hr.invoke("q") == dense_docs

    def test_dense_only_still_honours_k(self):
        dense_docs = [_doc(f"d{i}") for i in range(5)]
        dense = MagicMock()
        dense.invoke.return_value = dense_docs

        hr = _hr(dense, sparse=None, k=2)
        assert len(hr.invoke("q")) == 2

    def test_safe_invoke_returns_empty_on_error(self):
        """A failing channel must not break retrieval."""
        dense = MagicMock()
        dense.invoke.side_effect = RuntimeError("dense blew up")

        hr = _hr(dense, sparse=None, k=2)
        assert hr.invoke("q") == []

    def test_safe_invoke_swallows_sparse_error(self):
        """If the sparse channel errors, the dense channel still gets returned."""
        dense_docs = [_doc("a"), _doc("b")]
        dense = MagicMock()
        dense.invoke.return_value = dense_docs
        sparse = MagicMock()
        sparse.invoke.side_effect = RuntimeError("sparse blew up")

        hr = _hr(dense, sparse, k=2)
        results = hr.invoke("q")
        assert [d.metadata["citation_tag"] for d in results] == ["a", "b"]

    def test_weights_are_respected(self):
        """Bumping the dense weight should favour dense-only docs over sparse-
        only docs, all else equal."""
        # d1 top of dense; s1 top of sparse. Equal ranks => weights decide.
        dense_docs = [_doc("d1", "common")]
        sparse_docs = [_doc("s1", "other")]

        dense = MagicMock()
        dense.invoke.return_value = dense_docs
        sparse = MagicMock()
        sparse.invoke.return_value = sparse_docs

        # Heavily favour dense: dense docs should come first.
        hr_heavy = _hr(dense, sparse, k=10, weights=(0.9, 0.1))
        assert hr_heavy.invoke("q")[0].metadata["citation_tag"] == "d1"

        # Heavily favour sparse: sparse docs should come first.
        hr_heavy_sparse = _hr(dense, sparse, k=10, weights=(0.1, 0.9))
        assert hr_heavy_sparse.invoke("q")[0].metadata["citation_tag"] == "s1"


# ===================================================================
# 2. build_hybrid_retriever - construction & degradation
# ===================================================================
class TestBuildHybridRetriever:
    def test_returns_dense_only_when_store_empty(self):
        """Empty corpus => dense-only retriever (pre-hybrid behaviour)."""
        fake_store = MagicMock()
        fake_retriever = MagicMock()
        fake_store.as_retriever.return_value = fake_retriever

        with patch.object(vector_embed, "_load_all_docs_from_store", return_value=[]):
            result = vector_embed.build_hybrid_retriever(fake_store, k=5)

        assert result is fake_retriever
        fake_store.as_retriever.assert_called_once_with(
            search_type="similarity", search_kwargs={"k": 5}
        )

    def test_returns_hybrid_when_docs_exist(self):
        """Populated corpus => HybridRetriever wrapping both channels."""
        fake_store = MagicMock()
        fake_retriever = MagicMock()
        fake_store.as_retriever.return_value = fake_retriever
        docs = [_doc("a"), _doc("b")]

        with patch.object(vector_embed, "_load_all_docs_from_store", return_value=docs):
            result = vector_embed.build_hybrid_retriever(fake_store, k=7)

        assert isinstance(result, vector_embed.HybridRetriever)
        assert result.dense_retriever is fake_retriever
        assert result.sparse_retriever is not None
        assert result.k == 7

    def test_forwards_search_type(self):
        """The vector search type (e.g. mmr) is passed through to the dense side."""
        fake_store = MagicMock()
        fake_store.as_retriever.return_value = MagicMock()
        docs = [_doc("a")]

        with patch.object(vector_embed, "_load_all_docs_from_store", return_value=docs):
            vector_embed.build_hybrid_retriever(
                fake_store, k=20, vector_search_type="mmr"
            )

        fake_store.as_retriever.assert_called_once_with(
            search_type="mmr", search_kwargs={"k": 20}
        )

    def test_falls_back_to_dense_when_bm25_fails(self):
        """rank_bm25 missing/incompatible => degrade to dense-only."""
        fake_store = MagicMock()
        fake_retriever = MagicMock()
        fake_store.as_retriever.return_value = fake_retriever
        docs = [_doc("a")]

        bad_bm25 = MagicMock()
        bad_bm25.from_documents.side_effect = ImportError("Could not import rank_bm25")

        with patch.object(vector_embed, "_load_all_docs_from_store", return_value=docs), \
             patch.object(vector_embed, "BM25Retriever", bad_bm25):
            result = vector_embed.build_hybrid_retriever(fake_store, k=5)

        assert result is fake_retriever


# ===================================================================
# 3. _load_all_docs_from_store helper
# ===================================================================
class TestLoadAllDocsFromStore:
    def test_extracts_documents_and_metadata(self):
        store = MagicMock()
        store.get.return_value = {
            "documents": ["text one", "text two"],
            "metadatas": [{"source_file": "a.pdf"}, {"source_file": "b.pdf"}],
        }

        docs = vector_embed._load_all_docs_from_store(store)
        assert len(docs) == 2
        assert docs[0].page_content == "text one"
        assert docs[0].metadata["source_file"] == "a.pdf"
        assert docs[1].page_content == "text two"
        assert docs[1].metadata["source_file"] == "b.pdf"

    def test_returns_empty_on_get_failure(self):
        store = MagicMock()
        store.get.side_effect = RuntimeError("collection does not exist")
        assert vector_embed._load_all_docs_from_store(store) == []

    def test_returns_empty_on_second_get_failure(self):
        store = MagicMock()
        store.get.side_effect = RuntimeError("nope")
        assert vector_embed._load_all_docs_from_store(store) == []

    def test_skips_none_text_and_defaults_metadata(self):
        store = MagicMock()
        store.get.return_value = {
            "documents": ["ok", None, "also ok"],
            "metadatas": [{"a": 1}, None, {"b": 2}],
        }
        docs = vector_embed._load_all_docs_from_store(store)
        assert len(docs) == 2
        assert docs[0].page_content == "ok"
        assert docs[1].page_content == "also ok"

    def test_retries_with_limit_on_error(self):
        """On the first get() failing, the helper retries with a limit kwarg."""
        store = MagicMock()
        store.get.return_value = {
            "documents": ["fallback"],
            "metadatas": [{"source": "fallback.pdf"}],
        }
        # First call (no limit) raises; second call (with limit) succeeds.
        store.get.side_effect = [RuntimeError("boom"), store.get.return_value]

        docs = vector_embed._load_all_docs_from_store(store)
        assert len(docs) == 1
        assert docs[0].page_content == "fallback"


# ===================================================================
# 4. _doc_key deduplication identity
# ===================================================================
class TestDocKey:
    def test_prefers_citation_tag(self):
        hr = _hr(MagicMock())
        assert hr._doc_key(_doc("tag-001")) == "citation_tag:tag-001"

    def test_prefers_id_metadata_when_no_citation_tag(self):
        hr = _hr(MagicMock())
        d = Document(page_content="text", metadata={"id": "abc-123"})
        assert hr._doc_key(d) == "id:abc-123"

    def test_falls_back_to_source_file(self):
        hr = _hr(MagicMock())
        d = Document(page_content="text", metadata={"source_file": "report.pdf"})
        assert hr._doc_key(d) == "source_file:report.pdf"

    def test_falls_back_to_content(self):
        hr = _hr(MagicMock())
        d = Document(page_content="the quick brown fox", metadata={})
        assert hr._doc_key(d) == "content:the quick brown fox"
