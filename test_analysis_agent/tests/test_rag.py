"""
RAG & Vector Search Tests
=========================
Tests for document ingestion, chunking, embedding, vector storage,
similarity retrieval, and context injection.

Run:
    cd ulak-analyst-assist/test_analysis_agent
    pytest tests/test_rag.py -v
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain_core.documents import Document

# Ensure the package root is importable
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# Shared sample documents for retrieval tests
SAMPLE_DOCS: list[Document] = [
    Document(
        page_content=(
            "ISO/IEC/IEEE 29119-1:2022 defines concepts and provides a framework "
            "for software testing. It covers test processes, test documentation, "
            "and test techniques."
        ),
        metadata={"source": "29119-1-2022.pdf", "standard": "29119-1-2022", "category": "Requirements_and_quality"},
    ),
    Document(
        page_content=(
            "IEEE 829-2008 Standard for Software Test Documentation specifies "
            "formats for test plans, test design specifications, test case "
            "specifications, and test summary reports."
        ),
        metadata={"source": "IEEE-Test-Doc-829-2008.pdf", "standard": "IEEE-Test-Doc-829-2008", "category": "Requirements_and_quality"},
    ),
    Document(
        page_content=(
            "MIL-STD-882E establishes the DoD standard practice for system safety. "
            "It defines hazard risk assessment and mitigation requirements."
        ),
        metadata={"source": "MIL-STD-882E.pdf", "standard": "MIL-STD-882E", "category": "Security_and_safety"},
    ),
]


# ===================================================================
# 1. Document Ingestion & Chunking
# ===================================================================
class TestDocumentChunking:
    """Verify splitting logic, chunk sizes, and edge-case handling."""

    def test_text_loader_returns_documents(self):
        """TextLoader should produce at least one Document from a plain text file."""
        from langchain_community.document_loaders import TextLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Line one.\nLine two.\nLine three.\n")
            tmp = f.name
        try:
            docs = TextLoader(tmp).load()
            assert len(docs) >= 1
            assert isinstance(docs[0], Document)
            assert "Line one" in docs[0].page_content
        finally:
            os.unlink(tmp)

    def test_text_loader_empty_file(self):
        """Empty files should still return a Document (possibly empty content)."""
        from langchain_community.document_loaders import TextLoader

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            tmp = f.name
        try:
            docs = TextLoader(tmp).load()
            assert len(docs) >= 1
        finally:
            os.unlink(tmp)

    def test_recursive_text_splitter_chunk_sizes(self):
        """Verify chunk sizes respect configured limits."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        long_text = "word " * 200  # ~1000 chars
        chunks = splitter.split_text(long_text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 150  # some tolerance for word boundaries

    def test_recursive_text_splitter_overlap(self):
        """Chunks should overlap by the configured amount."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text = "abcdefghij " * 50  # repeating pattern
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-10:]
            head = chunks[i + 1][:30]
            assert any(c in head for c in tail)

    @pytest.mark.parametrize("content,expected_min", [
        ("Single line.", 1),
        ("Paragraph one.\n\nParagraph two.\n\nParagraph three.", 1),
        ("A" * 5000, 5),
    ])
    def test_splitter_parametrized(self, content, expected_min):
        """Parametrized chunking tests for various document sizes."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        assert len(chunks) >= expected_min

    def test_empty_document_handling(self):
        """Chunking an empty document should not crash."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text("")
        assert chunks == [] or chunks == [""]

    def test_whitespace_only_document(self):
        """Document with only whitespace should produce minimal chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text("   \n\n   ")
        assert len(chunks) <= 1


# ===================================================================
# 2. Collection Name Sanitization
# ===================================================================
class TestCollectionNameSanitization:
    """Test sanitize_collection_name from vector_embed."""

    def test_clean_name_unchanged(self):
        from vector_embed import sanitize_collection_name
        assert sanitize_collection_name("my_collection") == "my_collection"

    def test_special_chars_replaced(self):
        from vector_embed import sanitize_collection_name
        result = sanitize_collection_name("my/collection:name!@#")
        # Result should only contain alphanumerics, underscores, or hyphens
        assert re.match(r"^[A-Za-z0-9_-]+$", result), f"Invalid chars in: {result}"

    def test_empty_string(self):
        from vector_embed import sanitize_collection_name
        result = sanitize_collection_name("")
        assert len(result) >= 1

    def test_oversized_name_truncated(self):
        from vector_embed import sanitize_collection_name
        long_name = "a" * 300
        result = sanitize_collection_name(long_name)
        assert len(result) <= 255

    def test_starts_ends_with_letter(self):
        from vector_embed import sanitize_collection_name
        result = sanitize_collection_name("a_valid_name")
        assert result[0].isalnum()
        assert result[-1].isalnum()

    def test_only_special_chars(self):
        from vector_embed import sanitize_collection_name
        result = sanitize_collection_name("!@#$%")
        assert len(result) >= 1
        assert result[0].isalnum()
        assert result[-1].isalnum()


# ===================================================================
# 3. Embedding & Vector Storage
# ===================================================================
class TestEmbeddingAndStorage:
    """Verify embedding generation and batch insertion logic."""

    def test_fake_embeddings_dimensions(self, fake_embeddings):
        """Fake embeddings should return vectors of the correct dimension."""
        texts = ["hello", "world", "test query"]
        vectors = fake_embeddings.embed_documents(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == fake_embeddings.dim

    def test_fake_embeddings_query(self, fake_embeddings):
        """Query embedding should match document embedding dimension."""
        q = fake_embeddings.embed_query("find safety standards")
        assert len(q) == fake_embeddings.dim

    def test_add_in_batches_calls_vector_store(self, mock_vector_store):
        """add_in_batches should call add_documents in correct batch sizes."""
        from vector_embed import add_in_batches, EMBED_BATCH_SIZE

        n_docs = EMBED_BATCH_SIZE * 3 + 5
        docs = [Document(page_content=f"chunk-{i}", metadata={}) for i in range(n_docs)]

        with patch("vector_embed.vector_store", mock_vector_store):
            add_in_batches(docs, label="test_batch")

        expected_calls = 4  # 3 full batches + 1 partial
        assert mock_vector_store.add_documents.call_count == expected_calls

    def test_add_in_batches_empty_list(self, mock_vector_store):
        """Empty list should not trigger any add_documents call."""
        from vector_embed import add_in_batches

        with patch("vector_embed.vector_store", mock_vector_store):
            add_in_batches([], label="empty")
        mock_vector_store.add_documents.assert_not_called()

    @pytest.mark.parametrize("batch_size,total,expected_calls", [
        (32, 100, 4),
        (32, 32, 1),
        (32, 33, 2),
        (32, 0, 0),
    ])
    def test_batch_count_calculation(self, mock_vector_store, batch_size, total, expected_calls):
        """Verify correct number of batch calls for various document counts."""
        from vector_embed import add_in_batches

        docs = [Document(page_content=f"doc-{i}", metadata={}) for i in range(total)]

        with patch("vector_embed.vector_store", mock_vector_store), \
             patch("vector_embed.EMBED_BATCH_SIZE", batch_size):
            add_in_batches(docs, label="param_test")

        assert mock_vector_store.add_documents.call_count == expected_calls


# ===================================================================
# 4. Retrieval Quality & Top-K
# ===================================================================
class TestRetrieval:
    """Test similarity search, scoring, and empty-retrieval handling."""

    def test_retriever_returns_documents(self, mock_retriever):
        """Standard query should return a list of Documents."""
        results = mock_retriever.invoke("What is ISO 29119?")
        assert isinstance(results, list)
        assert len(results) > 0
        assert isinstance(results[0], Document)

    def test_retriever_returns_expected_count(self, mock_retriever):
        """Mock retriever should return all sample docs."""
        results = mock_retriever.invoke("test query")
        assert len(results) == len(SAMPLE_DOCS)

    def test_retriever_empty_result(self, mock_retriever_empty):
        """Query matching nothing should return an empty list."""
        results = mock_retriever_empty.invoke("obscure query about purple elephants")
        assert results == []

    def test_retrieval_metadata_preserved(self, mock_retriever):
        """Retrieved documents should retain their metadata."""
        results = mock_retriever.invoke("safety standards")
        for doc in results:
            assert "source" in doc.metadata or "standard" in doc.metadata

    def test_compression_retriever_returns_subset(self, mock_compression_retriever):
        """Compression retriever should return a filtered subset."""
        results = mock_compression_retriever.invoke("test plan format")
        assert len(results) <= len(SAMPLE_DOCS)


# ===================================================================
# 5. Context Injection
# ===================================================================
class TestContextInjection:
    """Verify retrieved context is correctly formatted for prompts."""

    def test_context_formatting_single_doc(self):
        """Single document context should be cleanly extractable."""
        doc = SAMPLE_DOCS[0]
        context = "\n---\n".join(d.page_content for d in [doc])
        assert "29119" in context

    def test_context_formatting_multiple_docs(self):
        """Multiple documents should be joined with delimiters."""
        context = "\n---\n".join(d.page_content for d in SAMPLE_DOCS)
        assert "ISO/IEC/IEEE" in context
        assert "IEEE 829" in context
        assert "MIL-STD-882E" in context
        # Delimiter should appear between docs
        assert context.count("---") >= 2

    def test_context_in_prompt_template(self):
        """Context should be injectable into a prompt without errors."""
        context = "\n".join(d.page_content for d in SAMPLE_DOCS)
        prompt = (
            f"Using these standards:\n{context}\n\n"
            "Generate test cases for: boundary value analysis"
        )
        assert "29119" in prompt
        assert "boundary value analysis" in prompt

    def test_empty_context_handling(self):
        """Empty context should not crash the prompt builder."""
        context = ""
        prompt = f"Context:\n{context}\n\nQuestion: What is testing?"
        assert "Question: What is testing?" in prompt

    def test_context_metadata_extraction(self):
        """Standard names should be extractable from metadata."""
        standards = [d.metadata.get("standard", "unknown") for d in SAMPLE_DOCS]
        assert "29119-1-2022" in standards
        assert "MIL-STD-882E" in standards


# ===================================================================
# 6. DOCS Registry & Metadata Lookup
# ===================================================================
class TestDocsRegistry:
    """Verify the DOCS and DOC_METADATA_LOOKUP constants."""

    def test_docs_tuple_structure(self):
        from vector_embed import DOCS
        for entry in DOCS:
            assert len(entry) == 3
            category, filename, label = entry
            assert isinstance(category, str)
            assert isinstance(filename, str)
            assert isinstance(label, str)

    def test_metadata_lookup_populated(self):
        from vector_embed import DOC_METADATA_LOOKUP, DOCS
        assert len(DOC_METADATA_LOOKUP) == len(DOCS)

    def test_metadata_lookup_has_required_keys(self):
        from vector_embed import DOC_METADATA_LOOKUP
        for filename, meta in DOC_METADATA_LOOKUP.items():
            assert "category" in meta
            assert "standard" in meta

    def test_metadata_lookup_known_entry(self):
        from vector_embed import DOC_METADATA_LOOKUP
        assert "MIL-STD-882E.pdf" in DOC_METADATA_LOOKUP
        meta = DOC_METADATA_LOOKUP["MIL-STD-882E.pdf"]
        assert meta["category"] == "Security_and_safety"
        assert meta["standard"] == "MIL-STD-882E"


# ===================================================================
# 7. Embeddings Model Integration Points
# ===================================================================
class TestEmbeddingsInterface:
    """Verify FakeEmbeddings satisfies the LangChain embeddings interface contract."""

    def test_embed_documents_returns_list_of_lists(self, fake_embeddings):
        result = fake_embeddings.embed_documents(["a", "b", "c"])
        assert isinstance(result, list)
        assert all(isinstance(v, list) for v in result)

    def test_embed_query_returns_list(self, fake_embeddings):
        result = fake_embeddings.embed_query("test")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_consistent_dimension_across_calls(self, fake_embeddings):
        q1 = fake_embeddings.embed_query("first query")
        q2 = fake_embeddings.embed_query("second query")
        assert len(q1) == len(q2)

    @pytest.mark.asyncio
    async def test_async_embed_query(self, fake_embeddings):
        result = await fake_embeddings.aembed_query("async test")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_async_embed_documents(self, fake_embeddings):
        result = await fake_embeddings.aembed_documents(["doc1", "doc2"])
        assert len(result) == 2
