"""
Shared fixtures and mocks for the test-analysis-agent test suite.

Run all tests:
    cd ulak-analyst-assist/test_analysis_agent
    pytest tests/ -v

Run a single file:
    pytest tests/test_rag.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Mock third-party modules that are not installed in the test environment.
# This MUST happen before any import of vector_embed / agent / bridge.
# ---------------------------------------------------------------------------
_MOCK_MODULES = [
    "langchain_docling",
    "langchain_docling.loader",
    "docling",
    "docling.chunking",
    "docling_core",
    "docling_core.transforms",
    "docling_core.transforms.chunker",
    "docling_core.transforms.chunker.tokenizer",
    "docling_core.transforms.chunker.tokenizer.huggingface",
    "docling_core.types",
    "docling_core.types.doc",
    "transformers",
    "sentence_transformers",
    "langchain_classic",
    "langchain_classic.retrievers",
    "langchain_classic.retrievers.parent_document_retriever",
    "langchain_classic.storage",
    "langchain_classic.retrievers.contextual_compression",
    "langchain_classic.retrievers.contextual_compression.contextual_compression_retriever",
    "langchain_classic.retrievers.document_compressors",
    "langchain_classic.retrievers.document_compressors.cross_encoder_rerank",
    "langchain_community.cross_encoders",
    "langchain_community.vectorstores",
    "langchain_community.vectorstores.utils",
]

for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Provide sensible defaults for things that vector_embed accesses at import time
sys.modules["langchain_classic"].retrievers = MagicMock()
sys.modules["langchain_classic"].storage = MagicMock()
sys.modules["langchain_community.vectorstores"].utils = MagicMock()
sys.modules["langchain_community.vectorstores.utils"].filter_complex_metadata = MagicMock(side_effect=lambda docs: docs)

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Ensure the package root is importable
# ---------------------------------------------------------------------------
AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ---------------------------------------------------------------------------
# Sample documents used across multiple test modules
# ---------------------------------------------------------------------------
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

EMPTY_DOCS: list[Document] = []

CORRUPT_DOCS: list[Document] = [
    Document(page_content="", metadata={}),
    Document(page_content="   ", metadata={}),
]


# ---------------------------------------------------------------------------
# Mock embedding model
# ---------------------------------------------------------------------------
class FakeEmbeddings:
    """Deterministic fake embeddings that return a fixed-dimensional vector."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 10) / 10.0 for i in range(self.dim)] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim

    # Async variants used by langchain
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------
MOCK_LLM_RESPONSE_TEXT = (
    "Test Case ID: TC-001\n"
    "Requirement: REQ-101\n"
    "Test Type: Boundary\n"
    "Preconditions: System is in idle state.\n"
    "Steps:\n"
    "1. Send input at boundary value.\n"
    "2. Observe output.\n"
    "Expected Result: System accepts the boundary input without error."
)

MOCK_LLM_NO_MATCH_RESPONSE = (
    "The retrieved sections don't cover the specific topic you asked about. "
    "No matching clause was found in the provided context."
)


@pytest.fixture
def mock_llm_response():
    """Return a mock that behaves like ChatOllama.invoke()."""
    mock = MagicMock()
    mock.content = MOCK_LLM_RESPONSE_TEXT
    return mock


@pytest.fixture
def mock_llm_no_match_response():
    mock = MagicMock()
    mock.content = MOCK_LLM_NO_MATCH_RESPONSE
    return mock


# ---------------------------------------------------------------------------
# Mock vector store / retriever
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_vector_store():
    """In-memory mock for a Chroma vector store."""
    store = MagicMock()
    store._collection = MagicMock()
    store._collection.count.return_value = len(SAMPLE_DOCS)

    def _add_documents(documents, **kwargs):
        return [f"id-{i}" for i in range(len(documents))]

    store.add_documents.side_effect = _add_documents
    return store


@pytest.fixture
def mock_retriever():
    """Mock retriever that returns sample documents for any query."""
    retriever = MagicMock()

    def _invoke(query, **kwargs):
        return SAMPLE_DOCS

    retriever.invoke.side_effect = _invoke
    return retriever


@pytest.fixture
def mock_retriever_empty():
    """Mock retriever that returns nothing — simulates no-match queries."""
    retriever = MagicMock()
    retriever.invoke.return_value = []
    return retriever


@pytest.fixture
def mock_compression_retriever():
    """Mock ContextualCompressionRetriever."""
    cr = MagicMock()
    cr.invoke.return_value = SAMPLE_DOCS[:2]
    return cr


# ---------------------------------------------------------------------------
# Mock agent / chain
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_agent():
    """Mock agent that returns a canned result dict (matches langchain agent output shape)."""
    agent_mock = MagicMock()

    final_message = MagicMock()
    final_message.content = MOCK_LLM_RESPONSE_TEXT
    final_message.type = "ai"

    agent_mock.invoke.return_value = {
        "messages": [final_message],
    }
    return agent_mock


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------
@pytest.fixture
def test_client():
    """Create a FastAPI TestClient with a pre-loaded mock agent."""
    from fastapi.testclient import TestClient

    with patch.dict(os.environ, {"PYTHONDONTWRITEBYTECODE": "1"}):
        import bridge

        mock_base = MagicMock()
        final_msg = MagicMock()
        final_msg.content = MOCK_LLM_RESPONSE_TEXT
        final_msg.type = "ai"
        mock_base.invoke.return_value = {"messages": [final_msg]}

        bridge.app_state["base_agent"] = mock_base
        bridge.app_state["thread_agents"] = {}
        bridge.app_state["startup_error"] = None

        client = TestClient(bridge.app)
        yield client
