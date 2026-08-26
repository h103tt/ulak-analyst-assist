"""
Citation-tagging regression tests for vector_embed.py
=====================================================
Exactly three tests, one per unverified / recently fixed behaviour:

1. ``test_tag_chunks_format_and_verbatim_text``
   Pure regression guard on ``_tag_chunks`` tag format. Fast, no mocks.

2. ``test_no_double_tagging_in_build_session_tool``
   Guards against the double-wrap bug (undefined ``exc`` + second
   ``_tag_chunks`` call) in ``build_session_retriever_tool``.
   Fast, mocked loader/store.

3. ``test_tags_survive_retrieval_pipeline``
   THE important unknown: verifies the reranker/compression stage does not
   strip the ``<chunk id="...">`` tags. Real Ollama embeddings + real
   BGE cross-encoder. Marked ``@pytest.mark.integration``.

Run:
    cd ulak-analyst-assist/test_analysis_agent
    uv run pytest tests/test_citation_tagging.py -v                 # fast (1-2)
    uv run pytest tests/test_citation_tagging.py -v -s -m integration  # probe (3)
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import vector_embed  # noqa: E402  (needs conftest's module mocks installed first)
from langchain_core.documents import Document  # noqa: E402


def is_ollama_online() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not is_ollama_online(),
    reason="Ollama server is not running on http://localhost:11434",
)


# ===================================================================
# 1. Tag format regression guard
# ===================================================================
def test_tag_chunks_format_and_verbatim_text():
    """_tag_chunks emits well-formed tags and never loses the original text.

    Pins the correct shape ``<chunk id="<stem>-NNN" source="test.pdf">...</chunk>``.
    (The id prefix comes from the filename stem, hence 'test-' for test.pdf.)
    The historical malformed-tag bug produced ``<chunk id>="..."``, which this
    full-string regex cannot match.
    """
    originals = [
        "Alpha paragraph about turbine lubrication schedules.\nSecond line.",
        "Beta paragraph quoting <angle brackets> & ampersands.",
    ]
    docs = [Document(page_content=text, metadata={}) for text in originals]

    tagged = vector_embed._tag_chunks(docs, "test.pdf")

    tag_re = re.compile(
        r'^<chunk id="(?P<id>[A-Za-z0-9]+-\d{3})" source="test\.pdf">'
        r"(?P<body>.*)</chunk>$",
        re.DOTALL,
    )

    assert len(tagged) == len(originals)
    for i, doc in enumerate(tagged):
        m = tag_re.match(doc.page_content)
        assert m, f"chunk {i} malformed page_content: {doc.page_content[:120]!r}"

        tag_id = m.group("id")
        assert doc.metadata["citation_tag"] == tag_id, "metadata drifted from tag id"
        assert m.group("body") == originals[i], "original text lost or truncated"


# ===================================================================
# 2. No double-tagging in build_session_retriever_tool
# ===================================================================
def test_no_double_tagging_in_build_session_tool(monkeypatch):
    """An already-tagged document must pass through storage untouched.

    Regression guard for the bug where load_document's output got wrapped a
    second time by _tag_chunks (and the failure path referenced undefined
    ``exc``).
    """
    already_tagged = Document(
        page_content=(
            '<chunk id="user-001" source="report.pdf">Original body text.</chunk>'
        ),
        metadata={"citation_tag": "user-001", "source": "report.pdf"},
    )

    monkeypatch.setattr(vector_embed, "load_document", lambda file_path: [already_tagged])

    captured: dict = {}

    def fake_add_with_storage_debug(store, collection_name, documents):
        captured["documents"] = list(documents)
        return [f"id-{i}" for i in range(len(documents))]

    monkeypatch.setattr(vector_embed, "_add_with_storage_debug", fake_add_with_storage_debug)
    # Nothing is embedded or queried in this test; keep Chroma/Ollama offline.
    monkeypatch.setattr(vector_embed, "Chroma", lambda **kwargs: MagicMock())

    _, report = vector_embed.build_session_retriever_tool(
        ["/tmp/fake/report.pdf"],
        session_id="double-tag-guard",
    )

    assert report["chunk_count"] == 1
    stored = captured["documents"]
    assert len(stored) == 1
    # IDENTICAL content => no second <chunk> wrap happened.
    assert stored[0].page_content == already_tagged.page_content
    assert stored[0].page_content.count("<chunk") == 1
    assert stored[0].metadata["citation_tag"] == "user-001"


# ===================================================================
# 3. Tags survive the retrieval pipeline (integration)
# ===================================================================
CITATION_FIXTURE_TEXT = (
    "The Zephyr-7 wind turbine gearbox requires a lubrication oil analysis every "
    "500 operating hours. Its vibration alarm threshold is set to exactly "
    "4.2 millimetres per second RMS, measured at the gearbox housing sensor.\n\n"
    "Quartzite extraction permits in the fictional province of Valdoria expire "
    "after eleven years and must be accompanied by a seasonal bat population survey.\n\n"
    "The Valdoria engineering cafeteria serves saffron risotto every Tuesday at a "
    "price of fourteen credits, including one complimentary cardamom bun."
)
CITATION_QUERY = "What is the vibration alarm threshold of the Zephyr-7 wind turbine gearbox?"

_RERANKER_MODULE_PREFIXES = ("langchain_classic", "langchain_community.cross_encoders")


def _load_real_reranker_classes():
    """Import the REAL CrossEncoderReranker / HuggingFaceCrossEncoder classes.

    conftest.py swaps these modules for MagicMocks so the rest of the suite
    runs fast and offline. This temporarily un-mocks them, grabs the real
    classes, then restores the mock state.
    """
    saved = {
        n: mod
        for n, mod in sys.modules.items()
        if n.startswith(_RERANKER_MODULE_PREFIXES)
    }
    for n in saved:
        del sys.modules[n]
    try:
        from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import (
            CrossEncoderReranker,
        )
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        return CrossEncoderReranker, HuggingFaceCrossEncoder
    finally:
        for n in list(sys.modules):
            if n.startswith(_RERANKER_MODULE_PREFIXES):
                del sys.modules[n]
        sys.modules.update(saved)


@pytest.mark.integration
@requires_ollama
def test_tags_survive_retrieval_pipeline(tmp_path, monkeypatch):
    """End-to-end probe: <chunk id="..."> tags must reach the LLM context.

    Builds a REAL session tool (real Ollama embeddings, real Chroma, real
    reranker) over a small .txt fixture, queries it, and asserts the returned
    documents still carry their citation tags. If this fails, the
    reranker/compression retriever is stripping tags and no tool-description
    wording will fix citations.
    """
    fixture = tmp_path / "citation_probe.txt"
    fixture.write_text(CITATION_FIXTURE_TEXT, encoding="utf-8")

    # Keep the session collection out of the repo's real chromadb directory.
    collections_dir = tmp_path / "user_collections"
    monkeypatch.setattr(vector_embed, "USER_COLLECTIONS_DIR", str(collections_dir))

    session_id = f"cite-probe-{uuid.uuid4().hex[:8]}"  # unique => no stale collection reuse
    tool, report = vector_embed.build_session_retriever_tool(
        [str(fixture)],
        session_id=session_id,
    )
    assert report["failed_files"] == [], f"ingest failed: {report['failed_files']}"

    retriever = getattr(tool, "retriever", None)
    assert retriever is not None, "retriever tool does not expose its .retriever"

    # Swap the conftest-mocked compressor for the real cross-encoder so the
    # reranking stage (the suspected tag-stripper) actually executes.
    CrossEncoderReranker, HuggingFaceCrossEncoder = _load_real_reranker_classes()
    retriever.compressor = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),
        top_n=5,
    )

    results = retriever.invoke(CITATION_QUERY)

    raw = "\n-----\n".join(d.page_content for d in results)
    print("\n===== RAW RETRIEVED CONTEXT (eye-ball sanity check) =====")
    print(raw if raw else "(empty)")
    print("=========================================================\n")

    assert results, "compression retriever returned no documents"
    assert '<chunk id="' in raw, (
        "Tags did not survive the compression/reranker pipeline - citations "
        "cannot work regardless of tool-description wording."
    )
