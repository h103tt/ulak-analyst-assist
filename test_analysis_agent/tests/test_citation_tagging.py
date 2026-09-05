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
   strip the ``<chunk id="...">`` tags. Real Gemini embeddings + real
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

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

import vector_embed  # noqa: E402  (needs conftest's module mocks installed first)
from langchain_core.documents import Document  # noqa: E402


def is_gemini_online() -> bool:
    try:
        import agent

        agent.get_llm()
        return True
    except Exception:
        return False


requires_gemini = pytest.mark.skipif(
    not is_gemini_online(),
    reason="Gemini API is not reachable (no working key, or every model in MODEL_CHAIN is down)",
)


# ===================================================================
# 1. Tag format regression guard
# ===================================================================
def test_tag_chunks_format_and_verbatim_text():
    """_tag_chunks emits well-formed tags and never loses the original text.

    Pins the correct shape
    ``<chunk id="<stem>-NNN" source="test.pdf" section="..." page="...">...</chunk>``.
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
        r'^<chunk id="(?P<id>[A-Za-z0-9]+-\d{3})" source="test\.pdf" '
        r'section="[^"]*" page="[^"]*">'
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
    # Nothing is embedded or queried in this test; keep Chroma/Gemini offline.
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


@pytest.mark.integration
@requires_gemini
def test_tags_survive_retrieval_pipeline(tmp_path, monkeypatch):
    """End-to-end probe: <chunk id="..."> tags must reach the LLM context.

    Builds a REAL session tool (real Gemini embeddings, real Chroma, real
    reranker) over a small .txt fixture, queries it, and asserts the returned
    documents still carry their citation tags. If this fails, the
    reranker/compression retriever is stripping tags and no tool-description
    wording will fix citations.

    Calls the tool itself (not some internal `.retriever` reached into and
    swapped) so this exercises the exact same object build_session_retriever_tool
    hands the agent. langchain_core's create_retriever_tool doesn't expose
    the wrapped retriever as a public attribute, and conftest.py only mocks
    the reranker/docling stack when the real packages fail to import (see
    conftest.py's _MOCK_MODULES comment) -- in any environment with the real
    ML deps installed, which this test requires anyway via @requires_gemini's
    sibling KB dependencies, the tool already runs the real cross-encoder.
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

    raw = tool.invoke(CITATION_QUERY)

    print("\n===== RAW RETRIEVED CONTEXT (eye-ball sanity check) =====")
    print(raw if raw else "(empty)")
    print("=========================================================\n")

    assert raw, "retriever tool returned no documents"
    assert '<chunk id="' in raw, (
        "Tags did not survive the compression/reranker pipeline - citations "
        "cannot work regardless of tool-description wording."
    )
