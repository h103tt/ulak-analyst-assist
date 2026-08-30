"""
KB Ingestion Tests (E2E, format-focused)
=========================================
Covers rag_e2e.feature @ingestion scenarios: PDF/Markdown ingestion through
the real Docling pipeline, metadata tagging, corrupted-file resilience, and
the DOCS-registry-vs-disk consistency check.

Run:
    cd test_analysis_agent
    pytest tests/e2e/test_ingestion_formats.py -v
    pytest tests/e2e/test_ingestion_formats.py -v -m integration   # PDF parsing only
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

KB_DIR = AGENT_DIR / "knowledge_base"

pytestmark = pytest.mark.e2e


# ===================================================================
# 1. DOCS registry vs. disk (fast, no parsing required)
# ===================================================================
class TestDocsRegistryConsistency:
    """Guards against the exact drift this audit found: DOCS/DOC_METADATA_LOOKUP
    referencing files that were removed from knowledge_base/ in cleanup commits."""

    def test_every_registered_doc_exists_on_disk(self):
        from vector_embed import DOCS

        missing = []
        for category, filename, _standard in DOCS:
            if not (KB_DIR / category / filename).exists():
                missing.append(f"{category}/{filename}")

        assert not missing, (
            "vector_embed.DOCS references files that no longer exist on disk "
            f"(remove them from DOCS or restore the files): {missing}"
        )

    def test_every_file_on_disk_is_registered(self):
        """Flags orphan KB files that are on disk but missing metadata --
        they'd still get embedded (load_concurrently_multi_format scans by
        extension), just without a 'standard'/'category' tag."""
        from vector_embed import DOC_METADATA_LOOKUP

        on_disk = {
            p.name
            for p in KB_DIR.rglob("*")
            if p.suffix.lower() in (".pdf", ".docx", ".xlsx", ".xls", ".md")
        }
        unregistered = on_disk - set(DOC_METADATA_LOOKUP.keys())
        assert not unregistered, (
            f"Files present in knowledge_base/ but missing from "
            f"DOC_METADATA_LOOKUP (chunks would lack standard/category "
            f"metadata): {unregistered}"
        )


# ===================================================================
# 2. Real PDF / Markdown ingestion through Docling (slow, live)
# ===================================================================
@pytest.mark.integration
class TestRealDocumentIngestion:
    KB_PDF_SAMPLES = [
        ("Security_and_safety", "MIL-STD-882E.pdf", "MIL-STD-882E"),
        ("Requirements_and_quality", "IEEE-Test-Doc-829-2008.pdf", "IEEE-Test-Doc-829-2008"),
    ]

    @pytest.mark.parametrize("category,filename,standard", KB_PDF_SAMPLES)
    def test_pdf_ingestion_produces_tagged_chunks(self, category, filename, standard):
        from vector_embed import process_single_file

        path = KB_DIR / category / filename
        if not path.exists():
            pytest.skip(f"{path} not present in this checkout")

        docs = process_single_file(path)

        assert len(docs) > 0, f"No chunks produced for {filename}"
        for doc in docs[:5]:  # sampling is enough; full-doc parsing is slow
            assert doc.metadata.get("source_file") == filename
            assert doc.metadata.get("standard") == standard
            assert doc.metadata.get("category") == category
            assert "citation_tag" in doc.metadata
            assert doc.page_content.startswith("<chunk ")
            assert f'standard="{standard}"' in doc.page_content

    def test_markdown_ingestion_via_load_document(self):
        from vector_embed import load_document

        path = KB_DIR / "Requirements_and_quality" / "requirements_and_testing.md"
        if not path.exists():
            pytest.skip(f"{path} not present in this checkout")

        docs = load_document(str(path))
        assert len(docs) > 0
        assert all(doc.page_content.startswith("<chunk ") for doc in docs)


# ===================================================================
# 3. Corrupted / empty file resilience (fast, no live model needed)
# ===================================================================
class TestCorruptedFileHandling:
    def test_empty_pdf_does_not_raise(self):
        """process_single_file must swallow Docling failures and return []
        so one bad file in a ThreadPoolExecutor batch doesn't kill the rest."""
        from vector_embed import process_single_file

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\nnot a real pdf body")
            tmp = Path(f.name)
        try:
            docs = process_single_file(tmp)  # should not raise
            assert docs == []
        finally:
            tmp.unlink(missing_ok=True)


# ===================================================================
# 4. Unsupported format: JSON (documents a real gap, does not assert
#    a specific outcome since none is currently guaranteed)
# ===================================================================
class TestJsonIngestionGap:
    def test_json_ingestion_documents_current_gap(self):
        """load_document() has explicit branches for .csv and .txt/.md, but
        NOT .json -- a .json file falls through to the generic Docling
        binary path (_load_binary_with_docling), which is designed for
        PDFs/Office documents, not structured JSON. This test documents
        that gap so it fails loudly (via the informative message) the
        moment someone changes the branching logic, rather than silently
        mis-parsing JSON requirement exports in production.

        Recommendation: add an explicit `elif ext == ".json":` branch using
        e.g. langchain_community.document_loaders.JSONLoader with a
        jq_schema appropriate to the requirement-export format, before
        JSON is advertised as a supported ingestion format.
        """
        import inspect

        import vector_embed

        source = inspect.getsource(vector_embed.load_document)
        assert ".json" not in source, (
            "load_document() now has an explicit .json branch -- update "
            "this test to exercise it directly instead of asserting the gap, "
            "and update rag_e2e.feature's 'Ingest a JSON document' scenario "
            "to describe the real (now-supported) behavior."
        )
