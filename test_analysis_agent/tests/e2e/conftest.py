"""Shared fixtures for the E2E RAG regression suite (tests/e2e).

tests/conftest.py (one level up) already mocks heavy optional deps
(docling, transformers, sentence_transformers, ...) so `import vector_embed`
/ `import agent` / `import bridge` stay import-safe even without those
packages installed. This file only adds what's specific to the E2E suite:
the golden dataset fixture and the KB-populated check. Plain helper
functions (skip_if_kb_empty, requires_gemini, ...) live in _helpers.py so
test modules can import them without pytest also picking up a second copy
via its own conftest auto-loading.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


@pytest.fixture(scope="session")
def golden_dataset() -> dict:
    """Load the curated (or synthesizer-generated) golden Q&A set.

    Point this at tests/e2e/golden_dataset.generated.json (produced by
    generate_golden_dataset.py) to run against a larger, KB-derived set.
    """
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def kb_populated() -> bool:
    """True if the real 'iso_files' Chroma collection has documents.

    A fresh checkout hasn't run vector_embed.ingest_knowledge_base() yet,
    so live retrieval/generation tests should skip loudly instead of
    failing confusingly on an empty collection.
    """
    try:
        import vector_embed

        return vector_embed.vector_store._collection.count() > 0
    except Exception:
        return False
