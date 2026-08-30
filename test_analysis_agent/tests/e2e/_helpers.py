"""Plain helper functions for the E2E suite (deliberately NOT conftest.py,
so test modules can `import` it directly without triggering pytest's
separate auto-loading of conftest.py as a second module object)."""
from __future__ import annotations

import httpx
import pytest


def is_ollama_online() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def is_model_pulled(model_tag: str) -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        r.raise_for_status()
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        return any(t.startswith(model_tag.split(":")[0]) for t in tags)
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not is_ollama_online(),
    reason="Ollama server is not running on http://localhost:11434",
)


def skip_if_kb_empty(kb_populated: bool) -> None:
    if not kb_populated:
        pytest.skip(
            "The 'iso_files' Chroma collection is empty -- run "
            "`python vector_embed.py` to ingest the knowledge base first."
        )
