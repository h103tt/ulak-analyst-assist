"""Plain helper functions for the E2E suite (deliberately NOT conftest.py,
so test modules can `import` it directly without triggering pytest's
separate auto-loading of conftest.py as a second module object)."""
from __future__ import annotations

import pytest


def is_gemini_online() -> bool:
    """Cheap liveness check: agent.get_llm() probes and caches a working
    key with one small 'ping' call the first time it's invoked in this
    process, so this only costs one real API call per test session, not
    per test -- and it already walks MODEL_CHAIN/rotates keys on failure,
    so a single working model+key anywhere in the chain is enough."""
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


JUDGE_MODEL = "gemini-3.5-flash-lite"


def get_judge_model():
    """A deepeval-compatible Gemini judge model, using whichever configured
    Gemini API key actually works right now. Kept separate from agent.get_llm()
    so a judge-purpose key gets probed/cached independently of the
    chat-purpose one (Google tracks quota per project+model, not globally)."""
    import gemini_keys
    from deepeval.models import GeminiModel

    def probe(api_key: str) -> None:
        GeminiModel(model=JUDGE_MODEL, api_key=api_key, temperature=0.0).generate("ping")

    api_key = gemini_keys.working_key(probe, purpose=f"judge:{JUDGE_MODEL}")
    return GeminiModel(model=JUDGE_MODEL, api_key=api_key, temperature=0.3)


def skip_if_kb_empty(kb_populated: bool) -> None:
    if not kb_populated:
        pytest.skip(
            "The 'iso_files' Chroma collection is empty -- run "
            "`python vector_embed.py` to ingest the knowledge base first."
        )
