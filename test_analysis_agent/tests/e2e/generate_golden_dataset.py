"""
Golden Q&A dataset generator for E2E RAG regression testing.

This is a MANUAL/CLI script, not a pytest test -- it talks to a live Ollama
instance and a docling parsing pipeline, both of which are slow and
non-deterministic, so it must never run as part of the automated suite.
It completes the synthesizer wiring that was left commented out in
test_analysis_agent/test_rag_metrics.py.

Usage (from test_analysis_agent/):
    uv run python tests/e2e/generate_golden_dataset.py --out tests/e2e/golden_dataset.generated.json

Requirements:
    - Ollama running locally with the judge model available
      (ollama pull gemma3:4b  -- or override with --judge-model)
    - The knowledge_base/ PDFs this script points at must exist on disk
      (see KB_SOURCE_FILES below; keep this list in sync with
      vector_embed.DOCS / DOC_METADATA_LOOKUP)

Output: a JSON file with the same shape as golden_dataset.json
(top-level "golden_set" list), suitable for direct use by
test_qa_regression.py's `golden_dataset` fixture.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

KB_DIR = AGENT_DIR / "knowledge_base"

# Keep this in sync with vector_embed.DOC_METADATA_LOOKUP. Only files that
# actually exist on disk should be listed here -- see the "DOCS registry
# stays in sync with files on disk" scenario in rag_e2e.feature for why.
KB_SOURCE_FILES = [
    ("Environmental_and_hardware", "MIL-STD-461.pdf", "MIL-STD-461"),
    ("Environmental_and_hardware", "MIL-STD-1586A.pdf", "MIL-STD-1586A"),
    ("Requirements_and_quality", "15288-2023-2.pdf", "15288-2023-2"),
    ("Requirements_and_quality", "29119-1-2022.pdf", "29119-1-2022"),
    ("Requirements_and_quality", "IEEE-Test-Doc-829-2008.pdf", "IEEE-Test-Doc-829-2008"),
    ("Security_and_safety", "MIL-STD-882E.pdf", "MIL-STD-882E"),
    ("Security_and_safety", "SP800-53_REV-3.PDF", "SP800-53_REV-3"),
]


def _resolve_present_files() -> list[tuple[str, str]]:
    """Return (path, standard_label) for every configured file that
    actually exists, skipping (with a warning) any that don't."""
    present = []
    for category, filename, standard in KB_SOURCE_FILES:
        path = KB_DIR / category / filename
        if path.exists():
            present.append((str(path), standard))
        else:
            print(f"[skip] {path} not found on disk", file=sys.stderr)
    return present


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/e2e/golden_dataset.generated.json")
    parser.add_argument("--judge-model", default="gemma3:4b")
    parser.add_argument("--max-goldens-per-doc", type=int, default=3)
    args = parser.parse_args()

    # Imported lazily: deepeval + its synthesizer pull in heavy optional
    # deps (docling, transformers, ...) that a pure pytest-collection run
    # should never have to pay for.
    from deepeval.models import OllamaModel
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import ContextConstructionConfig
    from deepeval.models.base_model import DeepEvalBaseEmbeddingModel

    import vector_embed

    class DeepEvalEmbedder(DeepEvalBaseEmbeddingModel):
        def __init__(self, langchain_embedder):
            self.embedder = langchain_embedder

        def load_model(self):
            return self.embedder

        def get_model_name(self) -> str:
            return "nomic-embed-text"

        def embed_text(self, text: str) -> list[float]:
            return self.embedder.embed_query(text)

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return self.embedder.embed_documents(texts)

        async def a_embed_text(self, text: str) -> list[float]:
            return self.embed_text(text)

        async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
            return self.embed_texts(texts)

    judge_model = OllamaModel(model=args.judge_model, base_url="http://localhost:11434", temperature=0.3)
    synthesizer = Synthesizer(model=judge_model)
    context_config = ContextConstructionConfig(
        embedder=DeepEvalEmbedder(vector_embed.embeddings),
        critic_model=judge_model,  # without this, deepeval defaults to OpenAI and needs OPENAI_API_KEY
        max_contexts_per_document=args.max_goldens_per_doc,
    )

    present_files = _resolve_present_files()
    if not present_files:
        print("No configured KB files found on disk -- nothing to generate.", file=sys.stderr)
        sys.exit(1)

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=[path for path, _standard in present_files],
        context_construction_config=context_config,
    )

    golden_set = [
        {
            "id": f"GEN-{i:03d}",
            "question": g.input,
            "expected_answer": g.expected_output,
            "expected_standard": None,  # not attributable 1:1 -- fill in by hand if needed
            "expected_category": None,
            "source": "deepeval.synthesizer",
        }
        for i, g in enumerate(goldens, start=1)
    ]

    out_path = AGENT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"_meta": {"source": "generate_golden_dataset.py"}, "golden_set": golden_set}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(golden_set)} generated goldens to {out_path}")


if __name__ == "__main__":
    main()
