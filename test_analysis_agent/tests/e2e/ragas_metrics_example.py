"""
RAGAS integration sketch (illustrative, not wired into the CI suite).
=======================================================================
The project already standardized on deepeval (see test_rag_metrics.py and
test_qa_regression.py) with a local Ollama judge model, which is the
lower-friction choice here since it reuses the same judge_model /
local_embedder plumbing everywhere else. This file shows the equivalent
RAGAS wiring in case the team wants to cross-check deepeval's scores with a
second framework, or standardize on RAGAS instead.

Install (not in requirements.txt by default):
    uv pip install ragas datasets

RAGAS expects a HuggingFace `datasets.Dataset` with columns:
    question, answer, contexts (list[str]), ground_truth
and evaluates with LangChain-wrapped LLM/embeddings (so ChatOllama /
OllamaEmbeddings from this project plug in directly, no adapter needed
unlike deepeval's OllamaModel/DeepEvalBaseEmbeddingModel wrappers).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


def build_ragas_dataset(golden_set: list[dict], trace_results: list[dict]):
    """golden_set: tests/e2e/golden_dataset.json's "golden_set" list.
    trace_results: one /trace response dict per golden item (same order),
    e.g. captured by TestGenerationFaithfulnessLive in test_qa_regression.py.
    """
    from datasets import Dataset

    rows = {
        "question": [g["question"] for g in golden_set],
        "answer": [t["answer"] for t in trace_results],
        "contexts": [
            [
                m["content"]
                for m in t["messages"]
                if m["type"] == "ToolMessage" and m.get("name") == "search_testing_standards"
            ]
            for t in trace_results
        ],
        "ground_truth": [g["expected_answer"] for g in golden_set],
    }
    return Dataset.from_dict(rows)


def run_ragas_evaluation(dataset):
    """Faithfulness = is the answer supported by the retrieved contexts
    (hallucination check); answer_relevancy = does the answer address the
    question; context_precision/recall = retrieval quality -- the RAGAS
    equivalents of the deepeval Contextual* metrics used in
    test_qa_regression.py::TestRetrievalQualityMetrics."""
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge_llm = ChatOllama(model="gemma4:12b", temperature=0.3)
    judge_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )


if __name__ == "__main__":
    # Example manual run: point at a JSON file of captured /trace responses
    # (one per golden item, produced by a small script around
    # TestGenerationFaithfulnessLive.live_client), then score with RAGAS.
    golden = json.loads((Path(__file__).parent / "golden_dataset.json").read_text())["golden_set"]
    traces_path = Path(__file__).parent / "captured_traces.json"
    if not traces_path.exists():
        print(f"Expected captured trace responses at {traces_path} -- see docstring above.")
        sys.exit(1)

    traces = json.loads(traces_path.read_text())
    ds = build_ragas_dataset(golden, traces)
    result = run_ragas_evaluation(ds)
    print(result)
