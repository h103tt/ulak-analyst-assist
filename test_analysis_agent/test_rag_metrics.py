import vector_embed
import os

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.dataset import EvaluationDataset
from langchain_ollama import ChatOllama
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseEmbeddingModel
from deepeval.synthesizer.config import ContextConstructionConfig

# Shared Ollama judge model used by all deepeval metrics & the synthesizer
judge_model = OllamaModel(model="gemma3:4b", base_url="http://localhost:11434", temperature=0.5)

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

class DeepEvalEmbedder(DeepEvalBaseEmbeddingModel):
    def __init__(self, langchain_embedder):
        self.embedder = langchain_embedder
    
    def load_model(model):
        return self.embedder
    
    def get_model_name(self):
        return "nomic-embed-text"
    
    def embed_text(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)
    
    def embed_texts(self, text: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)
    
    async def a_embed_text(self, text: str) -> list[[float]]:
        return self.embed_text(text)
    
    async def a_embed_texts(self, text: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)
    
local_embedder = DeepEvalEmbedder(vector_embed.embeddings)




# ---------------------------------------------------------------------------
# 1. Generate goldens from the knowledge base documents
# ---------------------------------------------------------------------------
# from deepeval.synthesizer import Synthesizer

# synthesizer = Synthesizer(model=judge_model)

# context_config = ContextConstructionConfig(
#     embedder=local_embedder,
# )

# goldens = synthesizer.generate_goldens_from_docs(
#     document_paths=[
#         "ulak-analyst-assist/test_analysis_agent/knowledge_base/04_requirements/requirements_engineering.txt",
#         "ulak-analyst-assist/test_analysis_agent/knowledge_base/03_safety_security_config/MIL-STD-1586A.pdf",
#         "ulak-analyst-assist/test_analysis_agent/knowledge_base/02_verification_and_testing/IEEE-Test-Doc-829-2008.pdf",
#         "ulak-analyst-assist/test_analysis_agent/knowledge_base/01_se_process_and_requirements/15288-2023-2.pdf",
#         "ulak-analyst-assist/test_analysis_agent/knowledge_base/03_safety_security_config/NIST_SP_800-171A.pdf",
#         "ulak-analyst-assist/test_analysis_agent/knowledge_base/03_safety_security_config/SP800-53_REV-3.PDF",
#     ],
#     context_construction_config=context_config
# )

# dataset = EvaluationDataset(goldens=goldens)

# Optional: push/pull to persist goldens across runs via Confident AI.
# Uncomment if you want to store goldens in the cloud.
# dataset.push(alias="Knowledge Base")
# dataset = EvaluationDataset()
# dataset.pull("Knowledge Base")


# ---------------------------------------------------------------------------
# 2. Agent wrapper — separate retrieval from generation so we can inspect both
# ---------------------------------------------------------------------------
class MyAgent:
    def __init__(self):
        self.retriever = vector_embed.kb_compression_retriever
        self.llm = ChatOllama(
            model="gemma4:12b",
            temperature=0.5,
            top_k=20,
            top_p=0.15,
            num_ctx=32768,
        )

    def retrieve(self, query: str):
        """Return original Document objects (with metadata) for deepeval metrics."""
        return self.retriever.invoke(query)

    def generate(self, query: str) -> str:
        """Retrieve context and generate a response. Returns the text output."""
        retrieved_docs = self.retrieve(query)
        prompt = (
            f"Using these standards:\n"
            f"{[doc.page_content for doc in retrieved_docs]}\n\n"
            f"Generate test cases for:\n{query}"
        )
        response = self.llm.invoke(prompt)
        return response.content  # AIMessage → str


agent = MyAgent()


# ---------------------------------------------------------------------------
# 3. Build LLMTestCases from the dataset goldens
# ---------------------------------------------------------------------------
# test_cases = []
# for golden in dataset.goldens:
#     # Retrieve context documents (keep as Document objects for metrics)
#     retrieved_docs = agent.retrieve(golden.input)

#     # Generate the actual output text
#     actual_output = agent.generate(golden.input)

#     test_case = LLMTestCase(
#         input=golden.input,
#         actual_output=actual_output,
#         retrieval_context=[doc.page_content for doc in retrieved_docs],
#         expected_output=golden.expected_output,
#     )
#     test_cases.append(test_case)

# print(f"Built {len(test_cases)} test cases for evaluation.")

test_cases = [
    LLMTestCase(
        input="What does MIL-STD-1586A say about safety configurations?",
        expected_output="MIL-STD-1586A requires specific safety configurations for...", # What you expect it to know
        actual_output=agent.generate("What does MIL-STD-1586A say about safety configurations?"),
        retrieval_context=[doc.page_content for doc in agent.retrieve("What does MIL-STD-1586A say about safety configurations?")]
    )
]

# ---------------------------------------------------------------------------
# 4. Define metrics
# ---------------------------------------------------------------------------
# --- RAG / Retrieval metrics ---
relevancy = ContextualRelevancyMetric(model=judge_model)
recall = ContextualRecallMetric(model=judge_model)
precision = ContextualPrecisionMetric(model=judge_model)

retriever_metrics = [relevancy, recall, precision]

# --- Custom LLM-judge metrics (G-Eval) ---
answer_correctness = GEval(
    name="Answer Correctness",
    criteria=(
        "Evaluate whether the actual output's answer is factually correct and "
        "complete based on the input and retrieved context. If the answer is "
        "not correct or is missing key information, reduce the score."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    model=judge_model,
)

citation_accuracy = GEval(
    name="Citation Accuracy",
    criteria=(
        "Check whether the citations or standard references in the actual "
        "output are correct and relevant given the input and retrieved context. "
        "If a cited clause, section, or standard name does not appear in the "
        "retrieved text, reduce the score."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    model=judge_model,
)

generator_metrics = [answer_correctness, citation_accuracy]


# ---------------------------------------------------------------------------
# 5. Run evaluations
# ---------------------------------------------------------------------------
print("\n=== Evaluating retrieval metrics ===")
retrieval_results = evaluate(
    test_cases=test_cases, 
    metrics=retriever_metrics,
    )

for test_result in retrieval_results:
    print(f"\n--- Breakdown for Input: '{test_result.input}' ---")
    for metric_data in test_result.metrics_data:
        print(f"🔹 Metric: {metric_data.name}")
        print(f"   Score:  {metric_data.score}")
        print(f"   Reason: {metric_data.reason}\n")


print("\n=== Evaluating generator metrics ===")
generator_results = evaluate(
    test_cases=test_cases, 
    metrics=generator_metrics,
)

for test_result in generator_results:
    print(f"\n--- Breakdown for Input: '{test_result.input}' ---")
    for metric_data in test_result.metrics_data:
        print(f"🔹 Metric: {metric_data.name}")
        print(f"   Score:  {metric_data.score}")
        print(f"   Reason: {metric_data.reason}\n")