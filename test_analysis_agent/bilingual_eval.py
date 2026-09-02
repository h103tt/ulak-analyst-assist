"""End-to-end bilingual QA benchmark for the ULAK agent.

Asks one question per embedded standard, in English and in Turkish, through
the real agent (agent.build_agent -> invoke), then scores each answer with
an LLM judge on Correctness/Completeness and Citation Accuracy against the
retrieved context -- mirroring test_rag_metrics.py's GEval criteria, but as
a single combined judge call per answer to keep this fast and cheap.

Writes results to bilingual_eval_results.json for the dashboard to render.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict

from langchain_core.utils.uuid import uuid7

import agent
import gemini_keys

QUESTIONS = [
    {
        "standard": "MIL-STD-461",
        "en": "What does MIL-STD-461 require for emission and susceptibility testing of electronic equipment?",
        "tr": "MIL-STD-461, elektronik ekipmanların emisyon ve duyarlılık testleri için neler gerektirir?",
    },
    {
        "standard": "MIL-STD-1586A",
        "en": "What safety configurations does MIL-STD-1586A require?",
        "tr": "MIL-STD-1586A hangi güvenlik konfigürasyonlarını gerektirir?",
    },
    {
        "standard": "15288-2023-2",
        "en": "What are the technical processes defined in ISO/IEC/IEEE 15288?",
        "tr": "ISO/IEC/IEEE 15288'de tanımlanan teknik süreçler nelerdir?",
    },
    {
        "standard": "29119-1-2022",
        "en": "According to ISO 29119-1, what are the fundamental test process elements?",
        "tr": "ISO 29119-1'e göre temel test süreç unsurları nelerdir?",
    },
    {
        "standard": "MIL-STD-882E",
        "en": "What does MIL-STD-882E say about safety verification under Task 401?",
        "tr": "MIL-STD-882E, Task 401 kapsamında güvenlik doğrulaması hakkında ne söylüyor?",
    },
    {
        "standard": "SP800-53_REV-3",
        "en": "What security control families does NIST SP 800-53 Rev 3 define?",
        "tr": "NIST SP 800-53 Rev 3 hangi güvenlik kontrol ailelerini tanımlıyor?",
    },
]

CITATION_RE = re.compile(r"\(([A-Za-z0-9/_.\- ]+),\s*(Section[^)]+|p\.\s*[\d\-]+)\)")

JUDGE_PROMPT = """You are grading one answer from a systems-engineering QA assistant.

QUESTION:
{question}

RETRIEVED CONTEXT (what the assistant was allowed to use):
{context}

ASSISTANT'S ANSWER:
{answer}

Score the answer on two axes, 0-100 each:
- correctness: is the answer factually correct and complete relative to the retrieved context? Penalize invented facts or missing key information.
- citation_accuracy: are the citations (standard name, section/page) accurate and do they appear in the retrieved context? Penalize invented or mismatched citations.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{{"correctness": <int>, "citation_accuracy": <int>, "reason": "<one sentence>"}}
"""


@dataclass
class TestResult:
    standard: str
    language: str
    question: str
    answer: str
    latency_s: float
    retrieved_chunks: int
    citations_found: list = field(default_factory=list)
    correctness: int | None = None
    citation_accuracy: int | None = None
    judge_reason: str = ""
    error: str | None = None


def _aggregate_tool_context(messages) -> tuple[str, int]:
    blocks, seen = [], set()
    for m in messages:
        if type(m).__name__ != "ToolMessage":
            continue
        content = m.content if isinstance(m.content, str) else str(m.content)
        content = content.strip()
        if not content or content in seen:
            continue
        seen.add(content)
        blocks.append(content)
    chunk_count = sum(c.count("<chunk id=") for c in blocks)
    return "\n\n---\n\n".join(blocks), chunk_count


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            (item.get("text", "") if isinstance(item, dict) else str(item))
            for item in content
        )
    return str(content)


MAX_KEY_ROTATIONS = 6


def _is_quota_error(e: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)


def run_one(standard: str, language: str, question: str) -> TestResult:
    """Runs the agent + judge for one question, rotating to the next Gemini
    key (via gemini_keys.invalidate) whenever either call hits a 429 --
    build_agent()/get_llm() picks up the newly-rotated key automatically."""
    result = TestResult(standard=standard, language=language, question=question, answer="", latency_s=0, retrieved_chunks=0)

    for attempt in range(MAX_KEY_ROTATIONS):
        try:
            a = agent.build_agent()
            config = {"configurable": {"thread_id": str(uuid7())}}
            ctx = agent.Context(user_id="bilingual-eval")

            start = time.perf_counter()
            out = a.invoke({"messages": [{"role": "user", "content": question}]}, config=config, context=ctx)
            result.latency_s = round(time.perf_counter() - start, 2)

            answer = _text_of(out["messages"][-1].content)
            result.answer = answer
            result.citations_found = [f"{m.group(1).strip()}, {m.group(2).strip()}" for m in CITATION_RE.finditer(answer)]

            context, chunk_count = _aggregate_tool_context(out["messages"])
            result.retrieved_chunks = chunk_count

            judge_prompt = JUDGE_PROMPT.format(question=question, context=context[:12000], answer=answer)
            judge_llm = agent.get_llm()
            judge_raw = _text_of(judge_llm.invoke(judge_prompt).content).strip()
            judge_raw = re.sub(r"^```(?:json)?|```$", "", judge_raw.strip(), flags=re.MULTILINE).strip()
            judge = json.loads(judge_raw)
            result.correctness = int(judge.get("correctness", 0))
            result.citation_accuracy = int(judge.get("citation_accuracy", 0))
            result.judge_reason = judge.get("reason", "")
            result.error = None
            return result
        except Exception as e:  # noqa: BLE001 - rotate on quota errors, otherwise record and stop
            if _is_quota_error(e) and attempt < MAX_KEY_ROTATIONS - 1:
                print(f"  quota hit, rotating key (attempt {attempt + 1}/{MAX_KEY_ROTATIONS})", flush=True)
                gemini_keys.invalidate("chat")
                continue
            result.error = str(e)[:300]
            return result
    return result


def main():
    existing = {}
    try:
        with open("bilingual_eval_results.json", encoding="utf-8") as fh:
            for r in json.load(fh):
                if not r.get("error"):
                    existing[(r["standard"], r["language"])] = r
    except FileNotFoundError:
        pass

    results: list[TestResult] = []

    for q in QUESTIONS:
        for lang in ("en", "tr"):
            key = (q["standard"], lang)
            if key in existing:
                print(f"=== {q['standard']} [{lang}] === already done, skipping", flush=True)
                results.append(TestResult(**existing[key]))
                continue
            print(f"=== {q['standard']} [{lang}] ===", flush=True)
            r = run_one(q["standard"], lang, q[lang])
            results.append(r)
            if r.error:
                print(f"  ERROR: {r.error}", flush=True)
            else:
                print(f"  latency={r.latency_s}s chunks={r.retrieved_chunks} correctness={r.correctness} citation_accuracy={r.citation_accuracy}", flush=True)
            with open("bilingual_eval_results.json", "w", encoding="utf-8") as fh:
                json.dump([asdict(x) for x in results], fh, ensure_ascii=False, indent=2)

    print("\nDone. Results in bilingual_eval_results.json", flush=True)


if __name__ == "__main__":
    main()
