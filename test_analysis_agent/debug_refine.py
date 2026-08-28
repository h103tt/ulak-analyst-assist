import time
import agent
import refine
from bridge import extract_answer, extract_question

# 1) Gerçek dosyanla, gerçek soruyla agent'ı NORMAL şekilde çalıştır
#    (hop1 + tool call + hop2) -- bu kısım refine'dan bağımsız.
base_agent = agent.build_agent()  # veya bridge'deki gerçek agent kurulum fonksiyonunu kullan
config = {"configurable": {"thread_id": "debug-isolate-1"}}
user_ctx = agent.Context(user_id="debug-isolate-1")

question = "write test cases for the requirements"
user_messages = [{"role": "user", "content": question}]

t0 = time.perf_counter()
result = base_agent.invoke({"messages": user_messages}, config=config, context=user_ctx)
t1 = time.perf_counter()
draft_answer = extract_answer(result)
print(f"[HOP1+HOP2] {t1 - t0:.1f}s, draft_chars={len(draft_answer)}")
print("draft preview:", draft_answer[:300])

# 2) Aynı sonuçtan context'i çıkar, refine_answer'ı TEK BAŞINA çalıştır
aggregated_context = refine.aggregate_tool_context(result.get("messages",[]))
print(f"context_chars={len(aggregated_context)}")

t2 = time.perf_counter()
refined = refine.refine_answer(question, aggregated_context, draft_answer)
t3 = time.perf_counter()
print(f"[REFINE] {t3 - t2:.1f}s, refined_chars={len(refined)}")
print("refined preview:", repr(refined[:300]))