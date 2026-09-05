from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.middleware import wrap_model_call
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langchain_core.tools import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain_core.messages import trim_messages
import time

import gemini_keys
import rag_debug
import vector_embed
import pipeline_logging

# Ordered fallback chain, tried in order for the life of the process: the
# first two are both confirmed 500 RPD "Flash Lite" tier (per AI Studio's
# Rate Limit dashboard -- most "*Flash Lite*" models are NOT this generous,
# e.g. gemini-2.5-flash-lite is only 20 RPD, so don't assume the naming
# implies the quota). The third is a last-resort full "Flash" tier model
# with only 20 RPD, kept only so the agent degrades instead of dying outright
# once both Lite pools are exhausted for every configured key.
MODEL_CHAIN = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]
MODEL_NAME = MODEL_CHAIN[0]  # kept for the stale is_model_pulled()/Ollama skip checks in tests/e2e
MODEL_PROVIDER = "google_genai"

HISTORY_TOKEN_BUDGET = 8000

_model_index = 0  # process-wide: which MODEL_CHAIN entry is currently in use


def current_model_name() -> str:
    return MODEL_CHAIN[_model_index]


def _advance_model() -> bool:
    """Permanently switch to the next model in MODEL_CHAIN (stays switched
    for the rest of the process -- once a model's quota is exhausted for
    today, falling back to it again a minute later would just fail again).
    Returns False if already on the last model in the chain."""
    global _model_index
    if _model_index >= len(MODEL_CHAIN) - 1:
        return False
    _model_index += 1
    return True


def get_llm():
    """Return a working ChatGoogleGenerativeAI, auto-advancing through
    MODEL_CHAIN if the current model has no working key. That covers both
    quota exhaustion and a model-wide outage at cold start (Gemini's
    "high demand" 503 fails every key's probe identically -- no key can
    fix that, only trying the next model tier can)."""
    while True:
        model_name = current_model_name()

        def probe(api_key: str) -> None:
            # max_retries=0: langchain_google_genai defaults to 6 internal
            # retries with growing backoff on 5xx -- fine for a real chat
            # turn, but it turns a liveness probe (meant to answer "does
            # this key/model work right now") into a 30-60s stall per key
            # during a model-wide outage, with 27 keys to get through
            # before the outage/model-unavailable fallback even kicks in.
            # One fast attempt is all a probe needs.
            ChatGoogleGenerativeAI(
                model=model_name, google_api_key=api_key, timeout=15.0, max_retries=0
            ).invoke("ping")

        # Keyed by model, not just "chat": a key exhausted/invalid for one
        # model in the chain may still work for another (Google tracks quota
        # per project+model, not per key), so switching models must start
        # each key's rotation fresh rather than inheriting the previous
        # model's exhausted/bad-key list.
        try:
            api_key = gemini_keys.working_key(probe, purpose=f"chat:{model_name}")
        except EnvironmentError:
            if not _advance_model():
                raise
            continue
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.1,
            top_k=20,
            top_p=0.15,
            timeout=45.0,
        )


@wrap_model_call
def trim_history_middleware(request, handler):
    try:
        trimmed = trim_messages(
            request.messages,
            strategy="last",
            token_counter=request.model,
            max_tokens=HISTORY_TOKEN_BUDGET,
            start_on="human",
            include_system=True,
        )
    except Exception:
        # token_counter calls Gemini's count_tokens endpoint, which has
        # failed with "API key not valid" for some keys even though the
        # same key works fine for generateContent (the probe call that
        # cached it as good) -- trimming is a nice-to-have bound on
        # history size, not something that should be able to take down
        # the whole turn. Skip trimming rather than propagate.
        trimmed = request.messages
    if not trimmed:
        # start_on="human" requires the kept window to *start* with a human
        # message; if the most recent human/AI/tool run alone exceeds
        # max_tokens (e.g. one large retrieved-context ToolMessage),
        # trim_messages can't satisfy both constraints and returns [] --
        # which would otherwise reach Gemini as a contents-less request and
        # crash with "contents are required." Falling back to the
        # untrimmed messages here is safe: request.messages is already
        # bounded to one turn's worth of tool-loop history, so exceeding
        # the budget occasionally is expected, not a runaway history.
        trimmed = request.messages
    return handler(request.override(messages=trimmed))


MAX_KEY_ATTEMPTS_PER_MODEL = 4


@wrap_model_call
def key_rotation_middleware(request, handler):
    """Retry a model call on failure, rotating through the configured
    Gemini keys for the current model and falling back through
    MODEL_CHAIN when needed. Two failure classes are handled differently:
    - a model-wide outage (gemini_keys.is_model_unavailable_error, e.g.
      Gemini's "high demand" 503/504) skips straight to the next model --
      no key can fix an overloaded model, so rotating keys would just
      waste attempts hitting the same wall.
    - a per-key error (gemini_keys.is_key_error: quota exhaustion or an
      invalid/revoked key) rotates keys for the current model first, and
      only advances model after MAX_KEY_ATTEMPTS_PER_MODEL failures.
    get_llm() itself already walks MODEL_CHAIN if a model has no working
    key at all (cold-start outage included), so both branches below just
    need to call it again after positioning _model_index correctly -- no
    need to loop here too. A key passing the startup probe doesn't
    guarantee it stays good for the rest of the process -- keys can hit
    per-minute limits or go bad mid-session -- and outside of
    bilingual_eval.py's own retry loop nothing else in the main agent path
    rotated away from that, let alone switched models."""
    attempts_this_model = 0
    while True:
        try:
            return handler(request)
        except Exception as e:  # noqa: BLE001 - rotate/fall back on recognized errors, otherwise propagate
            if gemini_keys.is_model_unavailable_error(e):
                if not _advance_model():
                    raise
                request = request.override(model=get_llm())
                attempts_this_model = 0
                continue
            if not gemini_keys.is_key_error(e):
                raise
            gemini_keys.invalidate(f"chat:{current_model_name()}")
            attempts_this_model += 1
            if attempts_this_model >= MAX_KEY_ATTEMPTS_PER_MODEL:
                if not _advance_model():
                    raise  # every model in MODEL_CHAIN is out of working keys
                attempts_this_model = 0
            request = request.override(model=get_llm())


@dataclass
class Context:
    user_id: str


def get_system_prompt(has_user_document: bool = False) -> str:
    scope_tools = (
        "`search_testing_standards` or `search_user_document`"
        if has_user_document
        else "`search_testing_standards`"
    )
    """Single source of truth for the system prompt (also used for debug
    prompt dumps and by bridge.py)."""
    system_prompt = (
        f'''
        You are a Senior System Test Engineer and Systems Validation Expert.
        Your task is to analyze project and software requirement documents and generate comprehensive test cases.
        You must strictly adhere to the verification and validation processes outlined in ISO/IEC/IEEE 29119.

        Do not provide conversational filler. Base your analysis solely on the provided context, and preserve the document's structure and terminology (e.g., distinguishing between functional/non-functional requirements, existing/proposed features).

        --- AGENT INSTRUCTIONS & WORKFLOW ---
        When analyzing requirements and generating test cases, strictly follow this workflow:

        1. AMBIGUITY CHECK
        EVERY requirement identified in the document MUST first be evaluated for objective testability. 
        Treat a requirement as ambiguous if it fails to specify: threshold/limits, duration/timing, error messaging, state persistence (session vs. account-level), or recovery/unlock procedure. Other examples of ambiguity include: undefined terminology, unclear expected behavior, missing input/output conditions, or unclear scope.
        - Flag each missing dimension separately.

        2. HANDLING AMBIGUOUS REQUIREMENTS
        If a requirement is ambiguous or not objectively testable:
        - Mark it explicitly as: "NOT TESTABLE AS WRITTEN".
        - Do NOT invent a measurable interpretation, invent acceptance criteria, or generate a test case for it.
        - State the specific reason for the ambiguity (e.g., what information is missing).
        - Provide concrete suggestions on how to rewrite the requirement to make it testable.

        3. TEST CASE GENERATION
        For EVERY requirement that is sufficiently specified and objectively testable, you MUST generate AT LEAST ONE test case. NEVER skip a testable requirement. For every testable requirement, output a structured test plan that includes:
        - **Test ID & Traceability**: Traceability must reference actual requirement IDs from the provided context (include section number, requirement ID, heading, and page number using retrieved metadata). Note that this relationship is many-to-many: one requirement may be tested by multiple test cases, and one test case can verify multiple requirements simultaneously. List all applicable requirement IDs. If no ID is present in the input, output 'No requirement ID provided' rather than inventing one.
        - **Test Type**: (e.g., Boundary, Edge Case, Integration)
        - **Preconditions**: Any preliminary setup, configuration, or sequences required before the actual execution of the test must go here.
        - **Test Steps**: Every test step MUST have a corresponding Expected Result. If a test case yields only one Expected Result, it must contain exactly ONE Test Step. Multi-step setups must be moved to Preconditions or consolidated logically into a single execution step.
        - **Expected Result**: The expected result must be directly supported by the literal wording of the requirement. Do not invent acceptance criteria.

        4. INTERNAL VERIFICATION, REVIEW, & COVERAGE CHECK
        Before finalizing the test plan, review all generated test cases as a set:
        - Pay strict attention to boundary conditions.
        - Ensure that no two test cases with the same or overlapping preconditions produce contradictory expected results. If a contradiction exists, resolve it based on the literal wording of the requirement.
        - Internally verify: "Can I point to a specific piece of the retrieved document that supports every factual claim?" If not, state that the document does not specify it.

        At the end of your response, perform a mandatory coverage check:
        - Total requirements identified: X
        - Testable requirements: Y
        - Ambiguous requirements: Z
        - Requirements with test cases: Y
        *(Note: The number of testable requirements with test cases MUST exactly equal the number of testable requirements identified.)*

        --- QUERY REFORMULATION (apply before every retriever tool call) ---
        - The user's latest message is often an elliptical follow-up (e.g. 'what about the timing requirement', 'and for the other standard?') that only makes sense combined with earlier turns. Before calling a search tool, silently rewrite it into a complete, standalone query that folds in the relevant entities/standard/topic from the conversation so far -- pass that rewritten query as the tool argument, not the raw follow-up text.
        - If a question spans more than one standard or compares two requirements, issue a separate, focused search call per standard/topic rather than one combined query.

        --- SCOPE (check this before calling any tool) ---
        - You are a test/QA standards assistant only. In scope: requirement analysis, test case generation, and questions about the ingested standards, testing methodology, or the user's uploaded documents.
        - If the user's message is clearly unrelated to this scope (small talk, general knowledge, creative writing, coding help unrelated to testing, or any other off-topic request), do NOT call {scope_tools} -- there is nothing in the knowledge base that could answer it. Politely decline in one or two sentences, state what you're for, and stop there.
        - If a message is ambiguous (could plausibly relate to testing/requirements), treat it as in scope and proceed normally rather than declining.

        --- GROUNDING RULES (apply to every response, not just test plan generation) ---
        - Answer ONLY using information explicitly supported by the provided documents.
        - For ANY factual claim about a standard (definitions, process names, clause structure, requirements), you MUST call `search_testing_standards` first and base your answer only on the retrieved text.
        - Never answer questions about standard content from memory, even if you believe you know the answer. Your training knowledge of these standards may be wrong or may blend one standard with another.
        - If the retrieved chunks don't contain the answer, say so explicitly (e.g. 'The retrieved sections don't cover this') rather than filling the gap with general knowledge.
        - Never cite a clause/section number that doesn't appear verbatim in the retrieved text.
        - Respond in the same language the user's question was asked in. If the question mixes languages, match the dominant one.

        --- SELF-CORRECTING RETRIEVAL (grade before you answer) ---
        - Every retrieved chunk carries a `[relevance score: N]` line (0-1, higher = more relevant). Before answering, judge whether the retrieved chunks actually address the question -- not just whether their scores are high, but whether their content covers the specific topic asked about.
        - If the chunks are weakly relevant, off-topic, or only tangentially related (low scores, or high scores but wrong subject matter), do NOT answer from them. First try ONE more `search_testing_standards` call with a reformulated or narrower query (different keywords, a more specific clause/topic name) before giving up.
        - Only after that retry still fails to surface relevant content should you fall back to stating the standards don't cover this, per the grounding rule above. Never stretch a weak or off-topic match into an answer just because it's the best one you found.

        --- ANSWER FORMAT FOR GENERAL QUESTIONS (not test-case generation) ---
        When answering a general question about a standard's content (not generating a
        test plan), structure your response in three parts:
        - **Answer**: A concise, direct answer to the question.
        - **Reasoning**: A brief explanation of how you arrived at that answer from the
          retrieved text.
        - **Citation**: The specific quote or paraphrase from the retrieved context that
          supports the answer, with its (standard, Section X) or (standard, p. N) reference.
        This structure does not apply to test-case generation, which already follows the
        Test ID/Type/Preconditions/Steps/Expected Result structure above.

        '''

    )
    if has_user_document:
        nl = chr(10)
        system_prompt += (
            f"{nl}The user has uploaded one or more files for this conversation. "
            "Use the search_user_document tool to retrieve the relevant sections "
            "of the user's uploaded files before answering. The uploaded files are "
            "specific to this conversation only."
        )
    return system_prompt


def build_agent(tools=None, has_user_document: bool = False, use_query_expansion: bool = False):
    """Build the QA agent around the shared system prompt.

    use_query_expansion=True swaps the KB retriever tool for one backed by
    MultiQueryRetriever (extra LLM call per retrieval to generate query
    variants -- higher latency/token cost, off by default). See
    vector_embed.build_expanded_retriever for the circular-import reason
    this is wired here at build time rather than at vector_embed's import
    time.
    """
    active_tools = tools if tools is not None else vector_embed.tools
    if use_query_expansion:
        expanded = vector_embed.build_expanded_retriever(
            vector_embed.vector_store, k=20, search_type="mmr", llm=get_llm()
        )
        expanded_tool = create_retriever_tool(
            expanded,
            name="search_testing_standards",
            description=vector_embed.retriever_tool.description,
            document_prompt=vector_embed.retriever_document_prompt,
        )
        active_tools = [
            expanded_tool if t.name == "search_testing_standards" else t
            for t in active_tools
        ]
    limited_tools = [
        pipeline_logging.limit_tool_calls(t) if t.name in (
            "search_testing_standards", "search_user_document"
        ) else t
        for t in active_tools
    ]
    return create_agent(
        model=get_llm(),
        tools=limited_tools,
        system_prompt=get_system_prompt(has_user_document),
        checkpointer=InMemorySaver(),
        middleware=[key_rotation_middleware, trim_history_middleware],
    )


if __name__ == "__main__":  #runs only if you execute this file
    agent = build_agent() #initialize agent

    config = {"configurable": {"thread_id": str(uuid7())}}
    user_context = Context(user_id="user-123")

    print("\n--- Senior QA Agent Ready ---")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("You: ")  #waits for the user message
        if user_text.lower() in ["exit", "quit"]: #termination
            print("Shutting down agent...")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_text}]}, #user's inout
            config=config, #thread_id..
            context=user_context, #user context
        )

        print(f"\nAgent:\n{result['messages'][-1].content}\n") #extracts the only last 
        print("-" * 50)

# if __name__ == "__main__":  #runs only if you execute this file
#     agent = build_agent() #initialize agent

#     config = {"configurable": {"thread_id": str(uuid7())}}
#     user_context = Context(user_id="user-123")

#     print()
#     print("--- Senior QA Agent Ready ---")
#     print("Type 'exit' to quit.")

#     while True:
#         user_text = input("You: ")  #waits for the user message
#         if user_text.lower() in ["exit", "quit"]: #termination
#             print("Shutting down agent...")
#             break

#         rag_debug.log_query(config["configurable"]["thread_id"], user_text)
#         start = time.perf_counter()
#         result = agent.invoke(
#             {"messages": [{"role": "user", "content": user_text}]}, #user's inout
#             config=config, #thread_id..
#             context=user_context, #user context
#         )
#         latency_s = time.perf_counter() - start

#         answer_message = result["messages"][-1]
#         rag_debug.log_generation(
#             MODEL_NAME,
#             latency_s,
#             str(answer_message.content),
#             usage=rag_debug.extract_usage(result["messages"]),
#         )

#         print()
#         print(f"Agent:{result['messages'][-1].content}")
#         print("-" * 50)
