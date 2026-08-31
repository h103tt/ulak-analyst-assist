from langchain_ollama import ChatOllama
from langchain.agents.middleware import wrap_model_call
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain_core.messages import trim_messages
import time

import rag_debug
import vector_embed
import pipeline_logging

MODEL_NAME = "qwen3.5:4b"
MODEL_PROVIDER = "ollama"

HISTORY_TOKEN_BUDGET = 8000

def get_llm():
    model = ChatOllama(
        model=MODEL_NAME,
        temperature=0.1,
        top_k=20,
        top_p=0.15,
        repeat_penalty=1.3,
        num_ctx=8192,
        request_timeout=45.0,
    )
    return model


@wrap_model_call
def trim_history_middleware(request, handler):
    trimmed = trim_messages(
        request.messages,
        strategy="last",
        token_counter=request.model,
        max_tokens=HISTORY_TOKEN_BUDGET,
        start_on="human",
    )
    return handler(request.override(messages=trimmed))

@dataclass
class Context:
    user_id: str


def get_system_prompt(has_user_document: bool = False) -> str:
    """Single source of truth for the system prompt (also used for debug
    prompt dumps and by bridge.py)."""
    system_prompt = (
        '''
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

        --- GROUNDING RULES (apply to every response, not just test plan generation) ---
        - Answer ONLY using information explicitly supported by the provided documents.
        - For ANY factual claim about a standard (definitions, process names, clause structure, requirements), you MUST call `search_testing_standards` first and base your answer only on the retrieved text.
        - Never answer questions about standard content from memory, even if you believe you know the answer. Your training knowledge of these standards may be wrong or may blend one standard with another.
        - If the retrieved chunks don't contain the answer, say so explicitly (e.g. 'The retrieved sections don't cover this') rather than filling the gap with general knowledge.
        - Never cite a clause/section number that doesn't appear verbatim in the retrieved text.
        
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


def build_agent(tools=None, has_user_document: bool = False):
    """Build the QA agent around the shared system prompt."""
    active_tools = tools if tools is not None else vector_embed.tools
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
        middleware=[trim_history_middleware],
    )


# def build_agent(tools=None, has_user_document: bool = False):
#     model = ChatOllama(
#         model=MODEL_NAME,
#         temperature=0.1,
#         top_k=20,
#         top_p=0.15,
#         num_ctx=16384,
#         request_timeout=45.0,
#     )
#     rag_debug.section("GENERATION", "Agent build", rag_debug.C.GENERATION)
#     rag_debug.field("model", MODEL_NAME)
#     rag_debug.field("temperature", 0.5)
#     rag_debug.field("num_ctx", 16334)

#     return create_agent(
#         model=model,
#         tools=tools if tools is not None else vector_embed.tools,
#         system_prompt=get_system_prompt(has_user_document),
#         checkpointer=InMemorySaver(),
#     )



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
