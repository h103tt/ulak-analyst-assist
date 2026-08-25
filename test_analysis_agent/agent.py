from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import trim_messages
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from model import get_model
import vector_embed

# num_ctx=32768 total. Reserve room for the system prompt (~600 tokens),
# retrieved tool context (5 chunks x ~512 tokens ~= 2500), and response
# headroom, leaving this much for conversation history sent to the model.
# The full history still lives in InMemorySaver -- this only bounds what
# gets sent to the model on each call, so /trace still sees everything.
HISTORY_TOKEN_BUDGET = 24000


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


def build_agent(tools=None, has_user_document: bool = False):
    model = get_model()

    system_prompt = (
        "You are a Senior System Test Engineer and Systems Validation Expert. "
        "Your task is to analyze project and software requirement documents and generate comprehensive test cases. "
        "You must strictly adhere to the verification and validation processes outlined in ISO/IEC/IEEE 29119. "
        "For every requirement provided in the context, output a structured test plan that includes:\n"
        "- Test ID & Traceability\n"
        "Traceability must reference an actual requirement ID from the provided context. "
        "If no ID is present in the input, output 'No requirement ID provided' rather than inventing one."
        "- Test Type (e.g., Boundary, Edge Case, Integration)\n"
        "- Preconditions\n"
        "- Test Steps\n"
        "- Expected Result\n"
        "\n\nQUERY REFORMULATION (apply before every retriever tool call):\n"
        "- The user's latest message is often an elliptical follow-up (e.g. 'what "
        "about the timing requirement', 'and for the other standard?') that only "
        "makes sense combined with earlier turns. Before calling a search tool, "
        "silently rewrite it into a complete, standalone query that folds in the "
        "relevant entities/standard/topic from the conversation so far -- pass "
        "that rewritten query as the tool argument, not the raw follow-up text.\n"
        "- If a question spans more than one standard or compares two requirements, "
        "issue a separate, focused search call per standard/topic rather than one "
        "combined query.\n"
        "\n\nGROUNDING RULES (apply to every response, not just test plan generation):\n"
        "- For ANY factual claim about a standard (definitions, process names, clause "
        "structure, requirements), you MUST call search_testing_standards first and "
        "base your answer only on the retrieved text.\n"
        "- Never answer questions about standard content from memory, even if you "
        "believe you know the answer. Your training knowledge of these standards may "
        "be wrong or may blend one standard with another.\n"
        "- If the retrieved chunks don't contain the answer, say so explicitly "
        "(e.g. 'The retrieved sections don't cover this') rather than filling the gap "
        "with general knowledge.\n"
        "- Never cite a clause/section number that doesn't appear verbatim in the "
        "retrieved text.\n"
        "Do not provide conversational filler. Base your analysis solely on the provided context. "
        "Pay strict attention to boundary conditions and ensure expected results do not contradict each other. "
        "Treat a requirement as ambiguous if it fails to specify: threshold/limits, duration/timing, error messaging, state persistence (session vs. account-level), or recovery/unlock procedure. "
        "Flag each missing dimension separately. "
        "If a requirement is ambiguous or untestable, flag it and state why. "
        "Before finalizing the test plan, review all generated test cases as a set. "
        "Ensure that no two test cases with the same or overlapping preconditions produce contradictory expected results. "
        "If a contradiction exists, resolve it based on the literal wording of the requirement."
    )
    if has_user_document:
        system_prompt += (
            "\nThe user has uploaded one or more files for this conversation. "
            "Use the search_user_document tool to retrieve the relevant sections "
            "of the user's uploaded files before answering. The uploaded files are "
            "specific to this conversation only."
        )

    return create_agent(
        model=model,
        tools=tools if tools is not None else vector_embed.tools,
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
        middleware=[trim_history_middleware],
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
