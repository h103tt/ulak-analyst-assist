from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
import vector_embed


@dataclass
class Context:
    user_id: str


def build_agent(tools=None, has_user_document: bool = False):
    model = ChatOllama(
        model="qwen3.5:9b",
        temperature=0.5,
        top_k=20,
        top_p=0.15,
    )

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
