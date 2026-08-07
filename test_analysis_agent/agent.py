from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
import vector_embed
from langchain.agents import create_agent


@dataclass
class Context:
    user_id: str


model = ChatOllama(
    model="qwen3:8b",
    temperature=0,
    top_k=20,
    top_p=0.15,
)

system_prompt = (
    "You are a Senior QA Engineer and Systems Validation Expert. "
    "Your task is to analyze project and software requirement documents and generate comprehensive test cases. "
    "You must strictly adhere to the verification and validation processes outlined in ISO/IEC/IEEE 29119."
    "For every requirement provided in the context, output a structured test plan that includes:\n"
    "- Test ID & Traceability\n"
    "Traceability must reference an actual requirement ID from the provided context. "
    "If no ID is present in the input, output 'No requirement ID provided' rather than inventing one."
    "- Test Type (e.g., Boundary, Edge Case, Integration)\n"
    "- Preconditions\n"
    "- Test Steps\n"
    "- Expected Result\n"
    "Do not provide conversational filler. Base your analysis solely on the provided context. "
    "Pay strict attention to boundary conditions and ensure expected results do not contradict each other."
    "Treat a requirement as ambiguous if it fails to specify: threshold/limits, duration/timing, error messaging, state persistence (session vs. account-level), or recovery/unlock procedure. "
    "Flag each missing dimension separately."
    "If a requirement is ambiguous or untestable, flag it and state why."
    "Before finalizing the test plan, review all generated test cases as a set. "
    "Ensure that no two test cases with the same or overlapping preconditions produce contradictory expected results. "
    "If a contradiction exists, resolve it based on the literal wording of the requirement."
    )

agent = create_agent(
    model=model,
    tools=vector_embed.tools,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

# config = {"configurable": {"thread_id": str(uuid7())}}
# user_context = Context(user_id="user-123")
# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "Requirement: The login page must lock the user out after 3 failed password attempts."}]},
#     config=config,
#     context=user_context
# )
# print(result["messages"][-1].content)

# result2 = agent.invoke(
#     {"messages": [{"role": "user", "content": "Write one more test case for what happens on the 4th attempt."}]},
#     config=config,
#     context=user_context
# )
# print(result2["messages"][-1].content)

config = {"configurable": {"thread_id": str(uuid7())}}
user_context = Context(user_id="user-123")

print("\n--- Senior QA Agent Ready ---")
print("Type 'exit' to quit.\n")

while True:
    user_text = input("You: ")
    if user_text.lower() in ['exit', 'quit']:
        print("Shutting down agent...")
        break
        
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=config,
        context=user_context
    )
    
    print(f"\nAgent:\n{result['messages'][-1].content}\n")
    print("-" * 50)