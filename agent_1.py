import os
from dotenv import load_dotenv
from typing import TypedDict

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END


load_dotenv()

HF_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY not found")


class AgentState(TypedDict):
    messages: list[HumanMessage | AIMessage]


base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
)

llm = ChatHuggingFace(llm=base_llm)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print("CURRENT STATE",state["messages"])
    return {
        "messages": state["messages"] + [response]
    }

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history: list[HumanMessage | AIMessage] = []

while True:
    user_input = input("Enter: ")
    if user_input.lower() == "exit":
        break

    conversation_history.append(HumanMessage(content=user_input))

    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]

    print("\nAssistant:", conversation_history[-1].content)
    print("-" * 50)


with open("logging.txt","w") as file:
    file.write("Conversatin history\n\n")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message,AIMessage):
            file.write(f"AI: {message.content}\n\n")

    file.write("End of Conversation")

print("Conversation saved to the logging file")