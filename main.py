import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict,List,Union
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: list[HumanMessage|AIMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state:AgentState) -> AgentState:
    """
    The process node which will call the API
    """
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))

    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")

while(user_input != "exit"):
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messaged":conversation_history})

    print(result["messages"])
    user_input = input("Enter ")

