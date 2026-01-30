import os
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # Reducer Funtiom
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()
HF_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY not found")

base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
)
llm = ChatHuggingFace(llm=base_llm)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int,b:int) -> int:
    """
    This is a tool that adds two numbers
    """
    return a + b

# Tool Binding section
tools = [add]
llm.bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    sys_prompt = SystemMessage(content=  "You are an AI assistant.\n"
        "You MUST use the provided tools to perform arithmetic.\n"
        "Do NOT calculate numbers mentally.\n"
        "If arithmetic is required, call the appropriate tool.")
    response = llm.invoke([sys_prompt] + state["messages"])
    return {"messages": [response]} # reducer functions helps append the repsonse to the state

def should_cont(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else: 
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("agent",model_call)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START,"agent")
graph.add_conditional_edges(source="agent",
                            path=should_cont,
                            path_map={
                                "end" : END,
                                "continue": "tools"
                                }
                            )
graph.add_edge("tools","agent")
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages" : [("user","Add 40 + 12 using the tool i have given you and then multiply the result by 6. Also tell me a joke please")]}
print_stream(app.stream(inputs,stream_mode="values"))
