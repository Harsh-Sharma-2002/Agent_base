from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
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

tools = [add]

