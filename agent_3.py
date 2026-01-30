import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # Reducer Funtiom
from langgraph.graph import START,END,StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# Using in place of injective state
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 

@tool 
def update(content: str) -> str:
    """
    Update the document content  with the provided content
    """
    global document_content
    document_content = content
    return f"Document has been updated successfully! the current content is :\n {document_content}"

@tool
def save(filename:str) -> str:
    """
    Save the current  document to a text file and finish the process
    
    Args:
        filename: Name of the text file with .txt extension
    """

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
