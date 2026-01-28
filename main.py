import os
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict,List,Union
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: list[HumanMessage]
    
