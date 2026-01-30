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
HF_TOKEN = os.environ.get("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY not found")

base_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
)



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

    try:
        with open(filename,'w') as file:
            file.write(document_content)
            return f"Document has been saved successfully to {filename} . "

    except Exception as e:
        return f"Error saving document : {str(e)}"
    
tools = [update,save]
model = ChatHuggingFace(llm=base_llm).bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a drafter, a helpful writing assistant. You are going to help the user update and modify documents.
            -  if the user wants to update or modify the content, use the "update" tool with complete updated content.
            - if the user wants to save and finish, you need to use the "save tool."
            - Make sure to always show the current document state after modifications.
                                   
            the current document content is: {document_content}
    """)
     
    if not state["messages"]:
        user_input = "I'm ready to help you update the document. What would you like to create"
        user_message = HumanMessage(content=user_input)
    
    else: 
        user_input = input("\n What would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list[state["messages"] + [user_message]]

    response = model.invoke(all_messages)
    print(f'AI: {response.content}')
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages" : list(state["messages"] + [user_message,response])}

def should_cont(state: AgentState) -> str:
    m = state["messages"]
    for msg in m:
        if (isinstance(msg,ToolMessage) and "saved" in msg.content.lower() and "document" in msg.content.lower()):
            return "end"
        return "continue"
    
def print_m(state: AgentState):
    """
    made to make the messages more readable 
    """

    for m in state["messages"][-3:]:
        if isinstance(m,ToolMessage):
            print(f"\n TOOL RESULT: {m.content}")


graph = StateGraph(AgentState)

graph.add_node("agent",agent)
graph.add_node("tools",ToolNode(tools))

graph.add_edge(START,"agent")
graph.add_edge("agent","tools")
graph.add_conditional_edges(source="agent",
                            path=should_cont,
                            path_map={
                                "end" : END,
                                "continue" : "agent"
                            })
app = graph.compile()

app.invoke()