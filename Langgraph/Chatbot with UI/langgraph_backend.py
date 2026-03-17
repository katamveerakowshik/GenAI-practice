from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

llm = ChatOllama(model = "llama3.2")


# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define chat functions
def llm_reply(state: ChatState):
    prompt = state["messages"]
    response = llm.invoke(prompt)
    return {"messages": [response]}

# graph

graph = StateGraph(ChatState)

graph.add_node("llm_reply", llm_reply)

graph.add_edge(START, "llm_reply")
graph.add_edge("llm_reply", END)

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)


