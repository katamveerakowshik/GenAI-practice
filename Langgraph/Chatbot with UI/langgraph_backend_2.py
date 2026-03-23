from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

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

conn = sqlite3.connect(database = "chat_history.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
workflow = graph.compile(checkpointer=checkpointer)


def generate_prev_threads(checkpointer):
    all_threads = set()
    if checkpointer.list(None) is None:
        return []
    for checkpointer in checkpointer.list(None):
        all_threads.add(checkpointer.config["configurable"]["thread_id"])
    return list(all_threads)
