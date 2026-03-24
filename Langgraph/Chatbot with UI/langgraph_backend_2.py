from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_ollama import ChatOllama
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import sqlite3
from dotenv import load_dotenv
import os

load_dotenv()

# Adding tools
#1. Searching tool
search_tool = DuckDuckGoSearchRun(region="us-en")

#2. Calculator tools
@tool
def calculator(first_num: float, second_num: float, operation: str):
    """ 
    Performs a basic arthimetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num*second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num/second_num
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    
    except Exception as e:
        return {"error": str(e)}

#3. Getting stock price from alphavantage
@tool
def get_stock_price(symbol: str):
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using ALPHA Vantage with API key in the URL
    """
    api_key = os.getenv("ALPHA_VANTAGE_API")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    return response.json()


# make tools list
tools = [search_tool, calculator, get_stock_price]
llm = ChatOllama(model = "llama3.2")
#bind llm with tools
llm_with_tools = llm.bind_tools(tools)

# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define chat functions
def llm_reply(state: ChatState):
    prompt = state["messages"]
    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# graph

graph = StateGraph(ChatState)

graph.add_node("llm_reply", llm_reply)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm_reply")
graph.add_conditional_edges("llm_reply", tools_condition)
graph.add_edge("tools", "llm_reply")

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
