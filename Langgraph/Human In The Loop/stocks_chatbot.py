from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.prebuilt import tools_condition, ToolNode 
from langgraph.types import interrupt, Command
from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatOllama(model = "llama3.2")

# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# adding tools
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


@tool
def purchase_stock(symbol: str, quantity: int):
    """
    Purchase stock for a given symbol and quantity, you will not buy the stock it's just for demo
    """
    decision = interrupt(
        f"Approve buying stock {symbol} for {quantity} units? yes/no"
    )
    
    if decision["approval"] == "yes":
        return {
            "symbol": symbol,
            "quantity": quantity,
            "status": "success",
            "message": f"Stock purchase successful for {symbol} for {quantity} units"
        }
    else:
        return {
            "symbol": symbol,
            "quantity": quantity,
            "status": "failed",
            "message": f"Stock purchase failed for {symbol} for {quantity} units"
        }

tools = [purchase_stock, get_stock_price]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# Node
def chat_node(state: ChatState):
    message = state["messages"]
    response = llm_with_tools.invoke(message)
    return {"messages": [response]}


# Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node )

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    thread_id = "1"

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit" or user_input.lower() == "quit":
            print("Good Bye!!")
            break

        input_state = ChatState(messages=[HumanMessage(user_input)])
        result = workflow.invoke(input_state, config={"configurable": {"thread_id": thread_id}})

        interrupts = result.get("__interrupt__", [])

        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f'"HITL:" {prompt_to_human}')
            user_input = input("User: ")
            result = workflow.invoke(Command(resume = {"approval": user_input}), config={"configurable": {"thread_id": thread_id}})
        
        messages = result["messages"]
        output = messages[-1].content

        print("AI: ", output)