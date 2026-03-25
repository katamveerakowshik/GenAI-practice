from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools import tool
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

import requests
import sqlite3
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

embeddings = OllamaEmbeddings(model = "Embeddinggemma")


## PDF retriever store (per thread)
_THREAD_RETHRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available"""
    if thread_id and thread_id in _THREAD_RETHRIEVERS:
        return _THREAD_RETHRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """ Build a Chroma retriever for the uploaded pdf and store it for the thread.
    Returns a summary dict that can be surfaced in the UI"""
    if not file_bytes:
        raise ValueError("No bytes received for ingestion")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix = ".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, separators = ["\n\n", "\n", " ", ""])
        chuncks = splitter.split_documents(docs)

        vector_store = Chroma.from_documents(chuncks, embeddings)
        retriever = vector_store.as_retriever(
            search_type = "similarity",
            seach_kwargs = {"k": 4}
        )

        _THREAD_RETHRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chuncks)
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chuncks)
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
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


@tool
def rag_tool(query:str, thread_id: Optional[str] = None):
    """Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool
    """

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query
        }
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadta": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    }


# make tools list
tools = [search_tool, calculator, get_stock_price, rag_tool]
llm = ChatOllama(model = "llama3.2")
#bind llm with tools
llm_with_tools = llm.bind_tools(tools)

# defining state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# define chat functions
def chat_node(state: ChatState, config = None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    
    system_message = SystemMessage(
        content = ("" \
        "You are a helpful assistant. For questions about the uploaded PDF, call" \
        "the 'rag_tool' function and include the thread_id" \
        f"'{thread_id}'. You can also use the web search, stock price, and calculator"
        "tools when needed. If no document is availabe, ask the user to upoad a PDF")
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config = config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# graph

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

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

def thread_has_document(thread_id: str):
    return str(thread_id) in _THREAD_RETHRIEVERS

def thread_document_metadata(thread_id: str):
    return _THREAD_METADATA.get(str(thread_id), {})
