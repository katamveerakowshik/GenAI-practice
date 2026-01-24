from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOllama(model = "llama3.2")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain what is reinforcement learning in simple terms.")]

response = llm.invoke(messages) 
messages.append(AIMessage(content=response.content))

print(messages)