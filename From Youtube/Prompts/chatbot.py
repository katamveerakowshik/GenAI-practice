from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOllama(model = "llama3.2")


chathistory = [
    SystemMessage(content="You are a helpful assistant."),
]
print("Welcome to the Research Tool Chatbot! Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    chathistory.append(HumanMessage(content=user_input))  # Add the user's input to the chat historyuser_input)
    if user_input == 'exit':
        print("Exiting chat...")
        break
    response = llm.invoke(chathistory)
    chathistory.append(AIMessage(content=response.content))  # Add the AI's response to the chat historyresponse.content)
    print("AI: ", response.content)

print(chathistory)