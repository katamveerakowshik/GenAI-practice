from langchain_core.prompts import ChatPromptTemplate
# ChatPrompt Template is used to create a dynamic list of messages for the prompt
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

# chat_template = ChatPromptTemplate([
# SystemMessage(content="You are a helpful {domain} expert."),
# HumanMessage(content= "Explain about {topic}")
# ])
# prompt = chat_template.invoke({"domain": "cricket", "topic": "doosra"})

# print(prompt) 
"""messages=[SystemMessage(content='You are a helpful {domain} expert.', additional_kwargs={}, response_metadata={}), HumanMessage(content='Explain about {topic}', additional_kwargs={}, response_metadata={})]"""

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain about {topic}')
])
prompt = chat_template.invoke({"domain": "cricket", "topic": "doosra"})

print(prompt)
"""messages=[SystemMessage(content='You are a helpful cricket expert.', additional_kwargs={}, response_metadata={}), HumanMessage(content='Explain about doosra', additional_kwargs={}, response_metadata={})]"""

llm = ChatOllama(model = "llama3.2")
response = llm.invoke(prompt)
print(response.content)