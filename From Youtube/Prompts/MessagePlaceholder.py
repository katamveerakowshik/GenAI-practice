#Message PLceholders are used to dynamically insert chat history to the prompt

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name="history"),
    ('human', "{query}")
])

history = [
    HumanMessage(content="I need an update on my order"),
    AIMessage(content="Your order has been shipped. And it will be delivered in 3 days"),
]
prompt = chat_template.invoke({"history": history, "query": "What's my order status?"})

print(prompt)
"""messages=[SystemMessage(content='You are a helpful customer support agent.', additional_kwargs={}, response_metadata={}), HumanMessage(content='I need an update on my order', additional_kwargs={}, response_metadata={}), AIMessage(content='Your order has been shipped. And it will be delivered in 3 days', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content="What's my order status?", additional_kwargs={}, response_metadata={})]"""