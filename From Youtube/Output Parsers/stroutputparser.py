from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model = "llama3.2")

parser = StrOutputParser()

template1 = PromptTemplate(
    template = "Give me detailed report on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "Give me 5 lines summary on the topic based on the detailed report. {text}", 
    input_variables = ["text"]
)

chain = template1 | llm | parser | template2 | llm | parser

print(chain.invoke({"topic": "bloack hole"}))

