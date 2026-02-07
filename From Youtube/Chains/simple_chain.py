from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = PromptTemplate(
    template= "Give me 5 interesting points on {topic}",
    input_variables=["topic"]
)
llm = ChatOllama(model = "llama3.2")

parser = StrOutputParser()

chain = prompt | llm | parser

print(chain.invoke({"topic": "cricket"}))

chain.get_graph().print_ascii()

