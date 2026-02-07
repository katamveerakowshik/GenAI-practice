from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model = "llama3.2")

prompt1 = PromptTemplate(
    template = "Give me detailed report on {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Give 5 points on {detaied_report}",
    input_variables = ["detailed_report"]
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

print(chain.invoke({"topic": "cricket"}))

chain.get_graph().print_ascii()