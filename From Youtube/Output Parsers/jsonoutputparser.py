from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model = "llama3.2")

parser = JsonOutputParser()


# We have to mention the format instructions in the template
template1 = PromptTemplate(
    template = "Give me detailed report on {topic} \n{format_instructions}" ,
    input_variables = ["topic"],
    partial_variables = {"format_instructions": parser.get_format_instructions()}
)

chain = template1 | llm | parser

print(chain.invoke({"topic": "bloack hole"}))
