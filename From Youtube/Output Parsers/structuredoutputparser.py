from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

llm = ChatOllama(model = "llama3.2")

schema = {
    ResponseSchema(name="fact 1", description="fact 1 about the topic"),
    ResponseSchema(name="fact 2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact 3", description="Fact 3 about the topic"),
}

parser = StructuredOutputParser.from_response_schemas(schema)

template1 = PromptTemplate(
    template = "Give me detailed report on {topic} \n{format_instructions}",
    input_variables = ["topic"],
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

chain = template1 | llm | parser

print(chain.invoke({"topic": "black hole"}))