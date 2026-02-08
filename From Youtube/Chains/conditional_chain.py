from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import Literal
from pydantic import BaseModel, Field

llm = ChatOllama(model="llama3.2")

# Pydantic Validation
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the review to be classified (positive, negative)."
    )

#Parsers
parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)


# prompts required
prompt1 = PromptTemplate(
    template="""Given a feedback of the product {feedback},
classify this feedback into either positive or negative. Only give the sentiment.
{format_instructions}""",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="Knowing the feedback is positive, using the {feedback}, give the appropriate response to the user",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Knowing the feedback is negative, using the {feedback}, give the appropriate response to the user",
    input_variables=["feedback"]
)

conditional_chain = prompt1 | llm | parser2

# Runnable Chains, note that they are tuple, cause only one of them will be executed
runnable_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | llm | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | llm | parser1),
    RunnableLambda(lambda x: "No sentiment found")
)

chain = conditional_chain | runnable_chain

# Invoking the chain
result = chain.invoke(
    {"feedback": "The mobile phone is very bad, I don't recommend it to anyone"}
)

print(result)
