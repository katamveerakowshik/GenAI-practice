from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, Literal, Optional

llm = ChatOllama(model = "llama3.2")


"""Anotated is used to specify the type and description of the field.
Literal is used to specify the possible values of the field.
Optional is used to specify that the field is optional."""
class Review(TypedDict):
    summary: Annotated[str, "A short summary of the review provided."]
    sentiment: Annotated[Literal["pos", "neg", "neut"], "The sentiment of the review (positive, negative, or neutral)."]
    pros: Annotated[Optional[list[str]], "List of positive points in the review."]
    cons: Annotated[Optional[list[str]], "List of negative points in the review."]
    reviewer: Annotated[Optional[str], "The name of the reviewer in the review."]

structured_llm = llm.with_structured_output(Review)
response = structured_llm.invoke("The hardware is good, but the softeare feels bloated.There are too many preinstalled apps that I cannot remove. Also, the UI looks outdated when compred to other brands. Hoping for a new software update to fix this.")

print(response)
print(type(response))
print(response["sentiment"])