from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

llm = ChatOllama(model = "llama3.2")


class Review(BaseModel):
    summary: str = Field(description="A short summary of the review provided."  )
    sentiment: Literal["pro", "neg", "neut"] = Field(description="The sentiment of the review (positive, negative, or neutral).")
    pros: Optional[list[str]] = Field(description="List of positive points in the review.")
    cons: Optional[list[str]] = Field(description="List of negative points in the review.")
    reviewer: Optional[str] = Field(description="The name of the reviewer in the review.")

structured_llm = llm.with_structured_output(Review)
response = structured_llm.invoke("The hardware is good, but the softeare feels bloated.There are too many preinstalled apps that I cannot remove. Also, the UI looks outdated when compred to other brands. Hoping for a new software update to fix this.")

print(response) #It will be a dictionary
print(type(response)) #class Review
print(response.sentiment) #neg