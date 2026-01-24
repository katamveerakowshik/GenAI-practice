
from langchain_core.prompts import PromptTemplate

template_to_store = PromptTemplate(
    template = "Please summarise the research paper titled '{paper}' with the following specifications: "
    "Explanation Style: {style}"
    "Explanation Length: {length_input}"
    "1. Mathematical Details:"
    "-Include relevant mathematical equations if present in the paper"
    "-Explain the mathematical concepts clearly"
    "2.Analogies: Use relatable analogies to explain complex mathematical concepts"
    "If certain information is not avaiable in the paper then respond with 'Information Not Available instead of guessing. Ensure the summary is clear and concise",
    input_variables = ["paper", "style", "length_input"])

template_to_store.save(r".\From Youtube\Prompts\template.json")
