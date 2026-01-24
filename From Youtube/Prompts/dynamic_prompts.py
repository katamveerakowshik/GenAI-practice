from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model = "llama3.2")

st.header("Research Tool")

paper = st.selectbox("Select Paper", options=["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style = st.selectbox("Select Style", options=["Beginner Friendly", "Code-Oriented", "Technical", "Mathematical", "Detailed"])
length_input = st.selectbox("Select Length", options=["Short (200 words)", "Medium (500 words)", "Long (1000 words)"])

template = PromptTemplate(
    template = "Please summarise the research paper titled '{paper}' with the following specifications: "
    "Explanation Style: {style}"
    "Explanation Length: {length_input}"
    "1. Mathematical Details:"
    "-Include relevant mathematical equations if present in the paper"
    "-Explain the mathematical concepts clearly"
    "2.Analogies: Use relatable analogies to explain complex mathematical concepts"
    "If certain information is not avaiable in the paper then respond with 'Information Not Available instead of guessing. Ensure the summary is clear and concise",
    input_variables = ["paper", "style", "length_input"])

prompt = template.invoke({
    "paper": paper, 
    "style": style,
    "length_input": length_input
    })

if st.button("Generate Summary"):
    response = llm.invoke(prompt)
    st.write(response.content)
    