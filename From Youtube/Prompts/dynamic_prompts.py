from langchain_ollama import ChatOllama
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

llm = ChatOllama(model = "llama3.2")

st.header("Research Tool")

paper = st.selectbox("Select Paper", options=["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style = st.selectbox("Select Style", options=["Beginner Friendly", "Code-Oriented", "Technical", "Mathematical", "Detailed"])
length_input = st.selectbox("Select Length", options=["Short (200 words)", "Medium (500 words)", "Long (1000 words)"])

template = load_prompt(r".\From Youtube\Prompts\template.json")

# prompt = template.invoke({
#     "paper": paper, 
#     "style": style,
#     "length_input": length_input
#     })

# if st.button("Generate Summary"):
#     response = llm.invoke(prompt)
#     st.write(response.content)
    

if st.button("Generate Summary"):
    chain = template | llm
    response = chain.invoke({
        "paper": paper, 
        "style": style,
        "length_input": length_input
        })
    st.write(response.content)