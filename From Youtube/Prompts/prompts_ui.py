#Static Prompt implementation
#It can bring a lot of issues because the even if user make small changes to the prompt
#the model will not be able to adapt to it.

from langchain_ollama import ChatOllama
import streamlit as st

llm = ChatOllama(model = "llama3.2")

st.header("Research Tool")
query = st.text_input("Enter your prompt here:")

if st.button("Summarise"):
    response = llm.invoke(query)
    st.write(response.content)