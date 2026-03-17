import streamlit as st
from langgraph_backend import workflow, ChatState
from langchain_core.messages import HumanMessage

CONFG = {"configurable": {"thread_id": "thread-123"}}
# creating a session state to store previous messages
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# displaying previous turns
for message in st.session_state.message_history:
    role = message["role"]
    text = message["text"]
    with st.chat_message(role):
        st.text(text)


user_input = st.chat_input("Ask something")

if user_input:
    st.session_state["message_history"].append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    input_state = ChatState(messages=[HumanMessage(user_input)])
    with st.chat_message("ai"):
        ai_messaege = st.write_stream(
        message_chunk.content for message_chunk, metadata in workflow.stream(input_state, config=CONFG, stream_mode="messages")
        )
        st.session_state.message_history.append({"role": "ai", "text": ai_messaege})
    
    