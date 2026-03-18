
#********************************Importing necessary libraries and modules********************************
import streamlit as st
from langgraph_backend import workflow, ChatState
from langchain_core.messages import HumanMessage
import uuid

#***********************************Utility functions***********************************
def generate_thread_id():
    return str(uuid.uuid4())

def new_chat():
    st.session_state.thread_id = generate_thread_id()
    st.session_state.thread_list.append(st.session_state.thread_id)
    st.session_state.message_history = []

# Used to load the previous messages in the chat UI when a thread is selected from the sidebar
def load_thread(thread_id):
    st.session_state.thread_id = thread_id
    temp_message_history = []
    messages = (workflow.get_state(config={"configurable": {"thread_id": thread_id}})).values.get("messages", [])
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "user"
        else:    
            role = "ai"
        temp_message_history.append({"role": role, "text": message.content})
    st.session_state.message_history = temp_message_history


#***********************************Session states ************************************
# creating a session state to store previous messages
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'thread_list' not in st.session_state:
    st.session_state['thread_list'] = [st.session_state.thread_id]

#********************************Displaying previous messages in the chat UI********************************
for message in st.session_state.message_history:
    role = message["role"]
    text = message["text"]
    with st.chat_message(role):
        st.text(text)


#*******************************side bar UI********************************
st.sidebar.title("Langgraph Chatbot")

if st.sidebar.button("New Chat"):
    new_chat()
    st.rerun()

st.sidebar.header("Chat History")

for thread_id in st.session_state.thread_list[::-1]:
    if st.sidebar.button(str(thread_id)):
        load_thread(thread_id)
        st.rerun()


#**********************************************Chat UI***********************************************
user_input = st.chat_input("Ask something")

if user_input:

    st.session_state["message_history"].append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.text(user_input)


    CONFG = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = ChatState(messages=[HumanMessage(user_input)])
    with st.chat_message("ai"):
        ai_messaege = st.write_stream(
        message_chunk.content for message_chunk, metadata in workflow.stream(input_state, config=CONFG, stream_mode="messages")
        )
        st.session_state.message_history.append({"role": "ai", "text": ai_messaege})
    