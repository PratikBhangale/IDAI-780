import os
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph import invoke_our_graph
from st_callable_util import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

load_dotenv()

# Set up OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("LangGraph LLM Chatbot")



# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize the messages list in session state



# Create the Sidebar
sidebar = st.sidebar

# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    st.session_state["messages"] = []   # Clear the messages list in session state
    # [AIMessage(content="How can I help you?")]
    st.toast('Conversation Deleted', icon='⚙️')


question = st.chat_input(placeholder="Ask Anything.", key=1, accept_file=True, file_type=[".png", ".jpg", ".jpeg"])

if question:

    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                if question and question["files"]:
                    st.image(question["files"][0])
                st.write(message.content)


    # Add the user question to the session state messages
    st.session_state["messages"].append(HumanMessage(content=question.text))
    with st.chat_message("Human"):
        st.write(question.text)
        if question and question["files"]:
            st.image(question["files"][0])

    # Invoke the graph with the user question and the callback handler
    with st.chat_message("AI"):
    # create a new placeholder for streaming messages and other events, and give it context
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state.messages, [st_callback])
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content)) 

