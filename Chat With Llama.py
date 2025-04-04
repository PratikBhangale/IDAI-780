import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent



# Get API keys from environment variables
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')


# Creating an Array for storing the chat history for the model.
context = []


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("LANG-CHAIN CHAT WITH LLAMA")

# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state.keys():
    st.session_state.messages = []

# Create the Sidebar
sidebar = st.sidebar

# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    context = []
    st.session_state.messages =[]



# Defining out LLM Model
llm = ChatOllama(model='llama3.1', temperature=0)


search = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)


tools = [search]

llm_tools = llm.bind_tools(tools)

# Create the function to stream output from llm
def get_response(question, context):

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. You must answer all questions to the best of your ability without using tools. Make sure to use the tavily_search_results_json tool for information on current events, but only if the user requires it."),
        ("memory", "{chat_history}"),
        ("human", "{input}"),
        ("thought", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, stream_runnable=False)

    response = agent_executor.invoke({"input":question, "chat_history":context, "tools":tools})

    return response["output"]


# ------------------------------------------------------------------------------------------------------------------------------

def start_app():

        try:
            OLLAMA_MODELS = ollama.list()["models"]
        except Exception as e:
            st.warning("Please make sure Ollama is installed and running first. See https://ollama.ai for more details.")
            st.stop()

        question = st.chat_input("Ask Anything.", key=1)

        if question:
            
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            context.append(HumanMessage(content=question))
            st.session_state.messages.append({"role": "user", "content": question})
            
            
            # response = get_response(PROMPT, llm).invoke({"input": question, "context": context})
            with st.chat_message("Human"):
                st.markdown(question)

            with st.spinner("Thinking..."):
                with st.chat_message("AI"):
                    response = get_response(question, context)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    context.append(AIMessage(content=str(response)))
                    st.write(response)


            # for message in st.session_state.messages:
            #     if isinstance(message, AIMessage):
            #         with st.chat_message("AI"):
            #             st.write(message.content)
            #     elif isinstance(message, HumanMessage):
            #         with st.chat_message("Human"):
            #             st.write(message.content)


            # for message in st.session_state.messages:
            #     with st.chat_message(message["role"]):
            #         st.write(message["content"])



if __name__ == "__main__":
    start_app()