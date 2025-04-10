from typing import Annotated, TypedDict, Literal

from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
import os
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_tavily import TavilySearch



# Load environment variables
load_dotenv()

# Get environment variables for AstraDB
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
    

# Tavily Search API Tool
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
search = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    include_images=False,
    # include_image_descriptions=False,
    search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    # A simplified weather response based on location
    if location.lower() in ["sf", "san francisco"]:
        return "It's 60 degrees and foggy."
    else:
        return "It's 90 degrees and sunny."

@tool
def query_knowledge_base(query: str):
    """Retrieve information about Brain tumors from the knowledge base using semantic search."""

    # Initialize the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Initialize the vector store
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="capstone_test",
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAMESPACE,
    )
    
    # Perform the similarity search
    results = vector_store.similarity_search(query, k=3)
    
    # Format and return the search results
    if results:
        formatted_results = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(results)])
        return f"Here's what I found in the knowledge base:\n\n{formatted_results}"
    else:
        return "I couldn't find any relevant information in the knowledge base."

# List of tools that will be accessible to the graph via the ToolNode
tools = [get_weather, query_knowledge_base, search]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Core invocation of the model
def _call_model(state: GraphsState):
    messages = state["messages"]
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True,
        # specifically for OpenAI we have to set parallel tool call to false
        # because of st primitively visually rendering the tool results
    ).bind_tools(tools, parallel_tool_calls=False)
    response = llm.invoke(messages)
    return {"messages": [response]}  # add the response to the messages using LangGraph reducer paradigm

# Define the structure (nodes and directional edges between nodes) of the graph
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)

graph.add_edge(START, "modelNode")
# Add conditional logic to determine the next step based on the state (to continue or to end)
graph.add_conditional_edges(
    "modelNode",
    should_continue,  # This function will decide the flow of execution
)
graph.add_edge("tools", "modelNode")

# Compile the state graph into a runnable object
graph_runnable = graph.compile()


# Function to invoke the graph with messages and callbacks
def invoke_our_graph(st_messages, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")
    return graph_runnable.invoke({"messages": st_messages}, config={"callbacks": callables})
