from typing import Annotated, TypedDict, Literal
from typing import Dict, Any
import aiohttp
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
from typing import Annotated, TypedDict, Literal, Optional, List
import base64
from langchain_core.messages import HumanMessage
import io
import sys
from contextlib import redirect_stdout



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
def python_repl(code: str) -> str:
    """
    Execute Python code in a secure environment and return the result.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: The output or result of the executed code
    """
    try:
        # Create a string buffer to capture stdout
        
        # Execute code and capture output
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            # Execute the code - use exec for statements, eval for expressions
            try:
                # First try to evaluate as an expression
                result = eval(code)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                # If it's not an expression, execute as statements
                exec(code)
        
        return buffer.getvalue() or "Code executed successfully with no output."
    except Exception as e:
        return f"Error: {str(e)}"




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
    results = vector_store.similarity_search(query, k=7)
    
    # Format and return the search results
    if results:
        formatted_results = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(results)])
        return f"Here's what I found in the knowledge base:\n\n{formatted_results}"
    else:
        return "I couldn't find any relevant information in the knowledge base."

# List of tools that will be accessible to the graph via the ToolNode
tools = [query_knowledge_base, search, python_repl]
tool_node = ToolNode(tools)

# This is the default state same as "MessageState" TypedDict but allows us accessibility to custom keys
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    images: Optional[List[str]]  # Base64 encoded image data
    # Custom keys for additional data can be added here such as - conversation_id: str

graph = StateGraph(GraphsState)

# Function to decide whether to continue tool usage or end the process
def should_continue(state: GraphsState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:  # Check if the last message has any tool calls
        return "tools"  # Continue to tool execution
    return "__end__"  # End the conversation if no tool is needed

# Updated model invocation function to handle images
def _call_model(state: GraphsState):
    messages = state["messages"]
    images = state.get("images", [])
    
    # Create a multimodal model
    llm = ChatOpenAI(
        model="gpt-4o",  # Change to GPT-4o which supports vision capabilities
        temperature=0.2,
        streaming=True,
    ).bind_tools(tools, parallel_tool_calls=False)
    
    # If there are images, modify the last user message to include the image
    if images and len(images) > 0:
        # Find the last human message
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                # Create a new message with image content
                content = [
                    {"type": "text", "text": messages[i].content}
                ]
                
                # Add each image as content
                for image_data in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    })
                
                # Replace the message with the multimodal version
                messages[i] = HumanMessage(content=content)
                break
    
    response = llm.invoke(messages)
    # Clear images after they've been processed
    return {"messages": [response], "images": []}


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


# Updated invoke function to handle images
def invoke_our_graph(st_messages, images=None, callables=None):
    if callables and not isinstance(callables, list):
        raise TypeError("callables must be a list")
    
    initial_state = {"messages": st_messages}
    if images:
        initial_state["images"] = images
        
    config = {}
    if callables:
        config["callbacks"] = callables
        
    return graph_runnable.invoke(initial_state, config=config)