import io
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import aiohttp
from typing import Dict, Any

from graph import invoke_our_graph
import asyncio
from st_callable_util_improved import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context

load_dotenv()

# Set up OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# Tumor Segmentation API URL
async def segment_tumor(base64_image: str, api_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Send a base64-encoded image to the tumor segmentation API and get the results.
    
    Args:
        base64_image (str): Base64-encoded image string
        api_url (str): API endpoint URL (default: http://localhost:8000)
        
    Returns:
        dict: Dictionary containing:
            - segmentation_image: Base64-encoded segmentation result
            - tumor_detection: Tumor detection result
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{api_url}/predict",
            json={"image": base64_image},
            timeout=30
        ) as response:
            response.raise_for_status()
            return await response.json()


# Set the title of the Streamlit app
st.set_page_config(layout="wide")#, page_title="Llama 3 Model Document Q&A"
st.title("LangGraph LLM Chatbot")

# Create the Sidebar
sidebar = st.sidebar

# Image upload section
uploaded_image = sidebar.file_uploader(
    "Upload an image for analysis", 
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

# Creating a Session State array to store and show a copy of the conversation to the user.
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Initialize the messages list in session state

# Initialize images list in session state
if "images" not in st.session_state:
    st.session_state["images"] = []


# Initialize image in session state
if "image_base64" not in st.session_state:
    st.session_state["image_base64"] = None



# Initialize the current_document boolean variable
current_document = False

if uploaded_image:current_document = True

# Process uploaded image
if current_document:
    # Convert image to base64
    img = Image.open(uploaded_image)
    
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Store in session state
    st.session_state.image_base64 = img_str
    st.sidebar.success("Image uploaded successfully!")
    
    # Display currently loaded image
    if current_document:
        try:
            # Create event loop and run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(segment_tumor(st.session_state.image_base64))
            
            # Convert the segmentation image from base64 to displayable format
            segmentation_image_bytes = base64.b64decode(result["segmentation_image"])
            segmentation_image = Image.open(io.BytesIO(segmentation_image_bytes))
            
            # Create the message content with images directly displayed here
            with st.chat_message("AI"):
                st.write(f"""
                Analysis Results:
                - Tumor Detection: {result['tumor_detection']}
                
                I've analyzed the image and generated a segmentation map showing the areas 
                of interest. The segmentation is displayed below, where the highlighted 
                regions indicate potential tumor locations.
                """)
                
                # Display original and segmented images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original Image")
                with col2:
                    st.image(segmentation_image, caption="Segmentation Result")
            
            # Just add a simple text message to session state (no image data)
            # This is just to record that the analysis happened
            # st.session_state["messages"].append(
            #     AIMessage(content="Image analysis completed. You can ask questions about the image now.")
            # )
            
            current_document = False
        except aiohttp.ClientError as e:
            error_message = f"Error during image analysis: {str(e)}"
            st.session_state["messages"].append(AIMessage(content=error_message))
            with st.chat_message("AI"):
                st.write(error_message)
            current_document = False





# Create the reset button for the chats
clear_chat = sidebar.button("Clear Chat")
if clear_chat:
    st.session_state["messages"] = []
    st.session_state["images"] = []
    st.session_state["image_base64"] = None
    if "segmentation_data" in st.session_state:
        st.session_state["segmentation_data"] = []
    st.toast('Conversation Deleted', icon='⚙️')


# Function to encode image to base64
def encode_image(image_file):
    if image_file is None:
        return None
    
    img = Image.open(image_file)
    
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        # Create a white background image
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image on the background using the alpha channel as mask
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
    
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str



# Display the chat messages in the Streamlit app
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)



question = st.chat_input(placeholder="Ask Anything.", key=1, accept_file=True, file_type=[".png", ".jpg", ".jpeg"])


if question:

    # Add the user question to the session state messages
    st.session_state["messages"].append(HumanMessage(content=question.text))

    # Update the image processing section in your code:
    with st.chat_message("Human"):
        st.write(question.text)
        
        # Process and display uploaded images
        uploaded_images = []
        if question and question.files:
            # Add the image to the session state messages                
            for file in question.files:
                # st.session_state["messages"].append(HumanMessage(content={"type": "image", "image_data": encode_image(file)}))
                st.image(file)
                encoded_image = encode_image(file)
                if encoded_image:
                    uploaded_images.append(encoded_image)

    # Invoke the graph with the user question, images, and the callback handler
    with st.chat_message("AI"):
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(
            st_messages=st.session_state.messages,
            images=uploaded_images if uploaded_images else None,
            callables=[st_callback]
        )
        st.session_state.messages.append(AIMessage(content=response["messages"][-1].content))
