import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings


# Load environment variables
load_dotenv()

# Set the title of the Streamlit app
st.set_page_config(layout="wide", page_title="Loading Data into Data Store")
st.title("Loading Data into Data Store")

st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Get environment variables
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")

vector_store = AstraDBVectorStore(
    embedding=st.session_state.embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="capstone_test",
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)


# Define the function to handle document embeddings
def vector_embedding(uploaded_files):

    docs = []
    for uploaded_file in uploaded_files:
        # Save the uploaded file to disk
        with open(os.path.join("uploaded_docs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Load PDF document
        loader = PyPDFLoader(os.path.join("uploaded_docs", uploaded_file.name))
        docs.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    # Create vector embeddings
    vector_store.add_documents(final_documents)


# Allow users to upload PDF documents
uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=["pdf"])

# Button to create document embeddings
if st.button("Create Document Embeddings") and uploaded_files:
    os.makedirs("uploaded_docs", exist_ok=True)
    with st.spinner('Creating embeddings...'):
        vector_embedding(uploaded_files)
        st.write("Vector Store DB is ready")