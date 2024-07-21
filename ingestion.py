import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# List to store all loaded documents
consulting_documents = []

# Loading the PDF content
consulting_paths = [
    "rag_chatbot_v0/data/Architecture Consulting.pdf",
    "rag_chatbot_v0/data/Energies and Utilities Consulting.pdf",
    "rag_chatbot_v0/data/Oracle.pdf",
    "rag_chatbot_v0/data/Salesforce Consulting KB - Pointer based.pdf",
    "rag_chatbot_v0/data/SAP Consulting.pdf",
]

for dcmt in consulting_paths:
    loader = PyPDFLoader(dcmt)
    documents = loader.load()
    consulting_documents.extend(documents)

# Splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Adjust chunk_size and chunk_overlap as needed
texts = text_splitter.split_documents(consulting_documents)

print(f"Created {len(texts)} chunks")

# Embedding and storing in Pinecone vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
