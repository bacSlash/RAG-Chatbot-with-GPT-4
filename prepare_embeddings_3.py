import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

# Set your OpenAI API key
oai_api_key = os.environ.get("OPENAI_API_KEY")

# Document class
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

# Custom TextLoader to handle encoding issues
class CustomTextLoader(TextLoader):
    def load(self):
        try:
            with open(self.file_path, encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(self.file_path, encoding='utf-8', errors='ignore') as f:
                text = f.read()
        return [Document(page_content=text, metadata={"source": self.file_path})]

# Load documents manually for .txt files
documents = []
directory = "CJ-GPT-BOT-main/data_new"
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if filename.endswith(".txt"):
        loader = CustomTextLoader(file_path)
        loaded_docs = loader.load()
        for doc in loaded_docs:
            documents.append(doc)

print(f"Loaded documents: {len(documents)}")

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
texts = text_splitter.split_documents(documents)

print(f"Number of text chunks: {len(texts)}")

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=oai_api_key)

# Create and persist the vector store
vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
vectorstore.persist()

print("Embeddings created and saved successfully.")
