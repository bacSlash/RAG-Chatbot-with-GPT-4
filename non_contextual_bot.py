import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain

warnings.filterwarnings("ignore")

load_dotenv()

# Load embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

# Define the ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(
    "Using only the context from the knowledge base:\n\n{context}\n, answer the following question:\n\n{question}\n"
)

# Initialize the ChatOpenAI model
chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-4")

# Create an LLMChain with the prompt template and the chat model
llm_chain = LLMChain(
    llm=chat,
    prompt=prompt_template
)

# Define a function to query the vector store and get context
def query_vectorstore(question):
    retriever = vectorstore.as_retriever()
    context = retriever.get_relevant_documents(question)
    return context

# Function to invoke the chain with a question
def ask_question(question):
    context_documents = query_vectorstore(question)
    context_text = "\n".join([doc.page_content for doc in context_documents])
    prompt_input = {"question": question, "context": context_text}
    response = llm_chain.run(prompt_input)
    return response

# Ask the question ex. 1
question = "What are the features of SAP Cloud?"
response = ask_question(question)
print(response)

# Ask the question ex. 2
question = "Why should I choose CloudJune for Utilites and Energy?"
response = ask_question(question)
print(response)