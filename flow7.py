import os
from mysql.connector import pooling
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from flask_session import Session
from datetime import datetime, timedelta
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.output_parsers import YamlOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.base_language import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI  # Ensure this is the correct import for your version
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any, Optional, Dict
import openai
import whisper
from gtts import gTTS
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load environment variables
load_dotenv()

# Set your OpenAI API key
oai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI embedding function
embeddings = OpenAIEmbeddings(api_key=oai_api_key,)

# Load the vector store from disk with the embedding function
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Initialize the OpenAI LLM
llm = OpenAI(api_key=oai_api_key)

# Replace the MySQL connection setup with a connection pool
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "ris***",
    "database": "cloud***"
}

connection_pool = pooling.MySQLConnectionPool(
    pool_name="cloudjune_pool",
    pool_size=5,  # Adjust this based on your needs
    **db_config
)

# Set up for inactivity detection
@app.before_request
def before_request():
    session.permanent = True # Make session permananent so it uses PERMANENT_SESSION_LIFETIME
    
    # Check if 'last_activity' exists in the session
    if 'last_activity' in session:
        last_activity = session['last_activity']
        now = datetime.now()
        
        # If more than 3 minutes have passed since the last activity, clear the session
        if now - last_activity > timedelta(minutes=3):
            session.clear()
            return redirect(url_for('home'))
        
    # Update the last activity time
    session['last_activity'] = datetime.now()


@app.route('/')
def home():
    return render_template('base.html')

# Define a custom prompt template for the RAG chain
rag_prompt_template = """
You are June, an AI assistant for CloudJune, a cloud service provider company. Use the following pieces of context to answer the question at the end. If the question is not related to CloudJune company or its products or services, or if you don't know the answer, politely explain that you can only provide information about CloudJune.

Context: {context}

Question: {question}

Answer (short,human like,conversational,precise,professional and polite):
"""
RAG_PROMPT = PromptTemplate(
    template=rag_prompt_template,
    input_variables=["context", "question"]
)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=RAG_PROMPT)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": RAG_PROMPT,
    }
)

# Define the prompts (using the COSTAR framework as per your example)
REPHRASING_PROMPT_TEMPLATE = """
# Context #
# Objective #
Evaluate the given user question and determine if it requires reshaping according to chat history to provide necessary context and information for answering, or if it can be processed as it is.

#########

# Style #
The response should be clear, concise, and in the form of a straightforward decision - either "Reshape required" or "No reshaping required".

#########

# Tone #
Professional and analytical.

#########

# Audience #
The audience is the internal system components that will act on the decision.

#########

# Response #
If the question should be rephrased return response in YAML file format:
```
    result: true
```
otherwise return in YAML file format:
```
    result: false
```

##################

# Chat History #
{chat_history}

#########

# User question #
{question}

#########

# Your Decision in YAML format # 
"""
REPHRASING_PROMPT = PromptTemplate(
    template=REPHRASING_PROMPT_TEMPLATE,
    input_variables=["chat_history", "question"]
)

STANDALONE_PROMPT_TEMPLATE = """

# Context #
This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions. 

#########

# Objective #
Take the original user question and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information.

#########

# Style #
The reshaped standalone question should be clear, concise, and self-contained, while maintaining the intent and meaning of the original query.

#########

# Tone #
Neutral and focused on accurately capturing the essence of the original question.

#########

# Audience #
The audience is the internal system components that will act on the decision.

#########

# Response #
If the original question requires reshaping, provide a new reshaped standalone question that includes all necessary context and information to be self-contained.
If no reshaping is required, simply output the original question as is.

##################

# Chat History #
{chat_history}

#########

# User original question #
{question}

#########

# The new Standalone question #
"""
STANDALONE_PROMPT = PromptTemplate(
    template=STANDALONE_PROMPT_TEMPLATE,
    input_variables=["chat_history", "question"]
)

ROUTER_DECISION_PROMPT_TEMPLATE = """
# Context #
This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions. 

#########

# Objective #
Evaluate the given question and decide whether the RAG application is required to provide a comprehensive answer by retrieving relevant information from a knowledge base, or if the chat model's inherent knowledge is sufficient to generate an appropriate response.

#########

# Style #
The response should be a clear and direct decision, stated concisely.

#########

# Tone #
Analytical and objective.

#########

# Audience #
The audience is the internal system components that will act on the decision.

#########

# Response #
If the question should be rephrased return response in YAML file format:
```
    result: true
```
otherwise return in YAML file format:
```
    result: false
```

##################

# Chat History #
{chat_history}

#########

# User question #
{question}

#########

# Your Decision in YAML format #
"""
ROUTER_DECISION_PROMPT = PromptTemplate(
    template=ROUTER_DECISION_PROMPT_TEMPLATE,
    input_variables=["chat_history", "question"]
)

# Define the pydantic model for YAML output parsing
class ResultYAML(BaseModel):
    result: bool

class EnhancedConversationalRagChain(Chain):
    """Enhanced chain that encapsulates RAG application enabling natural conversations with improved context awareness."""
    rag_chain: Chain
    rephrasing_chain: LLMChain
    standalone_question_chain: LLMChain
    router_decision_chain: LLMChain
    yaml_output_parser: YamlOutputParser
    memory: ConversationBufferMemory
    llm: BaseLanguageModel
    
    # input/output parameters
    input_key: str = "query"  
    chat_history_key: str = "chat_history" 
    output_key: str = "result"

    @property
    def input_keys(self) -> list[str]:
        """Input keys."""
        return [self.input_key, self.chat_history_key]

    @property
    def output_keys(self) -> list[str]:
        """Output keys."""
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "EnhancedConversationalRagChain"

    @classmethod
    def from_llm(
        cls,
        rag_chain: Chain,
        llm: BaseLanguageModel,
        callbacks: Optional[Callbacks] = None,
        **kwargs: Any,
    ) -> "EnhancedConversationalRagChain":
        """Initialize from LLM."""
        
        # Create the rephrasing chain
        rephrasing_chain = LLMChain(llm=llm, prompt=REPHRASING_PROMPT, callbacks=callbacks)
        
        # Create the standalone question chain
        standalone_question_chain = LLMChain(llm=llm, prompt=STANDALONE_PROMPT, callbacks=callbacks)
        
        # Create the router decision chain
        router_decision_chain = LLMChain(llm=llm, prompt=ROUTER_DECISION_PROMPT, callbacks=callbacks)
        
        # Initialize memory with specific input and output keys
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",
            output_key="result",
            return_messages=True
        )
        
        # Return the instantiated EnhancedConversationalRagChain
        return cls(
            rag_chain=rag_chain,
            rephrasing_chain=rephrasing_chain,
            standalone_question_chain=standalone_question_chain,
            router_decision_chain=router_decision_chain,
            yaml_output_parser=YamlOutputParser(pydantic_object=ResultYAML),
            memory=memory,
            llm=llm,
            callbacks=callbacks,
            **kwargs,
        )

    def _summarize_recent_context(self, chat_history):
        if not chat_history:
            return "No recent context available."
        
        summary_prompt = f"""
        Summarize the following conversation history in a concise manner:
        {chat_history[-5:]}  # Consider only the last 5 messages for summary
        """
        summary = self.llm.invoke(summary_prompt)
        return summary  # Remove .content as it's already a string

    def _extract_key_points(self, answer):
        extract_prompt = f"""
        Extract 2-3 key points from the following answer:
        {answer}
        
        Format the key points as a comma-separated string.
        """
        key_points = self.llm.invoke(extract_prompt)
        return key_points  # This will be a string, not a list

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """Call the chain."""
        chat_history = self.memory.chat_memory.messages
        question = inputs[self.input_key]
        answer = None

        try:
            # Summarize recent context
            recent_summary = self._summarize_recent_context(chat_history)

            # Update prompt with context
            context_prompt = f"""
            Recent conversation summary: {recent_summary}

            Current question: {question}

            Please provide a response that takes into account the recent conversation context.
            """

            # Use the RAG chain with the enhanced prompt
            result = self.rag_chain({"query": context_prompt})
            answer = result['result']

            # Extract key points from the answer
            key_points = self._extract_key_points(answer)

            # Update memory
            self.memory.save_context(
                inputs,
                {"result": answer}  # Only save the 'result' to memory
            )

            return {self.output_key: answer, "key_points": key_points}
        except Exception as e:
            answer = f"An error occurred while processing your request: {str(e)}"
            key_points = ""  # Empty string instead of empty list
            return {self.output_key: answer, "key_points": key_points}

# Set up OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file.save(temp_audio.name)
        
        # Creare a client for interacting with the OpenAI API
        client = openai
        # Transcribe the audio file using the updated API method
        with open(temp_audio.name, 'rb') as audio:
            transcript = client.audio.transcriptions.create(
                model='whisper-1',
                file=audio,
                response_format='text',
                language='en'
            )
            
        # Return the transcribed text
        return jsonify({"text": transcript})
    
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
# Create a mapping for language dialects (accents)
ACCENT_MAP = {
    'us': 'en', # US English (default)
    'uk': 'en-uk', # UK English
    'au': 'en-au', # Australian English
    'in': 'en-in', # Indian English
}

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        # Use gTTS for text to speech
        tts = gTTS(text=text, lang='en-uk')
        
        # Save the generated speech to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            
        return send_file(temp_audio.name, mimetype='audio/mpeg')
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return jsonify({'error': str(e)}), 500
 
@app.route('/query', methods=['POST'])
def query():
    data = request.json
    session_id = data.get('session_id', '')
    chat_history = data.get('chat_history', [])
    question = data.get('question', '')
    
    connection = None
    cursor = None
    try:
        # Get a connection from the pool
        connection = connection_pool.get_connection()
        cursor = connection.cursor()

        # Check if the user exists, if not create a new user
        cursor.execute("SELECT id FROM users WHERE session_id = %s", (session_id,))
        user = cursor.fetchone()
        if not user:
            cursor.execute("INSERT INTO users (session_id) VALUES (%s)", (session_id,))
            connection.commit()
            user_id = cursor.lastrowid
        else:
            user_id = user[0]

        conversational_chain = EnhancedConversationalRagChain.from_llm(
            rag_chain=rag_chain,
            llm=llm
        )

        # Clear the memory at the start of each session
        if 'conversation_started' not in session:
            conversational_chain.memory.clear()
            session['conversation_started'] = True

        # Load chat history into memory
        for message in chat_history:
            if isinstance(message, HumanMessage):
                conversational_chain.memory.chat_memory.add_user_message(message.content)
            elif isinstance(message, AIMessage):
                conversational_chain.memory.chat_memory.add_ai_message(message.content)

        result = conversational_chain({"query": question})
        answer = result.get('result', '')
        key_points = result.get('key_points', '')

        # Store the conversation in the database
        cursor.execute(
            "INSERT INTO conversations (user_id, user_query, bot_response, key_points) VALUES (%s, %s, %s, %s)",
            (user_id, question, answer, key_points)
        )
        connection.commit()

        return jsonify({"result": answer, "key_points": key_points})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Ensure the cursor and connection are closed even if an exception occurs
        if cursor:
            cursor.close()
        if connection:
            try:
                connection.close()
            except Exception:
                pass  # Ignore any errors when closing the connection
    

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)