import os
import random
from mysql.connector import pooling
from flask import Flask, request, jsonify, render_template, send_file
from langchain.chains import LLMChain, RetrievalQA
from flask_session import Session
from langchain.chains.base import Chain
from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.output_parsers import YamlOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv
from flask import session 
from langchain.base_language import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from typing import Any, Optional, Dict, List
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import openai
from gtts import gTTS
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

###
# INTRODUCING AUTONOMOUS AGENT SYSTEM

class AutonomousAgent:
    def __init__(self, llm, rag_chain, knowledge_base):
        self.llm = llm
        self.rag_chain = rag_chain
        self.knowledge_base = knowledge_base
        self.memory = ConversationBufferMemory(return_messages=True)
        self.state = {}

    def process_input(self, user_input):
        intent = self.understand_intent(user_input)
        
        if intent == 'goodbye':
            return self.handle_goodbye()
        
        if intent == 'appointment':
            return self.handle_appointment(user_input)
        
        # For other intents, use the RAG chain
        rag_response = self.rag_chain.invoke({"query": user_input})['result']
        
        # Rephrase the response for more natural conversation
        final_response = self.rephrase_response(rag_response, user_input)
        
        self.update_state_and_memory(user_input, final_response)
        
        return final_response

    def understand_intent(self, user_input):
        intent_prompt = f"""
        Determine the primary intent of the following user input:
        User input: {user_input}
        
        Possible intents: goodbye, appointment, general_query
        Return only the intent label.
        """
        response = self.llm.invoke([{"role": "user", "content": intent_prompt}])
        return response.content.strip().lower()

    def handle_goodbye(self):
        goodbye_responses = [
            "It was great chatting with you! Have a wonderful day!",
            "Thanks for the conversation. Feel free to reach out if you need anything else.",
            "Goodbye! Don't hesitate to contact us if you have any questions in the future."
        ]
        return random.choice(goodbye_responses)

    def handle_appointment(self, user_input):
        if 'booked_appointment' in self.state:
            return f"Your appointment is already booked for {self.state['booked_appointment']}. Is there anything else I can help you with?"
        
        available_slots = ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"]
        
        for slot in available_slots:
            if slot.lower() in user_input.lower():
                self.state['booked_appointment'] = slot
                return f"Great! I've booked your appointment for {slot}. Is there anything else you need assistance with?"
        
        return f"I'd be happy to help you book an appointment. Our available slots are: {', '.join(available_slots)}. Which time works best for you?"

    def rephrase_response(self, rag_response, user_input):
        rephrase_prompt = f"""
        Rephrase the following response to sound more natural and conversational. 
        Maintain the key information but vary the structure and wording.
        Only mention CloudJune services if directly relevant to the user's query.

        User input: {user_input}
        Original response: {rag_response}

        Rephrased response:
        """
        response = self.llm.invoke([{"role": "user", "content": rephrase_prompt}])
        return response.content.strip()

    def update_state_and_memory(self, user_input, response):
        self.memory.save_context({"input": user_input}, {"output": response})

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load environment variables
load_dotenv()

# Set your OpenAI API key
oai_api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the OpenAI embedding function
embeddings = OpenAIEmbeddings(api_key=oai_api_key)

# Load the vector store from disk with the embedding function
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Replace the MySQL connection setup with a connection pool
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "ris***",
    "database": "cloud***"
}

connection_pool = pooling.MySQLConnectionPool(
    pool_name="cloudjune_pool",
    pool_size=5,
    **db_config
)

@app.route('/')
def home():
    return render_template('base.html')

# Advanced RAG prompt template
rag_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are June, an AI assistant exclusively for CloudJune, a cloud service provider company. "
        "Your primary function is to assist with inquiries about CloudJune's products, services, and company information in multiple languages. "
        "Use the following context to answer the user's question:{context}"
        "IMPORTANT INSTRUCTIONS:"
        "1. Be conversational, polite, and adaptive. Respond appropriately to greetings, small talk, and CloudJune-related queries."
        "2. For greetings or small talk, engage briefly and naturally, then guide the conversation towards CloudJune topics."
        "3. Keep responses concise,professional and short, typically within 2-3 sentences unless more detail is necessary."
        "4. Use only the provided context for CloudJune-related information. Don't invent or assume details."
        "5. If a question isn't about CloudJune, politely redirect: 'I apologize, but I can only provide information about CloudJune, its products and services.Is there anything else you'd like to know?'"
        "6. For unclear questions, ask for clarification: 'To ensure I provide the most accurate information about CloudJune, could you please rephrase your question?'"
        "7. Adjust your language style to match the user's - formal or casual, but always maintain professionalism."
        "8. Always respond in the same language as the user's input."
        "9. If the context doesn't provide enough information for a comprehensive answer, be honest about the limitations and offer to assist with related topics you can confidently address."
        "10. Remember previous interactions within the conversation and maintain context continuity."
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=oai_api_key)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": rag_prompt_template,
    }
)

# Initialize the autonomous agent
autonomous_agent = AutonomousAgent(llm, rag_chain, vectorstore)

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file.save(temp_audio.name)

        with open(temp_audio.name, "rb") as audio:
            transcript = openai.audio.transcriptions.create(
                model='whisper-1',
                file=audio,
                response_format='text',
                language='en'
            )

        return jsonify({"text": transcript})
    
    except Exception as e:
        print(f"Error in speech-to-text: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        tts = gTTS(text=text, lang='en-uk')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
        
        return send_file(temp_audio.name, mimetype="audio/mpeg")
    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    session_id = data.get('session_id', '')
    question = data.get('question', '')

    connection = None
    cursor = None
    try:
        connection = connection_pool.get_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT id FROM users WHERE session_id = %s", (session_id,))
        user = cursor.fetchone()
        if not user:
            cursor.execute("INSERT INTO users (session_id) VALUES (%s)", (session_id,))
            connection.commit()
            user_id = cursor.lastrowid
        else:
            user_id = user[0]
            
        # Process the query through the autonomous agent
        response = autonomous_agent.process_input(question)

        cursor.execute(
            "INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)",
            (user_id, question, response)
        )
        connection.commit()

        return jsonify({"result": response})
    except Exception as e:
        print(f"Error in query route: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def update_chat_history(session_id, role, content):
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({"role": role, "content": content})
    session.modified = True
    
    # Limit chat history to last 10 messages (adjust as needed)
    session['chat_history'] = session['chat_history'][-10:]

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)