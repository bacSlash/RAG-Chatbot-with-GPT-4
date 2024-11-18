import os
import logging
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
from msal import ConfidentialClientApplication
import requests
from datetime import datetime, timedelta

# Define the constants for OAuth2
TENANT_ID = "e820***"
CLIENT_ID = "795***"
REDIRECT_URI = "http://localhost:8000"
CLIENT_SECRET = "sLb***"
SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_API_URL = "https://login.microsoft.com/v1.0"


###
# INTRODUCING AGENT SYSTEM

class Agent:
    """Base class for all agents."""
    def __init__(self, llm):
        self.llm = llm
    
    def handle(self, user_input):
        raise NotImplementedError("Agent needs to implement a handle method.")
    
class RAGAgent(Agent):
    """Agent for handling RAG-based queries"""
    def __init__(self, llm, rag_chain):
        super().__init__(llm)
        self.rag_chain = rag_chain

    def handle(self, user_input, chat_history):
        # Use the RAG chain to retrieve the answer based on the question
        formatted_history = self._format_chat_history(chat_history)
        context_prompt = f"""
        Recent conversation summary: {formatted_history}
        
        Current question: {user_input}
        """
        # Invoke the rag_chain to get the response
        response = self.rag_chain.invoke({"query": context_prompt})  # Correct method
        return response['result'] if isinstance(response, dict) else str(response)
    
    def _format_chat_history(self, chat_history):
        # Helper method to format the last few chat history entries for context
        formatted_history = []
        for message in chat_history:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_history.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_history[-5:]) # Adjust as necessary
    
class AppointmentAgent(Agent):
    """Agent for handling appointment booking"""
    def __init__(self, llm):
        super().__init__(llm)
        self.client_id = CLIENT_ID
        self.client_secret = CLIENT_SECRET
        self.tenant_id = TENANT_ID
        self.scope = SCOPE
        self.graph_api_url = "https://graph.microsoft.com/v1.0"  # Corrected API URL
        
    def _get_access_token(self):
        app = ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoft.com/{self.tenant_id}",
            client_credential=self.client_secret,
        )
        token_response = app.acquire_token_for_client(scopes=self.scope)
        return token_response.get("access_token")
    
    def _get_free_slots(self):
        access_token = self._get_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        start_time = (datetime.utcnow() + timedelta(days=1)).isoformat() + 'Z'  # Start from tomorrow
        end_time = (datetime.utcnow() + timedelta(days=8)).isoformat() + 'Z'  # End after one week
        
        response = requests.get(
            f"{self.graph_api_url}/me/calendar/calendarView?startDateTime={start_time}&endDateTime={end_time}&$select=start,end,subject",
            headers=headers
        )
        if response.status_code == 200:
            events = response.json().get('value', [])
            # Find free slots (this is a simple implementation and might need refinement)
            all_slots = [slot for slot in self._generate_all_slots(start_time, end_time)]
            busy_slots = [(event['start']['dateTime'], event['end']['dateTime']) for event in events]
            free_slots = [slot for slot in all_slots if not any(
                datetime.fromisoformat(busy_start) <= datetime.fromisoformat(slot) < datetime.fromisoformat(busy_end)
                for busy_start, busy_end in busy_slots
            )]
            return free_slots
        return []
    
    def _generate_all_slots(self, start_time, end_time):
        start = datetime.fromisoformat(start_time.rstrip('Z'))
        end = datetime.fromisoformat(end_time.rstrip('Z'))
        current = start
        while current < end:
            if current.hour >= 9 and current.hour < 17:  # Assuming 9 AM to 5 PM working hours
                yield current.isoformat()
            current += timedelta(hours=1)
        
    def handle(self, user_input, chat_history):
        # Check if this is the first interaction for appointment booking
        if "book an appointment" in user_input.lower() or "schedule a meeting" in user_input.lower():
            free_slots = self._get_free_slots()
            if free_slots:
                available_slots_str = ", ".join([slot.replace('T', ' ').split('+')[0] for slot in free_slots[:5]])  # Show first 5 slots
                return f"Certainly! Here are some available slots for the next week: {available_slots_str}. Which one would you prefer?"
            else:
                return "I'm sorry, but there are no available slots in the next week. Would you like to check for a later date?"
        
        # Check if the user is selecting a slot
        free_slots = self._get_free_slots()
        for slot in free_slots:
            if slot.replace('T', ' ').split('+')[0] in user_input:
                if self._book_appointment(slot):
                    return f"Great! Your appointment has been booked for {slot.replace('T', ' ').split('+')[0]}. Is there anything else you need?"
                else:
                    return "I'm sorry, there was an error booking your appointment. Could you please try again?"
        
        # If we reach here, we need more information from the user
        return "I'm sorry, I didn't catch that. Could you please specify which available slot you'd like to book, or ask to see available slots again?"
    
    def _book_appointment(self, selected_slot):
        access_token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        start_time = datetime.fromisoformat(selected_slot)
        end_time = start_time + timedelta(hours=1)
        appointment_data = {
            "subject": "Appointment with CloudJune",
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "UTC"
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "UTC"
            },
            "location": {
                "displayName": "CloudJune Office"
            }
        }
        response = requests.post(f"{self.graph_api_url}/me/events", headers=headers, json=appointment_data)
        return response.status_code == 201  # Return True if appointment was created successfully
    
class IntentDetectionAgent(Agent):
    """Agent for detecting intent based on conversation context."""
    def __init__(self, llm):
        super().__init__(llm)

    def handle(self, user_input, chat_history):
        # Format the chat history for the LLM
        formatted_history = self._format_chat_history(chat_history)
        
        # Create a prompt for the LLM to determine intent
        prompt = f"""
        Based on the following conversation history and the latest user input, determine the user's intent:
        
        Chat History:
        {formatted_history}
        
        Latest User Input: {user_input}
        
        Please respond with one of the following intents: 'appointment', 'rag'.
        Respond only with the intent word, nothing else.
        """
        
        # Invoke the LLM to get the intent
        response = self.llm.invoke(prompt)
        intent = response['result'] if isinstance(response, dict) else str(response)
        
        # Normalize the intent
        intent = intent.lower().strip()
        
        # Ensure the intent is one of the expected values
        valid_intents = ['appointment', 'rag']
        return intent if intent in valid_intents else 'other'

    def _format_chat_history(self, chat_history):
        formatted_history = []
        for message in chat_history[-5:]:  # Consider only the last 5 messages for context
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted_history.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted_history)
    
###

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

# Initialize the ChatOpenAI LLM (simulating GPT-4-mini)
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=oai_api_key)

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

REPHRASING_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #"
        "# Objective #"
        "Evaluate the given user question and determine if it requires reshaping according to chat history "
        "to provide necessary context and information for answering, or if it can be processed as it is."
        "#########"
        "# Style #"
        "The response should be clear, concise, and in the form of a straightforward decision - either 'Reshape required' or 'No reshaping required'."
        "#########"
        "# Tone #"
        "Professional and analytical."
        "#########"
        "# Audience #"
        "The audience is the internal system components that will act on the decision."
        "#########"
        "# Response #"
        "If the question should be rephrased return response in YAML file format:"
        "result: true"
        "Otherwise return in YAML file format:"
        "result: false"
    ),
    HumanMessagePromptTemplate.from_template(
        "##################"
        "# Chat History #"
        "{chat_history}"
        "#########"
        "# User question #"
        "{question}"
        "#########"
        "# Your Decision in YAML format: #"
    )
])

STANDALONE_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #"
        "This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions."
        "#########"
        "# Objective #"
        "Take the original user question and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information."
        "#########"
        "# Style #"
        "The reshaped standalone question should be clear, concise, and self-contained, while maintaining the intent and meaning of the original query."
        "#########"
        "# Tone #"
        "Neutral and focused on accurately capturing the essence of the original question."
        "#########"
        "# Audience #"
        "The audience is the internal system components that will act on the decision."
        "#########"
        "# Response #"
        "If the original question requires reshaping, provide a new reshaped standalone question that includes all necessary context and information to be self-contained."
        "If no reshaping is required, simply output the original question as is."
    ),
    HumanMessagePromptTemplate.from_template(
        "##################"
        "# Chat History #"
        "{chat_history}"
        "#########"
        "# User original question #"
        "{question}"
        "#########"
        "# The new Standalone question: #"
    )
])

ROUTER_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "# Context #"
        "This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions."
        "#########"
        "# Objective #"
        "Evaluate the given question and decide whether the RAG application is required to provide a comprehensive answer by retrieving relevant information from a knowledge base, or if the chat model's inherent knowledge is sufficient to generate an appropriate response."
        "#########"
        "# Style #"
        "The response should be a clear and direct decision, stated concisely."
        "#########"
        "# Tone #"
        "Analytical and objective."
        "#########"
        "# Audience #"
        "The audience is the internal system components that will act on the decision."
        "#########"
        "# Response #"
        "If the question should be rephrased return response in YAML file format:"
        "result: true"
        "Otherwise return in YAML file format:"
        "result: false"
    ),
    HumanMessagePromptTemplate.from_template(
        "##################"
        "# Chat History #"
        "{chat_history}"
        "#########"
        "# User question #"
        "{question}"
        "#########"
        "# Your Decision in YAML format: #"
    )
])

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
    
    input_key: str = "query"
    chat_history_key: str = "chat_history"
    output_key: str = "result"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key, self.chat_history_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
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
        rephrasing_chain = LLMChain(llm=llm, prompt=REPHRASING_PROMPT, callbacks=callbacks)
        standalone_question_chain = LLMChain(llm=llm, prompt=STANDALONE_PROMPT, callbacks=callbacks)
        router_decision_chain = LLMChain(llm=llm, prompt=ROUTER_DECISION_PROMPT, callbacks=callbacks)
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="query",
            output_key="result",
            return_messages=True
        )
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

    def _format_chat_history(self, chat_history):
        formatted_history = []
        for message in chat_history:
            if isinstance(message, dict):
                role = message.get('role', '')
                content = message.get('content', '')
                formatted_history.append(f"{role.capitalize()}: {content}")
            elif isinstance(message, (HumanMessage, AIMessage)):
                formatted_history.append(f"{message.__class__.__name__}: {message.content}")
            else:
                formatted_history.append(str(message))
        return "\n".join(formatted_history[-5:])

    def _summarize_recent_context(self, formatted_history):
        if not formatted_history:
            return "No recent context available."
        
        summary_prompt = f"Summarize the following conversation history in a concise manner:\n{formatted_history}"
        summary_messages = [
            {"role": "system", "content": "Summarize the given conversation history concisely."},
            {"role": "user", "content": summary_prompt}
        ]
        summary = self.llm.invoke(summary_messages)
        return summary if isinstance(summary, str) else str(summary)

    def _extract_key_points(self, answer):
        extract_prompt = f"Extract 2-3 key points from the following answer:\n{answer}\nFormat the key points as a comma-separated string."
        extract_messages = [
            {"role": "system", "content": "Extract key points from the given answer."},
            {"role": "user", "content": extract_prompt}
        ]
        key_points = self.llm.invoke(extract_messages)
        return key_points if isinstance(key_points, str) else str(key_points)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """Call the chain."""
        chat_history = self.memory.chat_memory.messages
        question = inputs[self.input_key]

        try:
            formatted_history = self._format_chat_history(chat_history)
            recent_summary = self._summarize_recent_context(formatted_history)

            context_prompt = f"""
            Recent conversation summary: {recent_summary}

            Current question: {question}

            Please provide a response that takes into account the recent conversation context.
            """

            result = self.rag_chain.invoke({"query": context_prompt})
            answer = result['result'] if isinstance(result, dict) else str(result)

            key_points = self._extract_key_points(answer)

            self.memory.save_context(inputs, {"result": answer})

            return {self.output_key: answer, "key_points": key_points}
        except Exception as e:
            print(f"Error in _call: {str(e)}")  # Add this line for debugging
            answer = f"An error occurred while processing your request: {str(e)}"
            key_points = ""
            return {self.output_key: answer, "key_points": key_points}

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
    chat_history = data.get('chat_history', [])
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
        
        # Use the IntentDetectionAgent to determine the intent based on chat history and current question
        intent_agent = IntentDetectionAgent(llm)
        intent = intent_agent.handle(question, chat_history)

        # Determine which agent to use based on detected intent
        if intent == "appointment":
            agent = AppointmentAgent(llm)
        elif intent == "information":
            agent = RAGAgent(llm, rag_chain)
        else:
            # For 'feedback' and 'other' intents, use a default response or a specific agent
            agent = RAGAgent(llm, rag_chain)  # You might want to create a separate agent for handling these intents
            
        # Handle the user_query through the chosen agent
        response = agent.handle(question, chat_history)

        cursor.execute(
            "INSERT INTO conversations (user_id, user_query, bot_response) VALUES (%s, %s, %s)",
            (user_id, question, response)
        )
        connection.commit()

        return jsonify({"result": response})
    except Exception as e:
        print(f"Error in query route: {str(e)}")  # Add this line for debugging
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)