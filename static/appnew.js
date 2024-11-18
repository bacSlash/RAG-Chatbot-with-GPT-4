class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),
            chatBox: document.querySelector('.chatbox__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
        this.args.voiceInputButton = document.querySelector('.voice-input-button');
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.sessionId = this.getSessionId();
    }

    getSessionId() {
        let sessionId = localStorage.getItem('chatSessionId');
        if (!sessionId) {
            sessionId = 'session_' + Date.now();
            localStorage.setItem('chatSessionId', sessionId);
        }
        return sessionId;
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;
        openButton.addEventListener('click', () => this.toggleState(chatBox));
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });

        this.args.voiceInputButton.addEventListener('click', () => this.toggleVoiceInput());
    }

    toggleState(chatbox) {
        this.state = !this.state;
        chatbox.classList.toggle('chatbox--active', this.state);
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value.trim();
        if (text1 === "") return;

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        this.updateChatText(chatbox);
        textField.value = '';

        this.sendMessage(text1);
    }

    sendMessage(message) {
        const chatHistory = this.messages.map(msg => ({
            role: msg.name === "User" ? "user" : "assistant",
            content: msg.message
        }));

        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: this.sessionId,
                question: message,
                chat_history: chatHistory
            })
        })
        .then(response => response.json())
        .then(data => {
            let botMessage = { name: "CLoudjune", message: data.result };
            this.messages.push(botMessage);
            this.updateChatText(this.args.chatBox);
            
            // Handle key points if needed
            if (data.key_points) {
                console.log("Key points:", data.key_points);
                // You can display these key points in the UI if desired
            }
        })
        .catch(error => {
            console.error('Error:', error);
            let errorMessage = { name: "CLoudjune", message: "Sorry, something went wrong." };
            this.messages.push(errorMessage);
            this.updateChatText(this.args.chatBox);
        });
    }

    toggleVoiceInput() {
        if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
            this.mediaRecorder.stop();
            this.args.voiceInputButton.innerHTML = '<i class="fas fa-microphone"></i>';
        } else {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    this.mediaRecorder = new MediaRecorder(stream);
                    this.mediaRecorder.ondataavailable = (event) => {
                        this.audioChunks.push(event.data);
                    };
                    this.mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                        this.sendAudioToServer(audioBlob);
                        this.audioChunks = [];
                    };
                    this.mediaRecorder.start();
                    this.args.voiceInputButton.innerHTML = '<i class="fas fa-stop"></i>';
                })
                .catch(error => console.error('Error accessing microphone:', error));
        }
    }

    sendAudioToServer(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob);

        fetch('/speech-to-text', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Speech-to-text request failed');
            }
            return response.json();
        })
        .then(data => {
            if (data.text) {
                this.args.chatBox.querySelector('input').value = data.text;
                this.onSendButton(this.args.chatBox);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to convert speech to text. Please try again.');
        });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item) {
            const className = item.name === "CLoudjune" ? "messages__item--visitor" : "messages__item--operator";
            html += '<div class="messages__item ' + className + '">' + item.message + '</div>';
        });
        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
        this.addTextToSpeechButtons();
    }

    addTextToSpeechButtons() {
        const botMessages = document.querySelectorAll('.messages__item--visitor');
        botMessages.forEach(message => {
            if (!message.querySelector('.text-to-speech-button')) {
                const button = document.createElement('button');
                button.className = 'text-to-speech-button';
                button.innerHTML = '<i class="fas fa-volume-up"></i>';
                button.addEventListener('click', () => {
                    const textToSpeak = message.textContent.replace(button.textContent, '').trim();
                    console.log("Text extracted for TTS:", textToSpeak);  // Add this line for debugging
                    this.textToSpeech(textToSpeak);
                });
                message.appendChild(button);
            }
        });
    }

    textToSpeech(text) {
        console.log("Text being sent for TTS:", text);  // Add this line for debugging
        fetch('/text-to-speech', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Text-to-speech request failed');
            }
            return response.blob();
        })
        .then(blob => {
            if (blob.size === 0) {
                throw new Error('Received empty audio data');
            }
            const audio = new Audio(URL.createObjectURL(blob));
            audio.play();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to convert text to speech. Please try again.');
        });
    }
}

const chatbox = new Chatbox();
chatbox.display();
