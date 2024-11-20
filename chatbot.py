# Import libraries
import openai
import google.generativeai as genai
import constants
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from fastapi import FastAPI

# Config
#OpenAI API
openai.api_base = "https://api.pawan.krd/cosmosrp/v1/chat/completions"
openai.api_key = constants.OPENAI_API_KEY

#Gemini API
genai.configure(api_key=constants.GEMINI_API_KEY)

#FastAPI
app = FastAPI()

origins = [
    'http://127.0.0.1:3000',
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],   
)

# Select model of Gemini
model = genai.GenerativeModel("gemini-1.5-flash")

history_gemini = []
history_openai = []

@app.get("/gemini")
def gemini_chat_bot(user_message: Optional[str] = None):

    # Maximum number of history messages to keep
    max_history_size = 2
    
    # Saving the user message
    history_gemini.append({"role": "user", "parts": user_message})

    # Limit history size by keeping only the last `max_history_size` messages
    if len(history_gemini) > max_history_size:
        history_gemini.pop(0)  # Remove the oldest message if the history exceeds the limit

    gemini_full_message = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in history_gemini])

    # Generating the response
    response = model.generate_content(
        f"Gemini: {gemini_full_message}",
        generation_config=genai.types.GenerationConfig(
            temperature=0.34,
        ),
    )
    
    history_gemini.append({"role": "model", "parts": response.text})
    gemini_model_response = response.text

    return {
        "user": user_message,
        "model": gemini_model_response
    }


@app.get("/openai")
def openai_chat_bot(user_message: Optional[str] = None):
    # Initialize conversation with at least one message
    openai_messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Maximum number of history messages to keep
    max_history_size = 4

    # Add the user message and any history of conversation
    openai_messages += [{"role": msg["role"], "content": msg["content"]} for msg in history_openai]

    # Limit history size by keeping only the last `max_history_size` messages
    if len(openai_messages) > max_history_size:
        openai_messages = openai_messages[-max_history_size:]

    openai_messages.append({"role": "user", "content": f"OpenAI: '{user_message}'"})

    try:
        # Make the OpenAI API call
        chat_completion = openai.ChatCompletion.create(
            model="gpt-3.5", messages=openai_messages, temperature=0.3
        )
        openai_model_response = chat_completion.choices[0].message["content"]
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        openai_model_response = "Error with OpenAI response."
        
    # Store the response in the history for future use
    history_openai.append({"role": "model", "content": openai_model_response})
    
    return {
        "user": user_message,
        "model": openai_model_response
    }

