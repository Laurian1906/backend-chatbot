# Import libraries
import openai
import google.generativeai as genai
import constants
from fastapi.middleware.cors import CORSMiddleware

from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

class APIRequest(BaseModel):
    user_message: str

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

# Create the model
generation_config = {
  "temperature": 0.45,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

history_gemini = []
history_openai = []

@app.get("/gemini")
def gemini_chat_bot(user_message: str):
    print("I got the request, yay!")
    
    # Saving the user message
    history_gemini.append({"role": "user", "parts": user_message})

    # Select model of Gemini
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config
    )

    gemini_full_message = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in history_gemini])

    try:
        print("I am now generating the response")
        
        chat_session = model.start_chat(
            history=history_gemini
        )
        
        # Generating the response
        response = chat_session.send_message(gemini_full_message + "If necessary format text with HTML as you would do in a div, put simply use <p> tags and <ul>, <li>, <b>, <i>, <u>. Do not use text formatting, just HTML, do not put ```html at the beggining of the response and ``` at the end of the response")
        gemini_model_response = response.text
        
        history_gemini.append({"role": "model", "parts": gemini_model_response})
        
        print("I am now printing the response...")
        return {
            "user": user_message,
            "model": gemini_model_response
        }
        
    except Exception as e:
        gemini_model_response = f"Error with Gemini response: {e}"
        history_gemini.append({"role": "model", "parts": gemini_model_response})
    
        return {
            "user": user_message,
            "model": gemini_model_response
        }
    
@app.get("/openai")
def openai_chat_bot(user_message: str):
    # Initialize conversation with at least one message
    openai_messages = [{"role": "system", "content": "DO NOT ROLEPLAY!! BE PROFESSIONAL"}]

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
            model="gpt-3.5", messages=openai_messages, temperature=0.5
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

