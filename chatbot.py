# Import libraries
import google.generativeai as genai
import constants
from fastapi.middleware.cors import CORSMiddleware

from typing import Union
from fastapi import FastAPI

origins = [
    'http://localhost:3000',
]

# Config
#Gemini API
genai.configure(api_key=constants.GEMINI_API_KEY)

#FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],   
)

# Select model of Gemini
model = genai.GenerativeModel("gemini-1.5-flash")

message = 'Salut!' 

@app.get("/chat")
def chat_bot(user_message: str = message):

    # Saving the user message
    history = [{"role": "user", "parts": user_message}]

    # Generating the response
    response = model.generate_content(
        message,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
        ),
    )

    chat = model.start_chat(
        history=[ 
            {"role": "model", "parts": response.text},
        ] + [{"role": msg["role"], "parts": msg["parts"]} for msg in history]
    )

    model_response = chat.history[0].parts[-1].text #latest response in the chat

    return {
        "user": user_message,
        "model": model_response
    }




