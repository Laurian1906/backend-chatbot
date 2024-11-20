import google.generativeai as genai
from app.config.settings import generation_config
from app.config.settings import GEMINI_API_KEY

history_gemini = []

def gemini_chat(user_message: str):
    genai.configure(api_key=GEMINI_API_KEY)
    history_gemini.append({"role": "user", "parts": user_message})
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-002",
        generation_config=generation_config
    )
    
    gemini_full_message = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in history_gemini])
    
    try:
        chat_session = model.start_chat(history=history_gemini)
        response = chat_session.send_message(gemini_full_message + "Please format your text like you will do in a div element in HTML with <p>,<ul>,<ol>,<li>,<b>,<i>,<u> and so on, when it is necessary to format. Do not put ```html at the beggining of the response or ``` at the end of the reponse")
        gemini_model_response = response.text
        history_gemini.append({"role": "model", "parts": gemini_model_response})
        
        return {"user": user_message, "model": gemini_model_response}
    except Exception as e:
        return {"user": user_message, "model": f"Error with Gemini response: {e}"}
    