import openai
from app.config.settings import OPENAI_API_KEY

history_openai = []

def openai_chat(user_message: str):
    openai.api_key = OPENAI_API_KEY
    openai_messages = [{"role": "system", "content": "Be a good AI assistant"}]
    
    max_history_size = 4
    openai_messages += [{"role": msg["role"], "content": msg["content"]} for msg in history_openai]
    
    if len(openai_messages) > max_history_size:
        openai_messages = openai_messages[-max_history_size:]
    openai_messages.append({"role": "user", "content": user_message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5", messages=openai_messages, temperature=0.5
        )
        openai_model_response = response.choices[0].message["content"]
        history_openai.append({"role":"user", "content": openai_model_response})
        return {"user": user_message, "model": openai_model_response}
    except Exception as e:
        print(f"Error: {e}")
        raise
        