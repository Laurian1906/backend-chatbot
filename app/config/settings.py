from app.constants import OPENAI_API_KEY, GEMINI_API_KEY

generation_config = {
    "temperature": 0.45,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

OPENAI_API_KEY = OPENAI_API_KEY
GEMINI_API_KEY = GEMINI_API_KEY