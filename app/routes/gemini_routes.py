from fastapi import APIRouter
from app.services.gemini_service import gemini_chat

gemini_router = APIRouter()

@gemini_router.get("/")
def gemini_chat_bot(user_message: str):
    return gemini_chat(user_message)