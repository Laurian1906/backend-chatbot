from fastapi import APIRouter
from app.services.openai_service import openai_chat

openai_router = APIRouter()

@openai_router.get("/")
def openai_chat_bot(user_message: str):
    return openai_chat(user_message)
