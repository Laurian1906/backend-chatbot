from fastapi import FastAPI
from app.config.cors import setup_cors
from app.routes.gemini_routes import gemini_router
from app.routes.openai_routes import openai_router

app = FastAPI()

# Configure CORS
setup_cors(app)

app.include_router(gemini_router, prefix="/gemini")
app.include_router(openai_router, prefix="/openai")
