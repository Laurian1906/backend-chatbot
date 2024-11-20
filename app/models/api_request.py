from pydantic import BaseModel

class APIRequest(BaseModel):
    user_message: str