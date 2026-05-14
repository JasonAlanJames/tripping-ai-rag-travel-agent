from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat_service import generate_chat_response

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str


@router.post("/")
def chat(request: ChatRequest):
    return generate_chat_response(request)