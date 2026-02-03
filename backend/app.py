# backend/app.py
import os
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from rag_pipeline import (
    load_hybrid_retriever,
    load_reranker,
    load_chat_model,
    answer_question,
)

# Load env vars from backend/.env
load_dotenv(Path(__file__).parent / ".env", override=True)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize once at startup
    app.state.retriever = load_hybrid_retriever()
    app.state.reranker = load_reranker()
    app.state.chat_model = load_chat_model()
    yield
    # Cleanup (optional) - nothing required here

app = FastAPI(title="RAG API (Hybrid + Reranker)", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in req.history]

    return answer_question(
        question=req.message,
        history=history,
        retriever=app.state.retriever,
        reranker=app.state.reranker,
        chat_model=app.state.chat_model,
    )
