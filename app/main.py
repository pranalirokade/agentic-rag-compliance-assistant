"""FastAPI service.  INDEX_DIR=index LLM=openai RERANKER=cross-encoder uvicorn app.main:app"""
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.agent import ask
from rag.cli import load_context

app = FastAPI(title="Agentic RAG Compliance Assistant", version="1.0.0")


@lru_cache(maxsize=1)
def ctx():
    try:
        return load_context(os.getenv("INDEX_DIR", "index"), os.getenv("LLM", "extractive"),
                            os.getenv("RERANKER", "lexical"))
    except FileNotFoundError:
        raise HTTPException(503, "Index not found. Run: python -m rag.cli ingest")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(q: Question):
    return ask(q.question, ctx())
