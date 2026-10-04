"""HTTP API for serving many users concurrently.

Run:  uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
The index is memory-mapped, so extra workers share one copy of the vectors.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import config
from rag import RAG

state = {}


@asynccontextmanager
async def lifespan(app):
    state["rag"] = RAG()
    yield
    state.clear()


app = FastAPI(title="RAG Teaching Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(config.TOP_K, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["rag"].index)}


@app.post("/search")
async def search(req: AskRequest):
    results = await run_in_threadpool(state["rag"].retrieve, req.question, req.top_k)
    return {"sources": results}


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        result = await run_in_threadpool(state["rag"].answer, req.question, req.top_k)
    except Exception as exc:  # Ollama down, model missing, timeout...
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}") from exc
    return {"answer": result["answer"], "sources": result["sources"]}
