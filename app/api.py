"""
app/api.py
──────────
FastAPI production API — wraps the LangGraph agent.

Endpoints
─────────
  POST /ingest        upload a PDF and index it
  POST /query         ask a question, get an answer + sources
  GET  /health        liveness probe (required for cloud deploy)
  GET  /metrics       RAGAs evaluation scores (if run)

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.agent import ask
from ingestion.ingest import load_pdfs
from vectorstore.store import build_index


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Document Assistant",
    description=(
        "Production RAG API with hybrid retrieval, "
        "cross-encoder reranking, LangGraph agentic loop, "
        "and Corrective RAG (query rewriting + web fallback)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR    = Path("data")
METRICS_FILE = Path("evaluation/latest_scores.json")
DATA_DIR.mkdir(exist_ok=True)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    history:    Optional[List[dict]] = None   # [{"role": "user"|"assistant", "content": "..."}]


class SourceItem(BaseModel):
    text:   str
    source: str
    page:   int


class QueryResponse(BaseModel):
    answer:        str
    sources:       List[SourceItem]
    rewrite_count: int
    used_web:      bool
    latency_ms:    int


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — required for Docker/cloud deploy."""
    return {"status": "ok", "version": app.version}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF and add it to the vector index.
    Existing index is rebuilt to include the new document.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = DATA_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    logger.info(f"Saved uploaded file: {dest}")

    try:
        chunks, metas = load_pdfs(DATA_DIR)
        build_index(chunks, metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "status":    "indexed",
        "filename":  file.filename,
        "chunks":    len(chunks),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question. The LangGraph agent:
    1. Routes (documents vs LLM-only)
    2. Retrieves with hybrid FAISS+BM25
    3. Reranks with cross-encoder
    4. Grades relevance — rewrites query if needed (Corrective RAG)
    5. Falls back to web search if docs fail after max rewrites
    6. Generates a cited answer
    """
    t0 = time.time()
    try:
        result = ask(req.question)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No index found. Upload a PDF via POST /ingest first."
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = int((time.time() - t0) * 1000)
    logger.info(f"Query answered in {latency}ms | rewrites={result['rewrite_count']}")

    return QueryResponse(
        answer        = result["answer"],
        sources       = [SourceItem(**s) for s in result["sources"]],
        rewrite_count = result["rewrite_count"],
        used_web      = result["used_web"],
        latency_ms    = latency,
    )


@app.get("/metrics")
def metrics():
    """Return latest RAGAs evaluation scores if available."""
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text())
    return {"detail": "No evaluation scores yet. Run: python -m evaluation.evaluate"}
