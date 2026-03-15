"""
embeddings/embedder.py
──────────────────────
Upgrade from all-MiniLM-L6-v2  →  BAAI/bge-m3
  - State-of-the-art multilingual embedding model (2024/25)
  - Supports dense, sparse, and multi-vector retrieval
  - Ships a cross-encoder reranker for 2-stage retrieval
"""

from __future__ import annotations
from functools import lru_cache
from typing import List

from loguru import logger
from sentence_transformers import CrossEncoder, SentenceTransformer


EMBED_MODEL  = "BAAI/bge-m3"           # best open-source embedding 2025
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Singleton: loads once, reused across all requests."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Singleton cross-encoder reranker."""
    logger.info(f"Loading reranker: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings → list of float vectors."""
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]


def rerank(query: str, chunks: List[str], top_n: int = 5) -> List[str]:
    """
    2-stage reranking:
    1. Retrieve top-k by vector similarity  (done in vectorstore layer)
    2. Rerank with cross-encoder            (done here)
    Returns top_n chunks sorted by relevance score, highest first.
    """
    reranker = get_reranker()
    pairs    = [(query, chunk) for chunk in chunks]
    scores   = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    logger.debug(f"Reranker scores: {[round(s, 3) for s, _ in ranked[:top_n]]}")
    return [chunk for _, chunk in ranked[:top_n]]
