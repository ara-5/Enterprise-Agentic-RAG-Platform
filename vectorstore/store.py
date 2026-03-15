"""
vectorstore/store.py
────────────────────
Hybrid retrieval: dense FAISS  +  sparse BM25
Results fused with Reciprocal Rank Fusion (RRF).

This replaces your original FAISS-only search and gives significantly
better recall — especially for keyword-heavy queries.
"""

from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from embeddings.embedder import embed_texts, embed_query, rerank


STORE_DIR   = Path("vectorstore/index")
FAISS_PATH  = STORE_DIR / "faiss.index"
META_PATH   = STORE_DIR / "metadata.pkl"
BM25_PATH   = STORE_DIR / "bm25.pkl"

EMBED_DIM   = 1024   # BAAI/bge-m3 output dimension
TOP_K_FETCH = 20     # fetch more, rerank down to fewer
TOP_K_FINAL = 5      # chunks returned to the LLM
RRF_K       = 60     # standard RRF constant


# ── Build / persist ──────────────────────────────────────────────────────────

def build_index(chunks: List[str], metadatas: List[Dict[str, Any]]) -> None:
    """
    Embed chunks, build FAISS + BM25 indexes, save to disk.
    metadatas: list of dicts like {"source": "file.pdf", "page": 3}
    """
    STORE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building index for {len(chunks)} chunks …")
    vectors = np.array(embed_texts(chunks), dtype="float32")

    # FAISS flat L2 index (exact, no approximation — fine up to ~100k chunks)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_PATH))

    # BM25 sparse index
    tokenised = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenised)

    with open(META_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    logger.success(f"Index saved → {STORE_DIR}")


# ── Load ─────────────────────────────────────────────────────────────────────

def _load_index():
    if not FAISS_PATH.exists():
        raise FileNotFoundError("No FAISS index found. Run ingestion first.")
    index   = faiss.read_index(str(FAISS_PATH))
    with open(META_PATH, "rb") as f:
        store = pickle.load(f)
    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
    return index, store["chunks"], store["metadatas"], bm25


# ── Hybrid search ────────────────────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = TOP_K_FINAL) -> List[Dict[str, Any]]:
    """
    1. Dense FAISS search     → ranked list A
    2. Sparse BM25 search     → ranked list B
    3. Reciprocal Rank Fusion → merged list
    4. Cross-encoder rerank   → final top_k
    Returns list of {"text": ..., "source": ..., "page": ..., "score": ...}
    """
    index, chunks, metadatas, bm25 = _load_index()

    # ── Dense retrieval ──────────────────────────────────────
    q_vec   = np.array([embed_query(query)], dtype="float32")
    _, idxs = index.search(q_vec, TOP_K_FETCH)
    dense_ids = idxs[0].tolist()

    # ── Sparse BM25 retrieval ────────────────────────────────
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_ids  = np.argsort(bm25_scores)[::-1][:TOP_K_FETCH].tolist()

    # ── RRF fusion ───────────────────────────────────────────
    rrf: Dict[int, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(sparse_ids):
        rrf[doc_id] = rrf.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    fused_ids = sorted(rrf, key=rrf.get, reverse=True)[:TOP_K_FETCH]

    # ── Collect candidate chunks ─────────────────────────────
    candidate_chunks = [chunks[i]     for i in fused_ids]
    candidate_metas  = [metadatas[i]  for i in fused_ids]

    # ── Cross-encoder rerank ─────────────────────────────────
    reranked_texts = rerank(query, candidate_chunks, top_n=top_k)

    results = []
    for text in reranked_texts:
        idx  = candidate_chunks.index(text)
        meta = candidate_metas[idx]
        results.append({
            "text":   text,
            "source": meta.get("source", "unknown"),
            "page":   meta.get("page", 0),
        })

    logger.debug(f"Hybrid search returned {len(results)} chunks for: '{query[:60]}'")
    return results
