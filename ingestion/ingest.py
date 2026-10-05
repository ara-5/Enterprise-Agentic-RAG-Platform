"""
ingestion/ingest.py
───────────────────
Loads PDF, DOCX, XLSX and PPTX files, chunks them with rich metadata
(page/slide/sheet, source), then builds the hybrid FAISS+BM25 index.

Drop your documents in the  data/  folder and call:
    python -m ingestion.ingest
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from ingestion.loaders import SUPPORTED_SUFFIXES, is_supported, load_document
from vectorstore.store import build_index


DATA_DIR     = Path("data")
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 150


def load_documents(data_dir: Path = DATA_DIR) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Loads all supported documents from data/ directory.
    Returns (chunks, metadatas).
    Each metadata dict: {"source": filename, "page": page_number}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[str]             = []
    all_metas:  List[Dict[str, Any]]  = []
    files = sorted(p for p in data_dir.glob("*") if p.is_file() and is_supported(p))

    if not files:
        logger.warning(f"No documents ({', '.join(sorted(SUPPORTED_SUFFIXES))}) found in {data_dir}.")
        return [], []

    for path in files:
        logger.info(f"Loading: {path.name}")
        for section in load_document(path):
            for chunk in splitter.split_text(section.text):
                if len(chunk.strip()) < 30:   # skip very short fragments
                    continue
                all_chunks.append(chunk)
                all_metas.append({
                    "source": path.name,
                    "page":   section.page,
                })

    logger.success(f"Loaded {len(all_chunks)} chunks from {len(files)} document(s)")
    return all_chunks, all_metas


def run_ingestion() -> None:
    chunks, metas = load_documents()
    if chunks:
        build_index(chunks, metas)
        logger.success("Ingestion complete. Index is ready.")


if __name__ == "__main__":
    run_ingestion()
