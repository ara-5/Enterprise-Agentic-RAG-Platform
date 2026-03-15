"""
ingestion/ingest.py
───────────────────
Loads PDFs, chunks them with rich metadata (page, source, section),
then builds the hybrid FAISS+BM25 index.

Drop your PDFs in the  data/  folder and call:
    python -m ingestion.ingest
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from vectorstore.store import build_index


DATA_DIR     = Path("data")
CHUNK_SIZE   = 800
CHUNK_OVERLAP = 150


def load_pdfs(data_dir: Path = DATA_DIR) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Loads all PDFs from data/ directory.
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
    pdf_files   = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDFs found in {data_dir}. Add files and re-run.")
        return [], []

    for pdf_path in pdf_files:
        logger.info(f"Loading: {pdf_path.name}")
        loader    = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        for doc in documents:
            page_num = doc.metadata.get("page", 0)
            chunks   = splitter.split_text(doc.page_content)
            for chunk in chunks:
                if len(chunk.strip()) < 30:   # skip very short fragments
                    continue
                all_chunks.append(chunk)
                all_metas.append({
                    "source": pdf_path.name,
                    "page":   page_num + 1,   # 1-indexed for display
                })

    logger.success(f"Loaded {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)")
    return all_chunks, all_metas


def run_ingestion() -> None:
    chunks, metas = load_pdfs()
    if chunks:
        build_index(chunks, metas)
        logger.success("Ingestion complete. Index is ready.")


if __name__ == "__main__":
    run_ingestion()
