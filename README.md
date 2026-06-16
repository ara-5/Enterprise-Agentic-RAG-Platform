# 🤖 Enterprise Agentic RAG Platform

> **Production-grade Retrieval-Augmented Generation system** with a LangGraph agentic loop, hybrid BM25+FAISS retrieval, cross-encoder reranking, Corrective RAG (query rewriting), web search fallback, RAGAs evaluation pipeline, FastAPI backend, and Docker deployment.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![RAGAs](https://img.shields.io/badge/Evaluated-RAGAs-4CAF50?style=flat)](https://docs.ragas.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [RAGAs Evaluation Results](#-ragas-evaluation-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Running with Docker](#-running-with-docker)
- [CI/CD Pipeline](#-cicd-pipeline)

---

## 🔍 Overview

This project goes beyond a basic RAG chatbot. It implements a **full agentic decision loop** using LangGraph — the system reasons about what to do with each query rather than blindly retrieving and generating.

**Key capabilities:**
- **Hybrid retrieval** — combines dense FAISS vector search with sparse BM25 keyword search, fused via Reciprocal Rank Fusion (RRF) for significantly better recall than vector-only approaches
- **Cross-encoder reranking** — 2-stage retrieval: retrieve top-20 candidates, rerank to top-5 using `ms-marco-MiniLM` cross-encoder
- **Corrective RAG** — if retrieved documents score poorly on relevance grading, the agent automatically rewrites the query and retries (up to 2 times)
- **Web search fallback** — when documents fail after retries, the agent falls back to live Tavily web search
- **Conversation memory** — maintains full chat history across turns for coherent multi-turn Q&A
- **Quantified evaluation** — RAGAs metrics tracked per run with a CI quality gate

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│   route_query   │  ← Decides: use documents OR answer from LLM knowledge
└────────┬────────┘
         │
    ┌────┴──────┐
    │           │
    ▼           ▼
retrieve    generate (LLM-only path)
    │
    ▼
┌──────────────────────┐
│  Hybrid Search       │  ← FAISS dense + BM25 sparse + RRF fusion
│  Cross-encoder rerank│  ← ms-marco-MiniLM reranker
└──────────┬───────────┘
           │
           ▼
    grade_documents      ← LLM judges chunk relevance
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
 generate    rewrite_query  ← Corrective RAG: rewrites bad queries
                 │
                 ▼ (after 2 rewrites)
            web_search       ← Tavily fallback
                 │
                 ▼
             generate
                 │
                 ▼
            Final Answer + Citations [Source, Page X]
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent Orchestration | LangGraph 0.4 | Stateful agentic graph with conditional routing |
| Embedding Model | BAAI/bge-m3 | State-of-the-art multilingual embeddings (1024-dim) |
| Dense Retrieval | FAISS | Fast vector similarity search |
| Sparse Retrieval | BM25 (rank_bm25) | Keyword-based retrieval |
| Result Fusion | Reciprocal Rank Fusion | Combines dense + sparse rankings |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | 2-stage relevance reranking |
| LLM | Ollama (llama3.2) | Local, privacy-preserving inference |
| Web Fallback | Tavily Search API | Live web search when docs insufficient |
| API Layer | FastAPI | Production REST API with Pydantic validation |
| Frontend | Streamlit | Interactive chat UI |
| Evaluation | RAGAs | Faithfulness, relevancy, precision, recall metrics |
| Containerisation | Docker + docker-compose | Reproducible deployment |
| CI/CD | GitHub Actions | Automated evaluation on every push |
| Observability | Loguru | Structured logging with request tracing |

---

## 📊 RAGAs Evaluation Results

Evaluated on a 5-question test set using RAGAs metrics:

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | 0.91 | Answer grounded in retrieved context |
| **Answer Relevancy** | 0.88 | Answer addresses the question asked |
| **Context Precision** | 0.85 | Retrieved chunks are useful |
| **Context Recall** | 0.83 | All necessary facts were retrieved |

> Run `python -m evaluation.evaluate` to regenerate scores. CI pipeline fails automatically if faithfulness drops below 0.70.

---

## 📁 Project Structure

```
Genai-rag-agent/
│
├── app/
│   ├── agent.py          # LangGraph Corrective RAG agent graph
│   ├── api.py            # FastAPI endpoints (POST /query, POST /ingest, GET /health)
│   └── main.py           # Streamlit chat UI
│
├── embeddings/
│   └── embedder.py       # BGE-M3 embeddings + cross-encoder reranker
│
├── ingestion/
│   └── ingest.py         # PDF loading with page/source metadata extraction
│
├── vectorstore/
│   ├── store.py          # Hybrid BM25+FAISS search with RRF fusion
│   └── index/            # Persisted FAISS index (auto-generated)
│
├── evaluation/
│   └── evaluate.py       # RAGAs evaluation suite
│
├── data/                 # Drop PDFs here for indexing
│
├── .github/
│   └── workflows/
│       └── evaluate.yml  # CI: runs RAGAs on every push, enforces quality gate
│
├── Dockerfile            # Multi-stage production Docker image
├── docker-compose.yml    # Runs API + Streamlit together
├── .env.example          # Environment variable template
└── requirements.txt      # All dependencies pinned
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- A free [Tavily API key](https://tavily.com) (for web search fallback)

### 1. Clone the repository

```bash
git clone https://github.com/ara-5/Genai-rag-agent.git
cd Genai-rag-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your TAVILY_API_KEY
```

### 4. Pull the LLM model

```bash
ollama pull llama3.2
```

### 5. Add your PDFs and index them

```bash
# Drop PDF files into the data/ folder, then:
python -m ingestion.ingest
```

### 6. Start the API and UI

```bash
# Terminal 1 — start FastAPI backend
uvicorn app.api:app --port 8000 --reload

# Terminal 2 — start Streamlit UI
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📡 API Reference

### `GET /health`
Liveness probe. Returns `{"status": "ok"}`.

### `POST /ingest`
Upload a PDF and index it.
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@your_document.pdf"
```
Response:
```json
{"status": "indexed", "filename": "your_document.pdf", "chunks": 142}
```

### `POST /query`
Ask a question. Returns answer + cited sources + performance metadata.
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```
Response:
```json
{
  "answer": "The main findings are... [Source: report.pdf, Page 4]",
  "sources": [{"text": "...", "source": "report.pdf", "page": 4}],
  "rewrite_count": 0,
  "used_web": false,
  "latency_ms": 1243
}
```

### `GET /metrics`
Returns latest RAGAs evaluation scores.

---

## 🐳 Running with Docker

```bash
# Build and start both API + Streamlit
docker-compose up --build
```

- API available at: `http://localhost:8000`
- UI available at: `http://localhost:8501`
- API docs at: `http://localhost:8000/docs`

---

## ⚙️ CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. Installs dependencies
2. Pulls a lightweight Ollama model (`llama3.2:1b`)
3. Runs ingestion on a sample document
4. Executes the full RAGAs evaluation suite
5. **Fails the pipeline** if `faithfulness < 0.70`
6. Uploads scores as a build artifact

This ensures retrieval quality never silently degrades across commits.

---

## 🔮 Roadmap

- [ ] LLM fine-tuning with LoRA/QLoRA on domain-specific documents
- [ ] Multi-document cross-referencing
- [ ] Cloud deployment (GCP Cloud Run)
- [ ] Streaming responses via Server-Sent Events
- [ ] Support for DOCX, CSV, and web URL ingestion

---

## 👩‍💻 Author

**Athira Anil Kumar** — AI Engineer | Generative AI | LLM Systems  
📍 Sharjah, UAE  
🔗 [LinkedIn](https://www.linkedin.com/in/athira-a-k) · [GitHub](https://github.com/ara-5)  
📄 [IEEE Publication — Phishing Detection with DistilBERT & Explainable AI](https://ieeexplore.ieee.org/document/11407086)
