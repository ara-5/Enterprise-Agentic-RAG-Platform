"""
app/agent.py
────────────
Corrective RAG Agent built with LangGraph 0.4+

Graph nodes
───────────
  route_query      → decides: use documents OR answer from LLM knowledge
  retrieve         → hybrid FAISS+BM25 search + cross-encoder rerank
  grade_documents  → LLM judges if retrieved chunks are relevant
  rewrite_query    → rewrites query if grading failed (Corrective RAG loop)
  generate         → produces final answer with citations
  web_search       → Tavily fallback when docs fail after rewrite

State machine
─────────────
  route_query
      ├─→ "documents"  → retrieve → grade_documents
      │                     ├─→ "generate"     → generate → END
      │                     └─→ "rewrite"      → rewrite_query → retrieve (loop, max 2x)
      │                           └─→ "web_search" (after 2 rewrites) → generate → END
      └─→ "llm_only"   → generate → END
"""

from __future__ import annotations
import os
from typing import TypedDict, List, Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from loguru import logger

from vectorstore.store import hybrid_search

load_dotenv()

# ── LLM (local Ollama — swap to any LangChain-compatible model) ───────────────
LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")   # change to mistral, qwen2.5, etc.
llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

# ── Web search fallback (free Tavily key at tavily.com) ──────────────────────
web_search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
)

MAX_REWRITES = 2   # prevent infinite loops


# ── Agent state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:      Annotated[List[BaseMessage], add_messages]   # full conversation history
    query:         str
    rewrite_count: int
    context:       List[dict]   # retrieved chunk dicts from hybrid_search
    web_results:   List[str]
    answer:        str


# ── Node helpers ─────────────────────────────────────────────────────────────

def _last_human_query(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return state.get("query", "")


# ── Nodes ────────────────────────────────────────────────────────────────────

def route_query(state: AgentState) -> AgentState:
    """
    Ask LLM whether the question needs document retrieval.
    Returns updated state; routing decision is in 'query' metadata.
    """
    query = _last_human_query(state)
    prompt = (
        "You are a routing assistant. Given the user question below, decide:\n"
        "- Reply 'documents' if the answer likely requires looking up specific documents.\n"
        "- Reply 'llm_only' if it's general knowledge you can answer without documents.\n\n"
        f"Question: {query}\n\nReply with exactly one word: documents OR llm_only"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    decision = response.content.strip().lower()
    if "documents" not in decision:
        decision = "llm_only"
    logger.info(f"[route_query] decision='{decision}' for query='{query[:60]}'")
    return {**state, "query": query, "_route": decision}


def retrieve(state: AgentState) -> AgentState:
    """Hybrid search → cross-encoder rerank → store chunks in state."""
    query   = state["query"]
    results = hybrid_search(query, top_k=5)
    logger.info(f"[retrieve] got {len(results)} chunks")
    return {**state, "context": results}


def grade_documents(state: AgentState) -> AgentState:
    """
    LLM grades each retrieved chunk for relevance.
    Marks state with '_grade': 'generate' or 'rewrite'.
    """
    query   = state["query"]
    context = state["context"]

    relevant = []
    for chunk in context:
        prompt = (
            f"Question: {query}\n\n"
            f"Document chunk:\n{chunk['text']}\n\n"
            "Is this chunk relevant to answering the question? Reply yes or no."
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        if "yes" in resp.content.lower():
            relevant.append(chunk)

    logger.info(f"[grade_documents] {len(relevant)}/{len(context)} chunks relevant")

    if relevant:
        return {**state, "context": relevant, "_grade": "generate"}
    else:
        return {**state, "_grade": "rewrite"}


def rewrite_query(state: AgentState) -> AgentState:
    """Corrective RAG: rewrite the query to improve retrieval."""
    query = state["query"]
    prompt = (
        f"The following question didn't retrieve useful documents:\n{query}\n\n"
        "Rewrite it to be more specific and likely to match technical documentation. "
        "Return only the rewritten question, nothing else."
    )
    response      = llm.invoke([HumanMessage(content=prompt)])
    new_query     = response.content.strip()
    rewrite_count = state.get("rewrite_count", 0) + 1
    logger.info(f"[rewrite_query] attempt {rewrite_count}: '{new_query[:80]}'")
    return {**state, "query": new_query, "rewrite_count": rewrite_count}


def web_search(state: AgentState) -> AgentState:
    """Tavily web search as last-resort fallback."""
    query   = state["query"]
    results = web_search_tool.invoke(query)
    snippets = [r["content"] for r in results if "content" in r]
    logger.info(f"[web_search] got {len(snippets)} web results")
    return {**state, "web_results": snippets}


def generate(state: AgentState) -> AgentState:
    """
    Final generation node — synthesises answer from context + conversation history.
    Adds citations (source + page) when using document context.
    """
    query       = state["query"]
    context     = state.get("context", [])
    web_results = state.get("web_results", [])
    history     = state.get("messages", [])

    # Build context block
    if context:
        ctx_block = "\n\n".join(
            f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
            for c in context
        )
        source_note = "Cite sources as [Source, Page X] in your answer."
    elif web_results:
        ctx_block   = "\n\n".join(web_results)
        source_note = "These results are from the web."
    else:
        ctx_block   = ""
        source_note = "Answer from your general knowledge."

    # Conversation history (last 6 turns for context window efficiency)
    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-6:]
    )

    system = (
        "You are a precise, helpful document assistant. "
        "Answer only from the provided context. "
        "If the context is insufficient, say so clearly. "
        f"{source_note}"
    )
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}"
    )
    response = llm.invoke([
        HumanMessage(content=f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}")
    ])
    answer = response.content
    logger.info(f"[generate] answer length={len(answer)} chars")
    return {
        **state,
        "answer":   answer,
        "messages": state["messages"] + [AIMessage(content=answer)],
    }


# ── Conditional edges ─────────────────────────────────────────────────────────

def route_after_routing(state: AgentState) -> Literal["retrieve", "generate"]:
    return "retrieve" if state.get("_route") == "documents" else "generate"


def route_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "web_search"]:
    grade = state.get("_grade", "generate")
    if grade == "generate":
        return "generate"
    # After MAX_REWRITES failed attempts, fall through to web search
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        return "web_search"
    return "rewrite_query"


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("route_query",     route_query)
    g.add_node("retrieve",        retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query",   rewrite_query)
    g.add_node("web_search",      web_search)
    g.add_node("generate",        generate)

    g.set_entry_point("route_query")

    g.add_conditional_edges("route_query",     route_after_routing,
                            {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve",         "grade_documents")
    g.add_conditional_edges("grade_documents", route_after_grading,
                            {"generate": "generate",
                             "rewrite_query": "rewrite_query",
                             "web_search": "web_search"})
    g.add_edge("rewrite_query",    "retrieve")
    g.add_edge("web_search",       "generate")
    g.add_edge("generate",         END)

    return g.compile()


# Singleton compiled graph
rag_agent = build_graph()


# ── Public interface ──────────────────────────────────────────────────────────

def ask(query: str, history: List[BaseMessage] | None = None) -> dict:
    """
    Main entry point. Call from FastAPI or Streamlit.
    Returns {"answer": str, "sources": list, "rewrite_count": int}
    """
    messages = (history or []) + [HumanMessage(content=query)]
    initial_state: AgentState = {
        "messages":      messages,
        "query":         query,
        "rewrite_count": 0,
        "context":       [],
        "web_results":   [],
        "answer":        "",
    }
    final_state = rag_agent.invoke(initial_state)
    return {
        "answer":        final_state["answer"],
        "sources":       final_state.get("context", []),
        "rewrite_count": final_state.get("rewrite_count", 0),
        "used_web":      bool(final_state.get("web_results")),
    }
