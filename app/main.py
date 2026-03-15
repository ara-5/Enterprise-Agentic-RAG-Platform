"""
app/main.py
───────────
Streamlit UI — calls the FastAPI backend at localhost:8000.
Run separately:  streamlit run app/main.py
(Start the API first: uvicorn app.api:app --port 8000)
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Agentic RAG Document Assistant")
st.caption("Hybrid retrieval · Cross-encoder reranking · Corrective RAG · Web fallback")

# ── Sidebar — upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")
    if uploaded and st.button("Index Document"):
        with st.spinner("Ingesting and indexing …"):
            resp = httpx.post(
                f"{API_BASE}/ingest",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=120,
            )
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Indexed {data['chunks']} chunks from {data['filename']}")
        else:
            st.error(f"Error: {resp.text}")

    st.divider()
    st.header("RAGAs Scores")
    if st.button("Load Latest Scores"):
        resp = httpx.get(f"{API_BASE}/metrics", timeout=10)
        scores = resp.json()
        if "detail" not in scores:
            for k, v in scores.items():
                st.metric(k.replace("_", " ").title(), f"{v:.3f}")
        else:
            st.info(scores["detail"])

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** — Page {s['page']}")
        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("Latency", f"{m['latency_ms']} ms")
            cols[1].metric("Rewrites", m['rewrite_count'])
            cols[2].metric("Web used", "Yes" if m['used_web'] else "No")

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/query",
                    json={
                        "question": prompt,
                        "history": [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[:-1]
                        ],
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown(data["answer"])

                    if data["sources"]:
                        with st.expander("📚 Sources"):
                            for s in data["sources"]:
                                st.markdown(f"- **{s['source']}** — Page {s['page']}")

                    cols = st.columns(3)
                    cols[0].metric("Latency",   f"{data['latency_ms']} ms")
                    cols[1].metric("Rewrites",  data["rewrite_count"])
                    cols[2].metric("Web used",  "Yes" if data["used_web"] else "No")

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                        "meta":    {
                            "latency_ms":    data["latency_ms"],
                            "rewrite_count": data["rewrite_count"],
                            "used_web":      data["used_web"],
                        },
                    })
                else:
                    err = resp.json().get("detail", resp.text)
                    st.error(f"API error: {err}")
            except httpx.ConnectError:
                st.error("Cannot reach API. Start it with: uvicorn app.api:app --port 8000")
