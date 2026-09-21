"""
app.py
------
Streamlit web chatbot for the Self-Correcting RAG system.

Launch with:
    streamlit run app.py

Environment variables (set in .env):
    OLLAMA_BASE_URL       — Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL          — generation model tag
    OLLAMA_JUDGE_MODEL    — grading/correction model tag
    ENABLE_FALSIFICATION  — set "true" to enable devil's-advocate pass
"""

from __future__ import annotations

import time
import sys
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — MUST be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SCRAG — Self-Correcting RAG",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode glassmorphism design
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ─── Base ─────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ─── Sidebar ───────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * { color: #c8d6e5 !important; }

    /* ─── Header ────────────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
    }
    .app-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #8899aa;
        font-size: 0.95rem;
        margin: 0.4rem 0 0;
    }

    /* ─── Chat messages ─────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.25rem;
        margin: 0.4rem 0;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: rgba(102, 126, 234, 0.12);
    }
    [data-testid="stChatMessage"] p {
        font-size: 0.97rem;
        line-height: 1.65;
    }

    /* ─── Expanders ─────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        margin-top: 0.5rem;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.85rem;
        color: #8899bb;
        font-weight: 500;
    }
    [data-testid="stExpander"] summary:hover { color: #aabbdd; }

    /* ─── Stat pills ────────────────────────────────────────────── */
    .stat-pills {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 0.6rem 0 0.2rem;
    }
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 500;
        background: rgba(102,126,234,0.15);
        border: 1px solid rgba(102,126,234,0.3);
        color: #aabbff;
    }
    .stat-pill.green {
        background: rgba(52,199,89,0.12);
        border-color: rgba(52,199,89,0.3);
        color: #6de098;
    }
    .stat-pill.amber {
        background: rgba(255,159,67,0.12);
        border-color: rgba(255,159,67,0.3);
        color: #ffbf6e;
    }
    .stat-pill.red {
        background: rgba(255,69,58,0.12);
        border-color: rgba(255,69,58,0.3);
        color: #ff8a80;
    }

    /* ─── Trace nodes ───────────────────────────────────────────── */
    .trace-node {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.83rem;
    }
    .trace-node:last-child { border-bottom: none; }
    .trace-badge {
        flex-shrink: 0;
        padding: 1px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        background: rgba(102,126,234,0.2);
        color: #99aaff;
    }
    .trace-decision { color: #99aabb; line-height: 1.45; }

    /* ─── Source chunks ─────────────────────────────────────────── */
    .source-chunk {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid rgba(102,126,234,0.5);
        border-radius: 0 8px 8px 0;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #99aabb;
        line-height: 1.55;
    }

    /* ─── Chat input ────────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        border-radius: 14px;
        background: rgba(255,255,255,0.06);
        border-color: rgba(102,126,234,0.4);
        color: #e0e8f0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(102,126,234,0.8);
        box-shadow: 0 0 0 2px rgba(102,126,234,0.2);
    }

    /* ─── Divider ───────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.06); }

    /* ─── Spinner ───────────────────────────────────────────────── */
    [data-testid="stStatusWidget"] { display: none; }

    /* ─── Scrollbar ─────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(102,126,234,0.3); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Lazy pipeline loader (cached for the session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="🔄 Loading RAG pipeline… (first run only)")
def _load_pipeline():
    """
    Imports and warms up the RAG pipeline. Called once per Streamlit session
    (cache_resource persists across reruns). The EmbeddingIndex is also built
    here on first call because _get_index() is called lazily inside
    run_corrected_query — explicitly triggering it here gives us a clean
    spinner instead of a silent delay on the first query.
    """
    from src.graph.correction_graph import run_corrected_query, _get_index
    _get_index()  # warm up embedding model + FAISS index
    return run_corrected_query


# ---------------------------------------------------------------------------
# Node label → display icon mapping
# ---------------------------------------------------------------------------

_NODE_ICONS: dict[str, str] = {
    "retrieve":            "🔍",
    "grade_chunks":        "⚖️",
    "rewrite_query":       "✏️",
    "generate":            "✨",
    "grade_groundedness":  "🔬",
    "regenerate":          "🔄",
    "grade_usefulness":    "✅",
    "falsify":             "🧪",
}

# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------


def _render_stat_pills(correction_rounds: int, reretrieval: bool) -> None:
    pills_html = '<div class="stat-pills">'
    if correction_rounds == 0:
        pills_html += '<span class="stat-pill green">✓ No corrections needed</span>'
    else:
        pills_html += (
            f'<span class="stat-pill amber">'
            f'🔄 {correction_rounds} correction{"s" if correction_rounds > 1 else ""}'
            f'</span>'
        )
    if reretrieval:
        pills_html += '<span class="stat-pill amber">🔁 Re-retrieval ran</span>'
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)


def _render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.caption("No trace available.")
        return
    html = ""
    for entry in trace:
        node = entry.get("node", "?")
        decision = entry.get("decision", "")
        icon = _NODE_ICONS.get(node, "▸")
        html += (
            f'<div class="trace-node">'
            f'<span class="trace-badge">{icon} {node}</span>'
            f'<span class="trace-decision">{decision}</span>'
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def _render_sources(chunks: list[str]) -> None:
    if not chunks:
        st.caption("No source chunks recorded.")
        return
    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:400].replace("\n", " ").strip()
        if len(chunk) > 400:
            preview += "…"
        st.markdown(
            f'<div class="source-chunk"><strong>[{i}]</strong> {preview}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🧠 SCRAG")
        st.markdown(
            "**Self-Correcting RAG** uses a LangGraph state machine to "
            "retrieve documents, verify answer groundedness, and fix "
            "unsupported claims automatically."
        )
        st.divider()
        st.markdown("#### Pipeline stages")
        stages = [
            ("🔍", "Retrieve", "Vector similarity search (FAISS)"),
            ("⚖️", "Grade Chunks", "Relevance filter"),
            ("✏️", "Rewrite Query", "Auto-reformulation on failure"),
            ("✨", "Generate", "Ollama LLM answer"),
            ("🔬", "Groundedness", "Claim-level faithfulness check"),
            ("🔄", "Correct", "Targeted claim patching"),
            ("✅", "Usefulness", "Final answer quality check"),
            ("🧪", "Falsify", "Devil's-advocate round (optional)"),
        ]
        for icon, name, desc in stages:
            st.markdown(f"{icon} **{name}** — {desc}")
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown(
            "<small style='color:#556'>For CLI use: `python chat.py --verbose`</small>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    _render_sidebar()

    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>🧠 Self-Correcting RAG</h1>
            <p>Ask anything about the loaded documents — the pipeline will retrieve,
            verify, and correct its own answers automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Session state ────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[dict]: role, content, meta

    # ── Pipeline loading ─────────────────────────────────────────────────────
    try:
        run_corrected_query = _load_pipeline()
    except Exception as exc:
        st.error(
            f"**Failed to load the RAG pipeline.**\n\n"
            f"```\n{exc}\n```\n\n"
            "Make sure Ollama is running and your `.env` is configured correctly.",
            icon="❌",
        )
        st.info(
            "1. Start Ollama: `ollama serve`\n"
            "2. Pull models: `ollama pull qwen3-vl:8b-instruct-q8_0` and "
            "`ollama pull qwen2.5:14b-instruct`\n"
            "3. Check your `.env` file matches `.env.example`",
        )
        return

    # ── Replay chat history ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                _render_stat_pills(
                    meta.get("correction_rounds", 0),
                    meta.get("reretrieval_happened", False),
                )
                col1, col2 = st.columns(2)
                with col1:
                    with st.expander("🔍 Decision trace"):
                        _render_trace(meta.get("trace", []))
                with col2:
                    with st.expander("📄 Source chunks"):
                        _render_sources(meta.get("correction_context", []))

    # ── Chat input ───────────────────────────────────────────────────────────
    query = st.chat_input("Ask a question about the documents…")

    if query:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Assistant response
        with st.chat_message("assistant"):
            status_container = st.empty()
            answer_container = st.empty()

            # Animated status messages while the pipeline runs
            _pipeline_stages = [
                "🔍 Retrieving relevant chunks…",
                "⚖️ Grading chunk relevance…",
                "✨ Generating answer…",
                "🔬 Checking groundedness…",
                "✅ Evaluating usefulness…",
            ]
            stage_idx = 0

            with st.spinner(""):
                # We can't truly stream LangGraph progress, so we animate the
                # status text in a thread-safe way using a container.
                import threading

                _result: dict[str, Any] = {}
                _error: list[Exception] = []

                def _run_pipeline() -> None:
                    try:
                        _result.update(run_corrected_query(query))
                    except Exception as e:  # noqa: BLE001
                        _error.append(e)

                thread = threading.Thread(target=_run_pipeline, daemon=True)
                thread.start()

                while thread.is_alive():
                    status_container.markdown(
                        f"<small style='color:#8899aa'>{_pipeline_stages[stage_idx % len(_pipeline_stages)]}</small>",
                        unsafe_allow_html=True,
                    )
                    stage_idx += 1
                    time.sleep(1.5)
                thread.join()

            status_container.empty()

            if _error:
                err = _error[0]
                answer_container.error(
                    f"**Pipeline error:** {err}\n\nCheck that Ollama is running.",
                    icon="❌",
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {err}",
                })
                st.stop()

            final_answer = _result.get("final_answer", "").strip()
            if not final_answer:
                final_answer = "I cannot answer this based on the provided context."

            # Typewriter-style reveal
            displayed = ""
            for char in final_answer:
                displayed += char
                answer_container.markdown(displayed + "▌")
                time.sleep(0.004)
            answer_container.markdown(final_answer)

            # Stats + collapsible panels
            meta = {
                "correction_rounds":   _result.get("correction_rounds", 0),
                "reretrieval_happened": _result.get("reretrieval_happened", False),
                "trace":               _result.get("trace", []),
                "correction_context":  _result.get("correction_context", []),
            }
            _render_stat_pills(meta["correction_rounds"], meta["reretrieval_happened"])
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("🔍 Decision trace"):
                    _render_trace(meta["trace"])
            with col2:
                with st.expander("📄 Source chunks"):
                    _render_sources(meta["correction_context"])

        # Persist to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "meta": meta,
        })


if __name__ == "__main__":
    main()
