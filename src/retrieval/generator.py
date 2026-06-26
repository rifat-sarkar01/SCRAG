"""
src/retrieval/generator.py
---------------------------
Context-grounded answer generation using a local Ollama model via langchain-ollama.

The generator is strictly instructed to answer ONLY from the provided context
chunks.  If the context is empty or insufficient it says so explicitly rather
than hallucinating.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q8_0")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

_GENERATOR_SYSTEM = """\
You are a precise question-answering assistant operating inside a RAG pipeline.

Rules:
- Answer the QUESTION using ONLY information present in the CONTEXT below.
- If the context does not contain enough information, respond exactly with:
  "I cannot answer this based on the provided context."
- Do NOT add facts, opinions, or world knowledge that are absent from the context.
- Be concise and directly responsive to the question asked.
- Do NOT reference the context explicitly (e.g., avoid "According to the context…").\
"""

# ---------------------------------------------------------------------------
# Lazy LLM singleton
# ---------------------------------------------------------------------------

_llm: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    """Returns a cached ChatOllama instance, creating it on first call."""
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=_DEFAULT_MODEL,
            base_url=_OLLAMA_BASE_URL,
            temperature=0.0,
        )
    return _llm


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generates a grounded answer to *query* using *chunks* as the sole context.

    Args:
        query:  The user's natural-language question.
        chunks: List of context chunk dicts.  Each must contain at minimum a
                ``"text"`` key.  Extra fields (source, score, …) are ignored.

    Returns:
        The generated answer string (stripped of leading/trailing whitespace).
        Returns the "cannot answer" fallback if *chunks* is empty.
    """
    if not chunks:
        return "I cannot answer this based on the provided context."

    # Concatenate chunk texts, separated by a clear delimiter.
    chunk_texts = [
        c["text"] if isinstance(c, dict) else str(c) for c in chunks
    ]
    context = "\n\n---\n\n".join(chunk_texts)

    messages = [
        SystemMessage(content=_GENERATOR_SYSTEM),
        HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {query}"),
    ]

    response = _get_llm().invoke(messages)
    return response.content.strip()
