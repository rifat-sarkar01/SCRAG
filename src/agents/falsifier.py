"""
src/agents/falsifier.py
-----------------------
Falsification-verification agent for the Self-Correcting RAG system.

Overview
--------
This module adds an optional "devil's advocate" pass that deliberately tries to
surface evidence CONTRADICTING the draft answer, then decides whether the answer
should be kept, revised, or overturned in light of that counter-evidence.

Performance note
----------------
ENABLE_FALSIFICATION roughly doubles LLM calls per query:
  - generate_falsification_queries()  → 1 ChatOllama call
  - retrieval of counter-chunks       → embedding-only (fast, no LLM)
  - verify_against_counterevidence()  → 1 ChatOllama call  (FRESH context)
  Total overhead: ~2× latency on CPU/GPU compared to a baseline RAG run.
Cap: exactly 1 falsification round (no recursive falsification).

All traces are written to eval/falsification_log.jsonl for debugging, including
raw model output and a "low_confidence" flag whenever the verdict didn't parse
cleanly on the first try.

Public API
----------
    generate_falsification_queries(draft_answer, original_query) -> list[str]
    verify_against_counterevidence(draft_answer, supporting_chunks,
                                   counter_chunks) -> Literal["keep","revise","overturn"]
    ENABLE_FALSIFICATION: bool  (read from env var, default False)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q8_0")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Gate flag: set ENABLE_FALSIFICATION=true in .env or environment to activate.
# Default is False — the node is a no-op unless explicitly opted in.
ENABLE_FALSIFICATION: bool = (
    os.environ.get("ENABLE_FALSIFICATION", "false").strip().lower() == "true"
)

# Path for JSONL trace log.  Created on first write if it doesn't exist.
_LOG_PATH = Path("eval") / "falsification_log.jsonl"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_llm(temperature: float = 0.0) -> ChatOllama:
    """Instantiates a ChatOllama client using the same pattern as graders.py."""
    return ChatOllama(
        model=_DEFAULT_MODEL,
        base_url=_OLLAMA_BASE_URL,
        temperature=temperature,
    )


def _write_trace(record: dict) -> None:
    """
    Appends *record* as a single JSON line to the falsification log.

    Silently skips if the log directory cannot be created (e.g., read-only FS),
    logging a warning instead of crashing the main pipeline.
    """
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("falsifier: could not write trace to %s: %s", _LOG_PATH, exc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# System prompt literals
# ---------------------------------------------------------------------------

_FALSIFICATION_QUERY_SYSTEM = """\
You are a critical-thinking research assistant tasked with STRESS-TESTING an answer.

Your goal: generate search queries specifically designed to find evidence that would
CONTRADICT or UNDERMINE the draft answer provided.

Rules:
- Produce exactly 2 to 3 search queries.
- Each query must target a different potential weakness, counterexample, or
  contradicting fact relative to the draft answer.
- Do NOT generate queries that would find supporting evidence — only queries that
  might surface contradictions, exceptions, or disconfirming data.
- Queries must be short (≤ 15 words), concrete, and retrieval-friendly (avoid
  question marks; use keyword-style phrasing).
- Return ONLY a JSON array of strings — no preamble, no explanation, no markdown.

Example output format:
["query one here", "query two here", "query three here"]"""


_VERIFICATION_SYSTEM = """\
You are a strict evidence arbitrator. Your task is to decide whether a DRAFT ANSWER
should be kept, revised, or overturned, given both supporting and counter evidence.

IMPORTANT: You are reading this as a completely fresh evaluation. Ignore any prior
conversation context. Base your decision solely on the material provided below.

Decision criteria — follow these steps in order:

STEP 1 — Assess counter-evidence strength:
  a. Does the COUNTER EVIDENCE contain factual claims that directly contradict a
     specific fact, number, name, causal claim, or conclusion in the DRAFT ANSWER?
  b. Is the counter-evidence from a source that would plausibly be more authoritative
     or more recent than the supporting evidence?

STEP 2 — Assess supporting evidence strength:
  a. Does the SUPPORTING EVIDENCE directly back up the key claims in the DRAFT ANSWER?
  b. Are there multiple independent supporting chunks, or just one?

STEP 3 — Apply the decision rule:
  - "overturn"  → The counter-evidence directly contradicts a CORE claim in the answer
                  AND the supporting evidence is weak or absent for that claim.
  - "revise"    → The counter-evidence reveals that the answer is INCOMPLETE, OVERLY
                  BROAD, or partially wrong, but the core claim is defensible.
  - "keep"      → The counter-evidence is tangential, does not directly contradict the
                  draft answer, or is clearly weaker than the supporting evidence.

STEP 4 — Output:
  Respond with EXACTLY ONE of these three words on its own line, nothing else:
  keep
  revise
  overturn"""


# ---------------------------------------------------------------------------
# 1. generate_falsification_queries
# ---------------------------------------------------------------------------


def generate_falsification_queries(
    draft_answer: str,
    original_query: str,
) -> list[str]:
    """
    Prompts the local Ollama model to produce 2-3 search queries designed to
    surface evidence that would CONTRADICT the draft answer.

    Strategy:
    - Primary path: ask the model with JSON format instruction, then parse the
      raw content as JSON (``json.loads``).
    - Fallback path: if JSON parsing fails, apply a regex to extract quoted
      strings from the response.  This handles models that add prose around the
      JSON array despite being told not to.

    A "low_confidence" flag is set in the trace whenever the primary JSON parse
    fails and the fallback is used — or when the fallback also returns fewer
    than 2 queries.

    Args:
        draft_answer:   The candidate answer generated by the RAG pipeline.
        original_query: The user's original natural-language query.

    Returns:
        A list of 2-3 query strings.  May be shorter if the model output was
        unparseable; in the worst case returns a single generic fallback query.
    """
    # Slight temperature for query diversity; not 0 so the three queries differ.
    llm = _build_llm(temperature=0.3)

    messages = [
        SystemMessage(content=_FALSIFICATION_QUERY_SYSTEM),
        HumanMessage(
            content=(
                f"ORIGINAL QUERY: {original_query}\n\n"
                f"DRAFT ANSWER:\n{draft_answer}\n\n"
                "Generate 2-3 falsification search queries as a JSON array of strings:"
            )
        ),
    ]

    raw_output: str = ""
    low_confidence: bool = False
    queries: list[str] = []

    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip() if response.content else ""

        # --- Primary path: direct JSON parse ---
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, list) and all(isinstance(q, str) for q in parsed):
                queries = [q.strip() for q in parsed if q.strip()]
            else:
                raise ValueError("Parsed JSON is not a flat list of strings.")
        except (json.JSONDecodeError, ValueError):
            # --- Fallback path: regex extraction of quoted strings ---
            low_confidence = True
            logger.debug(
                "falsifier: primary JSON parse failed; falling back to regex. "
                "raw=%r",
                raw_output[:300],
            )
            extracted = re.findall(r'"([^"]{5,})"', raw_output)
            queries = [q.strip() for q in extracted if q.strip()]

        # Clamp to max 3.
        queries = queries[:3]
        if len(queries) < 2:
            low_confidence = True

    except Exception as exc:  # noqa: BLE001
        logger.error("falsifier: generate_falsification_queries failed: %s", exc)
        low_confidence = True
        raw_output = str(exc)

    # Guarantee at least one query so downstream retrieval never receives an
    # empty list.
    if not queries:
        queries = [f"evidence against: {original_query[:120]}"]

    _write_trace({
        "event": "generate_falsification_queries",
        "timestamp": _utc_now(),
        "original_query": original_query,
        "draft_answer_preview": draft_answer[:200],
        "raw_model_output": raw_output,
        "parsed_queries": queries,
        "low_confidence": low_confidence,
    })

    return queries


# ---------------------------------------------------------------------------
# 2. verify_against_counterevidence
# ---------------------------------------------------------------------------

# Tokens the model may reasonably emit in place of the exact keyword.
# Maps alias → canonical verdict so we can survive minor prompt-following failures.
_VERDICT_ALIASES: dict[str, Literal["keep", "revise", "overturn"]] = {
    "keep": "keep",
    "kept": "keep",
    "maintain": "keep",
    "maintained": "keep",
    "revise": "revise",
    "revised": "revise",
    "update": "revise",
    "updated": "revise",
    "modify": "revise",
    "modified": "revise",
    "partial": "revise",
    "overturn": "overturn",
    "overturned": "overturn",
    "reject": "overturn",
    "rejected": "overturn",
    "wrong": "overturn",
    "incorrect": "overturn",
    "contradict": "overturn",
    "contradicted": "overturn",
}


def _parse_verdict(
    raw: str,
) -> tuple[Literal["keep", "revise", "overturn"], bool]:
    """
    Parses the model's raw output into a canonical verdict.

    Returns:
        (verdict, low_confidence)

    ``low_confidence`` is True when the exact expected token ("keep", "revise",
    or "overturn") does not appear on the very first line, meaning we had to
    fall back to alias scanning.  This surfaces genuine prompt-following failures
    or model uncertainty in the trace log.
    """
    first_line = raw.strip().splitlines()[0].strip().lower() if raw.strip() else ""

    # Ideal path: first line is exactly one of the three keywords.
    if first_line in ("keep", "revise", "overturn"):
        return first_line, False  # type: ignore[return-value]

    # Alias scan across the full response text.
    normalized = raw.lower()
    for token, verdict in _VERDICT_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            return verdict, True

    # Truly unparseable — default to "keep" (conservative: don't discard the
    # answer when the model is unable to render a coherent judgment).
    return "keep", True


def verify_against_counterevidence(
    draft_answer: str,
    supporting_chunks: List[str],
    counter_chunks: List[str],
) -> Literal["keep", "revise", "overturn"]:
    """
    Judges whether the draft answer should be kept, revised, or overturned after
    weighing both supporting and counter evidence.

    Design notes for 8B local models
    ----------------------------------
    - A FRESH LLM instance is created for each call so there is zero context
      leakage from the generation conversation.  8B models are sensitive to
      conversation history and tend to "agree" with prior turns rather than
      reason independently — using a fresh instance eliminates that bias.
    - The system prompt (``_VERIFICATION_SYSTEM``) spells out all decision
      criteria step-by-step rather than relying on the model to infer them.
      Smaller models perform significantly worse on open-ended comparative
      reasoning when the criteria are implicit.
    - temperature=0.0 for maximum determinism on a structured verdict.

    Args:
        draft_answer:      The candidate answer to evaluate.
        supporting_chunks: Context chunks used to produce the answer.
        counter_chunks:    Chunks retrieved by the falsification queries;
                           potentially contradicting evidence.

    Returns:
        One of: ``"keep"`` | ``"revise"`` | ``"overturn"``
    """
    def _numbered_block(chunks: List[str], label: str) -> str:
        if not chunks:
            return f"{label}:\n[none retrieved]"
        lines = "\n".join(
            f"[{i + 1}] {chunk[:400]}{'…' if len(chunk) > 400 else ''}"
            for i, chunk in enumerate(chunks)
        )
        return f"{label}:\n{lines}"

    supporting_block = _numbered_block(supporting_chunks, "SUPPORTING EVIDENCE")
    counter_block = _numbered_block(counter_chunks, "COUNTER EVIDENCE")

    # Fresh LLM instance — deliberately NOT reusing any cached chain or instance
    # from the main generation pipeline.
    llm = _build_llm(temperature=0.0)

    messages = [
        SystemMessage(content=_VERIFICATION_SYSTEM),
        HumanMessage(
            content=(
                f"DRAFT ANSWER:\n{draft_answer}\n\n"
                f"{supporting_block}\n\n"
                f"{counter_block}\n\n"
                "Based on the decision criteria above, output exactly one word:\n"
                "keep, revise, or overturn"
            )
        ),
    ]

    raw_output: str = ""
    verdict: Literal["keep", "revise", "overturn"] = "keep"
    low_confidence: bool = False

    try:
        response = llm.invoke(messages)
        raw_output = response.content.strip() if response.content else ""
        verdict, low_confidence = _parse_verdict(raw_output)

    except Exception as exc:  # noqa: BLE001
        logger.error("falsifier: verify_against_counterevidence failed: %s", exc)
        raw_output = str(exc)
        low_confidence = True
        verdict = "keep"  # conservative default on error

    _write_trace({
        "event": "verify_against_counterevidence",
        "timestamp": _utc_now(),
        "draft_answer_preview": draft_answer[:200],
        "num_supporting_chunks": len(supporting_chunks),
        "num_counter_chunks": len(counter_chunks),
        "raw_model_output": raw_output,
        "verdict": verdict,
        "low_confidence": low_confidence,
    })

    if low_confidence:
        logger.warning(
            "falsifier: low_confidence verdict=%r — model output did not cleanly "
            "parse into keep/revise/overturn on first try. raw=%r",
            verdict,
            raw_output[:200],
        )

    return verdict
