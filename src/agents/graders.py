"""
src/agents/graders.py
---------------------
Reflection and correction agents for the Self-Correcting RAG system.

Agents
------
RetrievalGrader         — (query, chunk)            → RetrievalGrading
AnswerGroundednessGrader — (answer, chunks)           → GroundednessGrading
AnswerUsefulnessGrader   — (query, answer)            → UsefulnessGrading
QueryRewriter            — (query, irrelevant_chunks) → str

All graders use the local Ollama model via langchain-ollama's ChatOllama with
with_structured_output (Pydantic models — no manual JSON parsing).  The Ollama
base URL and model tag are read from environment variables (loaded via
python-dotenv).
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q8_0")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class RetrievalGrading(BaseModel):
    """Structured output schema for retrieval relevance grading."""

    relevant: bool = Field(
        description=(
            "True if the chunk contributes useful signal for answering the query, "
            "even if it alone is insufficient to fully answer it."
        )
    )
    reason: str = Field(
        description="One concise sentence explaining the grading decision."
    )


class GroundednessGrading(BaseModel):
    """Structured output schema for answer groundedness grading."""

    grounded: bool = Field(
        description=(
            "True if every factual claim in the answer is directly supported "
            "by information present in the context chunks."
        )
    )
    unsupported_claims: List[str] = Field(
        description=(
            "Verbatim or near-verbatim quotes of claims that are not supported "
            "by any context chunk.  Empty list when grounded is True."
        )
    )


class UsefulnessGrading(BaseModel):
    """Structured output schema for answer usefulness grading."""

    useful: bool = Field(
        description=(
            "True if the answer genuinely and directly resolves the user's query."
        )
    )
    reason: str = Field(
        description="One concise sentence explaining the grading decision."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_llm(model: str = _DEFAULT_MODEL, temperature: float = 0.0) -> ChatOllama:
    """Instantiates a ChatOllama client.  Base URL is read from the environment."""
    return ChatOllama(
        model=model,
        base_url=_OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# System prompt literals (verbatim from the approved design document)
# ---------------------------------------------------------------------------

_RETRIEVAL_GRADER_SYSTEM = """\
You are a strict relevance judge for a retrieval-augmented generation (RAG) system.

Your task: decide whether the provided DOCUMENT CHUNK contains information that is
directly useful for answering the QUERY.

Rules:
- A chunk is RELEVANT if it contains facts, definitions, explanations, or data
  that the query is asking about — even partially.
- A chunk is NOT RELEVANT if it is on a related topic but does not address the
  specific question, or if it only shares surface keywords with the query.
- Do NOT consider whether the chunk alone is sufficient to answer the query;
  only judge whether it contributes useful signal.
- Do NOT hallucinate. Base your judgment solely on the text provided.

Respond ONLY in this exact JSON format, no other text:
{"relevant": <true|false>, "reason": "<one concise sentence>"}"""

_GROUNDEDNESS_GRADER_SYSTEM = """\
You are a rigorous fact-checking judge for a retrieval-augmented generation (RAG) system.

Your task: determine whether every factual claim in the ANSWER is directly supported
by information present in the CONTEXT CHUNKS.

Rules:
- A claim is SUPPORTED if the exact fact (or a logically equivalent statement)
  appears in at least one of the context chunks.
- A claim is UNSUPPORTED if it introduces facts, numbers, names, causal claims, or
  conclusions that are not present in any of the context chunks — even if the claim
  seems plausible or likely true.
- Common knowledge hedges (e.g., "water is wet") can be ignored only if they are
  truly non-substantive. When in doubt, flag it.
- Do NOT judge whether the answer is correct in the real world. Only judge whether
  it is grounded in the provided context.
- Reproduce unsupported claims verbatim or as short direct quotes.

Respond ONLY in this exact JSON format, no other text:
{"grounded": <true|false>, "unsupported_claims": [<"claim1">, <"claim2">, ...]}

If fully grounded, return an empty list: {"grounded": true, "unsupported_claims": []}"""

_USEFULNESS_GRADER_SYSTEM = """\
You are a quality-control judge for a retrieval-augmented generation (RAG) system.

Your task: decide whether the ANSWER genuinely and directly resolves the user's QUERY.

Rules:
- An answer is USEFUL if it:
    (a) addresses the specific question asked (not a related but different question),
    (b) provides actionable information, a clear explanation, or a direct response,
    (c) is not so vague, hedged, or incomplete that it would leave a user without
        the information they needed.
- An answer is NOT USEFUL if it:
    (a) answers a different question than the one asked,
    (b) only restates the question or says "I don't know,"
    (c) is so heavily caveated or generic that it provides no real guidance,
    (d) is factually responsive but critically incomplete (e.g., lists 1 of 5
        required steps and stops).
- Do NOT penalise appropriate uncertainty hedges (e.g., "consult a doctor") when
  those hedges are genuinely warranted by the domain.

Respond ONLY in this exact JSON format, no other text:
{"useful": <true|false>, "reason": "<one concise sentence>"}"""

_QUERY_REWRITER_SYSTEM = """\
You are a search query optimiser for a retrieval-augmented generation (RAG) system.

Context: a previous retrieval attempt returned document chunks that were judged
NOT relevant to the user's query.  Your job is to reformulate the query so that
a vector similarity search is more likely to return relevant documents.

Rules:
- Preserve the user's original intent completely — do not change what is being asked.
- Remove ambiguity: replace vague terms with more specific synonyms or technical vocabulary.
- Expand acronyms if doing so would help retrieval.
- You may refocus a complex multi-part question on its most retrieval-critical aspect.
- Do NOT add constraints or assumptions not present in the original query.
- Return ONLY the rewritten query string.  No preamble, no explanation, no quotes."""


# ---------------------------------------------------------------------------
# 1. RetrievalGrader
# ---------------------------------------------------------------------------


class RetrievalGrader:
    """
    Judges whether a retrieved document chunk is relevant to the user's query.

    Design principle ("partial contribution" rule): a chunk does **not** need to
    fully answer the query on its own — it only needs to contribute useful signal.
    This prevents over-pruning in multi-hop retrieval scenarios where evidence
    is spread across multiple chunks.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        """
        Initializes the grader and wires the structured-output chain.

        Args:
            model: Anthropic model identifier.  Defaults to Claude Sonnet.
        """
        llm = _build_llm(model)
        self._chain = llm.with_structured_output(RetrievalGrading)

    def grade(self, query: str, chunk: str) -> RetrievalGrading:
        """
        Grades whether *chunk* contributes useful signal for answering *query*.

        Args:
            query: The user's search query.
            chunk: Raw text content of the retrieved document chunk.

        Returns:
            ``RetrievalGrading`` with fields:
            - ``relevant`` (bool): True if the chunk is useful.
            - ``reason`` (str): One-sentence rationale.
        """
        messages = [
            SystemMessage(content=_RETRIEVAL_GRADER_SYSTEM),
            HumanMessage(content=f"QUERY: {query}\n\nDOCUMENT CHUNK:\n{chunk}"),
        ]
        return self._chain.invoke(messages)


# ---------------------------------------------------------------------------
# 2. AnswerGroundednessGrader
# ---------------------------------------------------------------------------


class AnswerGroundednessGrader:
    """
    Checks every factual claim in the generated answer against the retrieved chunks.

    Design principle: this is a **faithfulness** checker, not a hallucination
    detector.  A real-world true claim that does not appear in the context chunks
    must be flagged as unsupported — otherwise the grader cannot be used reliably
    to prevent the model from injecting out-of-context facts.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        """
        Initializes the grader and wires the structured-output chain.

        Args:
            model: Anthropic model identifier.  Defaults to Claude Sonnet.
        """
        llm = _build_llm(model)
        self._chain = llm.with_structured_output(GroundednessGrading)

    def grade(self, answer: str, chunks: List[str]) -> GroundednessGrading:
        """
        Checks every factual claim in *answer* against *chunks*.

        Args:
            answer: The LLM-generated answer to evaluate.
            chunks: The list of raw context chunk strings that were provided to
                    the generator (in the order they were retrieved).

        Returns:
            ``GroundednessGrading`` with fields:
            - ``grounded`` (bool): True if all claims are supported.
            - ``unsupported_claims`` (list[str]): Verbatim unsupported claims;
              empty when ``grounded`` is True.
        """
        numbered_chunks = "\n".join(
            f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks)
        )
        messages = [
            SystemMessage(content=_GROUNDEDNESS_GRADER_SYSTEM),
            HumanMessage(
                content=f"CONTEXT CHUNKS:\n{numbered_chunks}\n\nANSWER:\n{answer}"
            ),
        ]
        return self._chain.invoke(messages)


# ---------------------------------------------------------------------------
# 3. AnswerUsefulnessGrader
# ---------------------------------------------------------------------------


class AnswerUsefulnessGrader:
    """
    Grades whether the generated answer genuinely resolves the user's query.

    Design principle: domain-appropriate uncertainty hedges such as "consult a
    doctor" or "seek legal advice" are explicitly excluded from the "not useful"
    category.  Penalising them would cause spurious query rewrites in medical
    and legal domains where hedging is both correct and responsible.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        """
        Initializes the grader and wires the structured-output chain.

        Args:
            model: Anthropic model identifier.  Defaults to Claude Sonnet.
        """
        llm = _build_llm(model)
        self._chain = llm.with_structured_output(UsefulnessGrading)

    def grade(self, query: str, answer: str) -> UsefulnessGrading:
        """
        Grades whether *answer* meaningfully resolves *query*.

        Args:
            query: The user's original query string.
            answer: The generated answer to evaluate.

        Returns:
            ``UsefulnessGrading`` with fields:
            - ``useful`` (bool): True if the answer resolves the query.
            - ``reason`` (str): One-sentence rationale.
        """
        messages = [
            SystemMessage(content=_USEFULNESS_GRADER_SYSTEM),
            HumanMessage(content=f"QUERY: {query}\n\nANSWER:\n{answer}"),
        ]
        return self._chain.invoke(messages)


# ---------------------------------------------------------------------------
# 4. QueryRewriter
# ---------------------------------------------------------------------------


class QueryRewriter:
    """
    Reformulates a failing query to improve vector similarity search recall.

    The rewriter receives the original query *and* the irrelevant chunks so it
    can understand what vocabulary the index responded to and steer the new query
    toward more specific or technically precise terms.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        """
        Initializes the rewriter with a plain-text (non-structured) LLM chain.

        Args:
            model: Anthropic model identifier.  Defaults to Claude Sonnet.
        """
        self._llm = _build_llm(model)

    def rewrite(self, query: str, irrelevant_chunks: List[str]) -> str:
        """
        Produces an improved query string likely to retrieve more relevant chunks.

        Args:
            query: The current (failing) query string.
            irrelevant_chunks: The chunks returned for *query* that were graded
                               not relevant by ``RetrievalGrader``.  Shown to the
                               model so it can infer why retrieval failed.

        Returns:
            A rewritten query string (plain text, no surrounding quotes).
        """
        # Truncate each chunk to 300 chars to keep the prompt concise.
        snippet_lines = "\n".join(
            f"[{i + 1}] {chunk[:300]}" for i, chunk in enumerate(irrelevant_chunks)
        )
        messages = [
            SystemMessage(content=_QUERY_REWRITER_SYSTEM),
            HumanMessage(
                content=(
                    f"ORIGINAL QUERY: {query}\n\n"
                    "IRRELEVANT CHUNKS RETURNED "
                    "(context on why retrieval failed):\n"
                    f"{snippet_lines}\n\n"
                    "Rewritten query:"
                )
            ),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip()
