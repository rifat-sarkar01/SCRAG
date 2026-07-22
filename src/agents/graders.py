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
# Default models
# ---------------------------------------------------------------------------
#
# IMPORTANT — separate judge model from generation model:
# Previously every grader here defaulted to OLLAMA_MODEL, i.e. the EXACT SAME
# model used to generate the answer being judged. That means the model was
# grading its own output — a well-known source of self-evaluation bias (a
# model is systematically less likely to catch error patterns from its own
# distribution). It also happened to be a vision-language checkpoint
# (qwen3-vl:8b-instruct), which trades some pure-text reasoning capacity for
# vision support it doesn't need here — a bad fit for a strict text
# fact-checking task.
#
# OLLAMA_JUDGE_MODEL is now a distinct, independently-configurable model used
# for ALL grading/arbitration roles (retrieval relevance, groundedness,
# usefulness, query rewriting). Default is qwen2.5:14b-instruct — a dense,
# text-only, tool-calling-native instruct model with materially stronger
# instruction-following than the 8B VL checkpoint, and no "thinking" wrapper
# to fight with structured-output parsing. Pull it with:
#   ollama pull qwen2.5:14b-instruct
# Override via .env if your hardware needs a different size.

_JUDGE_MODEL = os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5:14b-instruct")
_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------


class RetrievalGrading(BaseModel):
    """
    Structured output schema for retrieval relevance grading.

    Field order matters: ``reasoning`` is declared before ``relevant`` so the
    model is forced to articulate its analysis before committing to a verdict
    (chain-of-thought via schema order), rather than deciding first and
    rationalizing after.
    """

    reasoning: str = Field(
        description=(
            "One or two sentences analyzing what the chunk actually says "
            "relative to what the query asks, BEFORE deciding relevance."
        )
    )
    relevant: bool = Field(
        description=(
            "True if the chunk contributes useful signal for answering the query, "
            "even if it alone is insufficient to fully answer it."
        )
    )


class GroundednessGrading(BaseModel):
    """
    Structured output schema for answer groundedness grading.

    ``claim_analysis`` is declared first and requires an explicit numbered,
    claim-by-claim walkthrough (each claim in the answer checked individually
    against the context chunks) before the model commits to the aggregate
    ``grounded`` verdict. Single-shot "is this grounded, yes/no" judgments are
    exactly what let a weak or lazy judge default to "yes" without doing real
    verification — decomposition is the fix.
    """

    claim_analysis: str = Field(
        description=(
            "Numbered list: every distinct factual claim in the ANSWER, and for "
            "each one, which context chunk (if any) supports it, or 'UNSUPPORTED' "
            "if no chunk supports it. Do this BEFORE deciding the final verdict. "
            "Example:\n"
            "1. \"Cas9 is guided by a synthetic gRNA\" — supported by chunk [1]\n"
            "2. \"discovered in 2012\" — UNSUPPORTED, no chunk mentions a date"
        )
    )
    grounded: bool = Field(
        description=(
            "True only if EVERY claim in claim_analysis above was marked "
            "supported. False if even one claim is UNSUPPORTED."
        )
    )
    unsupported_claims: List[str] = Field(
        description=(
            "Verbatim or near-verbatim quotes of claims marked UNSUPPORTED in "
            "claim_analysis. Empty list when grounded is True."
        )
    )


class ChunkVerdict(BaseModel):
    """One chunk's relevance verdict within a batch grading call."""

    chunk_index: int = Field(description="1-based index matching the numbered chunk in the prompt.")
    reasoning: str = Field(description="Brief analysis before the verdict.")
    relevant: bool = Field(description="True if this chunk contributes useful signal.")


class RetrievalGradingBatch(BaseModel):
    """Structured output schema for grading ALL retrieved chunks in one call."""

    verdicts: List[ChunkVerdict] = Field(
        description="One verdict per chunk, in the same order as the numbered chunks given."
    )


class UsefulnessGrading(BaseModel):
    """
    Structured output schema for answer usefulness grading.

    ``reasoning`` precedes ``useful`` for the same chain-of-thought-via-schema-
    order reason as the other graders.
    """

    reasoning: str = Field(
        description=(
            "One or two sentences on whether the answer actually resolves what "
            "was asked, BEFORE deciding the verdict."
        )
    )
    useful: bool = Field(
        description=(
            "True if the answer genuinely and directly resolves the user's query."
        )
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_llm(model: str = _JUDGE_MODEL, temperature: float = 0.0) -> ChatOllama:
    """
    Instantiates a ChatOllama client.  Base URL is read from the environment.

    keep_alive="30m": during a batch eval run, this pipeline alternates between
    the generation model and this judge model within every single query. Ollama
    unloads a model from VRAM when a *different* model is requested (or after
    the default 5-minute idle timeout) — each unload/reload of a multi-GB model
    is a real, measurable latency cost. Explicit keep_alive doesn't eliminate
    swaps forced by VRAM pressure when both models can't fit simultaneously,
    but it does prevent *unnecessary* extra unloads from idle timeouts stacking
    on top of that. See OLLAMA_JUDGE_MODEL sizing note in .env.example if both
    models don't fit in VRAM together on your hardware.
    """
    return ChatOllama(
        model=model,
        base_url=_OLLAMA_BASE_URL,
        temperature=temperature,
        keep_alive="30m",
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
- First write your reasoning, THEN decide relevant true/false. Do not decide
  the verdict before writing the reasoning."""

_RETRIEVAL_GRADER_BATCH_SYSTEM = """\
You are a strict relevance judge for a retrieval-augmented generation (RAG) system.

Your task: for EACH of the numbered DOCUMENT CHUNKS below, decide whether it
contains information directly useful for answering the QUERY. Grade every
chunk independently — do not let your verdict on one chunk influence another.

Rules (apply identically to every chunk):
- A chunk is RELEVANT if it contains facts, definitions, explanations, or data
  that the query is asking about — even partially.
- A chunk is NOT RELEVANT if it is on a related topic but does not address the
  specific question, or if it only shares surface keywords with the query.
- Do NOT consider whether a chunk alone is sufficient to answer the query;
  only judge whether it contributes useful signal.
- Do NOT hallucinate. Base every judgment solely on the text provided.
- For each chunk, write brief reasoning BEFORE the verdict.
- Return exactly one verdict per chunk, using the same chunk_index as given,
  and grade every chunk — do not skip any."""

_GROUNDEDNESS_GRADER_SYSTEM = """\
You are a rigorous fact-checking judge for a retrieval-augmented generation (RAG) system.

Your task: determine whether every factual claim in the ANSWER is directly supported
by information present in the CONTEXT CHUNKS.

Required approach — follow these steps in order, do not skip the decomposition:
STEP 1: Break the ANSWER into its individual factual claims (numbers, names,
        causal statements, definitions, conclusions — each one separately).
        COMPLETENESS CHECK: every sentence in the ANSWER must produce at least
        one claim. If the ANSWER has 3 sentences, your list must cover all 3 —
        do not silently skip a sentence because it looks fine at a glance.
STEP 2: For each claim, search the CONTEXT CHUNKS for direct or logically
        equivalent support. Record "supported by chunk [N]" or "UNSUPPORTED".
STEP 3: Only after completing that per-claim walkthrough, decide the overall
        ``grounded`` verdict: true only if every claim was supported.

Rules:
- A claim is SUPPORTED if the exact fact (or a logically equivalent statement)
  appears in at least one of the context chunks.
- A claim is UNSUPPORTED if it introduces facts, numbers, names, causal claims, or
  conclusions that are not present in any of the context chunks — even if the claim
  seems plausible, well-known, or likely true. Real-world truth is NOT grounding.
- PARAPHRASE vs. NEW INFORMATION — judge by facts, not wording:
    * Restating the SAME fact in different words, summarizing a longer passage,
      or combining two chunk statements into one sentence is SUPPORTED. Compare
      what the claim actually asserts to what the chunk actually asserts — if
      they describe the same underlying fact, wording differences don't matter.
      Example: chunk says "Types: a. Classification: Predicting a discrete
      category. b. Regression: Predicting a continuous value." → claim "aiming
      to predict categories (classification) or continuous values (regression)"
      is SUPPORTED (same facts, compressed wording), NOT unsupported.
    * Introducing a fact, number, name, or claim that the chunk does not state
      and does not logically entail is UNSUPPORTED. Example: chunk describes
      clustering and dimensionality reduction; claim adds "used primarily in
      fraud detection" → UNSUPPORTED (that's new information, not a paraphrase).
- Common knowledge hedges (e.g., "water is wet") can be ignored only if they are
  truly non-substantive.
- Do NOT judge whether the answer is correct in the real world. Only judge whether
  it is grounded in the provided context.
- Reproduce unsupported claims verbatim or as short direct quotes."""

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
        required steps and stops),
    (e) MULTI-PART QUESTIONS: if the query asks to compare, differentiate,
        list, or enumerate multiple named items (e.g. "difference between X
        and Y", "compare A and B", "what are the causes of Z"), the answer
        must address EVERY named item. An answer about only X when the query
        asked for "X vs Y" is NOT USEFUL, even if what it says about X is
        perfectly accurate — silently dropping half the question is exactly
        the kind of regression this check exists to catch.
- Do NOT penalise appropriate uncertainty hedges (e.g., "consult a doctor") when
  those hedges are genuinely warranted by the domain.
- First write your reasoning, THEN decide the verdict."""

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

    def __init__(self, model: str = _JUDGE_MODEL) -> None:
        """
        Initializes the grader and wires the structured-output chain.

        Args:
            model: Ollama model tag for this judge/grading role. Defaults to
                   OLLAMA_JUDGE_MODEL (qwen2.5:14b-instruct) — deliberately NOT
                   the generation model, to avoid self-grading bias.
        """
        llm = _build_llm(model)
        self._chain = llm.with_structured_output(RetrievalGrading)
        self._batch_chain = llm.with_structured_output(RetrievalGradingBatch)

    def grade(self, query: str, chunk: str) -> RetrievalGrading:
        """
        Grades whether *chunk* contributes useful signal for answering *query*.

        Args:
            query: The user's search query.
            chunk: Raw text content of the retrieved document chunk.

        Returns:
            ``RetrievalGrading`` with fields:
            - ``reasoning`` (str): Analysis written before the verdict.
            - ``relevant`` (bool): True if the chunk is useful.
        """
        messages = [
            SystemMessage(content=_RETRIEVAL_GRADER_SYSTEM),
            HumanMessage(content=f"QUERY: {query}\n\nDOCUMENT CHUNK:\n{chunk}"),
        ]
        return self._chain.invoke(messages)

    def grade_batch(self, query: str, chunks: List[str]) -> List[RetrievalGrading]:
        """
        Grades ALL *chunks* against *query* in a single LLM call instead of one
        call per chunk. This is the primary latency fix for ``node_grade_chunks``:
        with N retrieved chunks, the old per-chunk loop made N sequential blocking
        calls to the local model; this makes exactly 1 call regardless of N.

        Falls back to per-chunk ``grade()`` calls if the batch call fails to
        return a verdict for every chunk (e.g. the model dropped one under
        structured-output pressure) — correctness over speed.

        Args:
            query:  The user's search query.
            chunks: Raw text content of all retrieved chunks, in order.

        Returns:
            List of ``RetrievalGrading``, one per chunk, in the same order as
            *chunks*.
        """
        if not chunks:
            return []

        numbered = "\n\n".join(
            f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks)
        )
        messages = [
            SystemMessage(content=_RETRIEVAL_GRADER_BATCH_SYSTEM),
            HumanMessage(
                content=f"QUERY: {query}\n\nDOCUMENT CHUNKS:\n{numbered}"
            ),
        ]

        try:
            batch_result = self._batch_chain.invoke(messages)
            by_index = {v.chunk_index: v for v in batch_result.verdicts}
            results: List[RetrievalGrading] = []
            for i in range(len(chunks)):
                v = by_index.get(i + 1)
                if v is None:
                    raise ValueError(f"batch grader omitted chunk_index={i + 1}")
                results.append(RetrievalGrading(reasoning=v.reasoning, relevant=v.relevant))
            return results
        except Exception:
            # Correctness fallback: one call per chunk, slower but always complete.
            return [self.grade(query=query, chunk=c) for c in chunks]


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

    def __init__(self, model: str = _JUDGE_MODEL) -> None:
        """
        Initializes the grader and wires the structured-output chain.

        Args:
            model: Ollama model tag for this judge/grading role. Defaults to
                   OLLAMA_JUDGE_MODEL (qwen2.5:14b-instruct) — deliberately NOT
                   the generation model, to avoid self-grading bias.
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

    def __init__(self, model: str = _JUDGE_MODEL) -> None:
        """
        Initializes the grader and wires the structured-output chain.

        Args:
            model: Ollama model tag for this judge/grading role. Defaults to
                   OLLAMA_JUDGE_MODEL (qwen2.5:14b-instruct) — deliberately NOT
                   the generation model, to avoid self-grading bias.
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
            - ``reasoning`` (str): Analysis written before the verdict.
            - ``useful`` (bool): True if the answer resolves the query.
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

    def __init__(self, model: str = _JUDGE_MODEL) -> None:
        """
        Initializes the rewriter with a plain-text (non-structured) LLM chain.

        Args:
            model: Ollama model tag for this judge/grading role. Defaults to
                   OLLAMA_JUDGE_MODEL (qwen2.5:14b-instruct) — deliberately NOT
                   the generation model, to avoid self-grading bias.
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
