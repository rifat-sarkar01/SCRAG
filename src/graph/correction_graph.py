"""
src/graph/correction_graph.py
------------------------------
LangGraph state machine for the Self-Correcting RAG system.

Flow
----
retrieve → grade_chunks
  → [if <50% relevant AND retries < 2: rewrite_query → retrieve]
  → generate
  → grade_groundedness
  → [if not grounded AND retries < 2: regenerate → grade_groundedness]
  → grade_usefulness
  → [if not useful AND retries < 1: rewrite_query → retrieve → … full cycle]
  → [if ENABLE_FALSIFICATION: falsify]   ← optional devil's-advocate round
  → END

Design decisions (approved in Phase 3a)
----------------------------------------
- Single shared ``rewrite_query`` node for both the retrieval loop and the
  usefulness fallback; the conditional edge determines which cycle is active.
- Counters are incremented *inside* the action node, not in the router, so
  routers remain pure decision functions.
- After exhausting retries the graph always proceeds to END with the best-effort
  answer rather than raising an error; the trace log records the struggle.

Faithfulness Fix (Step 2)
--------------------------
- ``node_regenerate`` now uses a *hardened* system prompt that explicitly
  forbids the model from introducing any claim not present in the provided
  context — even if omitting it hurts fluency or completeness.
- Correction is framed as *targeted claim-level patching* (remove unsupported
  claims, keep supported ones verbatim) rather than a full answer regeneration,
  so already-faithful claims are not disturbed.
- The exact chunks used during correction are stored in ``correction_context``
  and returned by ``run_corrected_query`` so the eval harness can do exact
  context pairing (fixing the harness bug).

Instrumentation (Step 3)
-------------------------
- ``per_claim_verdicts`` — per-claim faithfulness detail from ``grade_groundedness``.
- ``correction_rounds`` — total correction (regenerate) rounds executed.
- ``reretrieval_happened`` — True if a new ``retrieve`` cycle ran after the
  initial generation (usefulness or retrieval retry).

Public API
----------
    run_corrected_query(query: str) -> dict
        {
          "final_answer": str,
          "trace": list[dict],
          "correction_context": list[str],   # full chunks used in correction
          "correction_rounds": int,
          "reretrieval_happened": bool,
        }
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.agents.graders import (
    AnswerGroundednessGrader,
    AnswerUsefulnessGrader,
    QueryRewriter,
    RetrievalGrader,
)
from src.agents.falsifier import (
    ENABLE_FALSIFICATION,
    generate_falsification_queries,
    verify_against_counterevidence,
)
from src.retrieval.document_store import DocumentStore
from src.retrieval.embedding_index import EmbeddingIndex
from src.retrieval.generator import generate_answer

load_dotenv()

# ---------------------------------------------------------------------------
# Retry caps (named constants — change here only)
# ---------------------------------------------------------------------------

_MAX_RETRIEVAL_RETRIES: int = 2   # rewrite_query → retrieve cycles
_MAX_GROUNDEDNESS_RETRIES: int = 2  # regenerate cycles
_MAX_USEFULNESS_RETRIES: int = 1   # usefulness-driven full cycles
_DATA_DIR: str = "data"

# ---------------------------------------------------------------------------
# Lazy singletons — instantiated once per process, not per query
# ---------------------------------------------------------------------------

_index: EmbeddingIndex | None = None
_retrieval_grader: RetrievalGrader | None = None
_groundedness_grader: AnswerGroundednessGrader | None = None
_usefulness_grader: AnswerUsefulnessGrader | None = None
_query_rewriter: QueryRewriter | None = None
_correction_llm: ChatOllama | None = None


def _get_index() -> EmbeddingIndex:
    global _index
    if _index is None:
        _index = EmbeddingIndex()
        store = DocumentStore(chunk_size=500, chunk_overlap=50)
        docs = store.load_documents(_DATA_DIR)
        corpus_docs = [d for d in docs if d.get("filename") != "eval_qa.json"]
        chunks = store.chunk_documents(corpus_docs)
        _index.build_index(chunks)
    return _index


def _get_retrieval_grader() -> RetrievalGrader:
    global _retrieval_grader
    if _retrieval_grader is None:
        _retrieval_grader = RetrievalGrader()
    return _retrieval_grader


def _get_groundedness_grader() -> AnswerGroundednessGrader:
    global _groundedness_grader
    if _groundedness_grader is None:
        _groundedness_grader = AnswerGroundednessGrader()
    return _groundedness_grader


def _get_usefulness_grader() -> AnswerUsefulnessGrader:
    global _usefulness_grader
    if _usefulness_grader is None:
        _usefulness_grader = AnswerUsefulnessGrader()
    return _usefulness_grader


def _get_query_rewriter() -> QueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter


def _get_correction_llm() -> ChatOllama:
    """
    Returns a cached ChatOllama instance used exclusively for correction.

    Uses OLLAMA_JUDGE_MODEL (not OLLAMA_MODEL): the correction step is a strict
    rule-following compliance task (remove exactly the flagged unsupported
    claims, touch nothing else) rather than open-ended generation, so it
    benefits from the stronger, non-self-grading judge model — this is the
    step that produces the final answer text the eval score is based on.
    """
    global _correction_llm
    if _correction_llm is None:
        _correction_llm = ChatOllama(
            model=os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5:14b-instruct"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
            keep_alive="30m",
        )
    return _correction_llm


# ---------------------------------------------------------------------------
# Hardened correction system prompt (Step 2)
# ---------------------------------------------------------------------------

_CORRECTION_SYSTEM = """\
You are a strict faithfulness editor for a retrieval-augmented generation (RAG) system.

Your ONLY job is to remove or soften claims in the ORIGINAL ANSWER that are NOT
supported by the CONTEXT provided below.

ABSOLUTE RULES — violating any of these is a critical failure:
1. Every factual claim in your revised answer MUST be directly supported by the
   CONTEXT below.  If you cannot find support, REMOVE the claim entirely rather
   than rephrasing or hedging it.
2. Do NOT add any information that does not appear in the CONTEXT — not even
   reasonable inferences, background knowledge, or fluency improvements.
3. Do NOT use your own parametric/world knowledge.  Treat every fact outside the
   CONTEXT as non-existent.
4. Do NOT rewrite or restructure sentences that are already supported — copy them
   verbatim or make only the minimal change required.
5. If removing the unsupported claims leaves a correct, shorter answer, that is
   the right output.  Do not pad.
6. SCOPE PRESERVATION — this is the rule most often violated, watch it closely:
   the QUESTION may name multiple entities/topics to cover (e.g. "difference
   between X and Y", "compare A and B", "list the causes of Z"). An unsupported
   claim about ONE of those entities is NOT grounds to drop that entire entity
   from the answer. Check the CONTEXT again for OTHER supported facts about
   that same entity before deciding to remove it — if the context supports
   *some* description of it, keep that description and remove only the specific
   unsupported clause. Only drop an entire entity/topic if the context contains
   NOTHING usable about it at all.
7. GRANULARITY OF EDITS — remove the smallest span that eliminates the
   unsupported claim (a clause, a number, a qualifier). Do NOT delete an entire
   sentence or an entire topic when only one word or phrase within it is
   actually unsupported.

Approach:
- Read each sentence/claim in the ORIGINAL ANSWER.
- Check whether the CONTEXT supports it.
- Keep it if supported; remove (or soften to "not mentioned in context") if not.
- Before finalizing, re-read the QUESTION: does your revised answer still
  address every entity/part the question named? If the question asked for a
  comparison and the context supports facts about both sides, your answer
  must still cover both sides.
- Return the corrected answer only — no preamble, no explanation, no meta-commentary.\
"""


def _generate_corrected_answer(
    original_query: str,
    original_answer: str,
    relevant_chunks: List[str],
    unsupported_claims: List[str],
    claim_analysis: str = "",
) -> str:
    """
    Targeted claim-level correction: removes unsupported claims while leaving
    supported claims intact.  Uses the hardened ``_CORRECTION_SYSTEM`` prompt
    that explicitly forbids parametric knowledge injection.

    Args:
        original_query:    The user's original question.
        original_answer:   The draft answer from the previous generation round.
        relevant_chunks:   The full context chunks (not truncated) that were used
                           for generation.  Passed to the correction LLM as the
                           sole allowed source of facts.
        unsupported_claims: Claims flagged by the groundedness grader.
        claim_analysis:     The groundedness grader's full claim-by-claim
                           walkthrough (which claims WERE supported, and by
                           which chunk — not just which were unsupported).
                           Handing this to the corrector directly, instead of
                           making it re-derive "what's still supported" from
                           scratch, is what actually fixes the over-deletion
                           failure mode (a prose "don't over-delete" rule
                           alone was not reliably followed — this makes the
                           already-verified answer key explicit instead).

    Returns:
        A corrected answer string with unsupported claims removed/softened.
    """
    numbered_chunks = "\n\n".join(
        f"[CHUNK {i + 1}]\n{chunk}" for i, chunk in enumerate(relevant_chunks)
    )
    claims_block = "\n".join(f"  - {c}" for c in unsupported_claims)

    analysis_block = (
        f"GROUNDEDNESS ANALYSIS (already verified — this tells you exactly "
        f"what IS supported and by which chunk; do not re-derive this, use "
        f"it directly to decide what to keep):\n{claim_analysis}\n\n"
        if claim_analysis
        else ""
    )

    user_content = (
        f"QUESTION: {original_query}\n\n"
        f"CONTEXT:\n{numbered_chunks}\n\n"
        f"ORIGINAL ANSWER:\n{original_answer}\n\n"
        f"{analysis_block}"
        f"CLAIMS FLAGGED AS UNSUPPORTED (remove ONLY these; everything else "
        f"the analysis above marked as supported must be kept):\n{claims_block}\n\n"
        "Produce the corrected answer now:"
    )

    messages = [
        SystemMessage(content=_CORRECTION_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = _get_correction_llm().invoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """
    Shared mutable state passed between every node in the correction graph.

    Required at graph entry
    -----------------------
    query           Active query string; overwritten by ``rewrite_query``.
    original_query  User's original query; never mutated — used for usefulness
                    grading and trace readability.

    Populated during execution
    --------------------------
    retrieved_chunks        Raw chunk texts from the most recent ``retrieve`` call.
    chunk_grades            Per-chunk grading records from ``grade_chunks``.
    relevant_chunks         Subset of retrieved_chunks graded relevant.
    irrelevant_chunks       Subset graded not relevant; fed to ``rewrite_query``.
    draft_answer            Current candidate answer; overwritten each generate cycle.
    final_answer            Copied from draft_answer when the graph reaches END.
    unsupported_claims      Claims flagged by the last ``grade_groundedness`` call.
    claim_analysis          Full claim-by-claim walkthrough from the last
                            ``grade_groundedness`` call (which claims were
                            SUPPORTED and by which chunk, not just which were
                            UNSUPPORTED). Forwarded to ``regenerate`` so the
                            correction step has an explicit answer key for what
                            to keep, instead of having to re-derive it from
                            scratch under prompt-instruction pressure alone.
    per_claim_verdicts      Detailed per-claim faithfulness log from groundedness grader.
    retrieval_retry_count   rewrite_query→retrieve cycles completed (cap: 2).
    groundedness_retry_count  regenerate cycles completed (cap: 2).
    usefulness_retry_count  Usefulness-driven full cycles completed (cap: 1).
    _usefulness_result      Routing-control flag written by ``grade_usefulness``
                            and read by ``route_after_grade_usefulness``. THIS
                            FIELD MUST STAY DECLARED HERE. LangGraph silently
                            drops any key a node returns that isn't part of the
                            declared schema — no error, no warning. Before this
                            field was added, ``grade_usefulness`` correctly
                            computed and logged "not useful" in the trace, but
                            the routing key itself never survived the state
                            merge, so ``route_after_grade_usefulness`` always
                            read the fallback default (True) and NEVER routed
                            to ``rewrite_query`` — the entire usefulness retry
                            loop was dead code regardless of what the grader
                            decided. Confirmed with a minimal LangGraph
                            reproduction before this fix; see tests/test_correction.py.
    correction_rounds       Total correction (regenerate) rounds executed.
    correction_context      Full chunk texts used during correction — returned to
                            the eval harness for exact RAGAS context pairing.
    reretrieval_happened    True if a second retrieve cycle ran after generation.
    trace                   Chronological log of every node visit and decision.
    """

    # --- required at entry ---
    query: str
    original_query: str

    # --- retrieval ---
    retrieved_chunks: List[str]
    chunk_grades: List[Dict[str, Any]]
    relevant_chunks: List[str]
    irrelevant_chunks: List[str]

    # --- generation ---
    draft_answer: str
    final_answer: str
    unsupported_claims: List[str]
    claim_analysis: str
    per_claim_verdicts: List[Dict[str, Any]]  # per-claim faithfulness detail

    # --- retry counters (incremented inside action nodes) ---
    retrieval_retry_count: int
    groundedness_retry_count: int
    usefulness_retry_count: int
    _usefulness_result: bool      # routing flag — see docstring above; must stay declared

    # --- instrumentation ---
    correction_rounds: int        # total regenerate rounds executed
    correction_context: List[str] # full chunks used in correction (for eval)
    reretrieval_happened: bool    # True if a new retrieve cycle ran post-generation

    # --- falsification (optional, gated by ENABLE_FALSIFICATION) ---
    falsification_done: bool   # True once the single allowed round has run

    # --- observability ---
    trace: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper: append a trace entry
# ---------------------------------------------------------------------------


def _trace_entry(
    node: str,
    query: str,
    decision: str,
    detail: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Returns one trace record in the standard format."""
    return {
        "node": node,
        "query": query,
        "decision": decision,
        "detail": detail or {},
    }


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------


def node_retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Embeds the current query and retrieves the top-k chunks from the FAISS index.

    Reads:  ``query``, ``draft_answer`` (to detect post-generation re-retrieval)
    Writes: ``retrieved_chunks``, ``reretrieval_happened``, ``trace``
    """
    query = state["query"]
    index = _get_index()

    # EmbeddingIndex.retrieve returns List[Tuple[Dict, float]]
    results = index.retrieve(query, k=4)
    chunks = [item[0]["text"] if isinstance(item[0], dict) else str(item[0])
              for item in results]

    # Flag re-retrieval if generation has already happened.
    reretrieval_happened = state.get("reretrieval_happened", False)
    if state.get("draft_answer"):
        reretrieval_happened = True

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="retrieve",
        query=query,
        decision=f"Retrieved {len(chunks)} chunks from index.",
        detail={
            "num_chunks": len(chunks),
            "reretrieval": reretrieval_happened,
        },
    ))

    return {
        "retrieved_chunks": chunks,
        "reretrieval_happened": reretrieval_happened,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: grade_chunks
# ---------------------------------------------------------------------------


def node_grade_chunks(state: GraphState) -> Dict[str, Any]:
    """
    Grades every retrieved chunk for relevance and splits them into
    ``relevant_chunks`` and ``irrelevant_chunks``.

    Reads:  ``query``, ``retrieved_chunks``
    Writes: ``chunk_grades``, ``relevant_chunks``, ``irrelevant_chunks``, ``trace``
    """
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    grader = _get_retrieval_grader()

    grades: List[Dict[str, Any]] = []
    relevant: List[str] = []
    irrelevant: List[str] = []

    # Single batched call instead of one call per chunk — was the main source
    # of the graph pipeline's latency overhead vs. baseline (N sequential
    # blocking LLM calls collapsed into 1).
    batch_results = grader.grade_batch(query=query, chunks=chunks)

    for chunk, result in zip(chunks, batch_results):
        grades.append({
            "chunk": chunk[:120] + ("…" if len(chunk) > 120 else ""),
            "chunk_full": chunk,  # full text preserved for eval context pairing
            "relevant": result.relevant,
            "reasoning": result.reasoning,
        })
        (relevant if result.relevant else irrelevant).append(chunk)

    total = len(chunks)
    relevant_frac = len(relevant) / total if total > 0 else 0.0
    decision = (
        f"{len(relevant)}/{total} chunks relevant "
        f"({relevant_frac:.0%}). "
        + ("→ rewrite_query" if relevant_frac < 0.5 else "→ generate")
    )

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="grade_chunks",
        query=query,
        decision=decision,
        detail={"relevant_fraction": relevant_frac, "grades": grades},
    ))

    return {
        "chunk_grades": grades,
        "relevant_chunks": relevant,
        "irrelevant_chunks": irrelevant,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: rewrite_query
# ---------------------------------------------------------------------------


def node_rewrite_query(state: GraphState) -> Dict[str, Any]:
    """
    Rewrites the active query using irrelevant chunks as negative signal.
    Increments the appropriate retry counter.

    - If entering from the retrieval loop → increments ``retrieval_retry_count``.
    - If entering from the usefulness loop → increments ``usefulness_retry_count``.

    Determination: the usefulness loop has already set ``usefulness_retry_count``
    to ≥ 1 *before* this node runs (the router checks the pre-increment value;
    the node increments after routing has already decided to call it).
    In practice we differentiate by whether a ``draft_answer`` exists yet.

    Reads:  ``query``, ``irrelevant_chunks``, ``draft_answer`` (optional)
    Writes: ``query``, ``retrieval_retry_count`` or ``usefulness_retry_count``, ``trace``
    """
    original_query = state["query"]
    irrelevant_chunks = state.get("irrelevant_chunks", [])
    rewriter = _get_query_rewriter()

    new_query = rewriter.rewrite(
        query=original_query,
        irrelevant_chunks=irrelevant_chunks,
    )

    # Determine which counter to increment based on whether generation has happened.
    in_usefulness_loop = bool(state.get("draft_answer"))
    retrieval_retry_count = state.get("retrieval_retry_count", 0)
    usefulness_retry_count = state.get("usefulness_retry_count", 0)

    if in_usefulness_loop:
        usefulness_retry_count += 1
        loop_label = "usefulness"
    else:
        retrieval_retry_count += 1
        loop_label = "retrieval"

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="rewrite_query",
        query=original_query,
        decision=(
            f"Query rewritten for {loop_label} loop. "
            f"retrieval_retries={retrieval_retry_count}, "
            f"usefulness_retries={usefulness_retry_count}."
        ),
        detail={"original_query": original_query, "new_query": new_query},
    ))

    return {
        "query": new_query,
        "retrieval_retry_count": retrieval_retry_count,
        "usefulness_retry_count": usefulness_retry_count,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------


def node_generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates a draft answer from the relevant chunks using the local LLM.
    Also initialises ``correction_context`` to the current relevant chunks
    so it is always populated even if no correction round runs.

    Reads:  ``original_query``, ``relevant_chunks``
    Writes: ``draft_answer``, ``correction_context``, ``trace``
    """
    original_query = state["original_query"]
    relevant_chunks = state.get("relevant_chunks", [])

    # generate_answer expects list of dicts with at least a "text" key; wrap if plain str.
    chunk_dicts = [
        c if isinstance(c, dict) else {"text": c}
        for c in relevant_chunks
    ]
    draft = generate_answer(query=original_query, chunks=chunk_dicts)

    # Store full chunks as correction_context baseline (overwritten in node_regenerate).
    full_chunks = [
        c["text"] if isinstance(c, dict) else str(c) for c in relevant_chunks
    ]

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="generate",
        query=original_query,
        decision=f"Draft answer generated ({len(draft)} chars).",
        detail={"draft_preview": draft[:200]},
    ))

    return {
        "draft_answer": draft,
        "correction_context": full_chunks,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: grade_groundedness
# ---------------------------------------------------------------------------


def node_grade_groundedness(state: GraphState) -> Dict[str, Any]:
    """
    Checks whether every factual claim in the draft answer is supported by the
    retrieved context chunks.  Logs per-claim verdicts for instrumentation.

    Reads:  ``draft_answer``, ``relevant_chunks``
    Writes: ``unsupported_claims``, ``per_claim_verdicts``, ``trace``
    """
    draft = state.get("draft_answer", "")
    relevant_chunks = state.get("relevant_chunks", [])
    grader = _get_groundedness_grader()

    chunk_texts = [
        c["text"] if isinstance(c, dict) else str(c)
        for c in relevant_chunks
    ]
    result = grader.grade(answer=draft, chunks=chunk_texts)

    # Build per-claim verdict records for instrumentation.
    per_claim_verdicts: List[Dict[str, Any]] = []
    for claim in result.unsupported_claims:
        per_claim_verdicts.append({"claim": claim, "verdict": "unsupported"})
    # Note: RAGAS faithfulness tracks claim-level; we record unsupported ones.

    decision = (
        "Answer is grounded. → grade_usefulness"
        if result.grounded
        else (
            f"Answer has {len(result.unsupported_claims)} unsupported claim(s). "
            f"→ regenerate"
        )
    )

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="grade_groundedness",
        query=state.get("query", ""),
        decision=decision,
        detail={
            "grounded": result.grounded,
            "claim_analysis": result.claim_analysis,
            "unsupported_claims": result.unsupported_claims,
            "per_claim_verdicts": per_claim_verdicts,
            "context_chunk_count": len(chunk_texts),
        },
    ))

    return {
        "unsupported_claims": result.unsupported_claims,
        "claim_analysis": result.claim_analysis,
        "per_claim_verdicts": per_claim_verdicts,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: regenerate  (FIXED — targeted claim-patching, hardened prompt)
# ---------------------------------------------------------------------------


def node_regenerate(state: GraphState) -> Dict[str, Any]:
    """
    Corrects the answer using *targeted claim-level patching*:

    - Only the claims flagged as unsupported are removed/softened.
    - Supported claims are kept verbatim — the model is NOT allowed to rewrite
      unrelated sentences.
    - The hardened ``_CORRECTION_SYSTEM`` prompt explicitly forbids adding any
      information not present in the context (fixing the grounding/faithfulness
      collapse bug).
    - The exact chunks used are stored in ``correction_context`` so the eval
      harness can do exact RAGAS context pairing (fixing the context-pairing bug).

    Increments ``groundedness_retry_count`` and ``correction_rounds``.

    Reads:  ``original_query``, ``relevant_chunks``, ``unsupported_claims``,
            ``claim_analysis``, ``draft_answer``
    Writes: ``draft_answer``, ``correction_context``, ``groundedness_retry_count``,
            ``correction_rounds``, ``trace``
    """
    original_query = state["original_query"]
    relevant_chunks = state.get("relevant_chunks", [])
    unsupported_claims = state.get("unsupported_claims", [])
    claim_analysis = state.get("claim_analysis", "")
    original_answer = state.get("draft_answer", "")

    groundedness_retry_count = state.get("groundedness_retry_count", 0) + 1
    correction_rounds = state.get("correction_rounds", 0) + 1

    # Extract full text from chunk dicts.
    chunk_texts = [
        c["text"] if isinstance(c, dict) else str(c)
        for c in relevant_chunks
    ]

    # Targeted correction using hardened prompt — no full regeneration.
    corrected = _generate_corrected_answer(
        original_query=original_query,
        original_answer=original_answer,
        relevant_chunks=chunk_texts,
        unsupported_claims=unsupported_claims,
        claim_analysis=claim_analysis,
    )

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="regenerate",
        query=original_query,
        decision=(
            f"Targeted correction (round #{correction_rounds}). "
            f"Removed/softened {len(unsupported_claims)} unsupported claim(s). "
            f"groundedness_retry={groundedness_retry_count}."
        ),
        detail={
            "groundedness_retry_count": groundedness_retry_count,
            "correction_rounds": correction_rounds,
            "excluded_claims": unsupported_claims,
            "context_chunk_count": len(chunk_texts),
            "draft_preview": corrected[:200],
        },
    ))

    return {
        "draft_answer": corrected,
        "correction_context": chunk_texts,   # exact chunks used for this correction
        "groundedness_retry_count": groundedness_retry_count,
        "correction_rounds": correction_rounds,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Node: grade_usefulness
# ---------------------------------------------------------------------------


def node_grade_usefulness(state: GraphState) -> Dict[str, Any]:
    """
    Grades whether the draft answer genuinely resolves the user's original query.

    Reads:  ``original_query``, ``draft_answer``
    Writes: ``trace``  (usefulness result stored in trace; routing reads state directly)
    """
    original_query = state["original_query"]
    draft = state.get("draft_answer", "")
    grader = _get_usefulness_grader()

    result = grader.grade(query=original_query, answer=draft)

    decision = (
        "Answer is useful. → END"
        if result.useful
        else "Answer is not useful. → rewrite_query"
    )

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="grade_usefulness",
        query=original_query,
        decision=decision,
        detail={"useful": result.useful, "reasoning": result.reasoning},
    ))

    # Stash the usefulness result so the router can read it without rerunning the grader.
    return {"_usefulness_result": result.useful, "trace": trace}


# ---------------------------------------------------------------------------
# Node: falsify  (optional — only active when ENABLE_FALSIFICATION=True)
# ---------------------------------------------------------------------------


def node_falsify(state: GraphState) -> Dict[str, Any]:
    """
    Devil's-advocate falsification round.

    Cost note: this node adds ~2× LLM call latency per query (one call to
    generate falsification queries, one call to verify against counter-evidence).
    Retrieval of counter-chunks is embedding-only and adds negligible overhead.
    Capped at 1 round — ``falsification_done`` prevents re-entry.

    Reads:  ``draft_answer``, ``original_query``, ``relevant_chunks``,
            ``falsification_done``
    Writes: ``draft_answer`` (possibly unchanged), ``falsification_done``,
            ``trace``
    """
    draft = state.get("draft_answer", "")
    original_query = state["original_query"]
    supporting_chunks = [
        c["text"] if isinstance(c, dict) else str(c)
        for c in state.get("relevant_chunks", [])
    ]

    # --- Step 1: generate falsification queries ---
    counter_queries = generate_falsification_queries(
        draft_answer=draft,
        original_query=original_query,
    )

    # --- Step 2: retrieve counter-evidence chunks ---
    index = _get_index()
    counter_chunks: List[str] = []
    for cq in counter_queries:
        results = index.retrieve(cq, k=2)
        for item, _ in results:
            text = item["text"] if isinstance(item, dict) else str(item)
            if text not in counter_chunks:
                counter_chunks.append(text)

    # --- Step 3: verify verdict with a FRESH LLM context ---
    verdict = verify_against_counterevidence(
        draft_answer=draft,
        supporting_chunks=supporting_chunks,
        counter_chunks=counter_chunks,
    )

    # --- Step 4: apply verdict ---
    new_draft = draft
    if verdict == "overturn":
        new_draft = (
            "[Answer overturned by falsification check] "
            "Counter-evidence contradicts the original answer. "
            "Please re-examine the source documents directly."
        )
    elif verdict == "revise":
        # Append a caveat; the graph has no further correction cycles at this
        # point so we surface the uncertainty to the caller rather than silently
        # keeping a potentially incomplete answer.
        new_draft = (
            draft
            + "\n\n[Falsification note: counter-evidence suggests this answer "
            "may be incomplete or partially incorrect. Treat with caution.]"
        )
    # verdict == "keep": no change to draft

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="falsify",
        query=original_query,
        decision=(
            f"Falsification verdict: {verdict}. "
            f"{len(counter_queries)} counter-queries, "
            f"{len(counter_chunks)} counter-chunks retrieved."
        ),
        detail={
            "verdict": verdict,
            "counter_queries": counter_queries,
            "num_counter_chunks": len(counter_chunks),
            "draft_changed": new_draft != draft,
        },
    ))

    return {
        "draft_answer": new_draft,
        "falsification_done": True,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Router functions (pure — no side effects, no counter increments)
# ---------------------------------------------------------------------------


def route_after_grade_chunks(state: GraphState) -> str:
    """
    Route after grade_chunks:
      - < 50% relevant AND retries remaining → rewrite_query
      - otherwise → generate
    """
    relevant = state.get("relevant_chunks", [])
    retrieved = state.get("retrieved_chunks", [])
    total = len(retrieved)
    relevant_frac = len(relevant) / total if total > 0 else 0.0
    retries = state.get("retrieval_retry_count", 0)

    if relevant_frac < 0.5 and retries < _MAX_RETRIEVAL_RETRIES:
        return "rewrite_query"
    return "generate"


def route_after_grade_groundedness(state: GraphState) -> str:
    """
    Route after grade_groundedness:
      - unsupported claims exist AND retries remaining → regenerate
      - otherwise → grade_usefulness
    """
    unsupported = state.get("unsupported_claims", [])
    retries = state.get("groundedness_retry_count", 0)

    if unsupported and retries < _MAX_GROUNDEDNESS_RETRIES:
        return "regenerate"
    return "grade_usefulness"


def route_after_grade_usefulness(state: GraphState) -> str:
    """
    Route after grade_usefulness:
      - not useful AND retries remaining → rewrite_query (triggers full cycle)
      - ENABLE_FALSIFICATION is True AND not yet done → falsify
      - otherwise → END
    """
    useful = state.get("_usefulness_result", True)  # default True = safe fallback
    retries = state.get("usefulness_retry_count", 0)

    if not useful and retries < _MAX_USEFULNESS_RETRIES:
        return "rewrite_query"
    if ENABLE_FALSIFICATION and not state.get("falsification_done", False):
        return "falsify"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    """Assembles and compiles the LangGraph StateGraph."""
    graph = StateGraph(GraphState)

    # --- nodes ---
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_chunks", node_grade_chunks)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("generate", node_generate)
    graph.add_node("grade_groundedness", node_grade_groundedness)
    graph.add_node("regenerate", node_regenerate)
    graph.add_node("grade_usefulness", node_grade_usefulness)
    # Optional falsification node — always registered; the router decides whether
    # to enter it based on the ENABLE_FALSIFICATION flag at runtime.
    graph.add_node("falsify", node_falsify)

    # --- fixed edges ---
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_chunks")
    graph.add_edge("rewrite_query", "retrieve")       # shared node → always back to retrieve
    graph.add_edge("generate", "grade_groundedness")
    graph.add_edge("regenerate", "grade_groundedness")
    graph.add_edge("falsify", END)                    # falsify is always the last node

    # --- conditional edges ---
    graph.add_conditional_edges(
        "grade_chunks",
        route_after_grade_chunks,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    graph.add_conditional_edges(
        "grade_groundedness",
        route_after_grade_groundedness,
        {"regenerate": "regenerate", "grade_usefulness": "grade_usefulness"},
    )
    graph.add_conditional_edges(
        "grade_usefulness",
        route_after_grade_usefulness,
        {"rewrite_query": "rewrite_query", "falsify": "falsify", END: END},
    )

    return graph.compile()


# Compile once at import time.
_compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_corrected_query(query: str) -> Dict[str, Any]:
    """
    Runs the self-correcting RAG workflow for *query* and returns the result.

    The state machine:
      1. Retrieves chunks from the FAISS index.
      2. Grades chunks for relevance; rewrites the query if < 50% are relevant
         (up to 2 retries).
      3. Generates a draft answer from relevant chunks.
      4. Grades the answer for groundedness; applies targeted claim-level
         correction using the hardened prompt if needed (up to 2 retries).
      5. Grades the answer for usefulness; rewrites the query and restarts the
         full cycle if the answer is unhelpful (up to 1 retry).
      6. Returns the best-effort final answer with a full decision trace and
         instrumentation data.

    Args:
        query: The user's natural-language query string.

    Returns:
        A dictionary with:
        - ``"final_answer"`` (str): The best answer the system could produce.
        - ``"trace"`` (list[dict]): Chronological log of every node visit,
          routing decision, and grading result.
        - ``"correction_context"`` (list[str]): Full chunk texts used during
          the correction step — use these (not trace previews) for RAGAS
          faithfulness scoring to ensure exact context pairing.
        - ``"correction_rounds"`` (int): Number of correction (regenerate)
          rounds executed (0 if no correction was needed).
        - ``"reretrieval_happened"`` (bool): True if a second retrieve cycle
          ran after the initial generation.

    Note:
        If retrieval, generation, or grading raise exceptions (e.g. API errors),
        they will propagate from this function. Callers should handle accordingly.
    """
    initial_state: GraphState = {
        "query": query,
        "original_query": query,
        "retrieved_chunks": [],
        "chunk_grades": [],
        "relevant_chunks": [],
        "irrelevant_chunks": [],
        "draft_answer": "",
        "final_answer": "",
        "unsupported_claims": [],
        "claim_analysis": "",        # must be seeded — node_regenerate reads it
        "per_claim_verdicts": [],
        "retrieval_retry_count": 0,
        "groundedness_retry_count": 0,
        "usefulness_retry_count": 0,
        "_usefulness_result": True,   # must be seeded — router reads it; True = safe default
        "correction_rounds": 0,
        "correction_context": [],
        "reretrieval_happened": False,
        "falsification_done": False,
        "trace": [],
    }

    final_state: GraphState = _compiled_graph.invoke(initial_state)

    return {
        "final_answer": final_state.get("draft_answer", ""),
        "trace": final_state.get("trace", []),
        "correction_context": final_state.get("correction_context", []),
        "correction_rounds": final_state.get("correction_rounds", 0),
        "reretrieval_happened": final_state.get("reretrieval_happened", False),
    }
