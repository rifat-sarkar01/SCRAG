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

Public API
----------
    run_corrected_query(query: str) -> dict
        {"final_answer": str, "trace": list[dict]}
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

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
    retrieval_retry_count   rewrite_query→retrieve cycles completed (cap: 2).
    groundedness_retry_count  regenerate cycles completed (cap: 2).
    usefulness_retry_count  Usefulness-driven full cycles completed (cap: 1).
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

    # --- retry counters (incremented inside action nodes) ---
    retrieval_retry_count: int
    groundedness_retry_count: int
    usefulness_retry_count: int

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

    Reads:  ``query``
    Writes: ``retrieved_chunks``, ``trace``
    """
    query = state["query"]
    index = _get_index()

    # EmbeddingIndex.retrieve returns List[Tuple[Dict, float]]
    results = index.retrieve(query, k=4)
    chunks = [item[0]["text"] if isinstance(item[0], dict) else str(item[0])
              for item in results]

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="retrieve",
        query=query,
        decision=f"Retrieved {len(chunks)} chunks from index.",
        detail={"num_chunks": len(chunks)},
    ))

    return {"retrieved_chunks": chunks, "trace": trace}


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

    for chunk in chunks:
        result = grader.grade(query=query, chunk=chunk)
        grades.append({
            "chunk": chunk[:120] + ("…" if len(chunk) > 120 else ""),
            "relevant": result.relevant,
            "reason": result.reason,
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
    Generates a draft answer from the relevant chunks using Claude Sonnet.

    Reads:  ``original_query``, ``relevant_chunks``
    Writes: ``draft_answer``, ``trace``
    """
    original_query = state["original_query"]
    relevant_chunks = state.get("relevant_chunks", [])

    # generate_answer expects list of dicts with at least a "text" key; wrap if plain str.
    chunk_dicts = [
        c if isinstance(c, dict) else {"text": c}
        for c in relevant_chunks
    ]
    draft = generate_answer(query=original_query, chunks=chunk_dicts)

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="generate",
        query=original_query,
        decision=f"Draft answer generated ({len(draft)} chars).",
        detail={"draft_preview": draft[:200]},
    ))

    return {"draft_answer": draft, "trace": trace}


# ---------------------------------------------------------------------------
# Node: grade_groundedness
# ---------------------------------------------------------------------------


def node_grade_groundedness(state: GraphState) -> Dict[str, Any]:
    """
    Checks whether every factual claim in the draft answer is supported by the
    retrieved context chunks.

    Reads:  ``draft_answer``, ``relevant_chunks``
    Writes: ``unsupported_claims``, ``trace``
    """
    draft = state.get("draft_answer", "")
    relevant_chunks = state.get("relevant_chunks", [])
    grader = _get_groundedness_grader()

    chunk_texts = [
        c["text"] if isinstance(c, dict) else str(c)
        for c in relevant_chunks
    ]
    result = grader.grade(answer=draft, chunks=chunk_texts)

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
            "unsupported_claims": result.unsupported_claims,
        },
    ))

    return {"unsupported_claims": result.unsupported_claims, "trace": trace}


# ---------------------------------------------------------------------------
# Node: regenerate
# ---------------------------------------------------------------------------


def node_regenerate(state: GraphState) -> Dict[str, Any]:
    """
    Re-generates the answer with an explicit instruction to avoid the previously
    flagged unsupported claims. Increments ``groundedness_retry_count``.

    Reads:  ``original_query``, ``relevant_chunks``, ``unsupported_claims``
    Writes: ``draft_answer``, ``groundedness_retry_count``, ``trace``
    """
    original_query = state["original_query"]
    relevant_chunks = state.get("relevant_chunks", [])
    unsupported_claims = state.get("unsupported_claims", [])
    groundedness_retry_count = state.get("groundedness_retry_count", 0) + 1

    # Build an augmented query that instructs the generator to exclude flagged claims.
    exclusion_note = (
        "\n\nIMPORTANT: Do NOT include or imply any of the following claims "
        "which are NOT supported by the provided context:\n"
        + "\n".join(f"  - {c}" for c in unsupported_claims)
    )
    augmented_query = original_query + exclusion_note

    chunk_dicts = [
        c if isinstance(c, dict) else {"text": c}
        for c in relevant_chunks
    ]
    draft = generate_answer(query=augmented_query, chunks=chunk_dicts)

    trace = list(state.get("trace", []))
    trace.append(_trace_entry(
        node="regenerate",
        query=original_query,
        decision=(
            f"Regenerated answer (groundedness retry #{groundedness_retry_count}). "
            f"Excluded {len(unsupported_claims)} claim(s)."
        ),
        detail={
            "groundedness_retry_count": groundedness_retry_count,
            "excluded_claims": unsupported_claims,
            "draft_preview": draft[:200],
        },
    ))

    return {
        "draft_answer": draft,
        "groundedness_retry_count": groundedness_retry_count,
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
        detail={"useful": result.useful, "reason": result.reason},
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
      4. Grades the answer for groundedness; regenerates excluding unsupported
         claims if needed (up to 2 retries).
      5. Grades the answer for usefulness; rewrites the query and restarts the
         full cycle if the answer is unhelpful (up to 1 retry).
      6. Returns the best-effort final answer with a full decision trace.

    Args:
        query: The user's natural-language query string.

    Returns:
        A dictionary with:
        - ``"final_answer"`` (str): The best answer the system could produce.
        - ``"trace"`` (list[dict]): Chronological log of every node visit,
          routing decision, and grading result.

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
        "retrieval_retry_count": 0,
        "groundedness_retry_count": 0,
        "usefulness_retry_count": 0,
        "falsification_done": False,
        "trace": [],
    }

    final_state: GraphState = _compiled_graph.invoke(initial_state)

    return {
        "final_answer": final_state.get("draft_answer", ""),
        "trace": final_state.get("trace", []),
    }
