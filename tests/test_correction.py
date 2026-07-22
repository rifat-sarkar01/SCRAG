"""
tests/test_correction.py
-------------------------
Regression test for the claim_analysis wiring fix in src/graph/correction_graph.py.

Bug this guards against
------------------------
node_regenerate() used to call _generate_corrected_answer() with ONLY the
`unsupported_claims` list — the groundedness grader's full claim-by-claim
walkthrough (which claims WERE supported, and by which chunk) was computed
by the grader but silently discarded. The correction model then had to
re-derive "what's still supported" from scratch under prompt pressure alone,
and reliably failed to: it deleted an entire supported topic (one full side
of a "difference between X and Y" comparison) instead of surgically removing
only the flagged clause — even after the correction PROMPT was hardened with
explicit "don't over-delete" rules. Handing the grader's own claim_analysis
directly into the correction prompt is the actual fix; this test asserts
that wiring stays in place.

All tests mock the ChatOllama LLM — no real API calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.graph.correction_graph import _generate_corrected_answer


@patch("src.graph.correction_graph._get_correction_llm")
def test_correction_prompt_includes_claim_analysis_when_provided(
    mock_get_llm: MagicMock,
) -> None:
    """
    When claim_analysis is supplied, it must appear in the prompt sent to the
    correction LLM, clearly separated from the raw unsupported_claims list —
    this is what lets the corrector keep supported content instead of
    re-deriving (and likely under-detecting) it from scratch.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="corrected text")
    mock_get_llm.return_value = mock_llm

    claim_analysis = (
        '1. "Supervised learning uses labeled data to map inputs to outputs" '
        "— supported by chunk [2]\n"
        '2. "reducing dimensions" — UNSUPPORTED, no chunk mentions dimensionality'
    )

    _generate_corrected_answer(
        original_query="What is the difference between supervised and unsupervised learning?",
        original_answer=(
            "Supervised learning uses labeled data to map inputs to outputs, "
            "while unsupervised learning finds patterns and reduces dimensions."
        ),
        relevant_chunks=[
            "Unsupervised Learning: finds patterns in unlabeled data.",
            "Supervised Learning: The model learns from labeled data. Goal: Map input (X) to output (Y).",
        ],
        unsupported_claims=["reducing dimensions"],
        claim_analysis=claim_analysis,
    )

    assert mock_llm.invoke.called
    sent_messages = mock_llm.invoke.call_args[0][0]
    user_message_text = sent_messages[1].content

    assert "GROUNDEDNESS ANALYSIS" in user_message_text
    assert claim_analysis in user_message_text
    # The analysis block must appear BEFORE the bare unsupported-claims list,
    # so the model reads "what's already verified as supported" first.
    assert user_message_text.index("GROUNDEDNESS ANALYSIS") < user_message_text.index(
        "CLAIMS FLAGGED AS UNSUPPORTED"
    )


@patch("src.graph.correction_graph._get_correction_llm")
def test_correction_prompt_omits_analysis_block_when_not_provided(
    mock_get_llm: MagicMock,
) -> None:
    """
    Backward compatibility: calling without claim_analysis (the old call
    signature) must still work and must not inject an empty/broken analysis
    section into the prompt.
    """
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="corrected text")
    mock_get_llm.return_value = mock_llm

    _generate_corrected_answer(
        original_query="q",
        original_answer="a",
        relevant_chunks=["chunk"],
        unsupported_claims=["bad claim"],
    )

    sent_messages = mock_llm.invoke.call_args[0][0]
    user_message_text = sent_messages[1].content

    assert "GROUNDEDNESS ANALYSIS" not in user_message_text
    assert "CLAIMS FLAGGED AS UNSUPPORTED" in user_message_text


# ---------------------------------------------------------------------------
# Test — the _usefulness_result silent state-drop bug
# ---------------------------------------------------------------------------


@patch("src.graph.correction_graph._get_usefulness_grader")
def test_usefulness_result_survives_langgraph_state_merge(
    mock_get_grader: MagicMock,
) -> None:
    """
    Regression test for a confirmed, silent, high-impact bug: GraphState did
    NOT declare `_usefulness_result` as a field. LangGraph drops any key a
    node returns that isn't part of the declared state schema — no error, no
    warning. node_grade_usefulness correctly computed and logged "not useful"
    in the trace, but the routing-control key itself never survived the state
    merge, so route_after_grade_usefulness always read the fallback default
    (True) and NEVER routed to rewrite_query — the entire usefulness retry
    loop was dead code regardless of what the grader decided, confirmed via a
    real eval run's raw trace where a correctly-flagged "not useful" answer
    was still returned as the final answer.

    IMPORTANT: this must run through the REAL langgraph StateGraph merge
    engine, not a plain function call or dict check — the bug does not
    reproduce with a naive `result["_usefulness_result"] == False` assertion,
    since node_grade_usefulness's raw Python return value is correct; only
    the graph's internal state-channel merge silently drops it. A test that
    doesn't exercise the actual compiled graph would not have caught this.
    """
    from langgraph.graph import StateGraph, START, END
    from src.graph.correction_graph import (
        GraphState,
        node_grade_usefulness,
        route_after_grade_usefulness,
    )

    mock_grader = MagicMock()
    mock_grader.grade.return_value = MagicMock(
        useful=False, reasoning="only covers half of the comparison asked for"
    )
    mock_get_grader.return_value = mock_grader

    visited: list[str] = []

    def node_stub_b(state: GraphState):
        visited.append("rewrite_query")
        return {}

    # Minimal 2-node graph using the REAL GraphState schema and the REAL
    # node/router pair under test — this is what actually exercises
    # LangGraph's channel-merge behavior.
    g = StateGraph(GraphState)
    g.add_node("grade_usefulness", node_grade_usefulness)
    g.add_node("rewrite_query", node_stub_b)
    g.add_edge(START, "grade_usefulness")
    g.add_conditional_edges(
        "grade_usefulness",
        route_after_grade_usefulness,
        {"rewrite_query": "rewrite_query", "falsify": END, END: END},
    )
    g.add_edge("rewrite_query", END)
    compiled = g.compile()

    compiled.invoke({
        "original_query": "difference between supervised and unsupervised learning",
        "draft_answer": "Unsupervised learning finds patterns in unlabeled data.",
        "usefulness_retry_count": 0,
        "trace": [],
    })

    assert visited == ["rewrite_query"], (
        "route_after_grade_usefulness did not route to rewrite_query for a "
        "not-useful answer — the _usefulness_result field was likely dropped "
        "from GraphState again."
    )

