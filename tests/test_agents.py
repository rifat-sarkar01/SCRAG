"""
tests/test_agents.py
--------------------
Unit tests for the four correction agents in src/agents/graders.py.

All tests mock the ChatOllama LLM — no real API calls are made.

Mocking strategy
----------------
ChatAnthropic is patched at import time in the graders module.
- For structured-output agents (Graders): the mock LLM's
  ``with_structured_output()`` returns a fake chain whose ``invoke()``
  returns the desired Pydantic model directly.
- For the plain-text agent (QueryRewriter): the mock LLM's ``invoke()``
  returns an object with a ``.content`` attribute.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.graders import (
    AnswerGroundednessGrader,
    AnswerUsefulnessGrader,
    GroundednessGrading,
    QueryRewriter,
    RetrievalGrader,
    RetrievalGrading,
    UsefulnessGrading,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_structured_llm_mock(return_value) -> MagicMock:
    """
    Builds a ChatOllama mock whose with_structured_output(...).invoke(...)
    returns *return_value*.
    """
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = return_value

    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


def _make_plain_llm_mock(content: str) -> MagicMock:
    """
    Builds a ChatOllama mock whose invoke(...) returns an object with
    .content = *content* (mimics a LangChain AIMessage).
    """
    mock_response = MagicMock()
    mock_response.content = content

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    return mock_llm


# ---------------------------------------------------------------------------
# Test 1 — RetrievalGrader: relevant chunk
# ---------------------------------------------------------------------------


@patch("src.agents.graders._build_llm")
def test_retrieval_grader_marks_relevant_chunk(mock_build_llm: MagicMock) -> None:
    """
    RetrievalGrader.grade() should return relevant=True when the mocked LLM
    indicates the chunk contributes useful signal for the query.
    """
    expected = RetrievalGrading(
        relevant=True,
        reason=(
            "The chunk directly describes metformin side effects in elderly patients "
            "with renal impairment."
        ),
    )
    mock_build_llm.return_value = _make_structured_llm_mock(expected)

    grader = RetrievalGrader()
    result = grader.grade(
        query="What are the side effects of metformin in elderly patients?",
        chunk=(
            "In older adults, metformin is associated with an elevated risk of lactic "
            "acidosis, particularly in those with renal impairment."
        ),
    )

    assert isinstance(result, RetrievalGrading)
    assert result.relevant is True
    assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# Test 2 — AnswerGroundednessGrader: ungrounded answer
# ---------------------------------------------------------------------------


@patch("src.agents.graders._build_llm")
def test_groundedness_grader_flags_unsupported_claims(mock_build_llm: MagicMock) -> None:
    """
    AnswerGroundednessGrader.grade() should return grounded=False and populate
    unsupported_claims when the answer contains facts not present in context.
    """
    expected = GroundednessGrading(
        grounded=False,
        unsupported_claims=[
            "attracts approximately 7 million visitors per year",
            "making it the most visited paid monument in the world",
        ],
    )
    mock_build_llm.return_value = _make_structured_llm_mock(expected)

    grader = AnswerGroundednessGrader()
    result = grader.grade(
        answer=(
            "The Eiffel Tower was completed in 1889. It attracts approximately "
            "7 million visitors per year, making it the most visited paid monument "
            "in the world."
        ),
        chunks=[
            "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
            "It was designed by Gustave Eiffel's engineering company.",
        ],
    )

    assert isinstance(result, GroundednessGrading)
    assert result.grounded is False
    assert len(result.unsupported_claims) == 2
    assert "7 million visitors" in result.unsupported_claims[0]


# ---------------------------------------------------------------------------
# Test 3 — AnswerUsefulnessGrader: not-useful answer
# ---------------------------------------------------------------------------


@patch("src.agents.graders._build_llm")
def test_usefulness_grader_marks_evasive_answer_not_useful(mock_build_llm: MagicMock) -> None:
    """
    AnswerUsefulnessGrader.grade() should return useful=False when the answer
    is too generic to resolve the user's specific question.
    """
    expected = UsefulnessGrading(
        useful=False,
        reason=(
            "The answer discusses linked lists in general but never explains how "
            "to actually reverse one."
        ),
    )
    mock_build_llm.return_value = _make_structured_llm_mock(expected)

    grader = AnswerUsefulnessGrader()
    result = grader.grade(
        query="How do I reverse a linked list in Python?",
        answer=(
            "Linked lists are a fundamental data structure. There are many approaches "
            "to working with them. Consider time and space complexity carefully."
        ),
    )

    assert isinstance(result, UsefulnessGrading)
    assert result.useful is False
    assert len(result.reason) > 0


# ---------------------------------------------------------------------------
# Test 4 (bonus) — QueryRewriter: produces a non-empty rewritten string
# ---------------------------------------------------------------------------


@patch("src.agents.graders._build_llm")
def test_query_rewriter_returns_non_empty_string(mock_build_llm: MagicMock) -> None:
    """
    QueryRewriter.rewrite() should return a plain string (not empty, not quoted)
    when the LLM is mocked to return a rewritten query.
    """
    rewritten = "metformin lactic acidosis risk factors renal impairment elderly"
    mock_build_llm.return_value = _make_plain_llm_mock(f'  {rewritten}  ')

    rewriter = QueryRewriter()
    result = rewriter.rewrite(
        query="metformin side effects old people",
        irrelevant_chunks=["Diabetes is a metabolic disorder affecting insulin."],
    )

    assert isinstance(result, str)
    assert result == rewritten   # strip() must have been applied
    assert len(result) > 0
