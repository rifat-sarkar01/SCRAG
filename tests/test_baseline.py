"""
tests/test_baseline.py
-----------------------
Smoke tests for the baseline RAG pipeline (DocumentStore + generator).

All LLM calls are mocked — no API keys or network access required.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.document_store import DocumentStore


# ---------------------------------------------------------------------------
# Test 1 — DocumentStore: chunking produces correct boundaries
# ---------------------------------------------------------------------------


def test_document_store_splits_long_text_into_multiple_chunks() -> None:
    """
    A text clearly longer than chunk_size should be split into more than one
    chunk, and every chunk should fit within chunk_size (with a small slack for
    overlap prepending).
    """
    store = DocumentStore(chunk_size=100, chunk_overlap=10)

    # 10 paragraphs × ~60 chars each → well beyond 100 chars total.
    paragraphs = [
        "The mitochondria is the powerhouse of the cell and produces ATP energy.",
        "Photosynthesis converts sunlight into glucose inside chloroplasts.",
        "DNA stores genetic information in a double helix structure of nucleotides.",
        "Neurons transmit electrical signals across synapses in the nervous system.",
        "Enzymes are biological catalysts that speed up chemical reactions in cells.",
        "RNA transcribes DNA sequences and carries them to ribosomes for translation.",
        "Osmosis is the diffusion of water through a semipermeable membrane.",
        "Meiosis produces four haploid cells from one diploid cell via two divisions.",
        "Proteins are folded amino acid chains whose shape determines their function.",
        "Cell division in prokaryotes occurs by binary fission without a nucleus.",
    ]
    text = "\n\n".join(paragraphs)

    chunks = store._split(text)

    assert len(chunks) > 1, "Expected more than one chunk for a long multi-paragraph text."
    # Each chunk must be at most chunk_size + chunk_overlap chars (overlap head room).
    for chunk in chunks:
        assert len(chunk) <= store.chunk_size + store.chunk_overlap + 1, (
            f"Chunk exceeds size limit: {len(chunk)} chars.\nChunk: {chunk!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — generate_answer: correct prompt structure and LLM call
# ---------------------------------------------------------------------------


@patch("src.retrieval.generator._get_llm")
def test_generate_answer_calls_llm_with_context_and_returns_stripped_response(
    mock_get_llm: MagicMock,
) -> None:
    """
    generate_answer() should:
    - Concatenate chunk texts into the prompt.
    - Call the LLM exactly once.
    - Return the response content with surrounding whitespace stripped.
    """
    # --- Arrange ---
    mock_response = MagicMock()
    mock_response.content = "   The boiling point of water is 100 °C at sea level.   "
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    # Reset the module-level singleton so the mock is picked up.
    import src.retrieval.generator as gen_module
    gen_module._llm = None

    from src.retrieval.generator import generate_answer

    chunks = [
        {"text": "Water boils at 100 degrees Celsius (212 °F) at standard atmospheric pressure."},
        {"text": "At higher altitudes, the boiling point decreases due to lower air pressure."},
    ]

    # --- Act ---
    result = generate_answer(query="What is the boiling point of water?", chunks=chunks)

    # --- Assert ---
    assert result == "The boiling point of water is 100 °C at sea level.", (
        "Expected stripped response from mock LLM."
    )
    mock_llm.invoke.assert_called_once()

    # Verify that both chunk texts appear somewhere in the prompt messages.
    call_args = mock_llm.invoke.call_args[0][0]  # positional arg: list of messages
    full_prompt = " ".join(m.content for m in call_args)
    assert "100 degrees Celsius" in full_prompt
    assert "higher altitudes" in full_prompt
