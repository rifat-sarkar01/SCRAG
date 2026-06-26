"""
src/eval/evaluate.py
====================
Self-contained evaluation harness.

Loads  data/eval_qa.json, runs every question through both pipelines,
scores the answers with RAGAS faithfulness + answer_relevancy, and writes
a Markdown comparison table to eval_results.md (project root).

Usage
-----
    # From the project root:
    python -m src.eval.evaluate

    # Or directly:
    python src/eval/evaluate.py

Test-set format  (data/eval_qa.json)
-------------------------------------
A JSON array of objects, each with:

    {
      "question": "...",
      "expected_facts": ["fact 1", "fact 2", ...]   # optional
      "expected_answer": "..."                       # optional alternative
    }

  ``expected_facts`` entries are joined and used as the RAGAS ``reference``
  field; ``expected_answer`` is used instead when present.

Pipelines compared
------------------
- **Baseline**  — ``generate_answer`` (Phase 1): retrieve → generate.
- **Corrected** — ``run_corrected_query`` (Phase 3): the full self-correcting
  LangGraph state machine (retrieve → grade → rewrite → generate →
  groundedness-check → usefulness-check).

RAGAS metrics scored
--------------------
- ``faithfulness``     : fraction of answer claims supported by context.
- ``answer_relevancy`` : how well the answer addresses the question.

Outputs
-------
- Console            : per-item answers (excerpts) + final summary table.
- eval_results.md    : Markdown comparison table (project root).
- data/eval_results_raw.json : Full per-item data (answers, contexts,
                               timing, RAGAS scores).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so this file works both as
# ``python -m src.eval.evaluate`` and ``python src/eval/evaluate.py``.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent   # …/SCRAG/src/eval → …/SCRAG
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env early so OLLAMA_MODEL / OLLAMA_BASE_URL are available.
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass


# ===========================================================================
# Section 1 — Test-set loader
# ===========================================================================


def _load_test_set(path: str | Path) -> List[Dict[str, Any]]:
    """
    Reads *path* as a JSON array and returns the list of QA items.

    Each item must contain at minimum a ``"question"`` key.
    ``"expected_facts"`` (list[str]) and ``"expected_answer"`` (str) are
    both optional; either is used as the RAGAS reference field.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not contain a non-empty JSON array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Test-set not found: {p.resolve()}\n"
            "Create data/eval_qa.json with a list of "
            '{"question": "...", "expected_facts": [...]} objects.'
        )
    items: List[Dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{p} must be a non-empty JSON array.")
    return items


def _reference_for_item(item: Dict[str, Any]) -> str:
    """Returns the ground-truth reference string for a test item."""
    if "expected_answer" in item:
        return item["expected_answer"]
    return " ".join(item.get("expected_facts", []))


# ===========================================================================
# Section 2 — Corpus index builder (baseline pipeline only)
# ===========================================================================


def _build_baseline_index(data_dir: str) -> Any:
    """
    Builds an EmbeddingIndex over every .txt/.md file in *data_dir*,
    excluding eval_qa.json.  Falls back to a single-chunk placeholder if no
    corpus documents are found, so ``index.retrieve()`` never raises.
    """
    from src.retrieval.document_store import DocumentStore
    from src.retrieval.embedding_index import EmbeddingIndex

    store = DocumentStore(chunk_size=500, chunk_overlap=50)
    index = EmbeddingIndex()
    corpus_docs: List[Dict[str, Any]] = []

    try:
        docs = store.load_documents(data_dir)
        corpus_docs = [d for d in docs if d.get("filename") != "eval_qa.json"]
    except Exception as exc:  # noqa: BLE001
        print(f"[Eval] Warning: could not load corpus documents ({exc}).")

    if corpus_docs:
        chunks = store.chunk_documents(corpus_docs)
        index.build_index(chunks)
        print(f"[Eval] Corpus index built: {len(corpus_docs)} doc(s), "
              f"{len(chunks)} chunk(s).")
    else:
        index.build_index([
            {"text": "No corpus documents are available.", "source": "placeholder"}
        ])
        print("[Eval] No corpus documents found — using placeholder index.")

    return index


# ===========================================================================
# Section 3 — Context extraction from the correction-graph trace
# ===========================================================================


def _contexts_from_trace(trace: List[Dict[str, Any]]) -> List[str]:
    """
    Walks the correction-graph trace produced by ``run_corrected_query``
    and returns chunk texts from the *last* ``grade_chunks`` node visit.

    The ``grade_chunks`` node stores chunk previews (truncated to 120 chars)
    in ``detail.grades[].chunk``.  These are used as the RAGAS context even
    though they may be slightly truncated.

    Returns a list with at least one element (sentinel string on miss).
    """
    contexts: List[str] = []
    for entry in trace:
        if entry.get("node") == "grade_chunks":
            grades = entry.get("detail", {}).get("grades", [])
            texts = [g["chunk"] for g in grades if g.get("chunk")]
            if texts:
                contexts = texts
    return contexts or ["(no context retrieved)"]


# ===========================================================================
# Section 4 — RAGAS scorer
# ===========================================================================


def _reset_asyncio_loop() -> None:
    """
    Closes and replaces the current asyncio event loop.

    When a RAGAS scoring call fails with ConnectErrors and leaves pending
    cancelled coroutines on the loop, the next ``asyncio.run()`` call (used
    internally by RAGAS) will raise ``CancelledError`` → ``KeyboardInterrupt``.
    Replacing the loop between calls prevents this dirty-state cascade.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass
    asyncio.set_event_loop(asyncio.new_event_loop())


def _score_with_ragas(
    samples: List[Dict[str, Any]],
    label: str = "",
) -> Dict[str, Any]:
    """
    Runs RAGAS ``faithfulness`` + ``answer_relevancy`` on *samples*.

    Uses RAGAS's native Ollama support via the OpenAI-compatible endpoint
    (``http://localhost:11434/v1``) instead of the deprecated
    ``LangchainLLMWrapper`` / ``LangchainEmbeddingsWrapper`` approach.

    Parameters
    ----------
    samples : list of dicts, each containing:
        ``user_input``, ``response``, ``retrieved_contexts``, ``reference``
    label   : used only in log messages.

    Returns
    -------
    ``{"faithfulness": float|None, "answer_relevancy": float|None}``
    Both values are ``None`` if scoring raised any exception, including
    ``KeyboardInterrupt`` and ``CancelledError`` from a dirty asyncio state.
    """
    tag = f" ({label})" if label else ""
    print(f"[RAGAS] Scoring{tag} — {len(samples)} sample(s)…")
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        # Fully local — no external API calls.
        ragas_llm = LangchainLLMWrapper(
            ChatOllama(model="qwen3-vl:8b-instruct-q8_0", temperature=0)
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model="nomic-embed-text")
        )

        # Explicitly bind llm/embeddings per-metric to prevent silent
        # fallback to OpenAI defaults inside RAGAS internals.
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        run_config = RunConfig(timeout=300, max_workers=1)

        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )
        df = result.to_pandas()
        scores: Dict[str, Any] = {
            "faithfulness": float(df["faithfulness"].mean()),
            "answer_relevancy": float(df["answer_relevancy"].mean()),
        }
        print(f"[RAGAS]{tag} faithfulness={scores['faithfulness']:.3f}, "
              f"answer_relevancy={scores['answer_relevancy']:.3f}")
        return scores
    except BaseException as exc:  # noqa: BLE001 — catch KeyboardInterrupt/CancelledError too
        print(f"[RAGAS]{tag} Scoring failed: {type(exc).__name__}: {exc}")
        return {"faithfulness": None, "answer_relevancy": None}


# ===========================================================================
# Section 5 — Markdown report builder
# ===========================================================================


def _fmt(val: float | None, decimals: int = 3) -> str:
    """Formats a metric value, returning 'N/A' for None."""
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def _build_report(
    results: List[Dict[str, Any]],
    baseline_scores: Dict[str, Any],
    corrected_scores: Dict[str, Any],
) -> str:
    """
    Builds the full Markdown report string.

    Sections
    --------
    1. Header with metadata.
    2. Per-item comparison table (question, answer excerpts, timing).
    3. Aggregate RAGAS scores table with Δ column.
    4. Brief interpretation notes.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    lines: List[str] = [
        "# Evaluation Results: Baseline RAG vs Self-Correcting RAG",
        "",
        f"> Generated: {ts}  ",
        f"> Test items: **{len(results)}**",
        "",
        "---",
        "",
        "## Per-Item Comparison",
        "",
        ("| # | Question "
         "| Baseline answer (excerpt) "
         "| Corrected answer (excerpt) "
         "| Baseline (s) | Corrected (s) |"),
        "|--:|---|---|---|--:|--:|",
    ]

    for i, r in enumerate(results, start=1):
        q = r["question"][:65].replace("|", "\\|")
        if len(r["question"]) > 65:
            q += "…"

        def _trunc(text: str, n: int = 95) -> str:
            t = text.replace("|", "\\|").replace("\n", " ")
            return (t[:n] + "…") if len(t) > n else t

        b = _trunc(r.get("baseline_answer") or "")
        c = _trunc(r.get("corrected_answer") or "")
        bt = r.get("baseline_elapsed_s", 0.0)
        ct = r.get("corrected_elapsed_s", 0.0)
        lines.append(f"| {i} | {q} | {b} | {c} | {bt:.1f} | {ct:.1f} |")

    lines += [
        "",
        "---",
        "",
        "## Aggregate RAGAS Scores",
        "",
        "| Metric | Baseline | Self-Correcting | Δ (Corrected − Baseline) |",
        "|---|--:|--:|--:|",
    ]
    for metric in ["faithfulness", "answer_relevancy"]:
        bv = baseline_scores.get(metric)
        cv = corrected_scores.get(metric)
        delta_str = (
            f"{cv - bv:+.3f}" if (bv is not None and cv is not None) else "N/A"
        )
        lines.append(
            f"| {metric} | {_fmt(bv)} | {_fmt(cv)} | {delta_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
        ("- **Faithfulness**: fraction of answer claims supported by the "
         "retrieved context (higher = fewer hallucinations)."),
        ("- **Answer relevancy**: how directly the answer addresses the "
         "question (higher = more on-topic)."),
        "- A positive Δ means the Self-Correcting pipeline outperformed the baseline.",
        ("- `N/A` means RAGAS raised an exception for that pipeline "
         "(see console output for details)."),
    ]

    return "\n".join(lines) + "\n"


# ===========================================================================
# Section 6 — Console summary printer
# ===========================================================================


def _print_summary(
    baseline: Dict[str, Any],
    corrected: Dict[str, Any],
) -> None:
    """Prints a concise summary table to stdout."""
    bar = "=" * 58
    print(f"\n{bar}")
    print("  EVALUATION SUMMARY")
    print(bar)
    print(f"  {'Metric':<24}  {'Baseline':>9}  {'Corrected':>11}  {'Δ':>8}")
    print(f"  {'-'*24}  {'-'*9}  {'-'*11}  {'-'*8}")
    for metric in ["faithfulness", "answer_relevancy"]:
        bv = baseline.get(metric)
        cv = corrected.get(metric)
        if bv is not None and cv is not None:
            delta = cv - bv
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "═")
            delta_str = f"{arrow} {delta:+.3f}"
        else:
            delta_str = "N/A"
        print(
            f"  {metric:<24}  {_fmt(bv):>9}  {_fmt(cv):>11}  {delta_str:>8}"
        )
    print(bar)


# ===========================================================================
# Section 7 — Public entry point
# ===========================================================================


def run_evaluation(
    test_set_path: str = "data/eval_qa.json",
    data_dir: str = "data",
    output_path: str = "eval_results.md",
    raw_output_path: str = "data/eval_results_raw.json",
) -> None:
    """
    Runs the full evaluation: load → run both pipelines → score → report.

    Step-by-step
    ------------
    1. Load ``test_set_path`` (list of {question, expected_facts}).
    2. Build an EmbeddingIndex over ``data_dir`` for the baseline pipeline.
    3. For every test item:
       a. **Baseline**  — retrieve k=4 chunks + ``generate_answer``.
       b. **Corrected** — ``run_corrected_query`` (full LangGraph graph).
    4. Score both answer sets with RAGAS faithfulness + answer_relevancy.
    5. Write Markdown comparison table to ``output_path``.
    6. Write raw per-item JSON to ``raw_output_path``.
    7. Print a summary table to stdout.

    Parameters
    ----------
    test_set_path   : path to the JSON evaluation set.
    data_dir        : directory searched for corpus documents (.txt/.md).
    output_path     : destination for the Markdown comparison table.
    raw_output_path : destination for raw JSON results.
    """
    # ---- Lazy imports: avoid LLM construction before we actually need it ----
    from src.retrieval.generator import generate_answer         # Phase 1 baseline
    from src.graph.correction_graph import run_corrected_query  # Phase 3

    # 1. Load test set.
    test_items = _load_test_set(test_set_path)
    n = len(test_items)
    print(f"[Eval] Loaded {n} test item(s) from '{test_set_path}'.")

    # 2. Build baseline index.
    print("[Eval] Building baseline corpus index…")
    index = _build_baseline_index(data_dir)

    per_item_results: List[Dict[str, Any]] = []
    baseline_ragas_samples: List[Dict[str, Any]] = []
    corrected_ragas_samples: List[Dict[str, Any]] = []

    # 3. Run both pipelines for every question.
    for idx, item in enumerate(test_items, start=1):
        question: str = item["question"]
        reference: str = _reference_for_item(item)
        print(f"\n[Eval] ── Item {idx}/{n} ─────────────────────────────────────")
        print(f"[Eval]   Q: {question}")

        # ── Baseline (Phase 1) ─────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            raw_hits = index.retrieve(question, k=4)
            baseline_chunks = [
                r[0] if isinstance(r[0], dict) else {"text": str(r[0])}
                for r in raw_hits
            ]
            baseline_contexts = [c["text"] for c in baseline_chunks]
            baseline_answer = generate_answer(query=question, chunks=baseline_chunks)
        except Exception as exc:  # noqa: BLE001
            baseline_answer = f"[Baseline error: {exc}]"
            baseline_contexts = []
        baseline_elapsed = time.perf_counter() - t0
        print(f"[Eval]   Baseline  ({baseline_elapsed:.1f}s): "
              f"{baseline_answer[:100].replace(chr(10), ' ')}…")

        # ── Self-Correcting (Phase 3) ──────────────────────────────────────
        t0 = time.perf_counter()
        try:
            corrected_result = run_corrected_query(query=question)
            corrected_answer: str = corrected_result["final_answer"]
            corrected_trace: List[Dict[str, Any]] = corrected_result["trace"]
        except Exception as exc:  # noqa: BLE001
            corrected_answer = f"[Corrected error: {exc}]"
            corrected_trace = []
        corrected_elapsed = time.perf_counter() - t0
        corrected_contexts = _contexts_from_trace(corrected_trace)
        print(f"[Eval]   Corrected ({corrected_elapsed:.1f}s): "
              f"{corrected_answer[:100].replace(chr(10), ' ')}…")

        # Accumulate per-item record.
        record: Dict[str, Any] = {
            "question": question,
            "expected_facts": item.get("expected_facts", []),
            "baseline_answer": baseline_answer,
            "baseline_contexts": baseline_contexts,
            "baseline_elapsed_s": round(baseline_elapsed, 2),
            "corrected_answer": corrected_answer,
            "corrected_contexts": corrected_contexts,
            "corrected_elapsed_s": round(corrected_elapsed, 2),
            "corrected_trace_len": len(corrected_trace),
        }
        per_item_results.append(record)

        baseline_ragas_samples.append({
            "user_input": question,
            "response": baseline_answer,
            "retrieved_contexts": baseline_contexts or ["(none)"],
            "reference": reference,
        })
        corrected_ragas_samples.append({
            "user_input": question,
            "response": corrected_answer,
            "retrieved_contexts": corrected_contexts,
            "reference": reference,
        })

    # 4. RAGAS scoring.
    print("\n[Eval] ── RAGAS Scoring ─────────────────────────────────────────")
    baseline_scores = _score_with_ragas(baseline_ragas_samples, label="Baseline")
    # Reset the asyncio event loop between calls: if the first scoring run left
    # pending cancelled coroutines (e.g. from ConnectErrors), a fresh loop
    # prevents CancelledError → KeyboardInterrupt on the second run.
    _reset_asyncio_loop()
    corrected_scores = _score_with_ragas(corrected_ragas_samples, label="Self-Correcting")

    # Attach aggregate scores to every record for raw JSON consumers.
    for r in per_item_results:
        r["baseline_scores"] = baseline_scores
        r["corrected_scores"] = corrected_scores

    # 5 & 6. Write outputs.
    md = _build_report(per_item_results, baseline_scores, corrected_scores)
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"\n[Eval] Markdown report  → {Path(output_path).resolve()}")

    Path(raw_output_path).write_text(
        json.dumps(per_item_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Eval] Raw JSON results → {Path(raw_output_path).resolve()}")

    # 7. Console summary.
    _print_summary(baseline_scores, corrected_scores)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_evaluation()
