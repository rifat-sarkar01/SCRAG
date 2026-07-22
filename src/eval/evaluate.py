"""
src/eval/evaluate.py
====================
Self-contained evaluation harness.

Loads  data/eval_qa.json, runs every question through both pipelines,
scores the answers with an extended metric suite, and writes a Markdown
comparison table to eval_results.md (project root).

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

Metrics scored
--------------
Generation quality (RAGAS):
  - ``faithfulness``        : fraction of answer claims supported by context.
  - ``answer_relevancy``    : how well the answer addresses the question.
  - ``answer_correctness``  : factual overlap + semantic similarity vs. ground truth.
  - ``context_entity_recall``: key entities from ground truth present in context.

Retrieval quality (RAGAS):
  - ``context_precision``   : fraction of retrieved chunks that are relevant.
  - ``context_recall``      : fraction of ground-truth-relevant chunks retrieved.

Self-correction-specific (computed locally):
  - ``hallucination_fix_rate``       : % of baseline-unfaithful claims fixed.
  - ``hallucination_injection_rate`` : % of baseline-faithful claims broken.
  - ``avg_correction_rounds``        : mean correction iterations per query.
  - ``max_rounds_hit_pct``           : % of queries that hit max correction rounds.
  - ``latency_overhead_s``           : mean added latency from correction loop.

Secondary:
  - ``bertscore_f1``  : BERTScore F1 vs. ground-truth answers.
  - ``rouge_l``       : ROUGE-L surface-overlap vs. ground-truth answers.

Context pairing fix
-------------------
The corrected-answer RAGAS samples use ``correction_context`` returned directly
by ``run_corrected_query`` — the **full, untruncated** chunks actually passed
to the correction agent.  This replaces the previous approach of scraping
120-char trace previews, which caused artificially low faithfulness scores.

Outputs
-------
- Console            : per-item answers (excerpts) + full summary table.
- eval_results.md    : Markdown comparison table (project root).
- data/eval_results_raw.json : Full per-item data (answers, contexts,
                               timing, all scores).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Maximum correction rounds cap — must match correction_graph._MAX_GROUNDEDNESS_RETRIES
_MAX_CORRECTION_ROUNDS: int = 2


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
# Section 3 — Context extraction helpers
# ===========================================================================


def _contexts_from_trace(trace: List[Dict[str, Any]]) -> List[str]:
    """
    LEGACY helper — walks the correction-graph trace and returns chunk texts
    from the *last* ``grade_chunks`` node visit.

    .. warning::
        This uses the ``chunk_full`` field (full text) when available,
        falling back to the 120-char truncated ``chunk`` preview.
        Prefer passing ``correction_context`` directly from the pipeline
        result to avoid any truncation.

    Returns a list with at least one element (sentinel string on miss).
    """
    contexts: List[str] = []
    for entry in trace:
        if entry.get("node") == "grade_chunks":
            grades = entry.get("detail", {}).get("grades", [])
            # Prefer full text if stored (new format); fall back to truncated preview.
            texts = [
                g.get("chunk_full") or g.get("chunk", "")
                for g in grades
                if g.get("chunk_full") or g.get("chunk")
            ]
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


_RAGAS_METRIC_COLS = [
    "faithfulness", "answer_relevancy", "answer_correctness",
    "context_entity_recall", "context_precision", "context_recall",
]


def _score_with_ragas(
    samples: List[Dict[str, Any]],
    label: str = "",
) -> "tuple[Dict[str, Any], List[Dict[str, Any]]]":
    """
    Runs RAGAS metrics on *samples* using the local Ollama model.

    Metrics attempted:
      - faithfulness, answer_relevancy (always)
      - answer_correctness, context_entity_recall (if available in installed RAGAS)
      - context_precision, context_recall (if available)

    Parameters
    ----------
    samples : list of dicts, each containing:
        ``user_input``, ``response``, ``retrieved_contexts``, ``reference``
    label   : used only in log messages.

    Returns
    -------
    Tuple of:
      - Dict with metric names as keys and float|None aggregate (mean) values.
        All values are ``None`` if scoring raised any exception.
      - List of per-row dicts (same order as *samples*), each mapping metric
        name -> float|None for that individual item. This is what per-item
        diagnostics and self-correction metrics MUST use instead of the
        aggregate — using the aggregate for every row silently collapses
        every item to the same number and makes all item-level comparisons
        meaningless (this was the bug: every item in the report showed the
        exact same baseline/corrected faithfulness pair).
    """
    tag = f" ({label})" if label else ""
    print(f"[RAGAS] Scoring{tag} — {len(samples)} sample(s)…")
    empty = {
        "faithfulness": None,
        "answer_relevancy": None,
        "answer_correctness": None,
        "context_entity_recall": None,
        "context_precision": None,
        "context_recall": None,
    }
    empty_rows = [dict(empty) for _ in samples]
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import faithfulness, answer_relevancy
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        # Fully local — no external API calls.
        # Uses OLLAMA_JUDGE_MODEL (same as the internal graph graders), NOT
        # OLLAMA_MODEL. Previously this was hardcoded to "qwen3-vl:8b-instruct-q8_0"
        # — the exact same model used to GENERATE the baseline answer — which
        # meant RAGAS was scoring the model's output using the model itself as
        # judge (self-evaluation bias), and silently ignored any env override.
        ragas_llm = LangchainLLMWrapper(
            ChatOllama(
                model=os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5:14b-instruct"),
                temperature=0,
                keep_alive="30m",
            )
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model="nomic-embed-text")
        )

        # Core metrics (always available).
        metrics_to_run = [faithfulness, answer_relevancy]
        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        # Extended generation metrics (RAGAS >= 0.2 required).
        try:
            from ragas.metrics import answer_correctness, context_entity_recall
            answer_correctness.llm = ragas_llm
            answer_correctness.embeddings = ragas_embeddings
            context_entity_recall.llm = ragas_llm
            metrics_to_run += [answer_correctness, context_entity_recall]
            print(f"[RAGAS]{tag} answer_correctness + context_entity_recall enabled.")
        except ImportError:
            print(f"[RAGAS]{tag} answer_correctness / context_entity_recall not "
                  "available in this RAGAS version — skipping.")

        # Retrieval metrics.
        try:
            from ragas.metrics import context_precision, context_recall
            context_precision.llm = ragas_llm
            context_recall.llm = ragas_llm
            metrics_to_run += [context_precision, context_recall]
            print(f"[RAGAS]{tag} context_precision + context_recall enabled.")
        except ImportError:
            print(f"[RAGAS]{tag} context_precision / context_recall not "
                  "available in this RAGAS version — skipping.")

        # Local-model reality check: the previous RunConfig only set
        # timeout=300 and max_workers=1, leaving max_retries at its default
        # of 10 and max_wait at its default of 60 — with log_tenacity off by
        # default. That meant a single stuck job (context_precision and
        # context_recall in particular make several chained sub-calls per
        # "job", not one) could silently retry up to 10 times with expon-
        # ential backoff (worst case ~10 x (300 + 60)s ≈ 1 hour) before
        # finally failing, with zero visibility into what was happening —
        # this is almost certainly what made jobs 24/26 dominate the total
        # run time. Raise the per-call ceiling (heavy metrics need more than
        # 300s on an 8B/14B local model), but cap retries low so a genuinely
        # stuck job fails fast instead of consuming the whole run, and turn
        # on retry logging so failures are visible instead of silent.
        run_config = RunConfig(
            timeout=600,
            max_retries=3,
            max_wait=30,
            max_workers=1,
            log_tenacity=True,
        )

        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_run,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )
        df = result.to_pandas()

        scores: Dict[str, Any] = {}
        for col in _RAGAS_METRIC_COLS:
            scores[col] = float(df[col].mean()) if col in df.columns else None

        # Per-row scores, aligned back to the *input* sample order. RAGAS
        # preserves row order 1:1 with the input dataset (a failed/timed-out
        # job becomes NaN in that row, it does not drop or reorder rows), so
        # positional zip against `samples` is safe here.
        per_row: List[Dict[str, Any]] = []
        for i in range(len(samples)):
            row: Dict[str, Any] = {}
            for col in _RAGAS_METRIC_COLS:
                if col in df.columns and i < len(df):
                    val = df[col].iloc[i]
                    row[col] = float(val) if val == val else None  # NaN check
                else:
                    row[col] = None
            per_row.append(row)

        print(
            f"[RAGAS]{tag} "
            + "  ".join(
                f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                for k, v in scores.items()
            )
        )
        return scores, per_row

    except BaseException as exc:  # noqa: BLE001 — catch KeyboardInterrupt/CancelledError too
        print(f"[RAGAS]{tag} Scoring failed: {type(exc).__name__}: {exc}")
        return empty, empty_rows


# ===========================================================================
# Section 5 — Secondary metric scorers (BERTScore, ROUGE-L)
# ===========================================================================


def _score_bertscore(
    predictions: List[str],
    references: List[str],
) -> Optional[float]:
    """
    Computes mean BERTScore F1 over all (prediction, reference) pairs.
    Returns None if ``bert-score`` is not installed.
    """
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            predictions, references, lang="en", verbose=False
        )
        return float(F1.mean().item())
    except Exception as exc:  # noqa: BLE001
        print(f"[BERTScore] Skipped: {exc}")
        return None


def _score_rouge_l(
    predictions: List[str],
    references: List[str],
) -> Optional[float]:
    """
    Computes mean ROUGE-L F1 over all (prediction, reference) pairs.
    Returns None if ``rouge-score`` is not installed.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for pred, ref in zip(predictions, references)
        ]
        return float(sum(scores) / len(scores)) if scores else None
    except Exception as exc:  # noqa: BLE001
        print(f"[ROUGE-L] Skipped: {exc}")
        return None


# ===========================================================================
# Section 6 — Self-correction-specific metrics
# ===========================================================================


def _compute_self_correction_metrics(
    per_item_results: List[Dict[str, Any]],
    baseline_ragas_df_rows: List[Dict[str, Any]],
    corrected_ragas_df_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes self-correction-specific metrics from per-item results.

    Metrics
    -------
    hallucination_fix_rate
        % of claims that were unsupported in the baseline answer and became
        supported after correction.  Approximated via per-item faithfulness scores
        (item-level: baseline_faithfulness < corrected_faithfulness → "fixed").

    hallucination_injection_rate
        % of queries where the corrected answer is *less* faithful than baseline.
        This is the key metric showing the current pipeline failure.

    avg_correction_rounds
        Average number of correction (regenerate) rounds per query.

    max_rounds_hit_pct
        Percentage of queries that hit the maximum correction rounds cap.

    avg_latency_overhead_s
        Mean additional wall-clock time from the correction loop vs. baseline.
    """
    n = len(per_item_results)
    if n == 0:
        return {
            "hallucination_fix_rate": None,
            "hallucination_injection_rate": None,
            "avg_correction_rounds": None,
            "max_rounds_hit_pct": None,
            "avg_latency_overhead_s": None,
        }

    # Per-item faithfulness from RAGAS (if available).
    base_faiths = [r.get("baseline_faithfulness") for r in per_item_results]
    corr_faiths = [r.get("corrected_faithfulness") for r in per_item_results]

    fix_count = 0
    inject_count = 0
    valid_faith_pairs = 0
    for bf, cf in zip(base_faiths, corr_faiths):
        if bf is None or cf is None:
            continue
        valid_faith_pairs += 1
        if bf < 1.0 and cf > bf:
            fix_count += 1          # was unfaithful, got better
        if cf < bf:
            inject_count += 1       # was faithful, got worse

    fix_rate = (fix_count / valid_faith_pairs * 100) if valid_faith_pairs > 0 else None
    inject_rate = (inject_count / valid_faith_pairs * 100) if valid_faith_pairs > 0 else None

    # Correction rounds.
    rounds_list = [r.get("correction_rounds", 0) for r in per_item_results]
    avg_rounds = sum(rounds_list) / n
    max_rounds_hit = sum(1 for r in rounds_list if r >= _MAX_CORRECTION_ROUNDS)
    max_rounds_pct = max_rounds_hit / n * 100

    # Latency overhead.
    overhead_list = [
        r.get("corrected_elapsed_s", 0.0) - r.get("baseline_elapsed_s", 0.0)
        for r in per_item_results
    ]
    avg_overhead = sum(overhead_list) / n

    return {
        "hallucination_fix_rate": fix_rate,
        "hallucination_injection_rate": inject_rate,
        "avg_correction_rounds": avg_rounds,
        "max_rounds_hit_pct": max_rounds_pct,
        "avg_latency_overhead_s": avg_overhead,
    }


# ===========================================================================
# Section 7 — Diagnostic logging (worst faithfulness samples)
# ===========================================================================


def _print_faithfulness_diagnostics(
    per_item_results: List[Dict[str, Any]],
    n_worst: int = 10,
) -> None:
    """
    Prints context side-by-side for the *n_worst* samples where faithfulness
    dropped hardest after correction.  This is Step 1 from the fix instructions:
    diagnose the context-pairing before drawing conclusions.
    """
    # Filter items where both scores are available.
    scoreable = [
        r for r in per_item_results
        if r.get("baseline_faithfulness") is not None
        and r.get("corrected_faithfulness") is not None
    ]
    if not scoreable:
        print("\n[Diagnostic] No per-item faithfulness scores available for diagnostics.")
        return

    # Sort by faithfulness drop (worst first).
    dropped = sorted(
        scoreable,
        key=lambda r: r["corrected_faithfulness"] - r["baseline_faithfulness"],
    )
    worst = dropped[:n_worst]

    bar = "=" * 80
    print(f"\n{bar}")
    print(f"  FAITHFULNESS DIAGNOSTICS — Top {len(worst)} worst-drop samples")
    print(bar)
    for i, r in enumerate(worst, start=1):
        bfaith = r.get("baseline_faithfulness", "N/A")
        cfaith = r.get("corrected_faithfulness", "N/A")
        drop = (cfaith - bfaith) if isinstance(bfaith, float) and isinstance(cfaith, float) else "N/A"
        print(f"\n  [{i}] Q: {r['question'][:80]}")
        print(f"       Faithfulness: baseline={bfaith:.3f}  corrected={cfaith:.3f}  "
              f"drop={drop:+.3f}" if isinstance(drop, float) else
              f"       Faithfulness: baseline={bfaith}  corrected={cfaith}  drop={drop}")

        print(f"\n  --- Baseline context ({len(r.get('baseline_contexts', []))} chunks) ---")
        for j, ctx in enumerate(r.get("baseline_contexts", [])[:3], 1):
            print(f"    [Chunk {j}] {ctx[:200].replace(chr(10), ' ')}…")

        print(f"\n  --- Correction context ({len(r.get('corrected_contexts', []))} chunks) ---")
        for j, ctx in enumerate(r.get("corrected_contexts", [])[:3], 1):
            print(f"    [Chunk {j}] {ctx[:200].replace(chr(10), ' ')}…")

        print(f"\n  Baseline answer  : {(r.get('baseline_answer') or '')[:150].replace(chr(10), ' ')}")
        print(f"  Corrected answer : {(r.get('corrected_answer') or '')[:150].replace(chr(10), ' ')}")
        print(f"  Correction rounds: {r.get('correction_rounds', 0)}")
        print(f"  Re-retrieval     : {r.get('reretrieval_happened', False)}")
    print(f"\n{bar}\n")


# ===========================================================================
# Section 8 — Markdown report builder
# ===========================================================================


def _fmt(val: float | None, decimals: int = 3) -> str:
    """Formats a metric value, returning 'N/A' for None."""
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def _build_report(
    results: List[Dict[str, Any]],
    baseline_scores: Dict[str, Any],
    corrected_scores: Dict[str, Any],
    self_correction_metrics: Dict[str, Any],
    secondary_scores: Dict[str, Any],
) -> str:
    """
    Builds the full Markdown report string.

    Sections
    --------
    1. Header with metadata.
    2. Per-item comparison table (question, answer excerpts, timing).
    3. Generation quality RAGAS scores table with Δ column.
    4. Retrieval quality RAGAS scores table.
    5. Self-correction-specific metrics.
    6. Secondary metrics (BERTScore, ROUGE-L).
    7. Brief interpretation notes.
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
         "| Baseline (s) | Corrected (s) | Corr. Rounds |"),
        "|--:|---|---|---|--:|--:|--:|",
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
        cr = r.get("correction_rounds", 0)
        lines.append(f"| {i} | {q} | {b} | {c} | {bt:.1f} | {ct:.1f} | {cr} |")

    # --- Generation quality ---
    lines += [
        "",
        "---",
        "",
        "## Generation Quality (RAGAS)",
        "",
        "| Metric | Baseline | Self-Correcting | Δ (Corrected − Baseline) |",
        "|---|--:|--:|--:|",
    ]
    for metric in ["faithfulness", "answer_relevancy", "answer_correctness",
                   "context_entity_recall"]:
        bv = baseline_scores.get(metric)
        cv = corrected_scores.get(metric)
        delta_str = (
            f"{cv - bv:+.3f}" if (bv is not None and cv is not None) else "N/A"
        )
        lines.append(
            f"| {metric} | {_fmt(bv)} | {_fmt(cv)} | {delta_str} |"
        )

    # --- Retrieval quality ---
    lines += [
        "",
        "---",
        "",
        "## Retrieval Quality (RAGAS)",
        "",
        "| Metric | Baseline | Self-Correcting | Δ (Corrected − Baseline) |",
        "|---|--:|--:|--:|",
    ]
    for metric in ["context_precision", "context_recall"]:
        bv = baseline_scores.get(metric)
        cv = corrected_scores.get(metric)
        delta_str = (
            f"{cv - bv:+.3f}" if (bv is not None and cv is not None) else "N/A"
        )
        lines.append(
            f"| {metric} | {_fmt(bv)} | {_fmt(cv)} | {delta_str} |"
        )

    # --- Self-correction-specific ---
    lines += [
        "",
        "---",
        "",
        "## Self-Correction Metrics",
        "",
        "| Metric | Value |",
        "|---|--:|",
    ]
    scm = self_correction_metrics
    lines += [
        f"| Hallucination Fix Rate (%) | {_fmt(scm.get('hallucination_fix_rate'), 1)} |",
        f"| Hallucination Injection Rate (%) | {_fmt(scm.get('hallucination_injection_rate'), 1)} |",
        f"| Avg Correction Rounds | {_fmt(scm.get('avg_correction_rounds'), 2)} |",
        f"| Max Rounds Hit (%) | {_fmt(scm.get('max_rounds_hit_pct'), 1)} |",
        f"| Avg Latency Overhead (s) | {_fmt(scm.get('avg_latency_overhead_s'), 1)} |",
    ]

    # --- Secondary ---
    lines += [
        "",
        "---",
        "",
        "## Secondary Metrics",
        "",
        "| Metric | Baseline | Self-Correcting |",
        "|---|--:|--:|",
    ]
    for key in ["bertscore_f1", "rouge_l"]:
        bv = secondary_scores.get(f"baseline_{key}")
        cv = secondary_scores.get(f"corrected_{key}")
        lines.append(f"| {key} | {_fmt(bv)} | {_fmt(cv)} |")

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
        ("- **Answer correctness**: factual overlap + semantic similarity "
         "vs. ground-truth answer."),
        ("- **Context entity recall**: key entities from ground truth "
         "present in retrieved context."),
        ("- **Hallucination Injection Rate**: the critical self-correction "
         "failure metric — should be near zero before reporting corrected faithfulness."),
        "- A positive Δ means the Self-Correcting pipeline outperformed the baseline.",
        ("- `N/A` means RAGAS raised an exception for that pipeline "
         "(see console output for details)."),
        ("- **Context pairing fix**: corrected-answer RAGAS contexts now use the "
         "full chunks returned by `run_corrected_query['correction_context']`, "
         "not the 120-char trace previews from the previous version."),
    ]

    return "\n".join(lines) + "\n"


# ===========================================================================
# Section 9 — Console summary printer
# ===========================================================================


def _print_summary(
    baseline: Dict[str, Any],
    corrected: Dict[str, Any],
    self_correction_metrics: Dict[str, Any],
    secondary_scores: Dict[str, Any],
) -> None:
    """Prints a concise summary table to stdout."""
    bar = "=" * 70
    print(f"\n{bar}")
    print("  EVALUATION SUMMARY")
    print(bar)
    print(f"  {'Metric':<34}  {'Baseline':>9}  {'Corrected':>11}  {'Δ':>8}")
    print(f"  {'-'*34}  {'-'*9}  {'-'*11}  {'-'*8}")

    # Generation + retrieval metrics.
    for metric in [
        "faithfulness", "answer_relevancy", "answer_correctness",
        "context_entity_recall", "context_precision", "context_recall",
    ]:
        bv = baseline.get(metric)
        cv = corrected.get(metric)
        if bv is not None and cv is not None:
            delta = cv - bv
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "═")
            delta_str = f"{arrow} {delta:+.3f}"
        else:
            delta_str = "N/A"
        print(
            f"  {metric:<34}  {_fmt(bv):>9}  {_fmt(cv):>11}  {delta_str:>8}"
        )

    # Self-correction metrics.
    print(f"\n  {'Self-Correction Metrics':<34}")
    scm = self_correction_metrics
    for label, key in [
        ("Hallucination Fix Rate (%)", "hallucination_fix_rate"),
        ("Hallucination Injection Rate (%)", "hallucination_injection_rate"),
        ("Avg Correction Rounds", "avg_correction_rounds"),
        ("Max Rounds Hit (%)", "max_rounds_hit_pct"),
        ("Avg Latency Overhead (s)", "avg_latency_overhead_s"),
    ]:
        val = scm.get(key)
        print(f"  {label:<34}  {_fmt(val, 2):>9}")

    # Secondary metrics.
    print(f"\n  {'Secondary Metrics':<34}")
    for key in ["bertscore_f1", "rouge_l"]:
        bv = secondary_scores.get(f"baseline_{key}")
        cv = secondary_scores.get(f"corrected_{key}")
        if bv is not None and cv is not None:
            delta = cv - bv
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "═")
            delta_str = f"{arrow} {delta:+.3f}"
        else:
            delta_str = "N/A"
        print(
            f"  {key:<34}  {_fmt(bv):>9}  {_fmt(cv):>11}  {delta_str:>8}"
        )

    print(bar)


# ===========================================================================
# Section 10 — Public entry point
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
    4. Score both answer sets with RAGAS (generation + retrieval metrics).
    5. Compute self-correction-specific metrics locally.
    6. Score BERTScore and ROUGE-L.
    7. Print faithfulness diagnostics for worst-drop samples.
    8. Write Markdown comparison table to ``output_path``.
    9. Write raw per-item JSON to ``raw_output_path``.
    10. Print a summary table to stdout.

    Context pairing fix
    -------------------
    The corrected-answer RAGAS samples use ``correction_context`` from the
    pipeline result directly — these are the full, untruncated chunks actually
    passed to the correction agent.  This replaces the previous approach of
    scraping 120-char trace previews.

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
    baseline_predictions: List[str] = []
    corrected_predictions: List[str] = []
    references_list: List[str] = []

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
            # CONTEXT PAIRING FIX: use correction_context directly (not trace previews).
            corrected_contexts: List[str] = corrected_result.get(
                "correction_context", []
            )
            correction_rounds: int = corrected_result.get("correction_rounds", 0)
            reretrieval_happened: bool = corrected_result.get("reretrieval_happened", False)
        except Exception as exc:  # noqa: BLE001
            corrected_answer = f"[Corrected error: {exc}]"
            corrected_trace = []
            corrected_contexts = []
            correction_rounds = 0
            reretrieval_happened = False
        corrected_elapsed = time.perf_counter() - t0

        # Fall back to trace-based context extraction if correction_context is empty.
        if not corrected_contexts:
            corrected_contexts = _contexts_from_trace(corrected_trace)

        print(f"[Eval]   Corrected ({corrected_elapsed:.1f}s, "
              f"rounds={correction_rounds}): "
              f"{corrected_answer[:100].replace(chr(10), ' ')}…")

        # Accumulate per-item record.
        record: Dict[str, Any] = {
            "question": question,
            "expected_facts": item.get("expected_facts", []),
            "reference": reference,
            "baseline_answer": baseline_answer,
            "baseline_contexts": baseline_contexts,
            "baseline_elapsed_s": round(baseline_elapsed, 2),
            "corrected_answer": corrected_answer,
            "corrected_contexts": corrected_contexts,
            "corrected_elapsed_s": round(corrected_elapsed, 2),
            "corrected_trace_len": len(corrected_trace),
            # Previously only the length was kept and the actual trace (which
            # holds claim_analysis, excluded_claims, and draft_preview for
            # every node visit) was discarded — making it impossible to see
            # *why* a correction round did what it did without re-running
            # with extra logging. Keep the real thing.
            "corrected_trace": corrected_trace,
            "correction_rounds": correction_rounds,
            "reretrieval_happened": reretrieval_happened,
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
            "retrieved_contexts": corrected_contexts or ["(none)"],
            "reference": reference,
        })

        baseline_predictions.append(baseline_answer)
        corrected_predictions.append(corrected_answer)
        references_list.append(reference)

    # 4. RAGAS scoring.
    print("\n[Eval] ── RAGAS Scoring ─────────────────────────────────────────")
    baseline_scores, baseline_rows = _score_with_ragas(baseline_ragas_samples, label="Baseline")
    # Reset the asyncio event loop between calls to prevent CancelledError cascade.
    _reset_asyncio_loop()
    corrected_scores, corrected_rows = _score_with_ragas(corrected_ragas_samples, label="Self-Correcting")

    # Attach TRUE per-item faithfulness scores to records (for self-correction
    # metrics and diagnostics). baseline_rows/corrected_rows are positionally
    # aligned with per_item_results (both built in the same test_items loop),
    # so zip is safe and gives each item its own real score instead of the
    # aggregate mean.
    for r, brow, crow in zip(per_item_results, baseline_rows, corrected_rows):
        r["baseline_faithfulness"] = brow.get("faithfulness")
        r["corrected_faithfulness"] = crow.get("faithfulness")
        r["baseline_scores"] = brow
        r["corrected_scores"] = crow

    # 5. Self-correction-specific metrics.
    print("\n[Eval] ── Self-Correction Metrics ──────────────────────────────")
    self_correction_metrics = _compute_self_correction_metrics(
        per_item_results, baseline_ragas_samples, corrected_ragas_samples
    )
    for k, v in self_correction_metrics.items():
        print(f"[Eval]   {k}: {_fmt(v, 2) if v is not None else 'N/A'}")

    # 6. Secondary metrics.
    print("\n[Eval] ── Secondary Metrics ─────────────────────────────────────")
    secondary_scores: Dict[str, Any] = {}
    if references_list:
        print("[Eval]   Computing BERTScore…")
        secondary_scores["baseline_bertscore_f1"] = _score_bertscore(
            baseline_predictions, references_list
        )
        secondary_scores["corrected_bertscore_f1"] = _score_bertscore(
            corrected_predictions, references_list
        )
        print("[Eval]   Computing ROUGE-L…")
        secondary_scores["baseline_rouge_l"] = _score_rouge_l(
            baseline_predictions, references_list
        )
        secondary_scores["corrected_rouge_l"] = _score_rouge_l(
            corrected_predictions, references_list
        )
        for k, v in secondary_scores.items():
            print(f"[Eval]   {k}: {_fmt(v, 3) if v is not None else 'N/A'}")
    for r in per_item_results:
        r["secondary_scores"] = secondary_scores

    # 7. Faithfulness diagnostics.
    print("\n[Eval] ── Faithfulness Diagnostics ─────────────────────────────")
    _print_faithfulness_diagnostics(per_item_results, n_worst=10)

    # 8 & 9. Write outputs.
    md = _build_report(
        per_item_results, baseline_scores, corrected_scores,
        self_correction_metrics, secondary_scores,
    )
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"\n[Eval] Markdown report  → {Path(output_path).resolve()}")

    Path(raw_output_path).write_text(
        json.dumps(per_item_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Eval] Raw JSON results → {Path(raw_output_path).resolve()}")

    # 10. Console summary.
    _print_summary(baseline_scores, corrected_scores, self_correction_metrics, secondary_scores)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    run_evaluation()
