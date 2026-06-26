# Self-Correcting RAG — Build Guide for Google Antigravity

> Goal: build a Self-RAG/CRAG-style self-correcting RAG system (reflection agents + LangGraph), using Antigravity's free preview, while spending Claude Sonnet/Opus credits carefully.
>
> ⚠️ **Check before you start:** Antigravity's free-tier quotas and model lineup have changed several times since its Nov 2025 launch (cuts in Dec 2025, Feb 2026, March 2026; a 2.0 relaunch at I/O 2026 with new paid tiers). Open Antigravity's model picker and quota indicator *right now* and confirm what's actually available to you — treat every number below as "true as of this writing," not gospel.

---

## 0. How Antigravity Credits Actually Work (read this first)

- Antigravity has two views: **Editor** (normal IDE) and **Manager** (orchestrates multiple parallel agents in "missions").
- You can assign a **different model to each agent**, even within the same mission.
- Typical model lineup: Gemini Flash (cheapest/fastest), Gemini Pro, Claude Sonnet, Claude Opus (Thinking), GPT-OSS. Opus produces the best reasoning but burns through credits fastest; Flash is nearly free but weak on multi-step reasoning; Sonnet is the middle ground.
- Free-tier rate limits have been cut repeatedly and refresh windows have been inconsistent (sometimes ~5hr cycles, sometimes weekly). **Don't plan a whole project around the free tier holding steady for weeks** — build in small, committable chunks.

**The two cardinal rules for this project, given limited credits:**
1. **Never let an expensive model do cheap work.** Boilerplate, file scaffolding, README writing, test stubs → cheapest model available.
2. **Never let a cheap model do hard reasoning.** Designing the correction-loop logic, writing grader prompts, debugging subtle agentic bugs → escalate to Sonnet, then Opus only if Sonnet stalls.

---

## 1. Which Free Agent to Select First

**Start with the cheapest/default model (Gemini Flash or whichever is set as default) for Phase 0 only** — pure scaffolding, zero reasoning required, no reason to spend Claude credits on it.

Then:
- **Claude Sonnet** becomes your main driver for almost the entire build (Phases 1–4). Best cost/quality ratio for sustained iterative coding.
- **Claude Opus (Thinking)** is reserved — not your default — for four specific moments: designing the LangGraph routing logic, writing the grader/verifier prompts, the optional falsification agent (Phase 5), and any bug Sonnet fails to fix after two tries. Treat Opus as a scalpel, not your daily driver.

| Task type | Model | Why |
|---|---|---|
| Repo scaffold, file structure, requirements.txt, README, .gitignore | Cheapest/default (Flash) | Zero reasoning needed |
| Baseline RAG implementation (chunking, embedding, retrieval, generation) | Sonnet | Standard coding, Sonnet handles it cleanly |
| Designing grader/verifier prompts & rubrics | Opus | Needs careful reasoning about edge cases |
| Implementing graders from an approved design | Sonnet | Mechanical once the design is settled |
| LangGraph state machine **design** | Opus | Control-flow correctness matters; mistakes here cascade |
| LangGraph **implementation** from an approved design | Sonnet | Wiring, not designing |
| Evaluation harness | Sonnet or Flash | Mostly boilerplate + library calls |
| Falsification/adversarial agent (Phase 5, optional) | Opus | Subtlest reasoning in the whole project |
| Write-up / README polish | Flash | Formatting only |

---

## 2. Credit-Saving Rules of Thumb

- **One phase = one mission.** Don't run the whole project as one giant mission. A bad mission wastes credits on output you'll discard anyway — small missions are cheap to retry.
- **Plan-first, always.** Every prompt below ends with "show me a plan before writing code" or similar. Read the plan. Only say "proceed" if it's right. Paying for a wrong-direction execution is the single biggest credit-waster.
- **Be explicit about file scope.** Vague prompts like "look at my whole project and fix X" make agents read far more files than needed. Every prompt below names exact files/folders.
- **Commit to git after every working phase.** If a mission goes sideways, you roll back with `git checkout` instead of spending more credits asking the agent to undo its own mess.
- **Batch trivial work.** Do README, .gitignore, docstrings, formatting in one cheap-model mission, not scattered across several expensive-model ones.
- **Mock LLM calls in tests.** Don't let test runs make real API calls — that's credits spent on repeated, identical requests.

---

## 3. Step-by-Step Build Plan with Exact Prompts

### Phase 0 — Repo Scaffold
**Model: cheapest/default (Gemini Flash)**

```
Create a Python project scaffold for a "Self-Correcting RAG" system.

Structure:
/src
  /retrieval      (embedding + vector store wrapper)
  /agents         (grader, rewriter, verifier agents — stub files only)
  /graph          (LangGraph state machine — stub)
  /eval           (evaluation harness — stub)
/data             (placeholder for corpus)
/tests
requirements.txt  (faiss-cpu, langchain, langgraph, langchain-anthropic,
                   sentence-transformers, ragas, python-dotenv, pydantic)
.env.example      (ANTHROPIC_API_KEY=)
.gitignore        (standard Python + .env)
README.md         (one paragraph only, I'll expand it later)

Stub functions only — docstrings + `raise NotImplementedError`, no real logic.
Show me the full file tree and wait for my confirmation before creating anything.
```
✅ Check: tree matches before approving. Commit to git once done.

---

### Phase 1 — Baseline RAG
**Model: Claude Sonnet**

```
Implement the baseline RAG pipeline in /src/retrieval only.

1. DocumentStore class: loads .txt/.md files from /data, chunks with
   recursive character splitting (chunk_size=500, overlap=50).
2. EmbeddingIndex class: embeds chunks locally with sentence-transformers
   (all-MiniLM-L6-v2, no API calls for embeddings) and builds a FAISS index.
3. retrieve(query, k=4) -> list of (chunk, score).
4. generate_answer(query, chunks) -> calls Claude Sonnet via
   langchain-anthropic with an "answer only from provided context" prompt.
5. tests/test_baseline.py with 2 smoke tests (mock the LLM call).

Only touch /src/retrieval and /tests. Don't modify the scaffold structure.
Show me a short plan before writing code.
```
✅ Check: run it on 2-3 real queries manually before moving on. Commit.

---

### Phase 2 — Reflection & Correction Agents

**Step 2a — design the grader prompts. Model: Claude Opus.**

```
I'm building a Self-RAG/CRAG-style correction layer. DESIGN three agent
prompts — do not write Python yet:

1. RetrievalGrader — given (query, chunk) -> JSON
   {"relevant": bool, "reason": str}
2. AnswerGroundednessGrader — given (answer, chunks) -> JSON
   {"grounded": bool, "unsupported_claims": [str]}
3. AnswerUsefulnessGrader — given (query, answer) -> JSON
   {"useful": bool, "reason": str}

For each: exact system prompt text + 2 worked examples (one pass, one fail).
Output as markdown only, for my review.
```
✅ Check: read the prompts carefully — this is the cheapest moment to fix
logic bugs, since no code has been written yet. Approve or request a tweak
before moving to 2b.

**Step 2b — implement from the approved design. Model: Claude Sonnet.**

```
Implement these three grader agents in /src/agents/graders.py, using the
prompts below exactly as given:

[paste the approved prompts from Step 2a]

Each function calls Claude Sonnet via langchain-anthropic's
with_structured_output (Pydantic models, not manual JSON parsing).
Add a QueryRewriter agent in the same file: given (query, irrelevant_chunks)
-> rewritten query string.
Write 3 unit tests with mocked LLM responses — no real API calls in tests.
```
✅ Check: run the graders on 2-3 hand-picked (query, chunk) pairs where you
already know the right answer, to sanity-check before wiring into the graph.

---

### Phase 3 — LangGraph Orchestration

**Step 3a — design only. Model: Claude Opus.**

```
Design (markdown description + state schema, NO code) a LangGraph state
machine for self-correcting RAG with this flow:

retrieve -> grade_chunks
  -> [if <50% relevant: rewrite_query -> retrieve, max 2 retries]
  -> generate
  -> grade_groundedness
  -> [if not grounded: regenerate excluding flagged claims, max 2 retries]
  -> grade_usefulness
  -> [if not useful: rewrite_query -> retrieve, max 1 retry]
  -> END

Define the GraphState TypedDict fields needed. Output design only.
```

**Step 3b — implement. Model: Claude Sonnet.**

```
Implement the LangGraph state machine described below in
/src/graph/correction_graph.py. Use the existing functions from
/src/retrieval and /src/agents — do not reimplement them.
Add one entry point: run_corrected_query(query: str) -> dict, returning
the final answer plus a full trace of every grading decision made.

[paste the approved design from Step 3a]
```
✅ Check: this is the riskiest phase to get wrong (infinite loops are the
classic failure mode). Test with a query that *should* trigger at least one
correction round, and confirm the iteration caps actually stop execution.
Commit immediately once it works.

---

### Phase 4 — Evaluation Harness
**Model: Sonnet (or Flash for the boilerplate parts)**

```
Build /src/eval/evaluate.py:
1. Load a small test set from /data/eval_qa.json
   (list of {question, expected_answer or expected_facts}).
2. Run both generate_answer (Phase 1 baseline) and run_corrected_query
   (Phase 3) on each item.
3. Score both with RAGAS faithfulness + answer_relevancy.
4. Output a comparison table (baseline vs corrected) to /eval_results.md.

Self-contained — don't touch any other module.
```
✅ Check: this comparison table is your actual "result" for any report —
keep the raw JSON outputs too, not just the table.

---

### Phase 5 — Optional Novelty: Adversarial Falsification Agent
**Model: Claude Opus (this is the subtlest reasoning in the project — worth the cost)**

```
Add a falsification-verification agent in /src/agents/falsifier.py:

1. generate_falsification_queries(draft_answer, original_query) -> list[str]
   Prompts Claude to produce 2-3 search queries specifically designed to
   surface evidence that would CONTRADICT the draft answer.
2. verify_against_counterevidence(draft_answer, supporting_chunks,
   counter_chunks) -> Literal["keep","revise","overturn"]
   A strict comparison prompt — call this with a FRESH context window,
   not a continuation of the generation conversation.

Wire this in as an optional final LangGraph node, gated behind a config
flag ENABLE_FALSIFICATION=True (it roughly doubles LLM calls per query —
make that cost visible in a comment). Cap at 1 falsification round.
Log every trace to /eval/falsification_log.jsonl.
```
✅ Check: run with the flag on for 5 queries that have a false-premise or
contested-claim angle, and 5 plain factual queries. Confirm it actually
catches at least one wrong draft, and doesn't flip correct answers on the
easy queries (over-triggering wastes credits and hurts accuracy).

---

### Phase 6 — Write-up
**Model: cheapest/default (Flash)**

```
Generate a README.md section "Architecture & Results" summarizing:
the pipeline (mermaid diagram), the /eval_results.md comparison table,
and 3 example queries where correction changed the final answer
(pull from the trace logs). Under 600 words.
```

---

## 4. Credit Budget Cheat-Sheet

| Phase | Model | Relative cost | Skippable if low on credits? |
|---|---|---|---|
| 0 — Scaffold | Flash | Minimal | No, do it first regardless |
| 1 — Baseline RAG | Sonnet | Low–Medium | No, it's the foundation |
| 2a — Grader prompt design | Opus | Low (text only, no execution loop) | No |
| 2b — Grader implementation | Sonnet | Medium | No |
| 3a — Graph design | Opus | Low | No |
| 3b — Graph implementation | Sonnet | Medium–High (iteration-heavy) | No |
| 4 — Evaluation | Sonnet/Flash | Low–Medium | No, it's your results |
| 5 — Falsification agent | Opus | Medium | **Yes** — cleanest thing to cut if credits run low |
| 6 — Write-up | Flash | Minimal | No |

If you're genuinely tight on credits, Phase 5 is the one to drop — the project is complete and demonstrable without it (Self-RAG + CRAG patterns alone are a solid submission). Add it back later if you get more credits.

---

## 5. If You Get Rate-Limited / Locked Out

Antigravity's free tier has had real reliability issues (multi-day lockouts have been reported after quota cuts). If it happens mid-project:

- You've been committing after every phase (Section 2), so nothing is lost — just paused.
- Anthropic's own **Claude Code** is a legitimate fallback for the coding work if you have separate access to it.
- You can also bring any stuck step back to a Claude chat directly — paste the relevant file + error, and work the design/debug step there instead of burning a retry in Antigravity.
