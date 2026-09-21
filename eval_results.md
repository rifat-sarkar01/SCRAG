# Evaluation Results: Baseline RAG vs Self-Correcting RAG

> Generated: 2026-07-23 02:38:25 UTC  
> Test items: **5**

---

## Per-Item Comparison

| # | Question | Baseline answer (excerpt) | Corrected answer (excerpt) | Baseline (s) | Corrected (s) | Corr. Rounds |
|--:|---|---|---|--:|--:|--:|
| 1 | What are the main causes of the French Revolution? | The main causes of the French Revolution include a combination of social, political, and econom… | The main causes of the French Revolution include a combination of social, political, and econom… | 56.2 | 276.9 | 0 |
| 2 | How does the CRISPR-Cas9 system edit DNA? | The CRISPR-Cas9 system edits DNA by using a Cas9 enzyme guided by a synthetic guide RNA (gRNA) … | The CRISPR-Cas9 system edits DNA by using the Cas9 enzyme complexed with a synthetic guide RNA … | 37.0 | 199.3 | 0 |
| 3 | What is the difference between supervised and unsupervised learni… | Supervised learning uses labeled data to train models that map inputs to outputs, while unsuper… | Supervised Learning uses labeled data to map inputs to outputs, aiming to predict categories (c… | 25.4 | 188.4 | 0 |
| 4 | What causes the northern lights (aurora borealis)? | Charged particles from the Sun colliding with atoms in Earth's atmosphere cause the northern li… | Charged particles from the Sun colliding with atoms in Earth's atmosphere cause the northern li… | 37.3 | 164.1 | 0 |
| 5 | How does the TCP/IP protocol suite work? | The TCP/IP protocol suite works by defining how data is packaged, addressed, transmitted, route… | The TCP/IP protocol suite works by defining how data is packaged, addressed, transmitted, route… | 59.6 | 248.2 | 0 |

---

## Generation Quality (RAGAS)

| Metric | Baseline | Self-Correcting | Δ (Corrected − Baseline) |
|---|--:|--:|--:|
| faithfulness | 0.943 | 0.881 | -0.062 |
| answer_relevancy | 0.992 | 0.995 | +0.002 |
| answer_correctness | 0.680 | 0.771 | +0.091 |
| context_entity_recall | 0.273 | 0.223 | -0.050 |

---

## Retrieval Quality (RAGAS)

| Metric | Baseline | Self-Correcting | Δ (Corrected − Baseline) |
|---|--:|--:|--:|
| context_precision | 0.800 | 0.800 | -0.000 |
| context_recall | 0.850 | 0.783 | -0.067 |

---

## Self-Correction Metrics

| Metric | Value |
|---|--:|
| Hallucination Fix Rate (%) | 0.0 |
| Hallucination Injection Rate (%) | 40.0 |
| Avg Correction Rounds | 0.00 |
| Max Rounds Hit (%) | 0.0 |
| Avg Latency Overhead (s) | 172.3 |

---

## Secondary Metrics

| Metric | Baseline | Self-Correcting |
|---|--:|--:|
| bertscore_f1 | 0.847 | 0.847 |
| rouge_l | 0.197 | 0.200 |

---

## Notes

- **Faithfulness**: fraction of answer claims supported by the retrieved context (higher = fewer hallucinations).
- **Answer relevancy**: how directly the answer addresses the question (higher = more on-topic).
- **Answer correctness**: factual overlap + semantic similarity vs. ground-truth answer.
- **Context entity recall**: key entities from ground truth present in retrieved context.
- **Hallucination Injection Rate**: the critical self-correction failure metric — should be near zero before reporting corrected faithfulness.
- A positive Δ means the Self-Correcting pipeline outperformed the baseline.
- `N/A` means RAGAS raised an exception for that pipeline (see console output for details).
- **Context pairing fix**: corrected-answer RAGAS contexts now use the full chunks returned by `run_corrected_query['correction_context']`, not the 120-char trace previews from the previous version.
