# Evaluation Results: Baseline RAG vs Self-Correcting RAG

> Generated: 2026-06-26 06:19:46 UTC  
> Test items: **5**

---

## Per-Item Comparison

| # | Question | Baseline answer (excerpt) | Corrected answer (excerpt) | Baseline (s) | Corrected (s) |
|--:|---|---|---|--:|--:|
| 1 | What are the main causes of the French Revolution? | The main causes of the French Revolution include a combination of social, political, and econom… | The main causes of the French Revolution include a combination of social, political, and econom… | 62.3 | 125.1 |
| 2 | How does the CRISPR-Cas9 system edit DNA? | The CRISPR-Cas9 system edits DNA by using a Cas9 enzyme guided by a synthetic guide RNA (gRNA) … | The CRISPR-Cas9 system edits DNA by using the Cas9 enzyme, guided by a synthetic guide RNA (gRN… | 25.4 | 88.7 |
| 3 | What is the difference between supervised and unsupervised learni… | Supervised learning uses labeled data to train models that map inputs to outputs, while unsuper… | Supervised Learning uses labeled data to learn mapping from input (X) to output (Y), aiming to … | 19.2 | 88.1 |
| 4 | What causes the northern lights (aurora borealis)? | Charged particles from the Sun collide with atoms in Earth's upper atmosphere, exciting them to… | Charged particles from the Sun colliding with atoms in Earth's atmosphere cause the northern li… | 29.2 | 86.4 |
| 5 | How does the TCP/IP protocol suite work? | The TCP/IP protocol suite works by defining how data is packaged, addressed, transmitted, route… | The TCP/IP protocol suite works by defining how data is packaged, addressed, transmitted, route… | 55.0 | 119.6 |

---

## Aggregate RAGAS Scores

| Metric | Baseline | Self-Correcting | Δ (Corrected − Baseline) |
|---|--:|--:|--:|
| faithfulness | 0.949 | 0.151 | -0.798 |
| answer_relevancy | 0.966 | 0.978 | +0.011 |

---

## Notes

- **Faithfulness**: fraction of answer claims supported by the retrieved context (higher = fewer hallucinations).
- **Answer relevancy**: how directly the answer addresses the question (higher = more on-topic).
- A positive Δ means the Self-Correcting pipeline outperformed the baseline.
- `N/A` means RAGAS raised an exception for that pipeline (see console output for details).
