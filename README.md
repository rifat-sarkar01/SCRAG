# Self-Correcting RAG System

This project implements a Self-Correcting Retrieval-Augmented Generation (RAG) system utilizing a LangGraph state machine to dynamically grade retrieved document relevance, verify answer groundedness, evaluate usefulness, and recursively correct the query or generate a revised answer when inconsistencies are detected.

## Architecture & Results

### Pipeline Architecture

```mermaid
graph TD
    START --> R[Retrieve]
    R --> GC[Grade Chunks]
    
    GC -- "< 50% relevant (retry < 2)" --> WQ[Rewrite Query]
    WQ --> R
    
    GC -- "Otherwise" --> G[Generate]
    G --> GG[Grade Groundedness]
    
    GG -- "Unsupported claims (retry < 2)" --> RG[Regenerate]
    RG --> GG
    
    GG -- "Otherwise" --> GU[Grade Usefulness]
    
    GU -- "Not useful (retry < 1)" --> WQ
    GU -- "If ENABLE_FALSIFICATION" --> F[Falsify (Devil's Advocate)]
    F --> END
    GU -- "Otherwise" --> END
```

### Evaluation Comparison

The following table summarizes the performance of the Baseline RAG vs Self-Correcting RAG based on a 5-item evaluation test set.

| # | Question | Baseline (s) | Corrected (s) |
|--:|---|--:|--:|
| 1 | What are the main causes of the French Revolution? | 62.8 | 140.1 |
| 2 | How does the CRISPR-Cas9 system edit DNA? | 27.1 | 88.0 |
| 3 | What is the difference between supervised and unsupervised learning? | 21.5 | 86.6 |
| 4 | What causes the northern lights (aurora borealis)? | 29.3 | 95.4 |
| 5 | How does the TCP/IP protocol suite work? | 55.9 | 129.8 |

*(Note: Corrected pipeline times are longer due to multi-step recursive validation loops. Both faithfulness and answer relevancy metrics were tested using RAGAS, but returned NaN in the current run).*

### Example Corrections

The trace logs demonstrate how the self-correction mechanism actively enhances answers:

1. **Machine Learning:**
   *Query:* What is the difference between supervised and unsupervised learning?
   *Baseline:* Mentioned unsupervised learning finds patterns in unlabeled data to discover underlying structures.
   *Corrected:* Expanded the explanation, explicitly noting it focuses on "discovering underlying structure or reducing dimensions without predefined outputs" based on retrieved text about PCA and clustering.

2. **Aurora Borealis:**
   *Query:* What causes the northern lights (aurora borealis)?
   *Baseline:* Primarily discussed electrons accelerating into the atmosphere and colliding.
   *Corrected:* Synthesized broader causes from the context, accurately attributing the phenomenon to "disturbances in Earth’s magnetosphere caused by enhanced solar wind speeds from coronal holes and coronal mass ejections."

3. **French Revolution:**
   *Query:* What are the main causes of the French Revolution?
   *Baseline:* Listed bullet points of factors that culminated in widespread unrest.
   *Corrected:* Re-framed the inability of the regime to manage "these combined pressures" instead of just "these crises effectively", ensuring tighter semantic alignment with the exact wording in the source context.
