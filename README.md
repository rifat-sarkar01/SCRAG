# Self-Correcting RAG System

A **Self-Correcting Retrieval-Augmented Generation (RAG)** system built with LangGraph. The pipeline automatically retrieves documents, grades chunk relevance, generates answers, verifies groundedness claim-by-claim, corrects unsupported claims, and evaluates usefulness — all with local Ollama models.

## Quick Start

### 1. Prerequisites

- [Ollama](https://ollama.ai) installed and running
- Python 3.10+ with a virtual environment

```bash
# Pull required models
ollama pull qwen3-vl:8b-instruct-q8_0     # generator
ollama pull qwen2.5:14b-instruct           # judge / grader / corrector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Eval only:** if you want to run the evaluation harness, also install:
> ```bash
> pip install -r requirements-eval.txt
> ```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env to match your Ollama setup (model names, base URL)
```

### 4. Run the chatbot

**Web UI (recommended):**
```bash
streamlit run app.py
```
Opens a browser at `http://localhost:8501` with the full chatbot interface.

**CLI:**
```bash
python chat.py             # interactive mode
python chat.py --verbose   # show the decision trace after each answer
```

---

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

| # | Question | Baseline (s) | Corrected (s) |
|--:|---|--:|--:|
| 1 | What are the main causes of the French Revolution? | 62.8 | 140.1 |
| 2 | How does the CRISPR-Cas9 system edit DNA? | 27.1 | 88.0 |
| 3 | What is the difference between supervised and unsupervised learning? | 21.5 | 86.6 |
| 4 | What causes the northern lights (aurora borealis)? | 29.3 | 95.4 |
| 5 | How does the TCP/IP protocol suite work? | 55.9 | 129.8 |

*(Corrected pipeline is slower due to multi-step validation loops. Both faithfulness and answer relevancy metrics were tested using RAGAS.)*

### Example Corrections

1. **Machine Learning:**
   *Baseline:* Mentioned unsupervised learning finds patterns in unlabeled data.
   *Corrected:* Expanded to explicitly note it focuses on "discovering underlying structure or reducing dimensions without predefined outputs."

2. **Aurora Borealis:**
   *Baseline:* Discussed electrons accelerating into the atmosphere and colliding.
   *Corrected:* Attributed the phenomenon to "disturbances in Earth's magnetosphere caused by enhanced solar wind speeds from coronal holes and coronal mass ejections."

3. **French Revolution:**
   *Baseline:* Listed bullet points of factors that culminated in widespread unrest.
   *Corrected:* Re-framed the inability of the regime to manage "these combined pressures" for tighter semantic alignment with the source context.

---

## Project Structure

```
SCRAG/
├── app.py                   # Streamlit web chatbot UI
├── chat.py                  # CLI chatbot (python chat.py --verbose)
├── requirements.txt         # Runtime dependencies
├── requirements-eval.txt    # Eval-only dependencies
├── .env.example             # Environment variable template
├── data/                    # Corpus documents (.txt / .md)
├── src/
│   ├── graph/
│   │   └── correction_graph.py   # LangGraph state machine (main pipeline)
│   ├── retrieval/
│   │   ├── document_store.py     # File loader + recursive text chunker
│   │   ├── embedding_index.py    # FAISS index (sentence-transformers)
│   │   └── generator.py          # Ollama answer generator
│   ├── agents/
│   │   ├── graders.py            # Retrieval / groundedness / usefulness graders
│   │   └── falsifier.py          # Optional devil's-advocate falsification
│   └── eval/
│       └── evaluate.py           # RAGAS + BERTScore evaluation harness
└── tests/
    ├── test_agents.py            # Grader unit tests (mocked LLM)
    ├── test_baseline.py          # DocumentStore + generator smoke tests
    └── test_correction.py        # Correction graph regression tests
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Adding Documents

Drop `.txt` or `.md` files into the `data/` directory. The index is rebuilt automatically on the next startup.

## VRAM Note

See `.env.example` for notes on fitting both models into VRAM simultaneously. On ≤16 GB VRAM cards, Ollama will hot-swap them; each swap adds latency but does not affect correctness.
