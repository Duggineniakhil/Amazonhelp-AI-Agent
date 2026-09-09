# 🤖 AmazonHelp AI Support Agent

> **Hiver SDE Intern — Take-Home Assignment**  
> An AI customer support agent for AmazonHelp that classifies intents, drafts grounded replies, and makes escalation decisions — with rigorous evaluation proving it works.

---

## 🏗️ Architecture

```
Customer Message
       │
       ├──→ Embedding (MiniLM-L6-v2) ──→ FAISS ──→ Top-5 Similar Conversations
       │
       ├──→ Intent Classifier (Llama 3.1 70B, few-shot) ──→ Intent + Confidence
       │
       └──→ Reply Drafter (Llama 3.1 70B, RAG) ←── Historical Examples + Intent
                     │
                     ▼
             Escalation Engine (rules + ML confidence)
              ├── AUTO-HANDLE → send reply
              └── ESCALATE → route to human + stated reason
```

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | Groq API → Llama 3.1 70B | Free tier, fast inference, GPT-4 comparable |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Runs locally, no API cost, fast |
| **Vector Store** | FAISS (IndexFlatIP) | In-memory, no external DB needed |
| **Evaluation** | LLM-as-judge + automated metrics | ROUGE, F1, Cohen's Kappa agreement |

---

## 🚀 Quick Start (Reproduce Results in ~15 Minutes)

### Prerequisites
- Python 3.10+
- Kaggle account (for dataset)
- Groq API key (free: [console.groq.com](https://console.groq.com))

### Step 1: Clone & Install (~1 min)

```bash
git clone <repo-url>
cd hiver-amazon-support-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Step 2: Set Environment Variables

```bash
# Option A: Export directly
export GROQ_API_KEY=your_groq_api_key_here
export KAGGLE_API_TOKEN=your_kaggle_api_token_here

# Option B: Create .env file
cp .env.example .env
# Edit .env with your keys
```

**On Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:KAGGLE_API_TOKEN="your_kaggle_api_token_here"
```

### Step 3: Download & Process Data (~3 min)

```bash
# Download dataset from Kaggle
python scripts/download_data.py

# Process: filter AmazonHelp, reconstruct threads, clean text
python -m src.data_pipeline
```

This produces:
- `data/processed/amazon_threads.json` — conversation threads
- `data/processed/amazon_pairs.json` — (customer, agent) pairs

### Step 4: Build Retrieval Index (~3 min)

```bash
python scripts/build_index.py
```

This embeds all customer messages and builds a FAISS index for similarity search.

### Step 5: Run the Agent (~2 min)

```bash
# Demo mode — processes 10 sample messages
python -m src.agent --demo

# Interactive mode — chat with the agent
python -m src.agent --interactive

# Single message
python -m src.agent --message "Where is my order?"
```

### Step 6: Run Evaluation (~8 min)

```bash
# First, create the golden evaluation set
python -m evaluation.golden_set

# Run full evaluation (agent + baselines + LLM judge)
python -m evaluation.run_evaluation

# Faster: skip LLM judge
python -m evaluation.run_evaluation --skip-judge

# Smaller judge subset
python -m evaluation.run_evaluation --judge-size 20
```

### Step 7: Run Smoke Tests (~1 min)

```bash
python -m tests.test_pipeline
```

---

## 📁 Project Structure

```
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── .gitignore
│
├── src/                          # Core agent code
│   ├── config.py                 # Central configuration
│   ├── data_pipeline.py          # Data download, filter, thread reconstruction
│   ├── retriever.py              # Embedding + FAISS retrieval
│   ├── intent_classifier.py      # LLM-based intent classification
│   ├── reply_drafter.py          # RAG-grounded reply generation
│   ├── escalation_engine.py      # Auto-handle vs escalate decision
│   └── agent.py                  # Full pipeline orchestrator
│
├── evaluation/                   # Evaluation infrastructure
│   ├── golden_set.py             # Golden set creation (200 examples)
│   ├── golden_set_data.json      # Hand-labelled evaluation data
│   ├── metrics.py                # Automated metrics (accuracy, F1, ROUGE)
│   ├── llm_judge.py              # LLM-as-judge (5-dimension rubric)
│   ├── judge_human_agreement.py  # Inter-rater agreement analysis
│   ├── baselines.py              # Trivial + simple baselines
│   └── run_evaluation.py         # Full evaluation harness
│
├── scripts/                      # Helper scripts
│   ├── download_data.py          # Kaggle dataset download
│   └── build_index.py            # Pre-build FAISS index
│
├── reports/                      # Generated reports
│   ├── REPORT.md                 # 6-page evaluation report
│   └── DECISION_LOG.md           # 15 non-obvious decisions
│
├── tests/                        # Smoke tests
│   └── test_pipeline.py
│
└── data/                         # Data (gitignored)
    ├── raw/                      # Downloaded CSV
    └── processed/                # Filtered threads & pairs
```

---

## 📊 Deliverables Checklist

| # | Deliverable | Location | Status |
|---|-------------|----------|--------|
| 1 | Runnable pipeline | `src/`, `scripts/` | ✅ |
| 2 | Golden evaluation set (200 examples) | `evaluation/golden_set_data.json` | ✅ |
| 3 | Evaluation harness + LLM judge | `evaluation/` | ✅ |
| 4 | Report (max 6 pages) | `reports/REPORT.md` | ✅ |
| 5 | Decision log (15 decisions) | `reports/DECISION_LOG.md` | ✅ |

---

## 🏷️ Intent Taxonomy (10 categories)

| Intent | Description |
|--------|-------------|
| `order_status` | Order tracking, shipment updates, delivery ETA |
| `refund_request` | Refund, money back, reimbursement |
| `return_exchange` | Return or exchange process |
| `delivery_issue` | Lost, late, wrong, or undelivered packages |
| `account_access` | Login, password, account locked |
| `payment_billing` | Charges, double billing, gift cards |
| `product_issue` | Defective, damaged, wrong item |
| `prime_membership` | Prime subscription, cancellation, benefits |
| `seller_complaint` | Third-party seller issues |
| `general_inquiry` | Other / doesn't fit above |

---

## 🔧 Golden Set: Sampling & Labelling Note

The 200-example golden set was created via:

1. **Stratified sampling** by keyword-predicted intent (~18 examples per intent)
2. **Edge cases**: 10 examples with very short/long messages or special characters
3. **Adversarial cases**: 10 examples with high emotional intensity or anger indicators

**Labelling protocol**: Keyword heuristics for first-pass labels, then hand-corrected by reviewing each message. Intent labels were corrected based on message content. Escalation labels were assigned based on: (1) safety keywords trigger escalation, (2) high anger triggers escalation, (3) ambiguous or multi-intent messages trigger escalation. Single annotator, with self-consistency checked on a subset.

---

## ⚠️ Limitations & Honest Assessment

See `reports/REPORT.md` §5 "What Is Misleading About My Headline Number?" for a full transparency section. Key limitations:

- ROUGE scores undercount valid paraphrases
- LLM judge may systematically favor LLM-generated text
- Golden set labels are single-annotator (no inter-annotator agreement)
- Subsample (5K of 30K+) may not represent full distribution
- No adversarial robustness testing (prompt injection, etc.)

---

## 📚 Citations & Acknowledgments

- **Dataset**: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) by Thought Vector (Kaggle)
- **LLM**: [Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) by Meta AI, served via [Groq](https://groq.com)
- **Embeddings**: [sentence-transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) by UKPLab
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) by Facebook Research
- **AI Assistants**: Code written with assistance from AI coding tools. All code reviewed and understood by the author.

---

*Built for the Hiver SDE Intern Take-Home Assignment.*
