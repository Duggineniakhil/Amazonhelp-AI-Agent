"""
Centralized configuration for the AmazonHelp AI Support Agent.

All paths, model names, API keys, and hyperparameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = PROJECT_ROOT / "index"
REPORTS_DIR = PROJECT_ROOT / "reports"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEX_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_CSV_PATH = RAW_DATA_DIR / "twcs.csv"
THREADS_JSON_PATH = PROCESSED_DATA_DIR / "amazon_threads.json"
FAISS_INDEX_PATH = INDEX_DIR / "amazon_faiss.index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"

# ──────────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN", "")

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
# Groq LLM models
LLM_MODEL_PRIMARY = "llama-3.1-70b-versatile"      # For classification & drafting
LLM_MODEL_FAST = "llama-3.1-8b-instant"             # For batch evaluation / judge
LLM_MODEL_JUDGE = "llama-3.1-70b-versatile"         # For LLM-as-judge

# Embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
# Data pipeline
BRAND_NAME = "AmazonHelp"
MAX_THREADS = 5000          # Subsample size for tractability
MIN_THREAD_LENGTH = 2       # At least 1 customer msg + 1 agent reply

# Retrieval
TOP_K_RETRIEVAL = 5         # Number of similar conversations to retrieve
SIMILARITY_THRESHOLD = 0.45  # Below this → escalate (no good precedent)

# Intent classification
NUM_INTENTS = 10
LLM_TEMPERATURE_CLASSIFY = 0.1   # Low temp for deterministic classification
LLM_TEMPERATURE_DRAFT = 0.3     # Slightly creative for reply drafting
LLM_TEMPERATURE_JUDGE = 0.0     # Deterministic for judging

# Escalation
ESCALATION_KEYWORDS = [
    "lawyer", "lawsuit", "legal", "attorney", "sue",
    "fraud", "scam", "stolen", "hacked", "unauthorized",
    "police", "report", "fbi", "consumer protection",
    "threat", "dangerous", "safety", "injury", "hurt",
]
MAX_THREAD_TURNS_BEFORE_ESCALATE = 3  # After 3 back-and-forths → escalate
ANGRY_INDICATORS = ["!!!", "WTF", "WORST", "TERRIBLE", "HORRIBLE", "NEVER AGAIN"]

# Rate limiting (Groq free tier: 30 RPM, 14400 RPD)
GROQ_RPM_LIMIT = 28        # Stay slightly under limit
GROQ_DELAY_SECONDS = 2.2   # ~27 requests per minute

# ──────────────────────────────────────────────
# Intent Taxonomy
# ──────────────────────────────────────────────
INTENT_TAXONOMY = {
    "order_status": "Customer asking about order status, tracking, shipment updates, or delivery ETA.",
    "refund_request": "Customer requesting a refund, money back, or reimbursement for an order.",
    "return_exchange": "Customer wanting to return or exchange a product, asking about return process.",
    "delivery_issue": "Package marked delivered but not received, wrong address, lost package, late delivery.",
    "account_access": "Customer cannot log in, forgot password, account locked, verification issues.",
    "payment_billing": "Charged incorrectly, double charged, payment declined, gift card or promo code issues.",
    "product_issue": "Product is defective, damaged, wrong item received, quality complaint.",
    "prime_membership": "Questions about Prime subscription, cancellation, benefits, Prime Video, Prime Day.",
    "seller_complaint": "Issues with third-party seller, seller not responding, counterfeit product suspicion.",
    "general_inquiry": "General questions, policy inquiries, feedback, or messages that don't fit other categories.",
}

INTENT_EXAMPLES = {
    "order_status": [
        "Where is my order? It was supposed to arrive yesterday.",
        "Can you give me a tracking update on order #12345?",
    ],
    "refund_request": [
        "I want my money back. The product was nothing like described.",
        "Please refund me for this order, I've been waiting 3 weeks.",
    ],
    "return_exchange": [
        "How do I return this item? It doesn't fit.",
        "I need to exchange this for a different size.",
    ],
    "delivery_issue": [
        "It says delivered but I never got the package.",
        "My package was left in the rain and everything is ruined.",
    ],
    "account_access": [
        "I can't log into my Amazon account, it says my password is wrong.",
        "My account got locked after too many login attempts.",
    ],
    "payment_billing": [
        "I was charged twice for the same order!",
        "My gift card balance isn't showing up.",
    ],
    "product_issue": [
        "The laptop I received has a cracked screen right out of the box.",
        "This product stopped working after 2 days.",
    ],
    "prime_membership": [
        "How do I cancel my Prime membership?",
        "I was charged for Prime but I never signed up.",
    ],
    "seller_complaint": [
        "The seller sent me a fake product and won't respond.",
        "I've been trying to reach the seller for a week with no reply.",
    ],
    "general_inquiry": [
        "What's your return policy for electronics?",
        "Do you price match with other retailers?",
    ],
}
