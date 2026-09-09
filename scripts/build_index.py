"""
Build the FAISS index from processed AmazonHelp conversation data.

Usage:
    python scripts/build_index.py

Requires processed data (run data_pipeline first).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.retriever import Retriever


def build():
    """Build the FAISS index from processed pairs."""
    pairs_path = PROCESSED_DATA_DIR / "amazon_pairs.json"
    
    if not pairs_path.exists():
        print("[✗] Pairs file not found. Run the data pipeline first:")
        print("    python -m src.data_pipeline")
        sys.exit(1)
    
    print("[→] Loading pairs...")
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    print(f"    Loaded {len(pairs):,} (customer, agent) pairs")
    
    # Build retriever index
    retriever = Retriever(load_existing=False)
    retriever.build_index(pairs)
    
    # Quick test
    print("\n[→] Testing retrieval with sample query...")
    test_query = "Where is my order? I've been waiting for a week."
    results = retriever.retrieve(test_query, top_k=3)
    
    print(f"\n    Query: \"{test_query}\"")
    for i, r in enumerate(results):
        print(f"\n    Match {i+1} (similarity: {r['similarity']:.3f}):")
        print(f"      Customer: {r['customer_text'][:100]}...")
        print(f"      Agent:    {r['agent_text'][:100]}...")
    
    print("\n[✓] Index built and tested successfully!")


if __name__ == "__main__":
    build()
