"""
Baseline implementations for comparison.

Two baselines as required by the assignment:

1. **Trivial Baseline**: Random intent + canned reply + always escalate
   - Represents the floor — what you get with zero effort
   
2. **Simple Baseline**: TF-IDF keyword matching for intent + nearest-neighbor 
   reply (no LLM) + rule-based escalation
   - Represents a reasonable non-ML or minimal-ML approach
"""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import INTENT_TAXONOMY
from src.intent_classifier import keyword_classify
from src.reply_drafter import simple_nearest_neighbor_reply, canned_reply
from src.escalation_engine import always_escalate, rule_only_escalate


class TrivialBaseline:
    """
    Trivial baseline: the absolute minimum.
    
    - Intent: random selection from the taxonomy
    - Reply: same canned response every time
    - Escalation: always escalate (safest but useless)
    
    This establishes the floor that any real system must beat.
    """
    
    def __init__(self, seed: int = 42):
        self.intents = list(INTENT_TAXONOMY.keys())
        self.rng = random.Random(seed)
        print("[✓] TrivialBaseline initialized (random intent + canned reply)")
    
    def process_message(self, customer_message: str, retrieved_examples: list = None) -> dict:
        """Process a message with the trivial baseline."""
        intent = self.rng.choice(self.intents)
        
        return {
            "customer_message": customer_message,
            "intent": {
                "label": intent,
                "confidence": 0.1,
                "method": "random",
            },
            "reply": canned_reply(),
            "escalation": always_escalate(),
        }


class SimpleBaseline:
    """
    Simple baseline: keyword matching + nearest-neighbor retrieval.
    
    - Intent: keyword-based classification (no LLM)
    - Reply: copy the agent reply from the most similar historical conversation
    - Escalation: rule-based only (no LLM confidence signals)
    
    This represents a reasonable approach without any LLM usage.
    """
    
    def __init__(self):
        print("[✓] SimpleBaseline initialized (keyword intent + NN reply)")
    
    def process_message(self, customer_message: str, retrieved_examples: list = None) -> dict:
        """Process a message with the simple baseline."""
        # Intent via keywords
        intent_result = keyword_classify(customer_message)
        
        # Reply via nearest neighbor (or canned if no retrieval)
        if retrieved_examples:
            reply = simple_nearest_neighbor_reply(retrieved_examples)
        else:
            reply = canned_reply()
        
        # Escalation via rules only
        escalation = rule_only_escalate(customer_message)
        
        return {
            "customer_message": customer_message,
            "intent": {
                "label": intent_result["intent"],
                "confidence": intent_result["confidence"],
                "method": "keyword",
            },
            "reply": reply,
            "escalation": escalation,
        }


def run_baseline_comparison(golden_set: list, retriever=None) -> dict:
    """
    Run both baselines on the golden set and return results for comparison.
    
    Args:
        golden_set: The golden evaluation set.
        retriever: Optional retriever for nearest-neighbor baseline.
    
    Returns:
        Dict with results for both baselines.
    """
    trivial = TrivialBaseline()
    simple = SimpleBaseline()
    
    trivial_results = []
    simple_results = []
    
    print("\n[→] Running baselines on golden set...")
    for example in golden_set:
        msg = example["customer_message"]
        
        # Get retrieval if available
        retrieved = []
        if retriever:
            try:
                retrieved = retriever.retrieve(msg)
            except Exception:
                pass
        
        trivial_results.append(trivial.process_message(msg, retrieved))
        simple_results.append(simple.process_message(msg, retrieved))
    
    print(f"[✓] Processed {len(golden_set)} examples with both baselines")
    
    return {
        "trivial": trivial_results,
        "simple": simple_results,
    }
