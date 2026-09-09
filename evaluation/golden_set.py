"""
Golden evaluation set creation and sampling utilities.

Provides tools for:
1. Stratified sampling of customer messages by intent
2. Export format for hand-labelling
3. Loading and validating the golden set
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    EVALUATION_DIR,
    INTENT_TAXONOMY,
)
from src.data_pipeline import load_threads, extract_pairs
from src.intent_classifier import keyword_classify


GOLDEN_SET_PATH = Path(EVALUATION_DIR) / "golden_set_data.json"


def create_golden_set(target_size: int = 200, seed: int = 42):
    """
    Create the golden evaluation set via stratified sampling.
    
    Sampling strategy:
    1. Run keyword classifier on all customer messages (cheap, no API calls)
    2. Stratify by predicted intent: ~18 examples per intent (10 × 18 = 180)
    3. Add 10 edge cases (very short messages, multi-topic, etc.)
    4. Add 10 adversarial examples (high emotion, sarcasm, non-English)
    
    Each example gets:
    - customer_message: The raw text
    - actual_agent_reply: What AmazonHelp actually said (for comparison)
    - ground_truth_intent: Hand-corrected intent label
    - ground_truth_escalation: Whether this should be auto-handled or escalated
    - escalation_reason: Why (if escalated)
    - sampling_stratum: How this example was sampled
    - difficulty: easy / medium / hard
    """
    random.seed(seed)
    
    print("[→] Creating golden evaluation set...")
    
    # Load data
    threads = load_threads()
    pairs = extract_pairs(threads)
    print(f"    Total pairs available: {len(pairs):,}")
    
    # Step 1: Keyword-classify all messages for stratification
    print("[→] Running keyword classification for stratification...")
    for pair in pairs:
        result = keyword_classify(pair["customer_text"])
        pair["predicted_intent"] = result["intent"]
        pair["keyword_confidence"] = result["confidence"]
    
    # Group by predicted intent
    by_intent = {}
    for pair in pairs:
        intent = pair["predicted_intent"]
        if intent not in by_intent:
            by_intent[intent] = []
        by_intent[intent].append(pair)
    
    print(f"    Intent distribution:")
    for intent, items in sorted(by_intent.items(), key=lambda x: -len(x[1])):
        print(f"      {intent}: {len(items):,}")
    
    # Step 2: Stratified sampling
    samples_per_intent = max(1, (target_size - 20) // len(INTENT_TAXONOMY))
    golden_set = []
    sample_id = 1
    
    for intent in INTENT_TAXONOMY.keys():
        pool = by_intent.get(intent, [])
        if not pool:
            # If no keyword matches, sample from general_inquiry
            pool = by_intent.get("general_inquiry", pairs[:50])
        
        # Sample with diversity (mix of short/long messages)
        pool_sorted = sorted(pool, key=lambda x: len(x["customer_text"]))
        n = min(samples_per_intent, len(pool))
        
        if len(pool) > n:
            # Take from different parts of the length distribution
            indices = [int(i * len(pool) / n) for i in range(n)]
            sampled = [pool_sorted[i] for i in indices]
        else:
            sampled = pool[:n]
        
        for pair in sampled:
            golden_set.append({
                "id": sample_id,
                "customer_message": pair["customer_text"],
                "actual_agent_reply": pair["agent_text"],
                "thread_id": pair.get("thread_id", ""),
                "ground_truth_intent": intent,  # Will be hand-corrected
                "ground_truth_reply_quality": "acceptable",  # Will be hand-rated
                "ground_truth_escalation": "auto_handle",  # Will be hand-decided
                "escalation_reason": "",
                "sampling_stratum": f"intent:{intent}",
                "difficulty": _assess_difficulty(pair["customer_text"]),
                "notes": "",
            })
            sample_id += 1
    
    # Step 3: Edge cases — very short or very long messages
    remaining = target_size - len(golden_set)
    edge_cases = _sample_edge_cases(pairs, n=min(10, remaining // 2))
    for pair in edge_cases:
        golden_set.append({
            "id": sample_id,
            "customer_message": pair["customer_text"],
            "actual_agent_reply": pair["agent_text"],
            "thread_id": pair.get("thread_id", ""),
            "ground_truth_intent": "general_inquiry",
            "ground_truth_reply_quality": "acceptable",
            "ground_truth_escalation": "auto_handle",
            "escalation_reason": "",
            "sampling_stratum": "edge_case",
            "difficulty": "hard",
            "notes": pair.get("edge_reason", ""),
        })
        sample_id += 1
    
    # Step 4: Adversarial — angry, sarcastic, or complex messages
    remaining = target_size - len(golden_set)
    adversarial = _sample_adversarial(pairs, n=min(10, remaining))
    for pair in adversarial:
        golden_set.append({
            "id": sample_id,
            "customer_message": pair["customer_text"],
            "actual_agent_reply": pair["agent_text"],
            "thread_id": pair.get("thread_id", ""),
            "ground_truth_intent": keyword_classify(pair["customer_text"])["intent"],
            "ground_truth_reply_quality": "acceptable",
            "ground_truth_escalation": "escalate",
            "escalation_reason": pair.get("adversarial_reason", "High emotional intensity"),
            "sampling_stratum": "adversarial",
            "difficulty": "hard",
            "notes": pair.get("adversarial_reason", ""),
        })
        sample_id += 1
    
    # Hand-correct intents based on message content heuristics
    golden_set = _auto_correct_labels(golden_set)
    
    print(f"\n[✓] Created golden set with {len(golden_set)} examples")
    print(f"    By stratum:")
    strata = Counter(ex["sampling_stratum"] for ex in golden_set)
    for stratum, count in sorted(strata.items()):
        print(f"      {stratum}: {count}")
    
    # Save
    with open(GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)
    
    print(f"[✓] Saved to {GOLDEN_SET_PATH}")
    return golden_set


def _assess_difficulty(text: str) -> str:
    """Assess how difficult this message is to handle."""
    text_lower = text.lower()
    
    if len(text) < 30:
        return "hard"  # Very short = ambiguous
    
    # Multiple topics = hard
    topics = ["order", "refund", "return", "account", "charge", "delivery", "prime"]
    topic_count = sum(1 for t in topics if t in text_lower)
    if topic_count >= 2:
        return "hard"
    
    # Strong emotion = medium
    if any(c in text for c in ["!!!", "???", "WTF"]):
        return "medium"
    
    return "easy"


def _sample_edge_cases(pairs: list, n: int = 10) -> list:
    """Sample edge cases: very short, very long, or ambiguous messages."""
    edge = []
    
    # Very short messages (< 30 chars)
    short = [p for p in pairs if len(p["customer_text"]) < 30 and len(p["customer_text"]) > 5]
    if short:
        selected = random.sample(short, min(n // 3, len(short)))
        for s in selected:
            s["edge_reason"] = "very_short_message"
        edge.extend(selected)
    
    # Very long messages (> 250 chars)
    long_msgs = [p for p in pairs if len(p["customer_text"]) > 250]
    if long_msgs:
        selected = random.sample(long_msgs, min(n // 3, len(long_msgs)))
        for s in selected:
            s["edge_reason"] = "very_long_message"
        edge.extend(selected)
    
    # Messages with URLs/emails (placeholder patterns)
    special = [p for p in pairs if "[link]" in p["customer_text"] or "[email]" in p["customer_text"]]
    if special:
        selected = random.sample(special, min(n // 3, len(special)))
        for s in selected:
            s["edge_reason"] = "contains_links_or_emails"
        edge.extend(selected)
    
    return edge[:n]


def _sample_adversarial(pairs: list, n: int = 10) -> list:
    """Sample adversarial examples: angry, sarcastic, or emotional messages."""
    adversarial = []
    
    anger_indicators = ["!!!", "wtf", "worst", "terrible", "horrible", "never again",
                       "disgusted", "furious", "incompetent", "joke", "scam", "fraud"]
    
    for pair in pairs:
        text_lower = pair["customer_text"].lower()
        matched = [ind for ind in anger_indicators if ind in text_lower]
        if matched:
            pair["adversarial_reason"] = f"Contains: {', '.join(matched[:3])}"
            adversarial.append(pair)
    
    if len(adversarial) > n:
        adversarial = random.sample(adversarial, n)
    
    return adversarial


def _auto_correct_labels(golden_set: list) -> list:
    """
    Apply heuristic label corrections to improve golden set quality.
    
    This is a first pass — the labels should still be manually reviewed.
    """
    for example in golden_set:
        text = example["customer_message"].lower()
        
        # Correct obvious mislabels
        if "refund" in text or "money back" in text:
            example["ground_truth_intent"] = "refund_request"
        elif "where is" in text and "order" in text:
            example["ground_truth_intent"] = "order_status"
        elif "return" in text and ("how" in text or "can i" in text):
            example["ground_truth_intent"] = "return_exchange"
        elif "delivered" in text and ("not" in text or "never" in text or "didn't" in text):
            example["ground_truth_intent"] = "delivery_issue"
        elif any(w in text for w in ["log in", "login", "password", "locked out", "can't access"]):
            example["ground_truth_intent"] = "account_access"
        elif any(w in text for w in ["charged twice", "double charge", "billing"]):
            example["ground_truth_intent"] = "payment_billing"
        elif any(w in text for w in ["broken", "defective", "damaged", "cracked", "doesn't work"]):
            example["ground_truth_intent"] = "product_issue"
        elif "prime" in text:
            example["ground_truth_intent"] = "prime_membership"
        elif "seller" in text:
            example["ground_truth_intent"] = "seller_complaint"
        
        # Correct escalation for adversarial examples
        anger_words = ["fraud", "scam", "lawyer", "lawsuit", "stolen", "hacked", "worst",
                       "terrible", "horrible", "disgusted", "furious"]
        if any(w in text for w in anger_words):
            example["ground_truth_escalation"] = "escalate"
            if not example["escalation_reason"]:
                example["escalation_reason"] = "High emotional intensity or safety concern"
    
    return golden_set


def load_golden_set() -> list:
    """Load the golden evaluation set from JSON."""
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found at {GOLDEN_SET_PATH}. "
            "Run 'python -m evaluation.golden_set' to create it."
        )
    
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    create_golden_set()
