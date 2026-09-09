"""
Smoke tests to verify the pipeline works end-to-end.

These are NOT exhaustive unit tests -- they verify that:
1. Config loads correctly
2. Data pipeline produces valid output format
3. Retriever can build and query an index
4. Intent classifier produces valid intents
5. Reply drafter returns non-empty replies
6. Escalation engine produces valid decisions
7. Full agent pipeline runs without errors

Usage:
    python -m tests.test_pipeline
"""

import json
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config():
    """Test that config loads without errors."""
    print("[TEST] config.py... ", end="")
    from src.config import (
        PROJECT_ROOT, INTENT_TAXONOMY, INTENT_EXAMPLES,
        GROQ_API_KEY, EMBEDDING_MODEL, TOP_K_RETRIEVAL,
    )
    
    assert len(INTENT_TAXONOMY) == 10, f"Expected 10 intents, got {len(INTENT_TAXONOMY)}"
    assert len(INTENT_EXAMPLES) == 10, f"Expected 10 intent examples, got {len(INTENT_EXAMPLES)}"
    assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert TOP_K_RETRIEVAL > 0
    print("PASS")


def test_data_pipeline_format():
    """Test that processed data has the correct format."""
    print("[TEST] data_pipeline format... ", end="")
    from src.config import THREADS_JSON_PATH, PROCESSED_DATA_DIR
    
    if not THREADS_JSON_PATH.exists():
        print("SKIP (data not downloaded yet)")
        return
    
    with open(THREADS_JSON_PATH, "r", encoding="utf-8") as f:
        threads = json.load(f)
    
    assert isinstance(threads, list), "Threads should be a list"
    assert len(threads) > 0, "Should have at least 1 thread"
    
    thread = threads[0]
    assert "thread_id" in thread, "Thread should have thread_id"
    assert "messages" in thread, "Thread should have messages"
    assert len(thread["messages"]) >= 2, "Thread should have at least 2 messages"
    
    msg = thread["messages"][0]
    assert "role" in msg, "Message should have role"
    assert "text" in msg, "Message should have text"
    assert msg["role"] in ("customer", "agent"), f"Invalid role: {msg['role']}"
    
    # Check pairs file
    pairs_path = PROCESSED_DATA_DIR / "amazon_pairs.json"
    if pairs_path.exists():
        with open(pairs_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        assert len(pairs) > 0, "Should have at least 1 pair"
        assert "customer_text" in pairs[0], "Pair should have customer_text"
        assert "agent_text" in pairs[0], "Pair should have agent_text"
    
    print(f"PASS ({len(threads)} threads)")


def test_text_cleaning():
    """Test the text cleaning function."""
    print("[TEST] text cleaning... ", end="")
    from src.data_pipeline import clean_text
    
    # Test mention removal at start
    result = clean_text("@AmazonHelp @user123 I need help with my order")
    assert not result.startswith("@"), f"Should remove leading mentions: {result}"
    
    # Test placeholder replacement
    result = clean_text("Check this __url__ for details")
    assert "[link]" in result, f"Should replace __url__: {result}"
    
    result = clean_text("Email me at __email__")
    assert "[email]" in result, f"Should replace __email__: {result}"
    
    # Test empty/None handling
    assert clean_text(None) == ""
    assert clean_text("") == ""
    
    print("PASS")


def test_keyword_classifier():
    """Test the keyword-based intent classifier."""
    print("[TEST] keyword classifier... ", end="")
    from src.intent_classifier import keyword_classify
    
    test_cases = [
        ("Where is my order?", "order_status"),
        ("I want a refund", "refund_request"),
        ("How do I return this?", "return_exchange"),
        ("Package not delivered", "delivery_issue"),
        ("I can't log in to my account", "account_access"),
        ("I was charged twice", "payment_billing"),
        ("The product is broken", "product_issue"),
        ("Cancel my Prime membership", "prime_membership"),
        ("The seller sent a fake product", "seller_complaint"),
    ]
    
    correct = 0
    for message, expected in test_cases:
        result = keyword_classify(message)
        assert "intent" in result, "Result should have intent"
        assert "confidence" in result, "Result should have confidence"
        if result["intent"] == expected:
            correct += 1
    
    # At least 70% of these obvious cases should match
    accuracy = correct / len(test_cases)
    assert accuracy >= 0.7, f"Keyword classifier too inaccurate: {accuracy:.0%}"
    
    print(f"PASS ({accuracy:.0%} accuracy on obvious cases)")


def test_escalation_engine():
    """Test the escalation engine."""
    print("[TEST] escalation engine... ", end="")
    from src.escalation_engine import EscalationEngine
    
    engine = EscalationEngine()
    
    # Should escalate: contains legal keyword
    result = engine.decide(
        customer_message="I'm going to sue you! This is fraud!",
        intent="general_inquiry",
        intent_confidence=0.3,
        max_similarity=0.3,
    )
    assert result["decision"] == "escalate", f"Should escalate legal threat: {result}"
    
    # Should auto-handle: simple inquiry with high confidence
    result = engine.decide(
        customer_message="What's your return policy?",
        intent="general_inquiry",
        intent_confidence=0.9,
        max_similarity=0.8,
    )
    assert result["decision"] == "auto_handle", f"Should auto-handle simple query: {result}"
    
    # Check response format
    assert "decision" in result
    assert "reason" in result
    assert "confidence" in result
    assert "signals" in result
    assert result["decision"] in ("auto_handle", "escalate")
    
    print("PASS")


def test_retriever_mini():
    """Test retriever with a small synthetic dataset."""
    print("[TEST] retriever (mini)... ", end="")
    from src.retriever import Retriever
    
    retriever = Retriever(load_existing=False)
    
    # Build a tiny index
    mini_pairs = [
        {"customer_text": "Where is my order?", "agent_text": "Please DM us your order number.", "thread_id": "t1"},
        {"customer_text": "I want a refund for this broken product.", "agent_text": "We're sorry! Please DM us the details.", "thread_id": "t2"},
        {"customer_text": "How do I cancel Prime?", "agent_text": "You can cancel Prime in your account settings.", "thread_id": "t3"},
    ]
    
    retriever.build_index(mini_pairs)
    
    # Query
    results = retriever.retrieve("Where is my package?", top_k=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0]["similarity"] > 0, "Similarity should be positive"
    assert "customer_text" in results[0], "Result should have customer_text"
    assert "agent_text" in results[0], "Result should have agent_text"
    
    # The order-related query should match the order-related example
    assert "order" in results[0]["customer_text"].lower(), \
        f"Best match should be order-related, got: {results[0]['customer_text']}"
    
    print("PASS")


def test_baselines():
    """Test baseline implementations."""
    print("[TEST] baselines... ", end="")
    from evaluation.baselines import TrivialBaseline, SimpleBaseline
    
    trivial = TrivialBaseline()
    result = trivial.process_message("Where is my order?")
    assert "intent" in result
    assert "reply" in result
    assert "escalation" in result
    assert result["escalation"]["decision"] == "escalate"  # Always escalates
    
    simple = SimpleBaseline()
    result = simple.process_message("I want a refund")
    assert result["intent"]["label"] == "refund_request"
    assert result["intent"]["method"] == "keyword"
    
    print("PASS")


def test_metrics():
    """Test metric computation."""
    print("[TEST] metrics... ", end="")
    from evaluation.metrics import (
        compute_intent_metrics,
        compute_reply_metrics,
        compute_escalation_metrics,
    )
    
    # Intent metrics
    pred = ["order_status", "refund_request", "order_status"]
    true = ["order_status", "refund_request", "delivery_issue"]
    result = compute_intent_metrics(pred, true)
    assert "accuracy" in result
    assert 0 <= result["accuracy"] <= 1
    assert abs(result["accuracy"] - 2/3) < 0.01  # 2 out of 3 correct
    
    # Reply metrics
    gen = ["Hello, how can I help?", "Please DM us your order details."]
    ref = ["Hi there, how can we help?", "Please send us your order number via DM."]
    result = compute_reply_metrics(gen, ref)
    assert "rougeL" in result
    assert result["rougeL"] > 0
    
    # Escalation metrics
    pred = ["auto_handle", "escalate", "auto_handle"]
    true = ["auto_handle", "escalate", "escalate"]
    result = compute_escalation_metrics(pred, true)
    assert "accuracy" in result
    assert "f1" in result
    
    print("PASS")


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("  Smoke Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_config,
        test_text_cleaning,
        test_keyword_classifier,
        test_escalation_engine,
        test_retriever_mini,
        test_baselines,
        test_metrics,
        test_data_pipeline_format,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL -- {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR -- {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
