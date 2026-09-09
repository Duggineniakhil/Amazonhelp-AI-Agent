"""
AI Support Agent orchestrator for AmazonHelp.

Combines all components into a single pipeline:
1. Retriever → finds similar historical conversations
2. Intent Classifier → categorizes the customer message
3. Reply Drafter → generates a grounded reply
4. Escalation Engine → decides auto-handle vs escalate

Usage:
    python -m src.agent --demo
    python -m src.agent --interactive
    python -m src.agent --message "Where is my order?"
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import THREADS_JSON_PATH
from src.retriever import Retriever
from src.intent_classifier import IntentClassifier
from src.reply_drafter import ReplyDrafter
from src.escalation_engine import EscalationEngine


class AmazonHelpAgent:
    """
    Full AI customer support agent for AmazonHelp.
    
    Orchestrates retrieval, classification, reply drafting, and
    escalation decisions into a single pipeline.
    """
    
    def __init__(self, use_fast_model: bool = False):
        """
        Initialize all components.
        
        Args:
            use_fast_model: Use faster/smaller LLM for batch processing.
        """
        print("=" * 60)
        print("  Initializing AmazonHelp AI Agent")
        print("=" * 60)
        
        self.retriever = Retriever(load_existing=True)
        self.classifier = IntentClassifier(use_fast_model=use_fast_model)
        self.drafter = ReplyDrafter()
        self.escalation = EscalationEngine()
        
        print("\n[✓] All components initialized successfully!")
        print("=" * 60)
    
    def process_message(
        self,
        customer_message: str,
        thread_context: list = None,
        thread_length: int = 1,
    ) -> dict:
        """
        Process a single customer message through the full pipeline.
        
        Args:
            customer_message: The incoming customer message.
            thread_context: Prior messages in the conversation (optional).
            thread_length: Number of turns so far in this thread.
        
        Returns:
            Complete response dict with all component outputs.
        """
        start_time = time.time()
        
        # Step 1: Retrieve similar conversations
        retrieved = self.retriever.retrieve(customer_message)
        max_similarity = retrieved[0]["similarity"] if retrieved else 0.0
        
        # Step 2: Classify intent
        classification = self.classifier.classify(customer_message)
        intent = classification["intent"]
        intent_confidence = classification["confidence"]
        
        # Step 3: Draft reply
        draft = self.drafter.draft_reply(
            customer_message=customer_message,
            intent=intent,
            retrieved_examples=retrieved,
            thread_context=thread_context,
        )
        
        # Step 4: Escalation decision
        escalation = self.escalation.decide(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            max_similarity=max_similarity,
            thread_length=thread_length,
            retrieved_examples=retrieved,
        )
        
        elapsed = time.time() - start_time
        
        return {
            "customer_message": customer_message,
            "intent": {
                "label": intent,
                "confidence": intent_confidence,
                "cached": classification.get("cached", False),
            },
            "reply": draft["reply"],
            "reply_metadata": {
                "model": draft["model"],
                "grounded_in": draft["grounded_in"],
            },
            "escalation": escalation,
            "retrieval": {
                "top_similarity": max_similarity,
                "num_retrieved": len(retrieved),
                "examples": [
                    {
                        "customer": r["customer_text"][:100],
                        "similarity": r["similarity"],
                    }
                    for r in retrieved[:3]
                ],
            },
            "processing_time_seconds": round(elapsed, 2),
        }
    
    def process_batch(self, messages: list) -> list:
        """Process a batch of customer messages."""
        results = []
        for i, msg in enumerate(messages):
            print(f"\n  Processing {i+1}/{len(messages)}: {msg[:60]}...")
            result = self.process_message(msg)
            results.append(result)
        return results


def format_result(result: dict) -> str:
    """Pretty-print a single agent result for terminal output."""
    lines = []
    lines.append("\n" + "─" * 60)
    lines.append(f"📩 Customer: {result['customer_message']}")
    lines.append("─" * 60)
    
    intent = result["intent"]
    lines.append(f"🏷️  Intent:     {intent['label']} (confidence: {intent['confidence']:.2f})")
    
    escalation = result["escalation"]
    emoji = "🤖" if escalation["decision"] == "auto_handle" else "👤"
    lines.append(f"{emoji} Decision:   {escalation['decision'].upper()}")
    lines.append(f"   Reason:     {escalation['reason']}")
    
    lines.append(f"\n💬 Agent Reply:")
    lines.append(f"   {result['reply']}")
    
    retrieval = result["retrieval"]
    lines.append(f"\n🔍 Retrieval:  top similarity = {retrieval['top_similarity']:.3f}")
    lines.append(f"⏱️  Time:       {result['processing_time_seconds']}s")
    lines.append("─" * 60)
    
    return "\n".join(lines)


def run_demo():
    """Run the agent on pre-defined demo messages."""
    demo_messages = [
        "Where is my order? I ordered 5 days ago and still haven't received anything!",
        "I want a full refund. The product was completely different from the listing.",
        "How do I return this item? It's the wrong size.",
        "My package says delivered but it's not here. Someone might have stolen it.",
        "I can't log into my account. I've tried resetting my password 3 times.",
        "I was charged TWICE for one order!!! This is unacceptable!!!",
        "The laptop I received has a cracked screen. This is ridiculous.",
        "Cancel my Prime membership immediately.",
        "The seller is selling counterfeit products and won't respond to my complaints.",
        "What's your return window for electronics?",
    ]
    
    agent = AmazonHelpAgent()
    
    print("\n" + "=" * 60)
    print("  AmazonHelp AI Agent — Demo Mode")
    print("=" * 60)
    
    results = []
    for msg in demo_messages:
        result = agent.process_message(msg)
        print(format_result(result))
        results.append(result)
    
    # Save results
    output_path = Path(PROJECT_ROOT) / "reports" / "demo_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[✓] Demo results saved to {output_path}")
    return results


def run_interactive():
    """Run the agent in interactive chat mode."""
    agent = AmazonHelpAgent()
    
    print("\n" + "=" * 60)
    print("  AmazonHelp AI Agent — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            message = input("\n📩 You: ").strip()
            if message.lower() in ("quit", "exit", "q"):
                print("Goodbye! 👋")
                break
            if not message:
                continue
            
            result = agent.process_message(message)
            print(format_result(result))
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break


def run_single(message: str):
    """Process a single message."""
    agent = AmazonHelpAgent()
    result = agent.process_message(message)
    print(format_result(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AmazonHelp AI Support Agent")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample messages")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--message", type=str, help="Process a single message")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.interactive:
        run_interactive()
    elif args.message:
        run_single(args.message)
    else:
        # Default to demo
        print("No mode specified. Running demo. Use --help for options.")
        run_demo()
