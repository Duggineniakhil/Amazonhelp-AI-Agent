"""
Escalation decision engine.

Determines whether a customer message should be auto-handled by the AI
or escalated to a human agent, with a stated reason.

Uses a hybrid approach:
1. Rule-based signals (keywords, patterns, thread length)
2. Retrieval confidence (similarity threshold)
3. LLM confidence from classification
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ESCALATION_KEYWORDS,
    ANGRY_INDICATORS,
    MAX_THREAD_TURNS_BEFORE_ESCALATE,
    SIMILARITY_THRESHOLD,
)


class EscalationEngine:
    """
    Decides whether to auto-handle or escalate customer messages.
    
    Produces a decision with:
    - decision: "auto_handle" or "escalate"
    - reason: Human-readable explanation
    - confidence: How confident we are in this decision
    - signals: List of signals that contributed to the decision
    """
    
    def __init__(self):
        self.escalation_keywords = [kw.lower() for kw in ESCALATION_KEYWORDS]
        self.angry_indicators = [ind.lower() for ind in ANGRY_INDICATORS]
        print("[✓] EscalationEngine initialized")
    
    def decide(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        max_similarity: float,
        thread_length: int = 1,
        retrieved_examples: list = None,
    ) -> dict:
        """
        Make an escalation decision.
        
        Args:
            customer_message: The customer's message text.
            intent: Classified intent.
            intent_confidence: Confidence of the intent classification.
            max_similarity: Best similarity score from retrieval.
            thread_length: Number of turns in the current conversation thread.
            retrieved_examples: Retrieved similar conversations (optional).
        
        Returns:
            Dict with decision, reason, confidence, and signals.
        """
        signals = []
        escalate_score = 0.0  # 0 = definitely auto-handle, 1 = definitely escalate
        
        msg_lower = customer_message.lower()
        
        # ── Signal 1: Sensitive Keywords ──
        matched_keywords = [kw for kw in self.escalation_keywords if kw in msg_lower]
        if matched_keywords:
            escalate_score += 0.4
            signals.append(f"sensitive_keywords: {', '.join(matched_keywords[:3])}")
        
        # ── Signal 2: Emotional Intensity ──
        anger_score = self._detect_anger(customer_message)
        if anger_score > 0.5:
            escalate_score += 0.25
            signals.append(f"high_emotional_intensity (score: {anger_score:.2f})")
        
        # ── Signal 3: Repeated Contact (long thread) ──
        if thread_length > MAX_THREAD_TURNS_BEFORE_ESCALATE:
            escalate_score += 0.3
            signals.append(f"repeated_contact ({thread_length} turns, threshold: {MAX_THREAD_TURNS_BEFORE_ESCALATE})")
        
        # ── Signal 4: Low Retrieval Similarity ──
        if max_similarity < SIMILARITY_THRESHOLD:
            escalate_score += 0.2
            signals.append(f"low_retrieval_confidence (similarity: {max_similarity:.2f}, threshold: {SIMILARITY_THRESHOLD})")
        
        # ── Signal 5: Low Intent Confidence ──
        if intent_confidence < 0.3:
            escalate_score += 0.15
            signals.append(f"low_intent_confidence ({intent_confidence:.2f})")
        
        # ── Signal 6: Account Security Intent ──
        if intent == "account_access" and any(kw in msg_lower for kw in ["hacked", "unauthorized", "stolen"]):
            escalate_score += 0.35
            signals.append("account_security_concern")
        
        # ── Signal 7: Complex Multi-Intent ──
        if self._is_multi_intent(customer_message):
            escalate_score += 0.1
            signals.append("possible_multi_intent")
        
        # ── Auto-handle boosters ──
        if max_similarity > 0.75:
            escalate_score -= 0.15
            signals.append(f"high_retrieval_confidence (similarity: {max_similarity:.2f})")
        
        if intent_confidence > 0.8:
            escalate_score -= 0.1
            signals.append(f"high_intent_confidence ({intent_confidence:.2f})")
        
        # Common, well-understood intents get a small boost toward auto-handle
        simple_intents = {"order_status", "general_inquiry", "prime_membership"}
        if intent in simple_intents and anger_score < 0.3:
            escalate_score -= 0.1
            signals.append(f"common_simple_intent ({intent})")
        
        # ── Final Decision ──
        escalate_score = max(0.0, min(1.0, escalate_score))
        
        if escalate_score >= 0.4:
            decision = "escalate"
            reason = self._generate_reason(signals, decision="escalate")
            confidence = escalate_score
        else:
            decision = "auto_handle"
            reason = self._generate_reason(signals, decision="auto_handle")
            confidence = 1.0 - escalate_score
        
        return {
            "decision": decision,
            "reason": reason,
            "confidence": round(confidence, 3),
            "escalate_score": round(escalate_score, 3),
            "signals": signals,
        }
    
    def _detect_anger(self, text: str) -> float:
        """
        Detect anger/frustration in the message.
        
        Returns a score from 0.0 (calm) to 1.0 (very angry).
        """
        score = 0.0
        text_lower = text.lower()
        
        # All caps detection (more than 50% uppercase when > 10 chars)
        alpha_chars = [c for c in text if c.isalpha()]
        if len(alpha_chars) > 10:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio > 0.5:
                score += 0.4
        
        # Excessive punctuation (!!!, ???)
        exclaim_count = text.count("!")
        question_count = text.count("?")
        if exclaim_count >= 3:
            score += 0.3
        if question_count >= 3:
            score += 0.1
        
        # Angry indicators
        for indicator in self.angry_indicators:
            if indicator in text_lower:
                score += 0.2
        
        # Profanity (basic check)
        profanity = ["wtf", "damn", "hell", "stupid", "idiot", "crap", "suck", "worst"]
        for word in profanity:
            if word in text_lower.split():
                score += 0.3
                break
        
        return min(score, 1.0)
    
    def _is_multi_intent(self, text: str) -> bool:
        """Check if the message likely contains multiple distinct issues."""
        # Heuristic: long messages with multiple distinct topic keywords
        if len(text) < 100:
            return False
        
        topics = {
            "order": ["order", "tracking", "shipment"],
            "refund": ["refund", "money back"],
            "return": ["return", "exchange"],
            "account": ["account", "login", "password"],
            "payment": ["charge", "billing", "payment"],
        }
        
        text_lower = text.lower()
        matched_topics = sum(
            1 for keywords in topics.values()
            if any(kw in text_lower for kw in keywords)
        )
        
        return matched_topics >= 2
    
    def _generate_reason(self, signals: list, decision: str) -> str:
        """Generate a human-readable reason for the decision."""
        if not signals:
            if decision == "auto_handle":
                return "Standard inquiry with high confidence — safe to auto-handle."
            return "Unable to determine confidence — escalating for safety."
        
        if decision == "escalate":
            primary_signals = [s for s in signals if not s.startswith("high_")]
            if primary_signals:
                return f"Escalating because: {'; '.join(primary_signals[:3])}."
            return "Multiple signals suggest human attention needed."
        else:
            return f"Auto-handling — {'; '.join(signals[:2])}."


def always_escalate() -> dict:
    """Trivial baseline: always escalate everything."""
    return {
        "decision": "escalate",
        "reason": "Trivial baseline — always escalates.",
        "confidence": 1.0,
        "escalate_score": 1.0,
        "signals": ["trivial_baseline"],
    }


def rule_only_escalate(customer_message: str) -> dict:
    """Simple baseline: rule-based escalation only (no LLM confidence)."""
    engine = EscalationEngine()
    return engine.decide(
        customer_message=customer_message,
        intent="general_inquiry",
        intent_confidence=0.5,
        max_similarity=0.5,
        thread_length=1,
    )
