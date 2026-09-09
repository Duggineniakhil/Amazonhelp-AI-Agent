"""
Intent classifier for customer support messages.

Uses few-shot prompting with Groq API (Llama 3.1) to classify
customer messages into one of 10 predefined intents.

Includes caching to avoid redundant API calls.
"""

import json
import time
import hashlib
import sys
from pathlib import Path
from typing import Optional

from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL_PRIMARY,
    LLM_MODEL_FAST,
    LLM_TEMPERATURE_CLASSIFY,
    INTENT_TAXONOMY,
    INTENT_EXAMPLES,
    GROQ_DELAY_SECONDS,
    PROCESSED_DATA_DIR,
)


# Cache for classification results
_classification_cache = {}
_CACHE_FILE = PROCESSED_DATA_DIR / "intent_cache.json"


def _load_cache():
    """Load classification cache from disk."""
    global _classification_cache
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            _classification_cache = json.load(f)


def _save_cache():
    """Save classification cache to disk."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_classification_cache, f, ensure_ascii=False, indent=2)


def _cache_key(text: str) -> str:
    """Generate a cache key from message text."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def _build_classification_prompt(message: str) -> str:
    """
    Build the few-shot classification prompt.
    
    The prompt includes:
    1. Task description
    2. Intent taxonomy with descriptions
    3. Two examples per intent
    4. The message to classify
    """
    # Build taxonomy section
    taxonomy_section = ""
    for intent, description in INTENT_TAXONOMY.items():
        examples = INTENT_EXAMPLES.get(intent, [])
        examples_str = "\n".join(f'    - "{ex}"' for ex in examples)
        taxonomy_section += f"\n**{intent}**: {description}\n  Examples:\n{examples_str}\n"
    
    prompt = f"""You are an intent classifier for AmazonHelp customer support messages.

Classify the following customer message into exactly ONE of these intents:

{taxonomy_section}

## Rules:
- Pick the SINGLE most specific intent that matches.
- If the message covers multiple issues, pick the PRIMARY one.
- Use "general_inquiry" only if no other intent fits.
- Return ONLY valid JSON with no additional text.

## Customer Message:
"{message}"

## Response Format (JSON only):
{{"intent": "<intent_name>", "confidence": <0.0-1.0>}}"""
    
    return prompt


class IntentClassifier:
    """
    LLM-based intent classifier using Groq API.
    
    Uses few-shot prompting with the full intent taxonomy and examples.
    Results are cached to avoid redundant API calls.
    """
    
    def __init__(self, use_fast_model: bool = False):
        """
        Initialize the classifier.
        
        Args:
            use_fast_model: If True, use the smaller/faster model (8B).
                           Use for batch processing. Default uses 70B.
        """
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set! Set it via environment variable or .env file."
            )
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL_FAST if use_fast_model else LLM_MODEL_PRIMARY
        self.valid_intents = set(INTENT_TAXONOMY.keys())
        
        _load_cache()
        print(f"[✓] IntentClassifier initialized (model: {self.model})")
        print(f"    Cache size: {len(_classification_cache)} entries")
    
    def classify(self, message: str, use_cache: bool = True) -> dict:
        """
        Classify a customer message into an intent.
        
        Args:
            message: The customer's message text.
            use_cache: Whether to use cached results.
        
        Returns:
            Dict with keys:
                - intent: str (one of the 10 intents)
                - confidence: float (0.0 to 1.0)
                - cached: bool (whether this was a cache hit)
        """
        # Check cache
        key = _cache_key(message)
        if use_cache and key in _classification_cache:
            result = _classification_cache[key]
            result["cached"] = True
            return result
        
        # Build prompt and call LLM
        prompt = _build_classification_prompt(message)
        
        try:
            time.sleep(GROQ_DELAY_SECONDS)  # Rate limiting
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise intent classifier. Return ONLY valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=LLM_TEMPERATURE_CLASSIFY,
                max_tokens=100,
            )
            
            raw_output = response.choices[0].message.content.strip()
            result = self._parse_response(raw_output)
            
        except Exception as e:
            print(f"[⚠] Classification API error: {e}")
            result = {
                "intent": "general_inquiry",
                "confidence": 0.0,
                "error": str(e),
            }
        
        # Cache the result only if it's not a complete failure
        result["cached"] = False
        if result["confidence"] > 0.0:
            _classification_cache[key] = {
                "intent": result["intent"],
                "confidence": result["confidence"],
            }
            _save_cache()
        
        return result
    
    def _parse_response(self, raw: str) -> dict:
        """Parse the LLM's JSON response, with fallbacks for malformed output."""
        # Try direct JSON parse
        try:
            # Handle cases where LLM wraps in markdown code blocks
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            # Validate intent
            intent = result.get("intent", "general_inquiry")
            if intent not in self.valid_intents:
                # Try fuzzy matching
                intent = self._fuzzy_match_intent(intent)
            
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            
            return {"intent": intent, "confidence": confidence}
            
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: try to extract intent from raw text
            return self._extract_from_text(raw)
    
    def _fuzzy_match_intent(self, candidate: str) -> str:
        """Try to match a malformed intent name to a valid one."""
        candidate_lower = candidate.lower().replace(" ", "_").replace("-", "_")
        
        for valid_intent in self.valid_intents:
            if valid_intent in candidate_lower or candidate_lower in valid_intent:
                return valid_intent
        
        return "general_inquiry"
    
    def _extract_from_text(self, raw: str) -> dict:
        """Last-resort extraction from free-text LLM output."""
        raw_lower = raw.lower()
        
        for intent in self.valid_intents:
            if intent in raw_lower:
                return {"intent": intent, "confidence": 0.3}
        
        return {"intent": "general_inquiry", "confidence": 0.1}
    
    def classify_batch(self, messages: list, use_cache: bool = True) -> list:
        """
        Classify a batch of messages.
        
        Args:
            messages: List of customer message strings.
            use_cache: Whether to use cached results.
        
        Returns:
            List of classification result dicts.
        """
        results = []
        cached_count = 0
        
        for msg in messages:
            result = self.classify(msg, use_cache=use_cache)
            if result.get("cached"):
                cached_count += 1
            results.append(result)
        
        print(f"    Classified {len(messages)} messages ({cached_count} from cache)")
        return results


def keyword_classify(message: str) -> dict:
    """
    Simple keyword-based intent classifier (used as baseline).
    
    No LLM calls — pure keyword matching.
    """
    msg_lower = message.lower()
    
    keyword_map = {
        "order_status": ["where is my order", "tracking", "shipment", "when will", "order status", "estimated delivery"],
        "refund_request": ["refund", "money back", "reimburse", "charged but", "want my money"],
        "return_exchange": ["return", "exchange", "send back", "return label", "return policy"],
        "delivery_issue": ["not delivered", "never received", "lost package", "wrong address", "says delivered", "missing package"],
        "account_access": ["can't log in", "password", "locked out", "account locked", "can't access", "login"],
        "payment_billing": ["charged twice", "double charge", "billing", "payment", "gift card", "promo code", "charged"],
        "product_issue": ["broken", "defective", "damaged", "doesn't work", "wrong item", "not working", "cracked"],
        "prime_membership": ["prime", "membership", "cancel prime", "prime video", "prime day"],
        "seller_complaint": ["seller", "third party", "marketplace", "counterfeit", "fake product"],
    }
    
    best_intent = "general_inquiry"
    best_score = 0
    
    for intent, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in msg_lower)
        if score > best_score:
            best_score = score
            best_intent = intent
    
    return {
        "intent": best_intent,
        "confidence": min(best_score * 0.3, 1.0),
        "method": "keyword",
    }
