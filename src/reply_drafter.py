"""
Reply drafter for AmazonHelp customer support agent.

Uses RAG (Retrieval-Augmented Generation) to draft replies:
1. Receives the customer message + detected intent
2. Gets top-K similar historical conversations from the retriever
3. Prompts the LLM to draft a reply grounded in the brand's historical tone/style

The reply should match AmazonHelp's actual communication patterns.
"""

import time
import sys
from pathlib import Path

from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL_PRIMARY,
    LLM_TEMPERATURE_DRAFT,
    GROQ_DELAY_SECONDS,
    INTENT_TAXONOMY,
)


class ReplyDrafter:
    """
    RAG-based reply drafter using retrieved historical conversations
    and Groq LLM to generate brand-consistent responses.
    """
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set!")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL_PRIMARY
        print(f"[✓] ReplyDrafter initialized (model: {self.model})")
    
    def draft_reply(
        self,
        customer_message: str,
        intent: str,
        retrieved_examples: list,
        thread_context: list = None,
    ) -> dict:
        """
        Draft a reply to the customer message.
        
        Args:
            customer_message: The incoming customer message.
            intent: The classified intent (e.g., "refund_request").
            retrieved_examples: List of similar historical (customer, agent) pairs
                               from the retriever.
            thread_context: Optional list of prior messages in the current thread.
        
        Returns:
            Dict with keys:
                - reply: str (the drafted reply)
                - model: str (which model was used)
                - grounded_in: list (thread_ids of examples used)
        """
        prompt = self._build_prompt(
            customer_message, intent, retrieved_examples, thread_context
        )
        
        try:
            time.sleep(GROQ_DELAY_SECONDS)  # Rate limiting
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE_DRAFT,
                max_tokens=280,  # Twitter-length constraint
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Clean up any artifacts
            reply = self._clean_reply(reply)
            
            return {
                "reply": reply,
                "model": self.model,
                "grounded_in": [ex.get("thread_id", "") for ex in retrieved_examples],
            }
            
        except Exception as e:
            print(f"[⚠] Reply drafting error: {e}")
            return {
                "reply": self._fallback_reply(intent),
                "model": "fallback",
                "grounded_in": [],
                "error": str(e),
            }
    
    def _system_prompt(self) -> str:
        return """You are AmazonHelp, the official Amazon customer support account on Twitter.

Your communication style:
- Friendly, empathetic, and professional
- Concise (keep replies under 280 characters when possible)
- Use first names when the customer provides theirs
- Offer specific actionable next steps
- Direct customers to DMs for sensitive account/order details
- Never share or ask for sensitive info publicly
- Use "We" and "our" to represent Amazon
- Express genuine concern for the customer's experience

IMPORTANT: Your reply should sound like it was written by a real Amazon support agent on Twitter. Match the tone and patterns from the provided examples."""
    
    def _build_prompt(
        self,
        customer_message: str,
        intent: str,
        retrieved_examples: list,
        thread_context: list = None,
    ) -> str:
        """Build the RAG prompt with historical examples as context."""
        
        # Historical examples section
        examples_section = ""
        for i, ex in enumerate(retrieved_examples[:5], 1):
            sim = ex.get("similarity", 0)
            examples_section += f"""
Example {i} (similarity: {sim:.2f}):
  Customer: "{ex['customer_text']}"
  AmazonHelp replied: "{ex['agent_text']}"
"""
        
        # Thread context section (if this is a multi-turn conversation)
        context_section = ""
        if thread_context:
            context_section = "\n## Prior Messages in This Conversation:\n"
            for msg in thread_context[-4:]:  # Last 4 messages for context
                role = "Customer" if msg["role"] == "customer" else "AmazonHelp"
                context_section += f"  {role}: \"{msg['text']}\"\n"
        
        # Intent description
        intent_desc = INTENT_TAXONOMY.get(intent, "General customer inquiry")
        
        prompt = f"""## Historical Examples of How AmazonHelp Has Handled Similar Issues:
{examples_section}
{context_section}
## Current Customer Message:
"{customer_message}"

## Detected Intent: {intent} — {intent_desc}

Draft a reply as AmazonHelp. Match the tone and style from the historical examples above.
- Address the customer's specific issue
- Provide actionable next steps
- Keep it concise (under 280 characters if possible)
- If the issue requires private details, suggest DMing

Reply ONLY with the response text (no quotes, no prefix like "AmazonHelp:")."""
        
        return prompt
    
    def _clean_reply(self, reply: str) -> str:
        """Clean up common LLM reply artifacts."""
        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "AmazonHelp:", "Amazon Help:", "@", "Reply:", "Response:",
            "Here's my reply:", "Draft:", '"',
        ]
        
        for prefix in prefixes_to_remove:
            if reply.startswith(prefix):
                reply = reply[len(prefix):].strip()
        
        # Remove trailing quote if present
        if reply.endswith('"'):
            reply = reply[:-1].strip()
        
        return reply
    
    def _fallback_reply(self, intent: str) -> str:
        """Generate a safe fallback reply when the API fails."""
        fallbacks = {
            "order_status": "We're sorry about the wait! Please DM us your order number and we'll check the status for you right away.",
            "refund_request": "We understand your frustration. Please DM us your order details so we can look into a refund for you.",
            "return_exchange": "We'd be happy to help with your return! Please DM us your order number and we'll get the process started.",
            "delivery_issue": "We're sorry to hear about this! Please DM us your order details so we can investigate the delivery issue.",
            "account_access": "We're sorry you're having trouble accessing your account. Please DM us and we'll help you get back in.",
            "payment_billing": "We apologize for the billing concern. Please DM us your order details and we'll look into this for you.",
            "product_issue": "We're sorry about the product issue. Please DM us your order number so we can make this right.",
            "prime_membership": "We'd be happy to help with your Prime membership question. Please DM us for assistance.",
            "seller_complaint": "We're sorry about your experience with this seller. Please DM us the order details so we can help.",
            "general_inquiry": "Thanks for reaching out! Please DM us with more details and we'll be happy to help.",
        }
        return fallbacks.get(intent, fallbacks["general_inquiry"])


def simple_nearest_neighbor_reply(retrieved_examples: list) -> str:
    """
    Simple baseline: return the agent reply from the most similar historical conversation.
    No LLM involved — pure nearest-neighbor.
    """
    if retrieved_examples:
        return retrieved_examples[0].get("agent_text", "Please DM us for assistance.")
    return "Thanks for reaching out! Please DM us your details and we'll help."


def canned_reply() -> str:
    """Trivial baseline: always return the same canned reply."""
    return "Thanks for reaching out! Please DM us your order details so we can assist you."
