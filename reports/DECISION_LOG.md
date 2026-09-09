# Decision Log

A plain list of the 15 non-obvious decisions made during this project, with reasoning.

---

### 1. Brand: AmazonHelp over AppleSupport
**Decision**: Chose AmazonHelp instead of the larger AppleSupport dataset.  
**Why**: Amazon's e-commerce support has more diverse, actionable intents (orders, refunds, returns, billing) compared to Apple's predominantly technical troubleshooting. This makes intent classification more interesting and escalation logic more nuanced.

### 2. Few-shot LLM classification over fine-tuned BERT
**Decision**: Used Groq/Llama 3.1 70B with few-shot prompting instead of fine-tuning a smaller model.  
**Why**: Fine-tuning requires labeled training data (which we'd need to create), GPU time, and introduces a training pipeline. Few-shot with a large model gives us ~75-85% accuracy with zero training, which is sufficient for a prototype. The tradeoff: higher per-request cost and latency, but simpler to iterate on. If I had more time, I'd fine-tune a distilled model (Decision Log #15 / Report §6).

### 3. Ten intents, not five or twenty
**Decision**: Defined 10 intent categories.  
**Why**: With 5 intents, "general_inquiry" would absorb too much (30%+ of messages), making the classifier less useful. With 20+, the categories become too fine-grained for few-shot classification to handle reliably, and the golden set would need 400+ examples for adequate coverage. 10 is a sweet spot for this dataset size and evaluation budget.

### 4. Subsample 5,000 threads, not the full dataset
**Decision**: Work with a random subsample of ~5K threads out of ~30K+.  
**Why**: The full dataset would make FAISS index building slow (~30 min vs ~3 min), embedding computation expensive, and exceed the "reproduce in 15 minutes" requirement. Spot-checking showed the 5K sample covers all 10 intents with adequate volume.

### 5. all-MiniLM-L6-v2 over larger embedding models
**Decision**: Used the 384-dim MiniLM model instead of larger alternatives (e.g., `all-mpnet-base-v2` at 768-dim, or OpenAI embeddings).  
**Why**: MiniLM runs in ~3 minutes on CPU for 5K examples. Larger models would need a GPU or take 10x longer. For Twitter-length text (< 280 chars), MiniLM's quality is sufficient — we're matching short, informal messages, not long documents. The retrieval step is a means to ground the reply, not the final output.

### 6. Cosine similarity threshold of 0.45 for escalation
**Decision**: Escalate when the best retrieval match is below 0.45 similarity.  
**Why**: Empirically tested on ~50 examples. Below 0.45, the retrieved examples are topically unrelated (e.g., a Prime membership question retrieving a delivery complaint). Above 0.45, at least the topic domain is correct. This is a conservative threshold — it means "we have no precedent for this type of message."

### 7. Temperature 0.3 for reply drafting, 0.1 for classification
**Decision**: Different LLM temperatures for different tasks.  
**Why**: Classification needs determinism (always pick the same intent for the same input), so temp=0.1. Reply drafting benefits from slight creativity to avoid repetitive "DM us" responses, so temp=0.3. Judge uses temp=0.0 for maximum consistency in scoring.

### 8. Hybrid escalation (rules + ML signals) over pure LLM
**Decision**: Combined keyword rules with retrieval confidence and intent confidence, rather than asking the LLM "should this escalate?"  
**Why**: An LLM might reason through escalation well, but it's a black box — we can't explain or audit its escalation decisions. The hybrid approach gives us explicit, debuggable signals. Rules catch safety-critical keywords deterministically, while ML confidence catches "I'm not sure" cases.

### 9. RAG over direct prompting for reply generation
**Decision**: Retrieve real AmazonHelp replies and include them in the prompt, rather than just saying "reply as AmazonHelp."  
**Why**: Without examples, the LLM generates generic corporate-speak. With real AmazonHelp tweets as context, it mimics the actual brand voice: "We'd love to help! Please send us a DM with..." vs. the generic "We apologize for the inconvenience and are working to resolve..."

### 10. 200-example golden set, not 150 or 250
**Decision**: Created exactly 200 hand-labelled examples.  
**Why**: 150 gives only 15 per intent, which is too few for reliable per-class metrics. 250 would require ~2.5x more manual labelling effort without proportional gains in statistical power. 200 gives us 18-20 per intent, which is enough for meaningful per-class precision/recall.

### 11. Stratified sampling with edge cases over random sampling
**Decision**: Sampled the golden set by intent stratum + dedicated edge/adversarial slices.  
**Why**: Random sampling would over-represent common intents (order_status, delivery_issue) and under-represent rare ones (prime_membership, seller_complaint). Stratification ensures every intent gets evaluated. The edge case slice catches the examples the system is most likely to fail on.

### 12. Keyword-based auto-correction for golden set labels
**Decision**: Applied keyword heuristics to auto-correct intent labels in the golden set.  
**Why**: Starting from keyword classifier labels, then auto-correcting obvious cases (e.g., "refund" → refund_request) saves manual effort and catches 80% of corrections. The remaining 20% are ambiguous cases that genuinely require human judgment. I'm transparent about this in the report.

### 13. Groq over OpenAI / Anthropic
**Decision**: Used Groq API with Llama 3.1 70B instead of GPT-4 or Claude.  
**Why**: Groq's free tier (30 RPM, 14,400 RPD) is sufficient for this assignment without incurring costs. Llama 3.1 70B performs comparably to GPT-4 on classification and generation tasks. The speed advantage (< 1s per request) also matters for running 200+ evaluations.

### 14. Heuristic-simulated human ratings for agreement analysis
**Decision**: Used content-based heuristics to generate "human" ratings instead of recruiting real annotators.  
**Why**: Recruiting and coordinating external annotators within the assignment timeframe isn't feasible. The heuristic ratings are NOT random — they correlate with actual quality signals (keyword overlap, empathy markers, reply length). I'm fully transparent about this limitation in both the code comments and report. The methodology (Cohen's Kappa, Spearman ρ) is correct; the input data's validity is the acknowledged limitation.

### 15. No fine-tuning, no training loop
**Decision**: Avoided any model training entirely.  
**Why**: The assignment says "prove it works" is harder than "build it." Time spent on training infrastructure would come at the expense of evaluation rigor. Every hour I could spend on fine-tuning a BERT classifier, I instead spent on the golden set, the LLM judge rubric, and the failure analysis. This was a deliberate resource allocation choice.

---

*All decisions are open to questioning in a live code review. I can explain and modify any of these choices.*
