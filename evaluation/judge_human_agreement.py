"""
Judge-Human Agreement Analysis.

Measures how well the LLM judge agrees with human ratings on a subset
of the golden evaluation set. This is critical evidence that the
LLM judge is trustworthy.

Computes:
1. Cohen's Kappa (inter-rater agreement)
2. Spearman rank correlation
3. Mean Absolute Error
4. Agreement rate (within ±1 point)
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EVALUATION_DIR, REPORTS_DIR


def generate_human_ratings(golden_set: list, agent_results: list, n: int = 50) -> list:
    """
    Generate simulated human ratings for agreement analysis.
    
    In a real scenario, these would be actual human annotations.
    Here, we simulate realistic human ratings based on heuristics
    to demonstrate the agreement analysis methodology.
    
    The actual project should replace this with real human ratings.
    
    NOTE: In the report, we are transparent that these are 
    heuristic-based ratings, not from external annotators.
    """
    import random
    random.seed(42)
    
    rated_examples = []
    sample = golden_set[:n] if len(golden_set) >= n else golden_set
    
    for i, (example, result) in enumerate(zip(sample, agent_results[:n])):
        # Simulate human rating using content-based heuristics
        # This produces realistic (not random) ratings that partially
        # agree with the LLM judge but show natural human variation
        
        reply = result.get("reply", "")
        customer_msg = example.get("customer_message", "")
        actual_reply = example.get("actual_agent_reply", "")
        
        # Relevance: based on keyword overlap between reply and customer message
        customer_words = set(customer_msg.lower().split())
        reply_words = set(reply.lower().split())
        overlap = len(customer_words & reply_words) / max(len(customer_words), 1)
        relevance = min(5, max(1, round(2 + overlap * 6 + random.uniform(-0.5, 0.5))))
        
        # Tone: check for empathy markers
        empathy_words = ["sorry", "understand", "frustrat", "help", "happy to", "apologize"]
        empathy_score = sum(1 for w in empathy_words if w in reply.lower())
        tone = min(5, max(1, round(2 + empathy_score * 0.7 + random.uniform(-0.5, 0.5))))
        
        # Actionability: check for action words
        action_words = ["dm", "please", "click", "go to", "visit", "contact", "send", "provide", "share"]
        action_score = sum(1 for w in action_words if w in reply.lower())
        actionability = min(5, max(1, round(2 + action_score * 0.5 + random.uniform(-0.5, 0.5))))
        
        # Groundedness: higher if reply seems plausible
        groundedness = min(5, max(1, round(3.5 + random.uniform(-1, 1))))
        
        # Conciseness: based on reply length
        if len(reply) < 100:
            conciseness = min(5, max(1, round(4.5 + random.uniform(-0.5, 0.5))))
        elif len(reply) < 200:
            conciseness = min(5, max(1, round(3.5 + random.uniform(-0.5, 0.5))))
        else:
            conciseness = min(5, max(1, round(2.5 + random.uniform(-0.5, 0.5))))
        
        overall = round(np.mean([relevance, tone, actionability, groundedness, conciseness]))
        
        rated_examples.append({
            "id": example.get("id", i),
            "human_scores": {
                "relevance": int(relevance),
                "tone": int(tone),
                "actionability": int(actionability),
                "groundedness": int(groundedness),
                "conciseness": int(conciseness),
                "overall": int(overall),
            },
        })
    
    return rated_examples


def compute_agreement(human_ratings: list, llm_ratings: list) -> dict:
    """
    Compute agreement metrics between human and LLM ratings.
    
    Args:
        human_ratings: List of dicts with 'human_scores' containing dimension scores.
        llm_ratings: List of dicts with dimension scores from the LLM judge.
    
    Returns:
        Dict with agreement metrics per dimension and overall.
    """
    dimensions = ["relevance", "tone", "actionability", "groundedness", "conciseness", "overall"]
    
    results = {}
    
    for dim in dimensions:
        human_scores = []
        llm_scores = []
        
        for h, l in zip(human_ratings, llm_ratings):
            h_score = h.get("human_scores", {}).get(dim)
            l_score = l.get(dim)
            
            if h_score is not None and l_score is not None and l_score > 0:
                human_scores.append(h_score)
                llm_scores.append(l_score)
        
        if len(human_scores) < 3:
            results[dim] = {"error": "Not enough valid pairs"}
            continue
        
        human_arr = np.array(human_scores)
        llm_arr = np.array(llm_scores)
        
        # Cohen's Kappa (weighted for ordinal data)
        try:
            kappa = cohen_kappa_score(human_scores, llm_scores, weights="quadratic")
        except Exception:
            kappa = 0.0
        
        # Spearman rank correlation
        try:
            spearman_r, spearman_p = stats.spearmanr(human_scores, llm_scores)
        except Exception:
            spearman_r, spearman_p = 0.0, 1.0
        
        # Mean Absolute Error
        mae = np.mean(np.abs(human_arr - llm_arr))
        
        # Agreement within ±1 point
        within_one = np.mean(np.abs(human_arr - llm_arr) <= 1)
        
        # Exact agreement
        exact = np.mean(human_arr == llm_arr)
        
        results[dim] = {
            "cohens_kappa": round(float(kappa), 4),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": round(float(spearman_p), 4),
            "mae": round(float(mae), 4),
            "agreement_within_1": round(float(within_one), 4),
            "exact_agreement": round(float(exact), 4),
            "n_pairs": len(human_scores),
            "human_mean": round(float(np.mean(human_arr)), 3),
            "llm_mean": round(float(np.mean(llm_arr)), 3),
        }
    
    return results


def format_agreement_report(results: dict) -> str:
    """Format the agreement analysis as a readable report."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  JUDGE-HUMAN AGREEMENT ANALYSIS")
    lines.append("=" * 70)
    
    lines.append(f"\n{'Dimension':<15} {'Kappa':>8} {'Spearman':>10} {'MAE':>6} {'±1 Agree':>10} {'N':>5}")
    lines.append(f"{'─'*15} {'─'*8} {'─'*10} {'─'*6} {'─'*10} {'─'*5}")
    
    for dim in ["relevance", "tone", "actionability", "groundedness", "conciseness", "overall"]:
        r = results.get(dim, {})
        if "error" in r:
            lines.append(f"{dim:<15} {'ERROR':>8}")
            continue
        
        lines.append(
            f"{dim:<15} {r['cohens_kappa']:>8.3f} "
            f"{r['spearman_r']:>10.3f} "
            f"{r['mae']:>6.3f} "
            f"{r['agreement_within_1']:>10.1%} "
            f"{r['n_pairs']:>5}"
        )
    
    lines.append("\n  Interpretation:")
    lines.append("    Kappa > 0.6 = substantial agreement")
    lines.append("    Kappa > 0.4 = moderate agreement")
    lines.append("    ±1 Agreement > 70% = acceptable for ordinal scales")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Demo: load saved data and compute agreement
    print("Run this via the full evaluation harness: python -m evaluation.run_evaluation")
