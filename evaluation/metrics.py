"""
Automated evaluation metrics for the AI support agent.

Metrics computed:
1. Intent Classification: accuracy, macro F1, per-class precision/recall
2. Reply Quality: BLEU-4, ROUGE-L (vs actual agent replies)
3. Escalation: accuracy, precision, recall, F1
4. Confusion matrix for intent classification
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from rouge_score import rouge_scorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_intent_metrics(predictions: list, ground_truth: list, labels: list = None) -> dict:
    """
    Compute intent classification metrics.
    
    Args:
        predictions: List of predicted intent labels.
        ground_truth: List of ground truth intent labels.
        labels: Optional list of all possible labels.
    
    Returns:
        Dict with accuracy, macro_f1, per_class metrics, and confusion matrix.
    """
    if labels is None:
        labels = sorted(set(ground_truth + predictions))
    
    accuracy = accuracy_score(ground_truth, predictions)
    macro_f1 = f1_score(ground_truth, predictions, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(ground_truth, predictions, labels=labels, average="weighted", zero_division=0)
    
    # Per-class metrics
    report = classification_report(
        ground_truth, predictions, labels=labels, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=labels)
    
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": report[label]["support"],
            }
            for label in labels
            if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def compute_reply_metrics(generated_replies: list, reference_replies: list) -> dict:
    """
    Compute reply quality metrics.
    
    Compares generated replies to actual AmazonHelp replies using:
    - ROUGE-L (longest common subsequence)
    - ROUGE-1 (unigram overlap)
    - ROUGE-2 (bigram overlap)
    - Average reply length
    
    Note: These are surface-level metrics. LLM-as-judge provides deeper quality assessment.
    
    Args:
        generated_replies: List of generated reply strings.
        reference_replies: List of reference (actual agent) reply strings.
    
    Returns:
        Dict with ROUGE scores and length statistics.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    gen_lengths = []
    ref_lengths = []
    
    for gen, ref in zip(generated_replies, reference_replies):
        if not gen or not ref:
            continue
        
        scores = scorer.score(ref, gen)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
        gen_lengths.append(len(gen))
        ref_lengths.append(len(ref))
    
    return {
        "rouge1": round(np.mean(rouge1_scores), 4) if rouge1_scores else 0.0,
        "rouge2": round(np.mean(rouge2_scores), 4) if rouge2_scores else 0.0,
        "rougeL": round(np.mean(rougeL_scores), 4) if rougeL_scores else 0.0,
        "avg_generated_length": round(np.mean(gen_lengths), 1) if gen_lengths else 0,
        "avg_reference_length": round(np.mean(ref_lengths), 1) if ref_lengths else 0,
        "num_evaluated": len(rouge1_scores),
    }


def compute_escalation_metrics(predictions: list, ground_truth: list) -> dict:
    """
    Compute escalation decision metrics.
    
    Args:
        predictions: List of predicted decisions ("auto_handle" or "escalate").
        ground_truth: List of ground truth decisions.
    
    Returns:
        Dict with accuracy, precision, recall, F1 for escalation.
    """
    # Convert to binary (escalate = positive class)
    pred_binary = [1 if p == "escalate" else 0 for p in predictions]
    truth_binary = [1 if t == "escalate" else 0 for t in ground_truth]
    
    accuracy = accuracy_score(truth_binary, pred_binary)
    
    # Handle edge case where there might be no positive examples
    precision = precision_score(truth_binary, pred_binary, zero_division=0)
    recall = recall_score(truth_binary, pred_binary, zero_division=0)
    f1 = f1_score(truth_binary, pred_binary, zero_division=0)
    
    # Count distribution
    pred_counts = Counter(predictions)
    truth_counts = Counter(ground_truth)
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_distribution": dict(pred_counts),
        "ground_truth_distribution": dict(truth_counts),
    }


def format_results_table(results: dict) -> str:
    """Format evaluation results as a readable table."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  EVALUATION RESULTS")
    lines.append("=" * 70)
    
    # Intent metrics
    intent = results.get("intent", {})
    lines.append("\n📊 Intent Classification:")
    lines.append(f"   Accuracy:    {intent.get('accuracy', 'N/A')}")
    lines.append(f"   Macro F1:    {intent.get('macro_f1', 'N/A')}")
    lines.append(f"   Weighted F1: {intent.get('weighted_f1', 'N/A')}")
    
    # Per-class breakdown
    per_class = intent.get("per_class", {})
    if per_class:
        lines.append("\n   Per-class breakdown:")
        lines.append(f"   {'Intent':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
        lines.append(f"   {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*5}")
        for label, metrics in sorted(per_class.items()):
            lines.append(
                f"   {label:<20} {metrics['precision']:>6.3f} "
                f"{metrics['recall']:>6.3f} {metrics['f1']:>6.3f} "
                f"{metrics['support']:>5}"
            )
    
    # Reply metrics
    reply = results.get("reply", {})
    lines.append("\n📝 Reply Quality (surface metrics):")
    lines.append(f"   ROUGE-1:     {reply.get('rouge1', 'N/A')}")
    lines.append(f"   ROUGE-2:     {reply.get('rouge2', 'N/A')}")
    lines.append(f"   ROUGE-L:     {reply.get('rougeL', 'N/A')}")
    lines.append(f"   Avg gen len: {reply.get('avg_generated_length', 'N/A')} chars")
    lines.append(f"   Avg ref len: {reply.get('avg_reference_length', 'N/A')} chars")
    
    # Escalation metrics
    esc = results.get("escalation", {})
    lines.append("\n🚨 Escalation Decision:")
    lines.append(f"   Accuracy:    {esc.get('accuracy', 'N/A')}")
    lines.append(f"   Precision:   {esc.get('precision', 'N/A')}")
    lines.append(f"   Recall:      {esc.get('recall', 'N/A')}")
    lines.append(f"   F1:          {esc.get('f1', 'N/A')}")
    
    # LLM Judge metrics (if present)
    judge = results.get("llm_judge", {})
    if judge:
        lines.append("\n🤖 LLM Judge Scores (1-5):")
        for dim, score in judge.get("dimension_averages", {}).items():
            lines.append(f"   {dim:<15} {score:.2f}")
        lines.append(f"   {'Overall':<15} {judge.get('overall_average', 'N/A')}")
    
    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
