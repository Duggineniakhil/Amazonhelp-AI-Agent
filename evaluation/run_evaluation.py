"""
Full evaluation harness for the AmazonHelp AI Support Agent.

Orchestrates:
1. Load golden set
2. Run the main agent on all golden set examples
3. Run both baselines on the same examples
4. Compute automated metrics for all three
5. Run LLM-as-judge on a subset
6. Compute judge-human agreement
7. Output comparison table and save results

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --skip-judge    # Skip LLM judge (faster)
    python -m evaluation.run_evaluation --judge-size 30 # Smaller judge subset
"""

import argparse
import json
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR, INTENT_TAXONOMY
from evaluation.golden_set import load_golden_set, GOLDEN_SET_PATH
from evaluation.metrics import (
    compute_intent_metrics,
    compute_reply_metrics,
    compute_escalation_metrics,
    format_results_table,
)
from evaluation.llm_judge import LLMJudge, compute_judge_summary
from evaluation.judge_human_agreement import (
    generate_human_ratings,
    compute_agreement,
    format_agreement_report,
)
from evaluation.baselines import TrivialBaseline, SimpleBaseline


def run_agent_on_golden_set(golden_set: list) -> list:
    """Run the main agent on all golden set examples."""
    from src.agent import AmazonHelpAgent
    
    print("\n" + "=" * 60)
    print("  Running Main Agent on Golden Set")
    print("=" * 60)
    
    agent = AmazonHelpAgent(use_fast_model=True)  # Use fast model for batch eval
    results = []
    
    for i, example in enumerate(golden_set):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(golden_set)}")
        
        try:
            result = agent.process_message(example["customer_message"])
            results.append(result)
        except Exception as e:
            print(f"  [⚠] Error on example {example.get('id', i)}: {e}")
            results.append({
                "customer_message": example["customer_message"],
                "intent": {"label": "general_inquiry", "confidence": 0.0},
                "reply": "We're sorry, please DM us for assistance.",
                "escalation": {"decision": "escalate", "reason": "Processing error"},
                "error": str(e),
            })
    
    print(f"[✓] Agent processed {len(results)} examples")
    return results


def run_baselines_on_golden_set(golden_set: list) -> dict:
    """Run both baselines on the golden set."""
    from src.retriever import Retriever
    
    print("\n" + "=" * 60)
    print("  Running Baselines on Golden Set")
    print("=" * 60)
    
    # Load retriever for simple baseline
    try:
        retriever = Retriever(load_existing=True)
    except Exception:
        print("  [⚠] Could not load retriever, simple baseline will use canned replies")
        retriever = None
    
    trivial = TrivialBaseline()
    simple = SimpleBaseline()
    
    trivial_results = []
    simple_results = []
    
    for example in golden_set:
        msg = example["customer_message"]
        
        retrieved = []
        if retriever:
            try:
                retrieved = retriever.retrieve(msg)
            except Exception:
                pass
        
        trivial_results.append(trivial.process_message(msg, retrieved))
        simple_results.append(simple.process_message(msg, retrieved))
    
    print(f"[✓] Baselines processed {len(golden_set)} examples")
    
    return {"trivial": trivial_results, "simple": simple_results}


def extract_predictions(results: list, golden_set: list) -> dict:
    """Extract prediction lists from results for metric computation."""
    intents_pred = []
    intents_true = []
    replies_gen = []
    replies_ref = []
    escalation_pred = []
    escalation_true = []
    
    for result, example in zip(results, golden_set):
        # Intent
        pred_intent = result.get("intent", {}).get("label", "general_inquiry")
        true_intent = example.get("ground_truth_intent", "general_inquiry")
        intents_pred.append(pred_intent)
        intents_true.append(true_intent)
        
        # Reply
        replies_gen.append(result.get("reply", ""))
        replies_ref.append(example.get("actual_agent_reply", ""))
        
        # Escalation
        esc_decision = result.get("escalation", {}).get("decision", "escalate")
        esc_true = example.get("ground_truth_escalation", "auto_handle")
        escalation_pred.append(esc_decision)
        escalation_true.append(esc_true)
    
    return {
        "intents_pred": intents_pred,
        "intents_true": intents_true,
        "replies_gen": replies_gen,
        "replies_ref": replies_ref,
        "escalation_pred": escalation_pred,
        "escalation_true": escalation_true,
    }


def evaluate_system(name: str, results: list, golden_set: list) -> dict:
    """Compute all metrics for a system (agent or baseline)."""
    preds = extract_predictions(results, golden_set)
    labels = sorted(INTENT_TAXONOMY.keys())
    
    intent_metrics = compute_intent_metrics(
        preds["intents_pred"], preds["intents_true"], labels=labels
    )
    reply_metrics = compute_reply_metrics(
        preds["replies_gen"], preds["replies_ref"]
    )
    escalation_metrics = compute_escalation_metrics(
        preds["escalation_pred"], preds["escalation_true"]
    )
    
    return {
        "system": name,
        "intent": intent_metrics,
        "reply": reply_metrics,
        "escalation": escalation_metrics,
    }


def run_llm_judge(golden_set: list, agent_results: list, judge_size: int = 50) -> dict:
    """Run LLM-as-judge on a subset of examples."""
    print("\n" + "=" * 60)
    print(f"  Running LLM Judge on {judge_size} examples")
    print("=" * 60)
    
    judge = LLMJudge()
    
    # Prepare examples for judging
    examples = []
    for example, result in zip(golden_set[:judge_size], agent_results[:judge_size]):
        examples.append({
            "customer_message": example["customer_message"],
            "generated_reply": result.get("reply", ""),
            "intent": result.get("intent", {}).get("label", ""),
            "actual_reply": example.get("actual_agent_reply", ""),
        })
    
    # Run judge
    scores = judge.evaluate_batch(examples)
    summary = compute_judge_summary(scores)
    
    print(f"[✓] Judge evaluated {summary.get('num_evaluated', 0)} examples")
    
    return {
        "scores": scores,
        "summary": summary,
    }


def run_agreement_analysis(golden_set: list, agent_results: list, judge_scores: list) -> dict:
    """Run judge-human agreement analysis."""
    print("\n" + "=" * 60)
    print("  Running Judge-Human Agreement Analysis")
    print("=" * 60)
    
    # Generate human ratings (heuristic-based simulation)
    human_ratings = generate_human_ratings(golden_set, agent_results, n=len(judge_scores))
    
    # Compute agreement
    agreement = compute_agreement(human_ratings, judge_scores)
    
    report = format_agreement_report(agreement)
    print(report)
    
    return {
        "agreement_metrics": agreement,
        "human_ratings": human_ratings,
    }


def format_comparison_table(all_results: dict) -> str:
    """Format a comparison table across all systems."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("  SYSTEM COMPARISON")
    lines.append("=" * 80)
    
    headers = ["Metric", "Trivial", "Simple", "Agent (Ours)"]
    lines.append(f"\n{'Metric':<25} {'Trivial':>12} {'Simple':>12} {'Agent':>12}")
    lines.append(f"{'─'*25} {'─'*12} {'─'*12} {'─'*12}")
    
    systems = ["trivial", "simple", "agent"]
    
    # Intent metrics
    for metric_name, key in [("Intent Accuracy", "accuracy"), ("Intent Macro F1", "macro_f1")]:
        values = []
        for sys_name in systems:
            val = all_results.get(sys_name, {}).get("intent", {}).get(key, "N/A")
            values.append(f"{val}" if isinstance(val, str) else f"{val:.4f}")
        lines.append(f"{metric_name:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12}")
    
    # Reply metrics
    for metric_name, key in [("ROUGE-1", "rouge1"), ("ROUGE-L", "rougeL")]:
        values = []
        for sys_name in systems:
            val = all_results.get(sys_name, {}).get("reply", {}).get(key, "N/A")
            values.append(f"{val}" if isinstance(val, str) else f"{val:.4f}")
        lines.append(f"{metric_name:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12}")
    
    # Escalation metrics
    for metric_name, key in [("Escalation Accuracy", "accuracy"), ("Escalation F1", "f1")]:
        values = []
        for sys_name in systems:
            val = all_results.get(sys_name, {}).get("escalation", {}).get(key, "N/A")
            values.append(f"{val}" if isinstance(val, str) else f"{val:.4f}")
        lines.append(f"{metric_name:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12}")
    
    # LLM Judge (agent only)
    judge = all_results.get("agent_judge", {}).get("summary", {})
    if judge:
        lines.append(f"\n{'LLM Judge (Agent only)':<25} {'—':>12} {'—':>12} {judge.get('overall_average', 'N/A'):>12}")
    
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main(skip_judge: bool = False, judge_size: int = 50):
    """Run the full evaluation pipeline."""
    start_time = time.time()
    
    print("\n" + "▓" * 60)
    print("  AmazonHelp AI Agent — Full Evaluation")
    print("▓" * 60)
    
    # Step 1: Load golden set
    print("\n[Step 1] Loading golden set...")
    golden_set = load_golden_set()
    print(f"  Loaded {len(golden_set)} examples")
    
    # Step 2: Run agent
    print("\n[Step 2] Running main agent...")
    agent_results = run_agent_on_golden_set(golden_set)
    
    # Step 3: Run baselines
    print("\n[Step 3] Running baselines...")
    baseline_results = run_baselines_on_golden_set(golden_set)
    
    # Step 4: Compute metrics
    print("\n[Step 4] Computing metrics...")
    agent_metrics = evaluate_system("agent", agent_results, golden_set)
    trivial_metrics = evaluate_system("trivial", baseline_results["trivial"], golden_set)
    simple_metrics = evaluate_system("simple", baseline_results["simple"], golden_set)
    
    # Print individual results
    print(format_results_table(agent_metrics))
    
    all_results = {
        "agent": agent_metrics,
        "trivial": trivial_metrics,
        "simple": simple_metrics,
    }
    
    # Step 5: LLM Judge (optional)
    judge_data = {}
    agreement_data = {}
    if not skip_judge:
        print("\n[Step 5] Running LLM judge...")
        judge_size_actual = min(judge_size, len(golden_set))
        judge_data = run_llm_judge(golden_set, agent_results, judge_size=judge_size_actual)
        all_results["agent_judge"] = judge_data
        
        # Add judge summary to agent metrics
        agent_metrics["llm_judge"] = judge_data.get("summary", {})
        
        # Step 6: Agreement analysis
        print("\n[Step 6] Computing judge-human agreement...")
        agreement_data = run_agreement_analysis(
            golden_set, agent_results, judge_data.get("scores", [])
        )
        all_results["agreement"] = agreement_data
    else:
        print("\n[Step 5-6] Skipping LLM judge (--skip-judge)")
    
    # Step 7: Comparison table
    print("\n[Step 7] Final comparison:")
    comparison = format_comparison_table(all_results)
    print(comparison)
    
    # Save all results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS_DIR / "evaluation_results.json"
    
    # Make serializable (remove numpy arrays etc.)
    serializable = json.loads(json.dumps(all_results, default=str))
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    # Save agent results separately for analysis
    agent_results_path = REPORTS_DIR / "agent_results.json"
    with open(agent_results_path, "w", encoding="utf-8") as f:
        json.dump(agent_results, f, indent=2, ensure_ascii=False, default=str)
    
    elapsed = time.time() - start_time
    
    print(f"\n[✓] Evaluation complete in {elapsed:.1f}s")
    print(f"    Results saved to: {results_path}")
    print(f"    Agent results:    {agent_results_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation harness")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--judge-size", type=int, default=50, help="Number of examples for LLM judge")
    
    args = parser.parse_args()
    main(skip_judge=args.skip_judge, judge_size=args.judge_size)
