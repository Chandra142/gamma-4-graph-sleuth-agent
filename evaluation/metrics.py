"""
Evaluation metrics for Code-Graph Fault Localization and Context Compression.
Includes: Top-k Accuracy, MRR, Token Compression %, Causal Edge Retention, and Latency.
"""

from typing import Dict, List, Tuple


def compute_localization_metrics(
    predictions: List[List[str]],
    ground_truths: List[List[str]],
) -> Dict[str, float]:
    """
    Computes Top-1, Top-3, Top-5 accuracy and Mean Reciprocal Rank (MRR).
    """
    assert len(predictions) == len(ground_truths)
    total = len(predictions)
    if total == 0:
        return {"top_1": 0.0, "top_3": 0.0, "top_5": 0.0, "mrr": 0.0}

    top_1_hits = 0
    top_3_hits = 0
    top_5_hits = 0
    reciprocal_ranks = []

    for preds, truths in zip(predictions, ground_truths):
        truth_set = set(truths)
        # Check Top 1
        if preds and preds[0] in truth_set:
            top_1_hits += 1
        # Check Top 3
        if any(p in truth_set for p in preds[:3]):
            top_3_hits += 1
        # Check Top 5
        if any(p in truth_set for p in preds[:5]):
            top_5_hits += 1

        # MRR
        rr = 0.0
        for rank_idx, p in enumerate(preds, start=1):
            if p in truth_set:
                rr = 1.0 / rank_idx
                break
        reciprocal_ranks.append(rr)

    return {
        "top_1": round((top_1_hits / total) * 100.0, 2),
        "top_3": round((top_3_hits / total) * 100.0, 2),
        "top_5": round((top_5_hits / total) * 100.0, 2),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
    }


def compute_compression_metrics(
    original_token_counts: List[int],
    pruned_token_counts: List[int],
) -> Dict[str, float]:
    """
    Computes average tokens, total token reduction, and compression percentage.
    """
    assert len(original_token_counts) == len(pruned_token_counts)
    n = len(original_token_counts)
    if n == 0:
        return {"avg_original_tokens": 0.0, "avg_pruned_tokens": 0.0, "compression_ratio": 0.0}

    total_orig = sum(original_token_counts)
    total_pruned = sum(pruned_token_counts)

    reduction = 0.0
    if total_orig > 0:
        reduction = (1.0 - (total_pruned / total_orig)) * 100.0

    return {
        "avg_original_tokens": round(total_orig / n, 1),
        "avg_pruned_tokens": round(total_pruned / n, 1),
        "compression_percentage": round(reduction, 2),
        "token_reduction_factor": round(total_orig / max(1, total_pruned), 2),
    }

