from typing import Dict, List, Set, Union, Tuple
import numpy as np
import pandas as pd
from src.io_utils import parse_matched_ids

def compute_entity_f_beta(
    true_ids: Union[Set[str], List[str]], 
    pred_ids: Union[Set[str], List[str]], 
    beta: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes precision, recall, and F_beta for a single Source 1 entity.
    Returns: (precision, recall, f_beta)

    Edge case rules from PRD & Implementation Plan:
    - true empty + predicted empty -> (1.0, 1.0, 1.0)
    - true non-empty + predicted empty -> (0.0, 0.0, 0.0)
    - true empty + predicted non-empty -> (0.0, 0.0, 0.0)
    - normal case: F_beta = ((1 + beta^2) * P * R) / (beta^2 * P + R)
      For beta=0.5: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    t_set = set(true_ids) if not isinstance(true_ids, set) else true_ids
    p_set = set(pred_ids) if not isinstance(pred_ids, set) else pred_ids

    # Both empty (correct singleton identification)
    if len(t_set) == 0 and len(p_set) == 0:
        return 1.0, 1.0, 1.0

    # Exactly one is empty
    if len(t_set) == 0 or len(p_set) == 0:
        return 0.0, 0.0, 0.0

    tp = len(t_set & p_set)
    precision = tp / len(p_set)
    recall = tp / len(t_set)

    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        f_score = 0.0
    else:
        f_score = ((1.0 + beta_sq) * precision * recall) / denom

    return precision, recall, f_score


def evaluate_predictions(
    ground_truth_dict: Dict[str, Union[Set[str], List[str]]],
    predictions_dict: Dict[str, Union[Set[str], List[str]]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates predictions across all Source 1 entities in ground_truth_dict.
    Computes macro precision, macro recall, and macro F_beta.
    Also reports singleton accuracy and non-singleton metrics.
    """
    all_s1_ids = list(ground_truth_dict.keys())
    if not all_s1_ids:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "total_entities": 0}

    precisions = []
    recalls = []
    f_scores = []

    singleton_correct = 0
    singleton_total = 0
    non_singleton_f_scores = []

    for s1_id in all_s1_ids:
        true_matches = ground_truth_dict[s1_id]
        pred_matches = predictions_dict.get(s1_id, [])

        p, r, f = compute_entity_f_beta(true_matches, pred_matches, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

        if len(true_matches) == 0:
            singleton_total += 1
            if len(pred_matches) == 0:
                singleton_correct += 1
        else:
            non_singleton_f_scores.append(f)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f = float(np.mean(f_scores))
    singleton_acc = float(singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    non_singleton_f = float(np.mean(non_singleton_f_scores)) if non_singleton_f_scores else 0.0

    return {
        "macro_f05": macro_f,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singleton_accuracy": singleton_acc,
        "singleton_count": singleton_total,
        "non_singleton_f05": non_singleton_f,
        "total_entities": len(all_s1_ids)
    }


def evaluate_files(
    ground_truth_path: str,
    predictions_path: str,
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Evaluates predictions TSV file against ground truth TSV file.
    Both files must have headers:
    - ground_truth: source1_entity_id, matched_entity_ids
    - predictions: source1_entity_id, matched_entity_ids
    """
    gt_df = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    pred_df = pd.read_csv(predictions_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)

    gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in gt_df.iterrows()
    }
    pred_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in pred_df.iterrows()
    }

    return evaluate_predictions(gt_dict, pred_dict, beta=beta)
