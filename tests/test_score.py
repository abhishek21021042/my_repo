import pytest
from src.score import compute_entity_f_beta, evaluate_predictions

def test_both_empty_singleton():
    # True empty + predicted empty -> 1.0 (PRD 4.1)
    p, r, f = compute_entity_f_beta([], [])
    assert (p, r, f) == (1.0, 1.0, 1.0)

def test_true_empty_predicted_non_empty():
    # False positive on singleton -> 0.0 (PRD 4.1)
    p, r, f = compute_entity_f_beta([], ["S2-123"])
    assert (p, r, f) == (0.0, 0.0, 0.0)

def test_true_non_empty_predicted_empty():
    # Missed match -> 0.0 (PRD 4.1)
    p, r, f = compute_entity_f_beta(["S2-123"], [])
    assert (p, r, f) == (0.0, 0.0, 0.0)

def test_exact_match():
    # Exact 1-1 match -> 1.0
    p, r, f = compute_entity_f_beta(["S2-123"], ["S2-123"])
    assert (p, r, f) == (1.0, 1.0, 1.0)

def test_partial_match_f05():
    # True: [A, B], Pred: [A]
    # Precision = 1.0, Recall = 0.5
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.8333...
    p, r, f = compute_entity_f_beta(["S2-A", "S3-B"], ["S2-A"])
    assert p == 1.0
    assert r == 0.5
    assert pytest.approx(f, 1e-4) == 5.0 / 6.0

def test_false_positive_f05():
    # True: [A], Pred: [A, B]
    # Precision = 0.5, Recall = 1.0
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.5555...
    # Notice: F0.5 penalizes lower precision (0.555) more than lower recall (0.833)!
    p, r, f = compute_entity_f_beta(["S2-A"], ["S2-A", "S3-B"])
    assert p == 0.5
    assert r == 1.0
    assert pytest.approx(f, 1e-4) == 5.0 / 9.0

def test_macro_evaluation():
    gt = {
        "S1-1": {"S2-A"},
        "S1-2": set(), # singleton
    }
    # Pred: S1-1 exact match, S1-2 correct singleton
    pred = {
        "S1-1": {"S2-A"},
        "S1-2": set(),
    }
    metrics = evaluate_predictions(gt, pred)
    assert metrics["macro_f05"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["singleton_accuracy"] == 1.0
