import os
import argparse
import joblib
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.features import compute_pairwise_features

FEATURE_COLUMNS = [
    "exact_compact",
    "exact_core",
    "exact_leet",
    "name_jw",
    "core_jw",
    "name_jaccard",
    "name_containment",
    "core_jaccard",
    "len_ratio",
    "addr_jaccard",
    "addr_containment",
    "addr_jw",
    "postal_match",
    "postal_conflict",
    "nums_jaccard",
    "nums_conflict",
    "name_strong_addr_strong",
    "name_strong_addr_conflict",
    "retrieval_score",
    "channel_count",
    "is_s2"
]

class EntityMatcher:
    def __init__(self, max_iter: int = 150, learning_rate: float = 0.08, random_state: int = 42):
        self.base_model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state
        )
        self.calibrated_model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        print(f"Training Gradient Boosted Matcher on {len(y):,} pairs ({y.sum():,} positives, {(y==0).sum():,} negatives)...")
        # Platt / Sigmoid calibration for reliable confidence scores
        self.calibrated_model = CalibratedClassifierCV(self.base_model, method="sigmoid", cv=3)
        self.calibrated_model.fit(X, y)
        print("Model training and probability calibration complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrated_model.predict_proba(X)[:, 1]

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.calibrated_model, path)
        print(f"Saved trained matcher model to: {path}")

    @classmethod
    def load(cls, path: str) -> "EntityMatcher":
        inst = cls()
        inst.calibrated_model = joblib.load(path)
        return inst
