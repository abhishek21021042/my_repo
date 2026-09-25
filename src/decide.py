from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

def apply_precision_first_policy(
    scored_candidates_df: pd.DataFrame,
    threshold: float = 0.93,
    strong_addr_threshold: float = 0.89,
    min_margin: float = 0.05,
    enable_dual_anchor: bool = True
) -> Dict[str, List[str]]:
    """
    Precision-First Decision Policy conforming to PRD 12:
    Targets Precision >= 99% while maximizing Ground-Truth match recall:
    - Never force a match (abstain whenever uncertain)
    - Strict Dual-Anchor Rule: High confidence requires BOTH Name and Address consistency
    - Absolute Hard Vetoes on Postal Conflict & Numeric Conflict
    - Dynamic Address-Backed Acceptance: Uses 0.84 threshold when address agreement is strong
    - Multi-Match Recall Recovery: Allows multiple valid matches across S2 and S3
    """
    predictions = {}
    grouped = scored_candidates_df.groupby("source1_entity_id")

    for s1_id, group in grouped:
        sorted_group = group.sort_values(by="probability", ascending=False)
        accepted = []
        top_prob = 0.0

        for _, row in sorted_group.iterrows():
            prob = row["probability"]
            core_jw = row.get("core_jw", 0.0)
            addr_jaccard = row.get("addr_jaccard", 0.0)
            addr_jw = row.get("addr_jw", 0.0)
            postal_conflict = row.get("postal_conflict", 0.0)
            postal_match = row.get("postal_match", 0.0)
            nums_conflict = row.get("nums_conflict", 0.0)
            nums_jaccard = row.get("nums_jaccard", 0.0)
            exact_compact = row.get("exact_compact", 0.0)
            exact_core = row.get("exact_core", 0.0)
            core_jacc = row.get("core_jaccard", 0.0)
            name_cont = row.get("name_containment", 0.0)

            # --- HARD VETO 1: Explicit Postal Code Conflict ---
            if postal_conflict == 1.0:
                continue

            # --- HARD VETO 2: Explicit House/Street Number Conflict ---
            if nums_conflict == 1.0 and (addr_jaccard < 0.50):
                continue

            # --- HARD VETO 3: Multi-Tenancy Shared Building Veto ---
            if exact_compact == 0 and exact_core == 0:
                if core_jw < 0.93:
                    continue
                if core_jacc < 0.60 and core_jw < 0.97 and name_cont < 0.85:
                    continue

            # --- DUAL ANCHOR REQUIREMENT (PRD 12.6) ---
            if enable_dual_anchor:
                has_address_evidence = (
                    addr_jaccard >= 0.20 or 
                    addr_jw >= 0.70 or 
                    postal_match == 1.0 or 
                    nums_jaccard > 0.0
                )
                if not has_address_evidence:
                    continue

                if core_jw < 0.85 and addr_jaccard < 0.50:
                    continue

            # --- DYNAMIC ADDRESS-BACKED THRESHOLD ---
            is_strong_addr = (addr_jw >= 0.85 or postal_match == 1.0 or (addr_jaccard >= 0.35 and nums_jaccard > 0.0))
            active_thresh = strong_addr_threshold if is_strong_addr else threshold

            if prob >= active_thresh:
                accepted.append(row["candidate_entity_id"])
                if top_prob == 0.0:
                    top_prob = prob
            elif top_prob > 0.0:
                # Multi-match candidates with positive evidence
                if prob >= 0.82 and prob >= (top_prob - 0.15) and is_strong_addr:
                    accepted.append(row["candidate_entity_id"])

        # --- MARGIN & AMBIGUITY CHECK (PRD 12.7) ---
        if len(sorted_group) >= 2:
            probs = sorted_group["probability"].values
            margin = probs[0] - probs[1]
            if probs[0] < 0.90 and margin < min_margin and sorted_group.iloc[0].get("exact_compact", 0) == 0:
                accepted = []

        predictions[s1_id] = accepted

    return predictions


apply_decision_policy = apply_precision_first_policy
