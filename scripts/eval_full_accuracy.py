import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np

from src.io_utils import read_tsv, parse_matched_ids
from src.disk_blocker import DiskBTreeBlocker, DEFAULT_DB_PATH
from src.features import compute_pairwise_features
from src.train_matcher import EntityMatcher, FEATURE_COLUMNS
from src.decide import apply_precision_first_policy
from src.score import evaluate_predictions

def main():
    print("=========================================================================")
    print("=== FULL GROUND-TRUTH ACCURACY & PRECISION EVALUATION ===")
    print("=== Using Source 1, Source 2, Source 3 against train_ground_truth.tsv ===")
    print("=========================================================================")
    t_start = time.time()

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cur = conn.cursor()

    # 1. Load Fold 0 Unseen Test Sample (2,000 entities: 1,000 India, 1,000 US)
    print("Loading test entities from Holdout Fold 0...")
    folds_df = read_tsv("artifacts/splits/s1_folds.tsv")
    folds_df["fold"] = pd.to_numeric(folds_df["fold"], errors="coerce")
    val_pool = folds_df[folds_df["fold"] == 0]

    val_india = list(val_pool[val_pool["country"] == "India"]["source1_entity_id"].iloc[:1000])
    val_us = list(val_pool[val_pool["country"] == "US"]["source1_entity_id"].iloc[:1000])
    val_sample_ids = val_india + val_us
    print(f"Sampled {len(val_sample_ids):,} unseen entities (1,000 India, 1,000 US).")

    # Fetch normalized records from Source 1
    placeholders = ",".join(["?"] * len(val_sample_ids))
    cur.execute(f"""
        SELECT entity_id, business_name, business_address, country,
               name_clean, name_compact, name_core_compact, name_sorted,
               address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
        FROM source1 WHERE entity_id IN ({placeholders})
    """, val_sample_ids)
    s1_raw = cur.fetchall()

    s1_dict = {}
    s1_list = []
    for r in s1_raw:
        item = {
            "entity_id": r[0], "business_name": r[1], "business_address": r[2], "country": r[3],
            "name_clean": r[4], "name_compact": r[5], "name_core_compact": r[6], "name_no_legal": r[6],
            "name_sorted": r[7], "address_clean": r[8], "postal_code": r[9],
            "numbers": r[10].split(",") if r[10] else [],
            "addr_anchor": r[11] if len(r) > 11 and r[11] else "",
            "tok1": r[12] if len(r) > 12 and r[12] else "",
            "addr_tok1": r[13] if len(r) > 13 and r[13] else ""
        }
        s1_dict[r[0]] = item
        s1_list.append(item)

    # Load Ground Truth
    print("Loading official ground truth labels from train_ground_truth.tsv...")
    gt_df = read_tsv("train_ground_truth.tsv")
    val_gt_df = gt_df[gt_df["source1_entity_id"].isin(set(val_sample_ids))]
    gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in val_gt_df.iterrows()
    }

    # 2. Candidate Retrieval from Source 2 and Source 3
    print("\n--- STAGE 1: Retrieving Candidates from Source 2 & Source 3 ---")
    t0 = time.time()
    blocker = DiskBTreeBlocker(DEFAULT_DB_PATH, max_cands_per_query=80)
    candidates = blocker.retrieve_candidates(s1_list)
    print(f"Retrieved {len(candidates):,} candidate pairs in {time.time() - t0:.2f}s.")

    cand_pairs = set((c["source1_entity_id"], c["candidate_entity_id"]) for c in candidates)
    all_true_pairs = set((s1, m) for s1, ms in gt_dict.items() for m in ms)
    captured = len(all_true_pairs & cand_pairs)
    print(f"Candidate Recall: {captured}/{len(all_true_pairs)} ({captured/len(all_true_pairs)*100:.2f}%)")

    # 3. Pairwise Feature Engineering
    print("\n--- STAGE 2: Computing Pairwise Features ---")
    t0 = time.time()
    feat_rows = []
    for c in candidates:
        s1_id = c["source1_entity_id"]
        f_vec = compute_pairwise_features(s1_dict[s1_id], c["cand_record"], c)
        f_vec["source1_entity_id"] = s1_id
        f_vec["candidate_entity_id"] = c["candidate_entity_id"]
        feat_rows.append(f_vec)

    feat_df = pd.DataFrame(feat_rows)
    print(f"Computed features for {len(feat_df):,} pairs in {time.time() - t0:.2f}s.")

    # 4. Model Scoring & Decision Policy
    print("\n--- STAGE 3: Model Scoring & Decision Policy (Threshold = 0.95) ---")
    t0 = time.time()
    matcher = EntityMatcher.load("artifacts/models/matcher.joblib")
    feat_df["probability"] = matcher.predict_proba(feat_df[FEATURE_COLUMNS].values)

    predictions = apply_precision_first_policy(
        feat_df,
        threshold=0.95,
        min_margin=0.10,
        enable_dual_anchor=True
    )
    print(f"Scoring and decisions completed in {time.time() - t0:.2f}s.")

    # 5. Exact Accuracy & Precision Calculation
    print("\n================ OFFICIAL ACCURACY & PRECISION METRICS ================")
    
    # Overall Pair-Level
    total_preds = 0
    correct_preds = 0
    fp_count = 0
    
    for s1_id, p_list in predictions.items():
        true_set = gt_dict.get(s1_id, set())
        for m in p_list:
            total_preds += 1
            if m in true_set:
                correct_preds += 1
            else:
                fp_count += 1

    pair_precision = correct_preds / total_preds if total_preds > 0 else 1.0

    # Macro F0.5
    macro_metrics = evaluate_predictions(gt_dict, predictions, beta=0.5)

    # Per-Country Breakdown
    for country_name, ids in [("India", val_india), ("US", val_us)]:
        c_gt = {eid: gt_dict.get(eid, set()) for eid in ids}
        c_preds = {eid: predictions.get(eid, []) for eid in ids}
        c_tot = sum(len(v) for v in c_preds.values())
        c_corr = sum(1 for eid, vs in c_preds.items() for v in vs if v in c_gt.get(eid, set()))
        c_fp = c_tot - c_corr
        c_prec = c_corr / c_tot if c_tot > 0 else 1.0
        c_macro = evaluate_predictions(c_gt, c_preds, beta=0.5)
        print(f"\n[{country_name.upper()} BREAKDOWN (1,000 Entities)]")
        print(f"  - Pair-Level Precision: {c_corr:,} / {c_tot:,} ({c_prec*100:.2f}%)")
        print(f"  - False Positives:     {c_fp}")
        print(f"  - Macro F0.5 Score:    {c_macro['macro_f05']:.4f}")
        print(f"  - Macro Precision:     {c_macro['macro_precision']*100:.2f}%")
        print(f"  - Singleton Accuracy:  {c_macro['singleton_accuracy']*100:.2f}% ({c_macro['singleton_count']} singletons)")

    print("\n----------------- OVERALL SUMMARY (2,000 ENTITIES) -----------------")
    print(f"  Total Entities Evaluated:      {len(val_sample_ids):,}")
    print(f"  Total Predicted Match Pairs:   {total_preds:,}")
    print(f"  Correct Match Pairs:           {correct_preds:,}")
    print(f"  False Positive Pairs:          {fp_count} (out of {total_preds:,})")
    print(f"  PAIR-LEVEL PRECISION:          {pair_precision * 100:.2f}%")
    print(f"  SINGLETON ACCURACY:            {macro_metrics['singleton_accuracy'] * 100:.2f}% ({macro_metrics['singleton_count']} singletons)")
    print(f"  MACRO PRECISION:               {macro_metrics['macro_precision'] * 100:.2f}%")
    print(f"  MACRO F0.5 SCORE:              {macro_metrics['macro_f05']:.4f}")
    print(f"  Candidate Recall:              {captured/len(all_true_pairs)*100:.2f}%")
    print(f"  Total Elapsed Time:            {time.time() - t_start:.1f}s")
    print("====================================================================")

    blocker.close()
    conn.close()

if __name__ == "__main__":
    main()
