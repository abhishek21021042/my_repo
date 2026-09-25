import os
import sys
import time
import json
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Set

sys.path.insert(0, os.path.abspath("."))
from src.io_utils import read_tsv, parse_matched_ids
from src.disk_blocker import DiskBTreeBlocker, DEFAULT_DB_PATH
from src.features import compute_pairwise_features
from src.train_matcher import EntityMatcher, FEATURE_COLUMNS
from src.decide import apply_precision_first_policy
from src.score import evaluate_predictions

def main():
    print("=" * 80)
    print("=== TESTING ON 2,000 COMPLETELY NEW & UNSEEN ENTITIES ===")
    print("=== Source 1 vs Source 2 & Source 3 against train_ground_truth.tsv ===")
    print("=" * 80)
    t_start = time.time()

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cur = conn.cursor()

    # 1. Identify previous entities to guarantee ZERO overlap
    prev_tested_ids = set()
    if os.path.exists("output/matching_results.tsv"):
        prev_df = read_tsv("output/matching_results.tsv")
        prev_tested_ids.update(prev_df["source1_entity_id"].tolist())
    if os.path.exists("artifacts/splits/rapid_eval_s1.tsv"):
        rapid_df = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
        prev_tested_ids.update(rapid_df["source1_entity_id"].iloc[:1000].tolist())
    print(f"Loaded {len(prev_tested_ids):,} previously tested IDs to exclude from this test.")

    # 2. Sample 2,000 FRESH, UNSEEN entities from Fold 1 (1,000 India + 1,000 US)
    print("Selecting 2,000 brand-new entities from Holdout Fold 1...")
    folds_df = read_tsv("artifacts/splits/s1_folds.tsv")
    folds_df["fold"] = pd.to_numeric(folds_df["fold"], errors="coerce")
    fold1_pool = folds_df[(folds_df["fold"] == 1) & (~folds_df["source1_entity_id"].isin(prev_tested_ids))]

    new_india_ids = list(fold1_pool[fold1_pool["country"] == "India"]["source1_entity_id"].iloc[:1000])
    new_us_ids = list(fold1_pool[fold1_pool["country"] == "US"]["source1_entity_id"].iloc[:1000])
    sample_ids = new_india_ids + new_us_ids

    # Verify zero overlap
    overlap = set(sample_ids) & prev_tested_ids
    assert len(overlap) == 0, f"Error: Overlap detected with previous runs: {len(overlap)}"
    assert len(sample_ids) == 2000, f"Error: Sample size is {len(sample_ids)}, expected 2000"
    print(f"Successfully selected {len(sample_ids):,} 100% UNSEEN entities (1,000 India, 1,000 US).")
    print(f"Verified Overlap with previous tests: EXACTLY 0 entities.")

    # 3. Fetch normalized records from Source 1 in SQLite
    print("\nFetching normalized query records from SQLite source1 table...")
    t0 = time.time()
    placeholders = ",".join(["?"] * len(sample_ids))
    cur.execute(f"""
        SELECT entity_id, business_name, business_address, country,
               name_clean, name_compact, name_core_compact, name_sorted,
               address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
        FROM source1 WHERE entity_id IN ({placeholders})
    """, sample_ids)
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
    print(f"Loaded {len(s1_list):,} query records in {time.time() - t0:.2f}s.")

    # 4. Load Official Ground Truth for these 2,000 entities
    print("Loading Ground Truth from train_ground_truth.tsv for these 2,000 entities...")
    gt_df = read_tsv("train_ground_truth.tsv")
    sample_gt_df = gt_df[gt_df["source1_entity_id"].isin(set(sample_ids))]
    gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in sample_gt_df.iterrows()
    }
    all_true_pairs = set((s1, m) for s1, ms in gt_dict.items() for m in ms)
    gt_singletons = sum(1 for ms in gt_dict.values() if len(ms) == 0)
    print(f"Ground Truth Statistics for this 2,000 slice:")
    print(f"  - Total Ground Truth Pairs: {len(all_true_pairs):,}")
    print(f"  - True Singletons (0 matches in GT): {gt_singletons:,}")
    print(f"  - Multi-match entities (>=1 match): {len(sample_ids) - gt_singletons:,}")

    # 5. Candidate Generation (Stage 1)
    print("\n--- STAGE 1: Fast B-Tree Candidate Retrieval from Source 2 & Source 3 ---", flush=True)
    t0 = time.time()
    blocker = DiskBTreeBlocker(DEFAULT_DB_PATH, max_cands_per_query=150)
    candidates = blocker.retrieve_candidates(s1_list)
    print(f"Retrieved {len(candidates):,} candidate pairs in {time.time() - t0:.2f}s.", flush=True)

    cand_pairs = set((c["source1_entity_id"], c["candidate_entity_id"]) for c in candidates)
    captured = len(all_true_pairs & cand_pairs)
    blocker_recall = (captured / len(all_true_pairs) * 100) if len(all_true_pairs) > 0 else 100.0
    print(f"Candidate Blocker Recall: {captured:,} / {len(all_true_pairs):,} ({blocker_recall:.2f}%)", flush=True)

    # 6. Feature Computation (Stage 2)
    print("\n--- STAGE 2: Computing Pairwise Similarity Features ---", flush=True)
    t0 = time.time()
    feat_rows = []
    for c in candidates:
        s1_id = c["source1_entity_id"]
        f_vec = compute_pairwise_features(s1_dict[s1_id], c["cand_record"], c)
        f_vec["source1_entity_id"] = s1_id
        f_vec["candidate_entity_id"] = c["candidate_entity_id"]
        feat_rows.append(f_vec)

    feat_df = pd.DataFrame(feat_rows)
    print(f"Computed features for {len(feat_df):,} candidate pairs in {time.time() - t0:.2f}s.")

    # 7. Model Scoring & Decision Policy (Stage 3)
    print("\n--- STAGE 3: Model Scoring & Decision Policy (threshold=0.93, strong_addr=0.89) ---")
    t0 = time.time()
    matcher = EntityMatcher.load("artifacts/models/matcher.joblib")
    feat_df["probability"] = matcher.predict_proba(feat_df[FEATURE_COLUMNS].values)

    predictions = apply_precision_first_policy(
        feat_df,
        threshold=0.93,
        strong_addr_threshold=0.89,
        min_margin=0.05,
        enable_dual_anchor=True
    )
    print(f"Decisions applied in {time.time() - t0:.2f}s.")

    # 8. Detailed Metric Calculation & Direct Proof
    print("\n" + "=" * 80)
    print("=== LIVE PROOF & EMPIRICAL RESULTS ON 2,000 NEW ENTITIES ===")
    print("=" * 80)

    total_pred_pairs = 0
    true_positives = 0
    false_positives = 0
    predicted_pairs_set = set()

    for s1_id, p_list in predictions.items():
        true_set = gt_dict.get(s1_id, set())
        for m in p_list:
            total_pred_pairs += 1
            predicted_pairs_set.add((s1_id, m))
            if m in true_set:
                true_positives += 1
            else:
                false_positives += 1

    false_negatives = len(all_true_pairs - predicted_pairs_set)
    pair_precision = (true_positives / total_pred_pairs * 100) if total_pred_pairs > 0 else 100.0
    pair_recall = (true_positives / len(all_true_pairs) * 100) if len(all_true_pairs) > 0 else 100.0
    macro_metrics = evaluate_predictions(gt_dict, predictions, beta=0.5)

    print(f"\n[OVERALL RESULTS (2,000 Unseen Entities)]")
    print(f"  Total S1 Entities Tested:        {len(sample_ids):,}")
    print(f"  Total Ground Truth Matches:      {len(all_true_pairs):,}")
    print(f"  Total Matches Predicted:         {total_pred_pairs:,}")
    print(f"  True Positives (Correct Matches): {true_positives:,}")
    print(f"  False Positives (Incorrect):     {false_positives:,}")
    print(f"  False Negatives (Missed):        {false_negatives:,}")
    print(f"  -------------------------------------------------------------")
    print(f"  ★ PAIR-LEVEL PRECISION:          {pair_precision:.3f}%  (Target >= 99.0%)")
    print(f"  ★ PAIR-LEVEL RECALL:             {pair_recall:.3f}%")
    print(f"  ★ MACRO PRECISION:               {macro_metrics['macro_precision']*100:.3f}%")
    print(f"  ★ MACRO RECALL:                  {macro_metrics['macro_recall']*100:.3f}%")
    print(f"  ★ MACRO F0.5 SCORE:              {macro_metrics['macro_f05']:.4f}")
    print(f"  ★ SINGLETON ACCURACY:            {macro_metrics['singleton_accuracy']*100:.2f}% ({macro_metrics['singleton_count']} singletons)")
    print(f"  -------------------------------------------------------------")

    # Per-Country Breakdown
    for country_name, ids in [("India", new_india_ids), ("US", new_us_ids)]:
        c_gt = {eid: gt_dict.get(eid, set()) for eid in ids}
        c_preds = {eid: predictions.get(eid, []) for eid in ids}
        c_all_true = set((s1, m) for s1, ms in c_gt.items() for m in ms)
        c_tot = sum(len(v) for v in c_preds.values())
        c_corr = sum(1 for eid, vs in c_preds.items() for v in vs if v in c_gt.get(eid, set()))
        c_fp = c_tot - c_corr
        c_prec = (c_corr / c_tot * 100) if c_tot > 0 else 100.0
        c_rec = (c_corr / len(c_all_true) * 100) if len(c_all_true) > 0 else 100.0
        c_macro = evaluate_predictions(c_gt, c_preds, beta=0.5)

        print(f"\n[{country_name.upper()} BREAKDOWN (1,000 Entities)]")
        print(f"  - Ground Truth True Pairs:   {len(c_all_true):,}")
        print(f"  - Predicted Pairs:           {c_tot:,}")
        print(f"  - True Positives (Correct):  {c_corr:,}")
        print(f"  - False Positives:           {c_fp:,}")
        print(f"  - Pair-Level Precision:      {c_prec:.3f}%")
        print(f"  - Pair-Level Recall:         {c_rec:.3f}%")
        print(f"  - Macro F0.5 Score:          {c_macro['macro_f05']:.4f}")
        print(f"  - Singleton Accuracy:        {c_macro['singleton_accuracy']*100:.2f}%")

    # 9. Extract and print side-by-side verification examples
    print("\n" + "=" * 80)
    print("=== SIDE-BY-SIDE VERIFICATION EXAMPLES FROM THIS 2,000 RUN ===")
    print("=" * 80)
    
    verified_examples = []
    # Find 5 verified true positives (matches)
    for s1_id in sample_ids:
        preds = predictions.get(s1_id, [])
        trues = gt_dict.get(s1_id, set())
        for cand_id in preds:
            if cand_id in trues:
                # Fetch candidate record from DB
                src_tbl = "source2" if cand_id.startswith("S2") else "source3"
                cur.execute(f"SELECT business_name, business_address FROM {src_tbl} WHERE entity_id = ?", (cand_id,))
                c_row = cur.fetchone()
                if c_row:
                    verified_examples.append({
                        "s1_id": s1_id,
                        "s1_name": s1_dict[s1_id]["business_name"],
                        "s1_addr": s1_dict[s1_id]["business_address"],
                        "cand_id": cand_id,
                        "cand_name": c_row[0],
                        "cand_addr": c_row[1],
                        "country": s1_dict[s1_id]["country"]
                    })
            if len(verified_examples) >= 6:
                break
        if len(verified_examples) >= 6:
            break

    for idx, ex in enumerate(verified_examples, 1):
        print(f"\n[Example {idx} - {ex['country']}]")
        print(f"  Query S1:    [{ex['s1_id']}] {ex['s1_name']}")
        print(f"               Address: {ex['s1_addr']}")
        print(f"  Matched Cand:[{ex['cand_id']}] {ex['cand_name']}")
        print(f"               Address: {ex['cand_addr']}")
        print(f"  Verification Status: MATCH CONFIRMED IN GROUND TRUTH")

    # 10. Save results for reproducibility
    results_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_entities_tested": len(sample_ids),
        "total_true_pairs": len(all_true_pairs),
        "total_predicted_pairs": total_pred_pairs,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "pair_precision_percent": pair_precision,
        "pair_recall_percent": pair_recall,
        "macro_f05": macro_metrics["macro_f05"],
        "macro_precision": macro_metrics["macro_precision"],
        "macro_recall": macro_metrics["macro_recall"],
        "singleton_accuracy": macro_metrics["singleton_accuracy"],
        "singleton_count": macro_metrics["singleton_count"],
        "country_breakdown": {
            "India": {"tested": 1000},
            "US": {"tested": 1000}
        }
    }
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/eval_2000_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nDetailed proof summary saved to scratch/eval_2000_results.json.")
    print("=" * 80)

if __name__ == "__main__":
    main()
