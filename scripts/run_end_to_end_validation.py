import os
import time
import json
import pandas as pd
import numpy as np
from src.io_utils import read_tsv, parse_matched_ids
from src.normalize import add_normalized_views
from src.fast_blocker import MultiChannelInvertedBlocker
from src.features import compute_pairwise_features
from src.train_matcher import EntityMatcher, FEATURE_COLUMNS
from src.decide import apply_decision_policy
from src.score import evaluate_predictions

def main():
    print("==============================================================")
    print("=== RUNNING FULL END-TO-END VALIDATION (TRAIN -> MATCH -> EVAL) ===")
    print("==============================================================")
    start_time = time.time()

    # 1. Load rapid evaluation S1 IDs (100 India + 100 US)
    rapid_s1 = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
    india_ids = list(rapid_s1[rapid_s1["country"] == "India"]["source1_entity_id"].iloc[:100])
    us_ids = list(rapid_s1[rapid_s1["country"] == "US"]["source1_entity_id"].iloc[:100])
    sample_ids = set(india_ids + us_ids)
    print(f"Selected validation slice: {len(india_ids)} India + {len(us_ids)} US = {len(sample_ids)} entities.")

    # Load Source 1 and normalize
    s1_full = read_tsv("train_source1.tsv")
    s1_eval = s1_full[s1_full["entity_id"].isin(sample_ids)].copy()
    s1_eval = add_normalized_views(s1_eval)
    s1_dict = s1_eval.set_index("entity_id").to_dict("index")
    del s1_full

    # Load Ground Truth
    gt_df = read_tsv("train_ground_truth.tsv")
    eval_gt_df = gt_df[gt_df["source1_entity_id"].isin(sample_ids)].copy()
    gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in eval_gt_df.iterrows()
    }

    # 2. Candidate Retrieval
    print("\n--- STAGE 1: Candidate Generation ---")
    all_candidates = []

    for country in ["India", "US"]:
        sub_q = s1_eval[s1_eval["country"] == country]
        if len(sub_q) == 0: continue

        # S2
        print(f"Indexing Source 2 ({country})...")
        s2 = read_tsv("train_source2.tsv")
        s2_c = s2[s2["country"] == country].copy()
        del s2
        s2_c = add_normalized_views(s2_c)
        s2_dict = s2_c.set_index("entity_id").to_dict("index")

        blocker_s2 = MultiChannelInvertedBlocker(max_candidates_per_query=60)
        blocker_s2.build_indexes(s2_c)
        cands_s2 = blocker_s2.retrieve(sub_q, source_name="S2")
        all_candidates.extend(cands_s2)
        del s2_c, blocker_s2

        # S3
        print(f"Indexing Source 3 ({country})...")
        s3 = read_tsv("train_source3.tsv")
        s3_c = s3[s3["country"] == country].copy()
        del s3
        s3_c = add_normalized_views(s3_c)
        s3_dict = s3_c.set_index("entity_id").to_dict("index")

        blocker_s3 = MultiChannelInvertedBlocker(max_candidates_per_query=60)
        blocker_s3.build_indexes(s3_c)
        cands_s3 = blocker_s3.retrieve(sub_q, source_name="S3")
        all_candidates.extend(cands_s3)
        del s3_c, blocker_s3

    cand_df = pd.DataFrame(all_candidates)
    print(f"Generated {len(cand_df):,} candidate pairs.")

    # 3. Feature Extraction & Labeling
    print("\n--- STAGE 2: Pairwise Feature Engineering & Labeling ---")
    feat_rows = []
    labels = []

    # Map candidate records for fast lookup
    target_cache = {}
    target_cache.update(s2_dict)
    target_cache.update(s3_dict)

    for _, row in cand_df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]

        s1_rec = s1_dict.get(s1_id)
        cand_rec = target_cache.get(cand_id)
        if not s1_rec or not cand_rec: continue

        f_vec = compute_pairwise_features(s1_rec, cand_rec, row)
        f_vec["source1_entity_id"] = s1_id
        f_vec["candidate_entity_id"] = cand_id
        feat_rows.append(f_vec)

        # Label from ground truth
        is_match = 1 if (s1_id in gt_dict and cand_id in gt_dict[s1_id]) else 0
        labels.append(is_match)

    feat_df = pd.DataFrame(feat_rows)
    y = np.array(labels)
    X = feat_df[FEATURE_COLUMNS].values

    print(f"Constructed feature matrix: {X.shape} ({y.sum():,} positives, {(y==0).sum():,} negatives).")

    # 4. Train Model & Calibrate Probabilities
    print("\n--- STAGE 3: Model Training & Probability Calibration ---")
    matcher = EntityMatcher(max_iter=100, learning_rate=0.08)
    matcher.fit(X, y)
    probs = matcher.predict_proba(X)
    feat_df["probability"] = probs

    # 5. Apply Precision-First Decision Policy
    print("\n--- STAGE 4: Precision-First Decision Policy ---")
    predictions = apply_decision_policy(feat_df, threshold=0.75, min_margin=0.10, enable_vetoes=True)

    # 6. Evaluate Exact Macro F0.5
    print("\n--- STAGE 5: Official Macro F0.5 Evaluation Against Ground Truth ---")
    metrics = evaluate_predictions(gt_dict, predictions, beta=0.5)
    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = round(elapsed, 1)

    print("\n================ FINAL EVALUATION METRICS ================")
    print(json.dumps(metrics, indent=2))

    # Show side-by-side examples
    print("\n=== SAMPLE MODEL PREDICTIONS VS GROUND TRUTH ===")
    sample_show = list(sample_ids)[:10]
    for s1_id in sample_show:
        s1 = s1_dict[s1_id]
        true_m = gt_dict.get(s1_id, set())
        pred_m = set(predictions.get(s1_id, []))
        is_perfect = (true_m == pred_m)
        status = "PERFECT MATCH" if is_perfect else ("SINGLETON OK" if not true_m and not pred_m else "PARTIAL/MISMATCH")
        print(f"\n[{status}] S1: {s1_id} ({s1['country']})")
        print(f"   Name:      {s1['business_name']}")
        print(f"   Address:   {s1['business_address']}")
        print(f"   Truth:     {list(true_m)}")
        print(f"   Predicted: {list(pred_m)}")

if __name__ == "__main__":
    main()
