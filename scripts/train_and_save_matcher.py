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
    print("=== TRAIN & PERSIST MATCHER (FOLDS 1-4 TRAIN -> FOLD 0 VALIDATION) ===")
    print("=========================================================================")
    t_start = time.time()

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cur = conn.cursor()

    # 1. Load folds split file
    print("Loading split definitions...")
    folds_df = read_tsv("artifacts/splits/s1_folds.tsv")
    
    # Train entities from folds 1, 2, 3, 4 (strictly avoiding fold 0)
    folds_df["fold"] = pd.to_numeric(folds_df["fold"], errors="coerce")
    train_pool = folds_df[folds_df["fold"] > 0]
    train_india = list(train_pool[train_pool["country"] == "India"]["source1_entity_id"].iloc[:1250])
    train_us = list(train_pool[train_pool["country"] == "US"]["source1_entity_id"].iloc[:1250])
    train_sample_ids = train_india + train_us
    print(f"Sampled {len(train_sample_ids):,} training entities from Folds 1-4 (1,250 India, 1,250 US).")

    # Fetch normalized S1 records for training
    placeholders = ",".join(["?"] * len(train_sample_ids))
    cur.execute(f"""
        SELECT entity_id, business_name, business_address, country,
               name_clean, name_compact, name_core_compact, name_sorted,
               address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
        FROM source1 
        WHERE entity_id IN ({placeholders})
    """, train_sample_ids)
    train_s1_raw = cur.fetchall()

    train_s1_dict = {}
    train_s1_list = []
    for r in train_s1_raw:
        item = {
            "entity_id": r[0],
            "business_name": r[1],
            "business_address": r[2],
            "country": r[3],
            "name_clean": r[4],
            "name_compact": r[5],
            "name_core_compact": r[6],
            "name_no_legal": r[6],
            "name_sorted": r[7],
            "address_clean": r[8],
            "postal_code": r[9],
            "numbers": r[10].split(",") if r[10] else [],
            "addr_anchor": r[11] if len(r) > 11 and r[11] else "",
            "tok1": r[12] if len(r) > 12 and r[12] else "",
            "addr_tok1": r[13] if len(r) > 13 and r[13] else ""
        }
        train_s1_dict[r[0]] = item
        train_s1_list.append(item)

    # Ground truth mapping for training
    print("Loading ground truth labels for training entities...")
    gt_df = read_tsv("train_ground_truth.tsv")
    train_gt_df = gt_df[gt_df["source1_entity_id"].isin(set(train_sample_ids))]
    train_gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in train_gt_df.iterrows()
    }

    # 2. Candidate generation for training set
    print("Generating candidate pairs via 6-channel B-tree blocker...")
    blocker = DiskBTreeBlocker(DEFAULT_DB_PATH, max_cands_per_query=80)
    train_cands = blocker.retrieve_candidates(train_s1_list)
    print(f"Retrieved {len(train_cands):,} candidate pairs for training.")

    # 3. Feature engineering
    print("Computing 21 pairwise features for training pairs...")
    t0 = time.time()
    train_feats = []
    train_labels = []

    for c in train_cands:
        s1_id = c["source1_entity_id"]
        cand_id = c["candidate_entity_id"]
        f_vec = compute_pairwise_features(train_s1_dict[s1_id], c["cand_record"], c)
        train_feats.append(f_vec)
        is_match = 1 if (s1_id in train_gt_dict and cand_id in train_gt_dict[s1_id]) else 0
        train_labels.append(is_match)

    train_feat_df = pd.DataFrame(train_feats)
    y_train = np.array(train_labels)
    X_train = train_feat_df[FEATURE_COLUMNS].values
    print(f"Training matrix built: {X_train.shape} ({y_train.sum():,} positives, {(y_train == 0).sum():,} negatives) in {time.time() - t0:.2f}s.")

    # 4. Train and calibrate matcher
    print("Training calibrated Gradient Boosted Matcher...")
    matcher = EntityMatcher(max_iter=200, learning_rate=0.07, random_state=42)
    matcher.fit(X_train, y_train)

    # Save model
    model_path = "artifacts/models/matcher.joblib"
    matcher.save(model_path)

    # 5. Out-of-fold validation on Fold 0 (1,000 entities: 500 India, 500 US)
    print("\n--- OUT-OF-FOLD VALIDATION ON FOLD 0 (1,000 UNSEEN ENTITIES) ---")
    rapid_s1 = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
    val_india = list(rapid_s1[rapid_s1["country"] == "India"]["source1_entity_id"].iloc[:500])
    val_us = list(rapid_s1[rapid_s1["country"] == "US"]["source1_entity_id"].iloc[:500])
    val_sample_ids = val_india + val_us

    placeholders = ",".join(["?"] * len(val_sample_ids))
    cur.execute(f"""
        SELECT entity_id, business_name, business_address, country,
               name_clean, name_compact, name_core_compact, name_sorted,
               address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
        FROM source1 
        WHERE entity_id IN ({placeholders})
    """, val_sample_ids)
    val_s1_raw = cur.fetchall()

    val_s1_dict = {}
    val_s1_list = []
    for r in val_s1_raw:
        item = {
            "entity_id": r[0],
            "business_name": r[1],
            "business_address": r[2],
            "country": r[3],
            "name_clean": r[4],
            "name_compact": r[5],
            "name_core_compact": r[6],
            "name_no_legal": r[6],
            "name_sorted": r[7],
            "address_clean": r[8],
            "postal_code": r[9],
            "numbers": r[10].split(",") if r[10] else [],
            "addr_anchor": r[11] if len(r) > 11 and r[11] else "",
            "tok1": r[12] if len(r) > 12 and r[12] else "",
            "addr_tok1": r[13] if len(r) > 13 and r[13] else ""
        }
        val_s1_dict[r[0]] = item
        val_s1_list.append(item)

    val_gt_df = gt_df[gt_df["source1_entity_id"].isin(set(val_sample_ids))]
    val_gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in val_gt_df.iterrows()
    }

    # Retrieve candidates for validation set
    t0 = time.time()
    val_cands = blocker.retrieve_candidates(val_s1_list)
    val_cand_pairs = set((c["source1_entity_id"], c["candidate_entity_id"]) for c in val_cands)
    all_true_pairs = set((s1, m) for s1, ms in val_gt_dict.items() for m in ms)
    captured = len(all_true_pairs & val_cand_pairs)
    print(f"Validation candidates: {len(val_cands):,} pairs in {time.time() - t0:.2f}s.")
    print(f"Candidate Recall: {captured}/{len(all_true_pairs)} ({captured/len(all_true_pairs)*100:.2f}%)")

    # Features for validation
    val_feats = []
    for c in val_cands:
        s1_id = c["source1_entity_id"]
        f_vec = compute_pairwise_features(val_s1_dict[s1_id], c["cand_record"], c)
        f_vec["source1_entity_id"] = s1_id
        f_vec["candidate_entity_id"] = c["candidate_entity_id"]
        val_feats.append(f_vec)

    val_feat_df = pd.DataFrame(val_feats)
    X_val = val_feat_df[FEATURE_COLUMNS].values
    val_feat_df["probability"] = matcher.predict_proba(X_val)

    # Decision policy
    predictions = apply_precision_first_policy(val_feat_df, threshold=0.95, min_margin=0.10, enable_dual_anchor=True)

    # Precision check
    total_preds = 0
    correct_preds = 0
    for s1_id, p_list in predictions.items():
        true_set = val_gt_dict.get(s1_id, set())
        for m in p_list:
            total_preds += 1
            if m in true_set: correct_preds += 1
    pair_precision = correct_preds / total_preds if total_preds > 0 else 1.0

    metrics = evaluate_predictions(val_gt_dict, predictions, beta=0.5)
    metrics["total_entities"] = len(val_sample_ids)
    metrics["total_candidates"] = len(val_cands)
    metrics["candidate_recall"] = float(captured / len(all_true_pairs))
    metrics["pair_precision"] = float(pair_precision)
    metrics["correct_pairs"] = correct_preds
    metrics["total_predicted_pairs"] = total_preds
    metrics["elapsed_seconds"] = round(time.time() - t_start, 1)

    print("\n================ OUT-OF-FOLD VALIDATION RESULTS ================")
    print(json.dumps(metrics, indent=2))

    # Save metrics report
    report_path = "artifacts/reports/model_validation_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Report saved to {report_path}")

    blocker.close()
    conn.close()

if __name__ == "__main__":
    main()
