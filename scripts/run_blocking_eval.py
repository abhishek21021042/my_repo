import os
import time
import json
import pandas as pd
from src.io_utils import read_tsv, write_tsv
from src.normalize import add_normalized_views
from src.blocking import MultiChannelBlocker, evaluate_blocking_recall

def main():
    print("=== STARTING CANDIDATE BLOCKING EVALUATION (RAPID SLICE) ===")
    start_time = time.time()

    # 1. Load rapid evaluation S1 IDs
    rapid_s1_path = "artifacts/splits/rapid_eval_s1.tsv"
    rapid_meta = read_tsv(rapid_s1_path)
    # Take first 1,000 entities (500 US, 500 India) for rapid benchmarking
    test_s1_ids = set(rapid_meta["source1_entity_id"].iloc[:1000])
    print(f"Evaluating {len(test_s1_ids):,} Source 1 entities across US and India...")

    # Load Source 1 and filter to evaluated subset
    s1_full = read_tsv("train_source1.tsv")
    s1_eval = s1_full[s1_full["entity_id"].isin(test_s1_ids)].copy()
    s1_eval = add_normalized_views(s1_eval)
    del s1_full
    print(f"Prepared normalized query views for {len(s1_eval):,} entities.")

    # Load Ground Truth
    gt_df = read_tsv("train_ground_truth.tsv")

    all_candidates = []

    # Process by country to maintain country partition integrity
    for country in ["US", "India"]:
        print(f"\n================ Processing Country: {country} ================")
        q_country = s1_eval[s1_eval["country"] == country].copy()
        if len(q_country) == 0:
            continue

        # --- SOURCE 2 ---
        print(f"Loading Source 2 partition for {country}...")
        s2_df = read_tsv("train_source2.tsv")
        s2_country = s2_df[s2_df["country"] == country].copy()
        del s2_df
        print(f"Found {len(s2_country):,} records in Source 2 ({country}). Adding normalized views...")
        s2_country = add_normalized_views(s2_country)

        blocker_s2 = MultiChannelBlocker(name_top_k=40)
        blocker_s2.build_indexes(s2_country)
        cands_s2 = blocker_s2.retrieve_candidates_for_queries(q_country, source_name="S2")
        all_candidates.extend(cands_s2)
        del s2_country, blocker_s2

        # --- SOURCE 3 ---
        print(f"\nLoading Source 3 partition for {country}...")
        s3_df = read_tsv("train_source3.tsv")
        s3_country = s3_df[s3_df["country"] == country].copy()
        del s3_df
        print(f"Found {len(s3_country):,} records in Source 3 ({country}). Adding normalized views...")
        s3_country = add_normalized_views(s3_country)

        blocker_s3 = MultiChannelBlocker(name_top_k=40)
        blocker_s3.build_indexes(s3_country)
        cands_s3 = blocker_s3.retrieve_candidates_for_queries(q_country, source_name="S3")
        all_candidates.extend(cands_s3)
        del s3_country, blocker_s3

    cand_df = pd.DataFrame(all_candidates)
    os.makedirs("artifacts/candidates", exist_ok=True)
    out_cand_path = "artifacts/candidates/rapid_candidates_1k.tsv"
    write_tsv(cand_df, out_cand_path)
    print(f"\nSaved {len(cand_df):,} total candidates to: {out_cand_path}")

    # Evaluate Blocking Recall
    metrics = evaluate_blocking_recall(cand_df, gt_df, test_s1_ids)
    elapsed = time.time() - start_time
    metrics["elapsed_seconds"] = round(elapsed, 1)

    print("\n================ CANDIDATE BLOCKING METRICS ================")
    print(json.dumps(metrics, indent=2))
    
    with open("artifacts/reports/blocking_rapid_1k_report.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
