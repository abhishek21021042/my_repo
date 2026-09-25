import os
import time
import json
import pandas as pd
from src.io_utils import read_tsv, write_tsv
from src.normalize import add_normalized_views
from src.fast_blocker import MultiChannelInvertedBlocker
from src.blocking import evaluate_blocking_recall

def benchmark():
    print("=== BENCHMARKING COMPREHENSIVE 7-CHANNEL BLOCKER ===")
    t0 = time.time()

    # Load 100 India + 100 US entities from rapid eval slice
    rapid_s1 = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
    india_ids = list(rapid_s1[rapid_s1["country"] == "India"]["source1_entity_id"].iloc[:100])
    us_ids = list(rapid_s1[rapid_s1["country"] == "US"]["source1_entity_id"].iloc[:100])
    sample_ids = set(india_ids + us_ids)
    print(f"Sample size: {len(india_ids)} India, {len(us_ids)} US = {len(sample_ids)} total entities.")

    s1_df = read_tsv("train_source1.tsv")
    q_df = s1_df[s1_df["entity_id"].isin(sample_ids)].copy()
    q_df = add_normalized_views(q_df)
    del s1_df

    gt_df = read_tsv("train_ground_truth.tsv")

    all_candidates = []

    for country in ["India", "US"]:
        print(f"\n--- Testing Country: {country} ---")
        sub_q = q_df[q_df["country"] == country]
        if len(sub_q) == 0:
            continue

        # Load Source 2
        print(f"Loading Source 2 ({country})...")
        s2 = read_tsv("train_source2.tsv")
        s2_c = s2[s2["country"] == country].copy()
        del s2
        s2_c = add_normalized_views(s2_c)

        blocker = MultiChannelInvertedBlocker(max_candidates_per_query=80)
        blocker.build_indexes(s2_c)
        cands_s2 = blocker.retrieve(sub_q, source_name="S2")
        all_candidates.extend(cands_s2)
        del s2_c, blocker

        # Load Source 3
        print(f"Loading Source 3 ({country})...")
        s3 = read_tsv("train_source3.tsv")
        s3_c = s3[s3["country"] == country].copy()
        del s3
        s3_c = add_normalized_views(s3_c)

        blocker = MultiChannelInvertedBlocker(max_candidates_per_query=80)
        blocker.build_indexes(s3_c)
        cands_s3 = blocker.retrieve(sub_q, source_name="S3")
        all_candidates.extend(cands_s3)
        del s3_c, blocker

    cand_df = pd.DataFrame(all_candidates)
    metrics = evaluate_blocking_recall(cand_df, gt_df, sample_ids)
    metrics["elapsed_seconds"] = round(time.time() - t0, 1)

    print("\n================ COMPREHENSIVE BENCHMARK RESULTS ================")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    benchmark()
