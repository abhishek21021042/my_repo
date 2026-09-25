import os
import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids
from src.normalize import add_normalized_views

# Load benchmark candidates and ground truth
cand_df = read_tsv("artifacts/candidates/rapid_candidates_1k.tsv") if os.path.exists("artifacts/candidates/rapid_candidates_1k.tsv") else None

rapid_s1 = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
sample_ids = set(rapid_s1["source1_entity_id"].iloc[:200])

gt_df = read_tsv("train_ground_truth.tsv")
s1_df = read_tsv("train_source1.tsv")
s1_dict = s1_df[s1_df["entity_id"].isin(sample_ids)].set_index("entity_id").to_dict("index")

# We can re-check by looking at 5 true matches that were missed
print("--- Diagnosing why candidates were missed ---")
# Check if address token indexing would have caught them
s2_df = read_tsv("train_source2.tsv")
s3_df = read_tsv("train_source3.tsv")

s2_dict = s2_df.set_index("entity_id").to_dict("index")
s3_dict = s3_df.set_index("entity_id").to_dict("index")

eval_gt = gt_df[gt_df["source1_entity_id"].isin(sample_ids)]

for _, row in eval_gt.head(10).iterrows():
    s1_id = row["source1_entity_id"]
    matches = parse_matched_ids(row["matched_entity_ids"])
    s1_info = s1_dict.get(s1_id)
    if not s1_info: continue
    
    print(f"\n[S1] {s1_id}:")
    print(f"  Name:    {s1_info['business_name']}")
    print(f"  Address: {s1_info['business_address']}")
    for m in matches:
        target_info = s2_dict.get(m) or s3_dict.get(m)
        if target_info:
            print(f"  -> Match {m}:")
            print(f"     Name:    {target_info['business_name']}")
            print(f"     Address: {target_info['business_address']}")
