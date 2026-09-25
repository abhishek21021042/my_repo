import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.io_utils import read_tsv, write_tsv, parse_matched_ids

def create_stratified_folds(
    data_dir: str,
    output_dir: str,
    n_splits: int = 5,
    seed: int = 42,
    rapid_sample_size: int = 5000
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    print(f"Loading Source 1 and Ground Truth from {data_dir}...")
    s1_df = read_tsv(s1_path)[["entity_id", "country"]]
    gt_df = read_tsv(gt_path)

    print("Computing match count buckets for stratification...")
    match_counts = gt_df["matched_entity_ids"].apply(lambda x: len(parse_matched_ids(x)))
    gt_df["match_count"] = match_counts

    df = pd.merge(s1_df, gt_df[["source1_entity_id", "match_count"]], left_on="entity_id", right_on="source1_entity_id")
    
    def get_bucket(count: int) -> str:
        if count == 0:
            return "0"
        elif count == 1:
            return "1"
        elif 2 <= count <= 3:
            return "2_3"
        else:
            return "4_plus"

    df["bucket"] = df["match_count"].apply(get_bucket)
    df["strata"] = df["country"] + "_" + df["bucket"]

    print(f"Total entities to split: {len(df):,}")
    print("Strata distribution:")
    print(df["strata"].value_counts())

    print(f"\nGenerating {n_splits} deterministic Stratified folds (seed={seed})...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["strata"])):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    output_folds_path = os.path.join(output_dir, "s1_folds.tsv")
    folds_export = df[["entity_id", "country", "match_count", "fold"]].rename(columns={"entity_id": "source1_entity_id"})
    write_tsv(folds_export, output_folds_path)
    print(f"Saved full folds to: {output_folds_path}")

    # Create rapid iteration slice from Fold 0
    fold0_df = folds_export[folds_export["fold"] == 0]
    rapid_parts = []
    for c, grp in fold0_df.groupby("country"):
        n_take = min(len(grp), rapid_sample_size // 2)
        rapid_parts.append(grp.sample(n=n_take, random_state=seed))
    rapid_sample = pd.concat(rapid_parts, ignore_index=True)
    rapid_path = os.path.join(output_dir, "rapid_eval_s1.tsv")
    write_tsv(rapid_sample[["source1_entity_id", "country", "match_count"]], rapid_path)
    print(f"Saved rapid evaluation slice ({len(rapid_sample):,} entities) to: {rapid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Directory containing TSV files")
    parser.add_argument("--output-dir", default="artifacts/splits", help="Directory to save fold assignments")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rapid-size", type=int, default=5000)
    args = parser.parse_args()

    create_stratified_folds(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        rapid_sample_size=args.rapid_size
    )
