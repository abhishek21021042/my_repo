import os
import json
import argparse
from typing import Dict, Any, Set
import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids

def audit_source_file(path: str, expected_prefix: str) -> Dict[str, Any]:
    print(f"Auditing source file: {path} (Expected prefix: {expected_prefix})...")
    df = read_tsv(path)
    
    expected_cols = ["entity_id", "business_name", "business_address", "country"]
    actual_cols = list(df.columns)
    assert actual_cols == expected_cols, f"Column mismatch in {path}: expected {expected_cols}, got {actual_cols}"

    total_rows = len(df)
    unique_ids = df["entity_id"].nunique()
    assert unique_ids == total_rows, f"Duplicate IDs detected in {path}: {unique_ids} unique vs {total_rows} total rows"

    # Check prefix
    invalid_prefixes = (~df["entity_id"].str.startswith(expected_prefix)).sum()
    assert invalid_prefixes == 0, f"Found {invalid_prefixes} rows with invalid prefix in {path}"

    # Missing fields
    empty_names = (df["business_name"].str.strip() == "").sum()
    empty_addresses = (df["business_address"].str.strip() == "").sum()
    empty_countries = (df["country"].str.strip() == "").sum()

    # Country distribution
    country_counts = df["country"].value_counts().to_dict()

    return {
        "file": os.path.basename(path),
        "total_rows": int(total_rows),
        "empty_names": int(empty_names),
        "empty_names_pct": float(empty_names / total_rows * 100) if total_rows > 0 else 0,
        "empty_addresses": int(empty_addresses),
        "empty_addresses_pct": float(empty_addresses / total_rows * 100) if total_rows > 0 else 0,
        "empty_countries": int(empty_countries),
        "country_distribution": country_counts
    }

def audit_ground_truth(
    gt_path: str, 
    s1_ids: Set[str], 
    s2_ids: Set[str], 
    s3_ids: Set[str]
) -> Dict[str, Any]:
    print(f"Auditing ground truth file: {gt_path}...")
    df = read_tsv(gt_path)

    expected_cols = ["source1_entity_id", "matched_entity_ids"]
    assert list(df.columns) == expected_cols, f"Column mismatch in {gt_path}"

    total_rows = len(df)
    unique_s1 = df["source1_entity_id"].nunique()
    assert unique_s1 == total_rows, f"Duplicate source1_entity_id in {gt_path}"

    # Coverage assertion
    gt_s1_set = set(df["source1_entity_id"])
    missing_in_s1 = len(gt_s1_set - s1_ids)
    assert missing_in_s1 == 0, f"Found {missing_in_s1} ground truth S1 IDs missing in train_source1"

    missing_in_gt = len(s1_ids - gt_s1_set)

    # Inspect matches
    singletons = 0
    match_counts = []
    invalid_matched_ids = 0
    duplicate_matches_in_row = 0
    s2_matches_count = 0
    s3_matches_count = 0

    for _, row in df.iterrows():
        raw_matches = row["matched_entity_ids"]
        parsed = parse_matched_ids(raw_matches)
        
        # Check duplicate IDs inside row
        if len(parsed) != len([x.strip() for x in str(raw_matches).split(",") if x.strip()]):
            duplicate_matches_in_row += 1

        if len(parsed) == 0:
            singletons += 1
        else:
            match_counts.append(len(parsed))
            for m in parsed:
                if m.startswith("S2-"):
                    s2_matches_count += 1
                    if m not in s2_ids:
                        invalid_matched_ids += 1
                elif m.startswith("S3-"):
                    s3_matches_count += 1
                    if m not in s3_ids:
                        invalid_matched_ids += 1
                else:
                    invalid_matched_ids += 1

    return {
        "file": os.path.basename(gt_path),
        "total_s1_ground_truth": int(total_rows),
        "s1_coverage_complete": (missing_in_gt == 0),
        "missing_s1_in_gt": int(missing_in_gt),
        "singletons_count": int(singletons),
        "singletons_pct": float(singletons / total_rows * 100),
        "non_singletons_count": int(len(match_counts)),
        "s2_matches_total": int(s2_matches_count),
        "s3_matches_total": int(s3_matches_count),
        "invalid_matched_ids_count": int(invalid_matched_ids),
        "duplicate_matches_in_row_count": int(duplicate_matches_in_row),
        "matches_per_entity_mean": float(pd.Series(match_counts).mean()) if match_counts else 0,
        "matches_per_entity_max": int(max(match_counts)) if match_counts else 0,
    }

def run_data_audit(data_dir: str, output_report_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_report_path)), exist_ok=True)
    
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")

    report = {}

    s1_audit = audit_source_file(s1_path, "S1-")
    report["train_source1"] = s1_audit

    s2_df = read_tsv(s2_path)
    s2_ids = set(s2_df["entity_id"])
    s2_audit = audit_source_file(s2_path, "S2-")
    report["train_source2"] = s2_audit
    del s2_df

    s3_df = read_tsv(s3_path)
    s3_ids = set(s3_df["entity_id"])
    s3_audit = audit_source_file(s3_path, "S3-")
    report["train_source3"] = s3_audit
    del s3_df

    s1_df = read_tsv(s1_path)
    s1_ids = set(s1_df["entity_id"])
    del s1_df

    gt_audit = audit_ground_truth(gt_path, s1_ids, s2_ids, s3_ids)
    report["train_ground_truth"] = gt_audit

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================== PHASE 0: AUDIT COMPLETE ==================")
    print(f"Report saved to: {output_report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Directory containing TSV files")
    parser.add_argument("--output", default="artifacts/reports/data_audit_report.json", help="Output JSON path")
    args = parser.parse_args()

    run_data_audit(args.data_dir, args.output)
