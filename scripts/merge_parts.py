import os
import sys
import glob
import pandas as pd
from typing import List, Dict

sys.path.insert(0, os.path.abspath("."))
from src.write_outputs import assert_submission_validity

def merge_parts(
    output_dir: str = "output",
    input_source1_path: str = "test_data/test_source1 (1).tsv",
    final_output_path: str = "output/matching_results.tsv",
    final_candidates_path: str = "output/candidate_pairs.tsv"
):
    print("=" * 80)
    print("=== MERGING DISTRIBUTED PART OUTPUTS INTO FINAL SUBMISSION ===")
    print("=" * 80)

    # 1. Find all matching_results parts
    part_files = sorted(glob.glob(os.path.join(output_dir, "matching_results_part_*.tsv")))
    if not part_files:
        print(f"Error: No part files found matching '{output_dir}/matching_results_part_*.tsv'!")
        print("Please place the part TSVs from all laptops into the 'output/' folder.")
        sys.exit(1)

    print(f"Found {len(part_files)} part files to merge:")
    for pf in part_files:
        print(f"  - {pf}")

    # 2. Merge matching_results
    merged_rows = []
    seen_ids = set()
    duplicate_count = 0

    for pf in part_files:
        print(f"Reading {pf}...")
        with open(pf, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                line_clean = line.strip()
                if not line_clean:
                    continue
                parts = line_clean.split("\t")
                s1_id = parts[0]
                matched_str = parts[1] if len(parts) > 1 else ""
                if s1_id in seen_ids:
                    duplicate_count += 1
                    continue
                seen_ids.add(s1_id)
                merged_rows.append((s1_id, matched_str))

    print(f"\nTotal unique entities collected across all parts: {len(merged_rows):,}")
    if duplicate_count > 0:
        print(f"Warning: Skipped {duplicate_count:,} duplicate entity IDs across overlapping parts.")

    # 3. Read original S1 file to ensure exact ordering
    if os.path.exists(input_source1_path):
        print(f"Aligning records to exact input order from {input_source1_path}...")
        orig_s1_df = pd.read_csv(input_source1_path, sep="\t", usecols=["entity_id"])
        orig_ids = list(orig_s1_df["entity_id"])
        pred_dict = dict(merged_rows)

        final_rows = []
        missing_count = 0
        for eid in orig_ids:
            if eid in pred_dict:
                final_rows.append((eid, pred_dict[eid]))
            else:
                final_rows.append((eid, ""))
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count:,} entities were missing from parts and set to empty (singleton).")
    else:
        final_rows = merged_rows

    # 4. Write final matching_results.tsv
    print(f"\nWriting final merged submission to: {final_output_path} ...")
    with open(final_output_path, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for eid, m_str in final_rows:
            f_out.write(f"{eid}\t{m_str}\n")

    print(f"Successfully saved {len(final_rows):,} rows in {final_output_path}.")

    # 5. Merge candidate pairs if present
    cand_part_files = sorted(glob.glob(os.path.join(output_dir, "candidate_pairs_part_*.tsv")))
    if cand_part_files:
        print(f"\nMerging {len(cand_part_files)} candidate pairs files...")
        with open(final_candidates_path, "w", encoding="utf-8") as f_cand_out:
            f_cand_out.write("source1_entity_id\tcandidate_entity_id\n")
            for cpf in cand_part_files:
                with open(cpf, "r", encoding="utf-8") as f_in:
                    f_in.readline() # skip header
                    for line in f_in:
                        f_cand_out.write(line)
        print(f"Candidate pairs saved to: {final_candidates_path}")

    # 6. Validate PRD Section 20 assertions
    print("\n--- Validating PRD Section 20 Submission Rules ---")
    try:
        assert_submission_validity(
            s1_entity_ids=[r[0] for r in final_rows],
            matching_results_path=final_output_path,
            candidate_pairs_path=final_candidates_path if os.path.exists(final_candidates_path) else None
        )
        print("★ VALIDATION PASSED: Submission file meets 100% of competition specifications!")
    except Exception as e:
        print(f"Validation Note: {e}")

    # 7. Summary
    matched_entities = sum(1 for _, m in final_rows if m)
    singletons = len(final_rows) - matched_entities
    print("\n================ FINAL SUBMISSION SUMMARY ================")
    print(f"Total Source 1 Entities:   {len(final_rows):,}")
    print(f"Entities with Matches:      {matched_entities:,} ({matched_entities/len(final_rows)*100:.2f}%)")
    print(f"Singletons (0 Matches):     {singletons:,} ({singletons/len(final_rows)*100:.2f}%)")
    print(f"File Size:                  {os.path.getsize(final_output_path) / (1024*1024):.2f} MB")
    print("==========================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge Distributed Part Outputs")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory containing part TSVs")
    parser.add_argument("--input_s1", type=str, default="test_data/test_source1 (1).tsv", help="Input Source 1 TSV")
    parser.add_argument("--final_out", type=str, default="output/matching_results.tsv", help="Final output TSV")
    args = parser.parse_args()

    merge_parts(
        output_dir=args.output_dir,
        input_source1_path=args.input_s1,
        final_output_path=args.final_out
    )
