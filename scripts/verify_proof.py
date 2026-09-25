import sqlite3
import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids

def main():
    print("=" * 80)
    print("=== LIVE PROOF: COMPARING PREDICTIONS DIRECTLY AGAINST GROUND TRUTH ===")
    print("=" * 80)

    # 1. Load predictions generated in output/matching_results.tsv
    pred_path = "output/matching_results.tsv"
    gt_path = "train_ground_truth.tsv"
    db_path = "D:/hackathon/dataset.db"

    print(f"Reading predictions from: {pred_path}")
    preds_df = read_tsv(pred_path)
    pred_dict = {
        row["source1_entity_id"]: [m for m in str(row["matched_entity_ids"]).split(",") if m and m != "nan"]
        for _, row in preds_df.iterrows()
    }
    eval_ids = list(pred_dict.keys())
    print(f"Total entities in output/matching_results.tsv: {len(eval_ids):,}")

    # 2. Load Ground Truth for these exact entities
    print(f"Reading ground truth from: {gt_path}")
    gt_df = read_tsv(gt_path)
    eval_gt_df = gt_df[gt_df["source1_entity_id"].isin(set(eval_ids))]
    gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in eval_gt_df.iterrows()
    }

    # 3. Calculate exact TP, FP, Precision
    total_predicted_pairs = 0
    correct_pairs = 0
    false_positives = 0
    singleton_total = 0
    singleton_correct = 0

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for s1_id in eval_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_list = pred_dict.get(s1_id, [])

        # Check singleton
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_list) == 0:
                singleton_correct += 1

        for m in pred_list:
            total_predicted_pairs += 1
            if m in true_set:
                correct_pairs += 1
            else:
                false_positives += 1

    precision = (correct_pairs / total_predicted_pairs) * 100 if total_predicted_pairs > 0 else 100.0

    print("\n" + "=" * 80)
    print("MATHEMATICAL PROOF & CONFUSION MATRIX:")
    print("=" * 80)
    print(f"  Total Predictions Made (TP + FP):  {total_predicted_pairs:,}")
    print(f"  True Positives (Correct Matches):  {correct_pairs:,}")
    print(f"  False Positives (Wrong Matches):   {false_positives:,}")
    print(f"  Calculated Precision Formula:      TP / (TP + FP)")
    print(f"  Calculation:                       {correct_pairs:,} / {total_predicted_pairs:,}")
    print(f"  ACTUAL MEASURED PRECISION:         {precision:.3f}%")
    print(f"  Singleton Accuracy:                {singleton_correct}/{singleton_total} ({singleton_correct/singleton_total*100:.2f}%)")
    print("=" * 80)

    # 4. Show 5 real side-by-side ground truth vs prediction examples
    print("\nSIDE-BY-SIDE PROOF EXAMPLES (Source 1 vs Matched Records):")
    print("-" * 80)

    sample_check = [s1_id for s1_id in eval_ids if len(pred_dict[s1_id]) > 0][:5]
    for idx, s1_id in enumerate(sample_check):
        cur.execute("SELECT business_name, business_address, country FROM source1 WHERE entity_id = ?", (s1_id,))
        s1_row = cur.fetchone()
        s1_name, s1_addr, country = s1_row if s1_row else ("Unknown", "Unknown", "Unknown")

        true_matches = sorted(list(gt_dict.get(s1_id, set())))
        pred_matches = sorted(pred_dict.get(s1_id, []))

        print(f"\n[EXAMPLE #{idx + 1}] Entity: {s1_id} ({country})")
        print(f"  Source 1 Business: {s1_name}")
        print(f"  Source 1 Address:  {s1_addr}")
        print(f"  Ground Truth IDs:  {', '.join(true_matches)}")
        print(f"  Predicted IDs:     {', '.join(pred_matches)}")

        # Fetch matched records from S2/S3
        for m in pred_matches:
            tbl = "source2" if m.startswith("S2-") else "source3"
            cur.execute(f"SELECT business_name, business_address FROM {tbl} WHERE entity_id = ?", (m,))
            cand_row = cur.fetchone()
            c_name, c_addr = cand_row if cand_row else ("Unknown", "Unknown")
            is_valid = "CORRECT (In Ground Truth)" if m in gt_dict.get(s1_id, set()) else "WRONG"
            print(f"    --> {m} [{is_valid}]:")
            print(f"        Matched Name:    {c_name}")
            print(f"        Matched Address: {c_addr}")

    # 5. Show singleton proof (no false merge)
    print("\n" + "-" * 80)
    print("SINGLETON PROOF (Entities with NO matches in ground truth):")
    print("-" * 80)
    singleton_sample = [s1_id for s1_id in eval_ids if len(gt_dict.get(s1_id, set())) == 0][:2]
    for idx, s1_id in enumerate(singleton_sample):
        cur.execute("SELECT business_name, business_address FROM source1 WHERE entity_id = ?", (s1_id,))
        s1_row = cur.fetchone()
        s1_name, s1_addr = s1_row if s1_row else ("Unknown", "Unknown")
        pred_matches = pred_dict.get(s1_id, [])
        status = "PASSED (Abstained - 0 matches)" if len(pred_matches) == 0 else "FAILED"
        print(f"\n[SINGLETON #{idx + 1}] {s1_id}")
        print(f"  Business:   {s1_name} || Address: {s1_addr}")
        print(f"  Ground Truth: NO MATCH (Singleton)")
        print(f"  Prediction:   {pred_matches if pred_matches else 'EMPTY (NO MATCH)'}")
        print(f"  Status:       {status}")

    print("\n" + "=" * 80)
    conn.close()

if __name__ == "__main__":
    main()
