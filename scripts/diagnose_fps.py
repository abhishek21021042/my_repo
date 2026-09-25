import sqlite3
import pandas as pd
import numpy as np
from src.io_utils import read_tsv, parse_matched_ids
from src.disk_blocker import DiskBTreeBlocker, DEFAULT_DB_PATH
from src.features import compute_pairwise_features
from src.train_matcher import EntityMatcher, FEATURE_COLUMNS
from src.decide import apply_precision_first_policy

def main():
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    cur = conn.cursor()

    rapid_s1 = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
    val_india = list(rapid_s1[rapid_s1["country"] == "India"]["source1_entity_id"].iloc[:500])
    val_us = list(rapid_s1[rapid_s1["country"] == "US"]["source1_entity_id"].iloc[:500])
    val_sample_ids = val_india + val_us

    placeholders = ",".join(["?"] * len(val_sample_ids))
    cur.execute(f"""
        SELECT entity_id, business_name, business_address, country,
               name_clean, name_compact, name_core_compact, name_sorted,
               address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
        FROM source1 WHERE entity_id IN ({placeholders})
    """, val_sample_ids)
    val_s1_raw = cur.fetchall()

    val_s1_dict = {r[0]: {
        "entity_id": r[0], "business_name": r[1], "business_address": r[2], "country": r[3],
        "name_clean": r[4], "name_compact": r[5], "name_core_compact": r[6], "name_no_legal": r[6],
        "name_sorted": r[7], "address_clean": r[8], "postal_code": r[9],
        "numbers": r[10].split(",") if r[10] else [],
        "addr_anchor": r[11] if len(r) > 11 and r[11] else "",
        "tok1": r[12] if len(r) > 12 and r[12] else "",
        "addr_tok1": r[13] if len(r) > 13 and r[13] else ""
    } for r in val_s1_raw}

    gt_df = read_tsv("train_ground_truth.tsv")
    val_gt_df = gt_df[gt_df["source1_entity_id"].isin(set(val_sample_ids))]
    val_gt_dict = {row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"])) for _, row in val_gt_df.iterrows()}

    blocker = DiskBTreeBlocker(DEFAULT_DB_PATH, max_cands_per_query=80)
    val_cands = blocker.retrieve_candidates(list(val_s1_dict.values()))

    val_feats = []
    cand_map = {}
    for c in val_cands:
        s1_id = c["source1_entity_id"]
        cand_id = c["candidate_entity_id"]
        f_vec = compute_pairwise_features(val_s1_dict[s1_id], c["cand_record"], c)
        f_vec["source1_entity_id"] = s1_id
        f_vec["candidate_entity_id"] = cand_id
        val_feats.append(f_vec)
        cand_map[(s1_id, cand_id)] = c

    val_feat_df = pd.DataFrame(val_feats)
    matcher = EntityMatcher.load("artifacts/models/matcher.joblib")
    val_feat_df["probability"] = matcher.predict_proba(val_feat_df[FEATURE_COLUMNS].values)

    predictions = apply_precision_first_policy(val_feat_df, threshold=0.90, min_margin=0.10, enable_dual_anchor=True)

    fp_list = []
    for s1_id, p_list in predictions.items():
        true_set = val_gt_dict.get(s1_id, set())
        for m in p_list:
            if m not in true_set:
                row = val_feat_df[(val_feat_df["source1_entity_id"] == s1_id) & (val_feat_df["candidate_entity_id"] == m)].iloc[0]
                cand_c = cand_map[(s1_id, m)]
                fp_list.append({
                    "s1_id": s1_id, "cand_id": m,
                    "s1_name": val_s1_dict[s1_id]["business_name"],
                    "cand_name": cand_c["cand_record"]["business_name"],
                    "s1_addr": val_s1_dict[s1_id]["business_address"],
                    "cand_addr": cand_c["cand_record"]["business_address"],
                    "prob": row["probability"],
                    "exact_compact": row["exact_compact"],
                    "exact_core": row["exact_core"],
                    "core_jw": row["core_jw"],
                    "name_jaccard": row["name_jaccard"],
                    "core_jaccard": row["core_jaccard"],
                    "name_containment": row["name_containment"],
                    "addr_jaccard": row["addr_jaccard"],
                    "addr_jw": row["addr_jw"],
                    "postal_match": row["postal_match"],
                    "nums_conflict": row["nums_conflict"]
                })

    print(f"Total False Positives: {len(fp_list)}")
    for idx, fp in enumerate(fp_list[:15]):
        print(f"--- FP #{idx+1} ---")
        print(f"S1 Name: {fp['s1_name']} || Cand Name: {fp['cand_name']}")
        print(f"S1 Addr: {fp['s1_addr']} || Cand Addr: {fp['cand_addr']}")
        print(f"Prob: {fp['prob']:.4f} | core_jw: {fp['core_jw']:.3f} | name_jaccard: {fp['name_jaccard']:.3f} | core_jaccard: {fp['core_jaccard']:.3f} | addr_jaccard: {fp['addr_jaccard']:.3f}")
        print()

    blocker.close()
    conn.close()

if __name__ == "__main__":
    main()
