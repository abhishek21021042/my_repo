import os
import argparse
import time
import sqlite3
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

from src.io_utils import read_tsv
from src.normalize import (
    normalize_text, normalize_compact, remove_legal_suffixes,
    sort_tokens, extract_postal_code, extract_numbers
)
from src.disk_blocker import DiskBTreeBlocker, DEFAULT_DB_PATH
from src.features import compute_pairwise_features
from src.train_matcher import EntityMatcher, FEATURE_COLUMNS
from src.decide import apply_precision_first_policy
from src.write_outputs import write_submission_outputs

def run_inference(
    s1_path: str = "train_source1.tsv",
    db_path: str = DEFAULT_DB_PATH,
    model_path: str = "artifacts/models/matcher.joblib",
    output_dir: str = "output",
    threshold: float = 0.95,
    min_margin: float = 0.10,
    limit: int = None
):
    print("=========================================================================")
    print("=== HIGH-PRECISION ENTITY RESOLUTION INFERENCE PIPELINE ===")
    print(f"=== Model: {model_path} | Threshold: {threshold} ===")
    print("=========================================================================")
    t_start = time.time()

    # 1. Load trained matcher model
    print(f"Loading trained matcher from {model_path}...")
    matcher = EntityMatcher.load(model_path)

    # 2. Connect to database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 3. Load Source 1 query entities
    print(f"Loading Source 1 entities from {s1_path}...")
    s1_df = read_tsv(s1_path)
    if limit is not None and limit > 0:
        s1_df = s1_df.iloc[:limit]
    print(f"Processing {len(s1_df):,} Source 1 entities.")

    id_col = "source1_entity_id" if "source1_entity_id" in s1_df.columns else "entity_id"
    s1_ids = list(s1_df[id_col])
    
    # Check if normalized records can be fetched from DB
    s1_dict = {}
    s1_list = []
    
    # Query database in batches to retrieve normalized representations
    batch_size = 5000
    for i in range(0, len(s1_ids), batch_size):
        batch_ids = s1_ids[i:i + batch_size]
        placeholders = ",".join(["?"] * len(batch_ids))
        cur.execute(f"""
            SELECT entity_id, business_name, business_address, country,
                   name_clean, name_compact, name_core_compact, name_sorted,
                   address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
            FROM source1 
            WHERE entity_id IN ({placeholders})
        """, batch_ids)
        rows = cur.fetchall()
        for r in rows:
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
            s1_dict[r[0]] = item
            s1_list.append(item)

    # For any entities not yet in DB, compute normalized views dynamically
    missing_ids = [eid for eid in s1_ids if eid not in s1_dict]
    if missing_ids:
        print(f"Normalizing {len(missing_ids):,} entities on-the-fly...")
        missing_df = s1_df[s1_df[id_col].isin(set(missing_ids))]
        for _, row in missing_df.iterrows():
            eid = row[id_col]
            name = str(row["business_name"]) if pd.notna(row["business_name"]) else ""
            addr = str(row["business_address"]) if pd.notna(row["business_address"]) else ""
            country = str(row["country"]) if pd.notna(row["country"]) else ""
            
            n_clean = normalize_text(name)
            n_no_leg = remove_legal_suffixes(n_clean)
            n_comp = normalize_compact(name)
            n_core_comp = normalize_compact(n_no_leg)
            n_sort = sort_tokens(n_no_leg)
            a_clean = normalize_text(addr)
            postal = extract_postal_code(addr, country)
            nums = extract_numbers(addr)
            
            # Extract anchor
            first_num = nums[0] if nums else ""
            addr_words = [w for w in a_clean.split() if w.isalpha() and len(w) >= 3]
            street_word = addr_words[0] if addr_words else ""
            anc = f"{first_num}_{street_word}" if (first_num and street_word) else ""
            
            name_words = [w for w in n_no_leg.split() if w.isalpha() and len(w) >= 3]
            tok1 = name_words[0] if name_words else ""
            addr_tok1 = street_word

            item = {
                "entity_id": eid,
                "business_name": name,
                "business_address": addr,
                "country": country,
                "name_clean": n_clean,
                "name_compact": n_comp,
                "name_core_compact": n_core_comp,
                "name_no_legal": n_core_comp,
                "name_sorted": n_sort,
                "address_clean": a_clean,
                "postal_code": postal,
                "numbers": nums,
                "addr_anchor": anc,
                "tok1": tok1,
                "addr_tok1": addr_tok1
            }
            s1_dict[eid] = item
            s1_list.append(item)

    print(f"All {len(s1_list):,} Source 1 records prepared.")

    # 4. Multi-channel candidate retrieval
    print("\n--- STAGE 1: Multi-Channel Candidate Retrieval ---")
    t0 = time.time()
    blocker = DiskBTreeBlocker(db_path, max_cands_per_query=80)
    candidates = blocker.retrieve_candidates(s1_list)
    print(f"Retrieved {len(candidates):,} candidate pairs in {time.time() - t0:.2f}s.")

    # Group candidate IDs per S1
    candidates_dict = {eid: [] for eid in s1_ids}
    for c in candidates:
        s1_id = c["source1_entity_id"]
        candidates_dict[s1_id].append(c["candidate_entity_id"])

    # 5. Feature Engineering
    print("\n--- STAGE 2: Pairwise Feature Computation ---")
    t0 = time.time()
    feat_rows = []
    for c in candidates:
        s1_id = c["source1_entity_id"]
        f_vec = compute_pairwise_features(s1_dict[s1_id], c["cand_record"], c)
        f_vec["source1_entity_id"] = s1_id
        f_vec["candidate_entity_id"] = c["candidate_entity_id"]
        feat_rows.append(f_vec)

    feat_df = pd.DataFrame(feat_rows)
    X = feat_df[FEATURE_COLUMNS].values if len(feat_df) > 0 else np.empty((0, len(FEATURE_COLUMNS)))
    print(f"Computed features for {len(feat_df):,} pairs in {time.time() - t0:.2f}s.")

    # 6. Scoring & Decision Policy
    print("\n--- STAGE 3: Scoring & Precision-First Decision Policy ---")
    t0 = time.time()
    if len(feat_df) > 0:
        feat_df["probability"] = matcher.predict_proba(X)
        predictions = apply_precision_first_policy(
            feat_df,
            threshold=threshold,
            min_margin=min_margin,
            enable_dual_anchor=True
        )
    else:
        predictions = {eid: [] for eid in s1_ids}
    print(f"Decisions made in {time.time() - t0:.2f}s.")

    # 7. Write Official Outputs & Validate PRD 20.3 Assertions
    print("\n--- STAGE 4: Generating Submission Files ---")
    outputs = write_submission_outputs(
        s1_entity_ids=s1_ids,
        predictions=predictions,
        candidates_dict=candidates_dict,
        output_dir=output_dir,
        validate=True
    )

    total_matches = sum(len(m) for m in predictions.values())
    singletons = sum(1 for m in predictions.values() if len(m) == 0)
    print("\n================ INFERENCE SUMMARY ================")
    print(f"Total Source 1 entities: {len(s1_ids):,}")
    print(f"Total predicted matches: {total_matches:,}")
    print(f"Total singletons:        {singletons:,} ({singletons/len(s1_ids)*100:.2f}%)")
    print(f"Total elapsed time:      {time.time() - t_start:.1f}s")
    print("===================================================")

    blocker.close()
    conn.close()
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Precision Entity Resolution Inference")
    parser.add_argument("--s1_path", type=str, default="artifacts/splits/rapid_eval_s1.tsv", help="Path to Source 1 TSV file")
    parser.add_argument("--db_path", type=str, default=DEFAULT_DB_PATH, help="Path to dataset.db")
    parser.add_argument("--model_path", type=str, default="artifacts/models/matcher.joblib", help="Path to trained matcher")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.95, help="Probability acceptance threshold")
    parser.add_argument("--min_margin", type=float, default=0.10, help="Minimum margin for ambiguous candidates")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of entities (None for full set)")

    args = parser.parse_args()
    run_inference(
        s1_path=args.s1_path,
        db_path=args.db_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_margin=args.min_margin,
        limit=args.limit
    )
