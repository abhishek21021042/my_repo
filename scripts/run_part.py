import os
import sys
import time
import argparse
import sqlite3
import pandas as pd
from typing import Dict, List, Set

sys.path.insert(0, os.path.abspath("."))
from src.io_utils import read_tsv
from src.normalize import (
    normalize_text, normalize_compact, remove_legal_suffixes,
    sort_tokens, extract_postal_code, extract_numbers
)
from src.disk_blocker import DiskBTreeBlocker, strip_accents, to_leet, clean_extra_legal, GENERIC_NAME_TOKENS
from src.features import compute_pairwise_features
from src.train_matcher import EntityMatcher, FEATURE_COLUMNS
from src.decide import apply_precision_first_policy

def get_default_db() -> str:
    candidates = [
        "D:/hackathon/test_dataset.db",
        "test_dataset.db",
        "D:/hackathon/dataset.db",
        "dataset.db"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "test_dataset.db"

def extract_addr_anchor(addr_clean: str) -> str:
    import re
    if not addr_clean: return ""
    m = re.search(r'\b(\d+)\s+([a-z]{3,})\b', addr_clean.lower())
    return f"{m.group(1)} {m.group(2)}" if m else ""

def extract_tok1(name_clean: str) -> str:
    import re
    if not name_clean: return ""
    words = re.findall(r'[a-z0-9]{4,}', name_clean.lower())
    for w in words:
        if w not in GENERIC_NAME_TOKENS and not w.isdigit():
            return w
    return words[0] if words else ""

def extract_addr_tok1(addr_clean: str) -> str:
    import re
    if not addr_clean: return ""
    words = re.findall(r'[a-z]{4,}', addr_clean.lower())
    for w in words:
        if w not in {'road', 'street', 'lane', 'avenue', 'nagar', 'floor', 'block', 'plot'}:
            return w
    return words[0] if words else ""

def run_part(
    input_file: str,
    db_path: str,
    model_path: str,
    part_num: int,
    total_parts: int,
    start_row: int = None,
    end_row: int = None,
    batch_size: int = 1000,
    output_dir: str = "output"
):
    print("=" * 80)
    print("=== DISTRIBUTED ENTITY RESOLUTION RUNNER ===")
    print("=" * 80)
    print(f"Input File:   {input_file}")
    print(f"Database:     {db_path}")
    print(f"Model File:   {model_path}")
    print(f"Partition:    Part {part_num} of {total_parts}")

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found!")
        print("Tip: Run 'python scripts/build_test_db.py' or check DB path.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)

    # 1. Calculate row boundaries
    print("\nReading input file metadata...")
    total_file_rows = sum(1 for _ in open(input_file, "r", encoding="utf-8")) - 1 # exclude header
    print(f"Total rows in {input_file}: {total_file_rows:,}")

    if start_row is None or end_row is None:
        chunk_size = (total_file_rows + total_parts - 1) // total_parts
        start_row = (part_num - 1) * chunk_size
        end_row = min(start_row + chunk_size, total_file_rows)

    part_total = end_row - start_row
    print(f"This Laptop will process Rows: {start_row:,} to {end_row:,} ({part_total:,} records).")

    # 2. Output file setup (exact Ground Truth format)
    os.makedirs(output_dir, exist_ok=True)
    out_tsv = os.path.join(output_dir, f"matching_results_part_{part_num}_of_{total_parts}.tsv")
    cand_tsv = os.path.join(output_dir, f"candidate_pairs_part_{part_num}_of_{total_parts}.tsv")

    # Check for existing progress (Auto-resume feature)
    processed_eids = set()
    if os.path.exists(out_tsv):
        with open(out_tsv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip().split("\t")
                    if parts and parts[0]:
                        processed_eids.add(parts[0])
        print(f"Resuming: Found {len(processed_eids):,} previously processed records in {out_tsv}.")
    else:
        # Write header
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        with open(cand_tsv, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_id\n")

    # 3. Load matcher & blocker
    print("\nInitializing Matcher & Blocker...")
    matcher = EntityMatcher.load(model_path)
    blocker = DiskBTreeBlocker(db_path=db_path, max_cands_per_query=150)

    # 4. Stream and process the allocated chunk in batches
    print(f"\nStarting Inference on Part {part_num}/{total_parts}...")
    t_start = time.time()
    batch = []
    current_row_idx = 0
    total_processed_now = len(processed_eids)
    total_matches_found = 0

    with open(input_file, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        eid_idx = header.index("entity_id") if "entity_id" in header else 0
        name_idx = header.index("business_name") if "business_name" in header else 1
        addr_idx = header.index("business_address") if "business_address" in header else 2
        cntry_idx = header.index("country") if "country" in header else 3

        for line in f:
            if current_row_idx < start_row:
                current_row_idx += 1
                continue
            if current_row_idx >= end_row:
                break

            current_row_idx += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) <= eid_idx: continue

            eid = fields[eid_idx].strip()
            if eid in processed_eids:
                continue

            name = fields[name_idx].strip() if len(fields) > name_idx else ""
            addr = fields[addr_idx].strip() if len(fields) > addr_idx else ""
            country = fields[cntry_idx].strip() if len(fields) > cntry_idx else ""

            # Prepare normalized record
            n_clean = normalize_text(name)
            n_core = remove_legal_suffixes(n_clean)
            n_compact = normalize_compact(name)
            n_core_compact = normalize_compact(n_core)
            n_sorted = sort_tokens(n_core)
            a_clean = normalize_text(addr)
            postal = extract_postal_code(addr, country)
            nums = extract_numbers(addr)

            rec = {
                "entity_id": eid,
                "business_name": name,
                "business_address": addr,
                "country": country,
                "name_clean": n_clean,
                "name_compact": n_compact,
                "name_core_compact": n_core_compact,
                "name_no_legal": n_core,
                "name_sorted": n_sorted,
                "address_clean": a_clean,
                "postal_code": postal,
                "numbers": nums,
                "addr_anchor": extract_addr_anchor(a_clean),
                "tok1": extract_tok1(n_clean),
                "addr_tok1": extract_addr_tok1(a_clean)
            }
            batch.append(rec)

            if len(batch) >= batch_size:
                matches_in_batch = process_batch(batch, blocker, matcher, out_tsv, cand_tsv)
                total_matches_found += matches_in_batch
                total_processed_now += len(batch)
                elapsed = time.time() - t_start
                rate = total_processed_now / elapsed if elapsed > 0 else 0
                pct = (total_processed_now / part_total) * 100
                rem_sec = (part_total - total_processed_now) / rate if rate > 0 else 0
                print(f"[{pct:5.1f}%] Processed {total_processed_now:,}/{part_total:,} | Matches: {total_matches_found:,} | Speed: {rate:.1f} rec/s | ETA: {rem_sec/60:.1f}m")
                batch = []

    # Final remaining batch
    if batch:
        matches_in_batch = process_batch(batch, blocker, matcher, out_tsv, cand_tsv)
        total_matches_found += matches_in_batch
        total_processed_now += len(batch)

    blocker.close()
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"=== PART {part_num}/{total_parts} COMPLETED IN {elapsed_total/60:.1f} MINUTES ===")
    print(f"  Records Processed: {total_processed_now:,} / {part_total:,}")
    print(f"  Matches Found:     {total_matches_found:,}")
    print(f"  Results Saved To:  {out_tsv}")
    print(f"  Candidates Saved:  {cand_tsv}")
    print("=" * 80)

def process_batch(
    batch: List[Dict],
    blocker: DiskBTreeBlocker,
    matcher: EntityMatcher,
    out_tsv_path: str,
    cand_tsv_path: str
) -> int:
    # 1. Retrieve candidates
    candidates = blocker.retrieve_candidates(batch)
    s1_dict = {r["entity_id"]: r for r in batch}

    predictions = {r["entity_id"]: [] for r in batch}
    cand_pairs_to_write = []

    if candidates:
        feat_rows = []
        for c in candidates:
            s1_id = c["source1_entity_id"]
            cid = c["candidate_entity_id"]
            cand_pairs_to_write.append(f"{s1_id}\t{cid}\n")
            f_vec = compute_pairwise_features(s1_dict[s1_id], c["cand_record"], c)
            f_vec["source1_entity_id"] = s1_id
            f_vec["candidate_entity_id"] = cid
            feat_rows.append(f_vec)

        if feat_rows:
            feat_df = pd.DataFrame(feat_rows)
            feat_df["probability"] = matcher.predict_proba(feat_df[FEATURE_COLUMNS].values)
            preds_dict = apply_precision_first_policy(
                feat_df,
                threshold=0.93,
                strong_addr_threshold=0.89,
                min_margin=0.05,
                enable_dual_anchor=True
            )
            for eid, plist in preds_dict.items():
                predictions[eid] = plist

    # Append to output TSV in exact Ground Truth format
    total_matches = 0
    with open(out_tsv_path, "a", encoding="utf-8") as f_out:
        for r in batch:
            eid = r["entity_id"]
            matched_list = predictions.get(eid, [])
            matched_str = ",".join(matched_list) if matched_list else ""
            if matched_list:
                total_matches += len(matched_list)
            f_out.write(f"{eid}\t{matched_str}\n")

    # Append candidates
    if cand_pairs_to_write:
        with open(cand_tsv_path, "a", encoding="utf-8") as f_cand:
            f_cand.writelines(cand_pairs_to_write)

    return total_matches

def interactive_cli():
    print("=" * 80)
    print("=== AMAZON ML HACKATHON: DISTRIBUTED INFERENCE INTERFACE ===")
    print("=" * 80)
    
    default_input = "test_data/test_source1 (1).tsv" if os.path.exists("test_data/test_source1 (1).tsv") else "train_source1.tsv"
    default_db = get_default_db()

    print(f"\n1. Input Source 1 TSV File:")
    print(f"   [Default: {default_input}]")
    user_inp = input("   Enter path (or Press Enter to use default): ").strip()
    input_file = user_inp if user_inp else default_input

    print(f"\n2. Database Path (.db):")
    print(f"   [Default: {default_db}]")
    user_db = input("   Enter DB path (or Press Enter to use default): ").strip()
    db_path = user_db if user_db else default_db

    print("\n3. How many Parts / Laptops to divide into?")
    print("   [1] 2 Parts  (~866k records each) -> For 2 Laptops")
    print("   [2] 3 Parts  (~577k records each) -> For 3 Laptops")
    print("   [3] 4 Parts  (~433k records each) -> [RECOMMENDED]")
    print("   [4] 5 Parts  (~346k records each) -> For 5 Laptops")
    print("   [5] Custom Number of Parts")
    p_choice = input("   Choose option (1-5, Default: 3): ").strip()

    if p_choice == "1":
        total_parts = 2
    elif p_choice == "2":
        total_parts = 3
    elif p_choice == "4":
        total_parts = 5
    elif p_choice == "5":
        total_parts = int(input("   Enter total number of parts (e.g. 8): ").strip())
    else:
        total_parts = 4

    print(f"\n4. Which Part should THIS laptop run? (1 to {total_parts}):")
    part_num = int(input(f"   Enter Part Number (1-{total_parts}): ").strip())

    run_part(
        input_file=input_file,
        db_path=db_path,
        model_path="artifacts/models/matcher.joblib",
        part_num=part_num,
        total_parts=total_parts,
        batch_size=1000
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Part-by-Part Entity Resolution")
    parser.add_argument("--part", type=int, default=None, help="Part number to run on this machine (e.g. 1)")
    parser.add_argument("--total_parts", type=int, default=4, help="Total parts to divide dataset into")
    parser.add_argument("--input_file", type=str, default="test_data/test_source1 (1).tsv", help="Path to Source 1 TSV")
    parser.add_argument("--db_path", type=str, default=None, help="Path to indexed SQLite database")
    parser.add_argument("--model_path", type=str, default="artifacts/models/matcher.joblib", help="Path to trained matcher")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")

    args = parser.parse_args()

    if args.part is None:
        interactive_cli()
    else:
        chosen_db = args.db_path if args.db_path else get_default_db()
        run_part(
            input_file=args.input_file,
            db_path=chosen_db,
            model_path=args.model_path,
            part_num=args.part,
            total_parts=args.total_parts,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
