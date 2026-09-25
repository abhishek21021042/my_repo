import re
import sqlite3
import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids

ADDRESS_STOPS = {
    "road", "street", "avenue", "drive", "lane", "near", "opposite", "behind",
    "floor", "building", "block", "sector", "nagar", "colony", "apartment",
    "suite", "unit", "plot", "town", "city", "post", "dist", "district", "state",
    "first", "second", "third", "main", "cross", "phase", "house", "door", "no"
}

RE_WORDS = re.compile(r"\b[a-zA-Z]{4,}\b")

def get_address_anchor(addr_clean: str, numbers_str: str) -> str:
    """
    Extracts a high-precision spatial anchor: primary_number + primary_street_word.
    e.g. '85 Wayne Ave, Ticonderoga' -> '85_wayne'
    e.g. '901 Howard St, Shelbyville' -> '901_howard'
    """
    if not numbers_str:
        return ""
    first_num = numbers_str.split(",")[0].strip()
    if not first_num:
        return ""
        
    words = RE_WORDS.findall(addr_clean)
    street_word = ""
    for w in words:
        w_lower = w.lower()
        if w_lower not in ADDRESS_STOPS:
            street_word = w_lower
            break
            
    if not street_word:
        return ""
    return f"{first_num}_{street_word}"

def test_anchor():
    conn = sqlite3.connect("D:/hackathon/dataset.db")
    cur = conn.cursor()

    rapid_s1 = read_tsv("artifacts/splits/rapid_eval_s1.tsv")
    sample_ids = list(rapid_s1["source1_entity_id"].iloc[:500])

    gt_df = read_tsv("train_ground_truth.tsv")
    eval_gt_df = gt_df[gt_df["source1_entity_id"].isin(sample_ids)]
    gt_dict = {
        row["source1_entity_id"]: set(parse_matched_ids(row["matched_entity_ids"]))
        for _, row in eval_gt_df.iterrows()
    }

    placeholders = ",".join(["?"] * len(sample_ids))
    cur.execute(f"SELECT entity_id, address_clean, numbers FROM source1 WHERE entity_id IN ({placeholders})", sample_ids)
    rows = cur.fetchall()

    print("=== TESTING ADDRESS ANCHOR EXTRACTION ===")
    anchors = {}
    for r in rows:
        eid, addr, nums = r
        anc = get_address_anchor(addr, nums)
        if anc:
            anchors[eid] = anc

    print(f"Entities with strong address anchor: {len(anchors)}/{len(sample_ids)} ({len(anchors)/len(sample_ids)*100:.1f}%)")
    sample_items = list(anchors.items())[:10]
    for eid, anc in sample_items:
        print(f"  {eid} -> {anc}")

    conn.close()

if __name__ == "__main__":
    test_anchor()
