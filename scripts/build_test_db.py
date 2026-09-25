import os
import sqlite3
import time
import re
import pandas as pd
from typing import List, Tuple
from src.normalize import (
    normalize_text,
    normalize_compact,
    remove_legal_suffixes,
    sort_tokens,
    extract_postal_code,
    extract_numbers
)
from src.disk_blocker import GENERIC_NAME_TOKENS

DEFAULT_TEST_DB = os.environ.get(
    "TEST_DB_PATH",
    "D:/hackathon/test_dataset.db" if os.path.exists("D:/hackathon") else "test_dataset.db"
)

def extract_addr_anchor(addr_clean: str) -> str:
    """Extracts street number + primary street word (e.g. '1064 newton')."""
    if not addr_clean:
        return ""
    m = re.search(r'\b(\d+)\s+([a-z]{3,})\b', addr_clean.lower())
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return ""

def extract_tok1(name_clean: str) -> str:
    """Extracts first distinctive non-generic word in business name."""
    if not name_clean:
        return ""
    words = re.findall(r'[a-z0-9]{4,}', name_clean.lower())
    for w in words:
        if w not in GENERIC_NAME_TOKENS and not w.isdigit():
            return w
    return words[0] if words else ""

def extract_addr_tok1(addr_clean: str) -> str:
    """Extracts first distinctive word in clean address."""
    if not addr_clean:
        return ""
    words = re.findall(r'[a-z]{4,}', addr_clean.lower())
    for w in words:
        if w not in {'road', 'street', 'lane', 'avenue', 'nagar', 'floor', 'block', 'plot'}:
            return w
    return words[0] if words else ""

def process_chunk(df: pd.DataFrame) -> List[tuple]:
    records = []
    for _, row in df.iterrows():
        eid = str(row["entity_id"]).strip()
        name = str(row["business_name"]).strip()
        addr = str(row["business_address"]).strip()
        country = str(row["country"]).strip()

        n_clean = normalize_text(name)
        n_core = remove_legal_suffixes(n_clean)
        n_compact = normalize_compact(name)
        n_core_compact = normalize_compact(n_core)
        n_sorted = sort_tokens(n_core)

        a_clean = normalize_text(addr)
        postal = extract_postal_code(addr, country)
        nums = ",".join(extract_numbers(addr))
        anc = extract_addr_anchor(a_clean)
        tok1 = extract_tok1(n_clean)
        addr_tok1 = extract_addr_tok1(a_clean)

        records.append((
            eid, name, addr, country,
            n_clean, n_compact, n_core_compact, n_sorted,
            a_clean, postal, nums, anc, tok1, addr_tok1
        ))
    return records

def build_test_db(
    s2_path: str = "test_data/test_source2 (1).tsv",
    s3_path: str = "test_data/test_source3.tsv",
    db_path: str = DEFAULT_TEST_DB,
    chunksize: int = 150000
):
    print("=" * 80)
    print(f"=== BUILDING TEST DATABASE: {db_path} ===")
    print("=" * 80)
    t_start = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -128000;") # 128MB cache

    for tbl in ["source2", "source3"]:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {tbl} (
                entity_id TEXT PRIMARY KEY,
                business_name TEXT,
                business_address TEXT,
                country TEXT,
                name_clean TEXT,
                name_compact TEXT,
                name_core_compact TEXT,
                name_sorted TEXT,
                address_clean TEXT,
                postal_code TEXT,
                numbers TEXT,
                addr_anchor TEXT,
                tok1 TEXT,
                addr_tok1 TEXT
            );
        """)
    conn.commit()

    for tsv_path, tbl in [(s2_path, "source2"), (s3_path, "source3")]:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        existing_count = cur.fetchone()[0]
        if existing_count > 0:
            print(f"Table '{tbl}' already contains {existing_count:,} rows. Skipping insertion.")
            continue

        print(f"\n--- Ingesting {tsv_path} into table '{tbl}' in chunks of {chunksize:,} ---")
        t0 = time.time()
        insert_sql = f"""
            INSERT OR IGNORE INTO {tbl} (
                entity_id, business_name, business_address, country,
                name_clean, name_compact, name_core_compact, name_sorted,
                address_clean, postal_code, numbers, addr_anchor, tok1, addr_tok1
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        total = 0
        for chunk_df in pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, chunksize=chunksize):
            records = process_chunk(chunk_df)
            conn.executemany(insert_sql, records)
            conn.commit()
            total += len(records)
            print(f"  Inserted {total:,} rows ({time.time() - t0:.1f}s)...")
        print(f"Done table '{tbl}': {total:,} records inserted in {time.time() - t0:.1f}s.")

    # Create Indexes
    print("\n--- Creating High-Speed B-Tree Indexes on Test Database ---")
    t_idx = time.time()
    for tbl in ["source2", "source3"]:
        print(f"Indexing {tbl}...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_compact ON {tbl}(country, name_compact);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_core_c ON {tbl}(country, name_core_compact);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_sorted ON {tbl}(country, name_sorted);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_anc ON {tbl}(country, addr_anchor);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_tok1 ON {tbl}(country, tok1);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_addrtok ON {tbl}(country, addr_tok1);")
        conn.commit()
    print(f"All B-Tree Indexes created in {time.time() - t_idx:.1f}s.")

    conn.close()
    print(f"\n=== TEST DATABASE READY: {db_path} in {time.time() - t_start:.1f}s ===")

if __name__ == "__main__":
    build_test_db()
