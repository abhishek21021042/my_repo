import os
import sqlite3
import time
from typing import Dict, List, Optional
import pandas as pd
from src.normalize import (
    normalize_text,
    normalize_compact,
    remove_legal_suffixes,
    sort_tokens,
    extract_postal_code,
    extract_numbers
)

DB_PATH = "artifacts/dataset.db"

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Enable WAL mode for high-speed concurrent reads and fast writing
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -64000;") # 64MB cache
    return conn

def init_tables(conn: sqlite3.Connection) -> None:
    for table_name in ["source1", "source2", "source3"]:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
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
                numbers TEXT
            );
        """)
    conn.commit()

def process_chunk(df: pd.DataFrame) -> List[tuple]:
    records = []
    for _, row in df.iterrows():
        eid = str(row["entity_id"]).strip()
        name = str(row["business_name"]).strip()
        addr = str(row["business_address"]).strip()
        country = str(row["country"]).strip()

        # Compute views
        n_clean = normalize_text(name)
        n_core = remove_legal_suffixes(n_clean)
        n_compact = normalize_compact(name)
        n_core_compact = normalize_compact(n_core)
        n_sorted = sort_tokens(n_core)
        
        a_clean = normalize_text(addr)
        postal = extract_postal_code(addr, country)
        nums = ",".join(extract_numbers(addr))

        records.append((
            eid, name, addr, country,
            n_clean, n_compact, n_core_compact, n_sorted,
            a_clean, postal, nums
        ))
    return records

def ingest_tsv_in_chunks(
    tsv_path: str,
    table_name: str,
    conn: sqlite3.Connection,
    chunksize: int = 150000
) -> None:
    print(f"\n--- Ingesting {tsv_path} into table '{table_name}' in chunks of {chunksize:,} ---")
    t0 = time.time()
    
    # Check if table already has rows
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cur.fetchone()[0]
    if count > 0:
        print(f"Table '{table_name}' already contains {count:,} records. Skipping ingestion.")
        return

    insert_sql = f"""
        INSERT OR IGNORE INTO {table_name} (
            entity_id, business_name, business_address, country,
            name_clean, name_compact, name_core_compact, name_sorted,
            address_clean, postal_code, numbers
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    total_inserted = 0
    for chunk_df in pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, chunksize=chunksize):
        records = process_chunk(chunk_df)
        conn.executemany(insert_sql, records)
        conn.commit()
        total_inserted += len(records)
        print(f"Inserted {total_inserted:,} rows ({time.time() - t0:.1f}s)...")

    print(f"Completed ingestion of {total_inserted:,} rows into '{table_name}' in {time.time() - t0:.1f} seconds.")

def create_indexes(conn: sqlite3.Connection) -> None:
    print("\n--- Building B-Tree Indexes on Disk for Instant Lookups ---")
    t0 = time.time()
    for table_name in ["source2", "source3"]:
        print(f"Indexing {table_name}...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_compact ON {table_name}(country, name_compact);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_core_c ON {table_name}(country, name_core_compact);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_sorted ON {table_name}(country, name_sorted);")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_postal ON {table_name}(country, postal_code);")
        conn.commit()
    print(f"All B-Tree indexes created in {time.time() - t0:.1f} seconds!")

if __name__ == "__main__":
    conn = get_connection()
    init_tables(conn)
    ingest_tsv_in_chunks("train_source1.tsv", "source1", conn)
    ingest_tsv_in_chunks("train_source2.tsv", "source2", conn)
    ingest_tsv_in_chunks("train_source3.tsv", "source3", conn)
    create_indexes(conn)
    conn.close()
    print("\n=== DATASET DATABASE READY WITH ZERO RAM OVERHEAD ===")
