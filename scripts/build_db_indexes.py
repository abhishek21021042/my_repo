import os
import sqlite3
import time

DB_PATH = "D:/hackathon/dataset.db"

def build_indexes():
    print(f"Connecting to database at {DB_PATH} (on D: drive with 188GB free space)...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -128000;") # 128MB cache

    t0 = time.time()
    for table_name in ["source2", "source3"]:
        print(f"\n--- Creating B-Tree Indexes for {table_name} ---")
        
        idx_t0 = time.time()
        print(f"1. Indexing ({table_name}: country, name_compact)...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_compact ON {table_name}(country, name_compact);")
        conn.commit()
        print(f"   Done in {time.time() - idx_t0:.1f}s")

        idx_t0 = time.time()
        print(f"2. Indexing ({table_name}: country, name_core_compact)...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_core_c ON {table_name}(country, name_core_compact);")
        conn.commit()
        print(f"   Done in {time.time() - idx_t0:.1f}s")

        idx_t0 = time.time()
        print(f"3. Indexing ({table_name}: country, name_sorted)...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_sorted ON {table_name}(country, name_sorted);")
        conn.commit()
        print(f"   Done in {time.time() - idx_t0:.1f}s")

        idx_t0 = time.time()
        print(f"4. Indexing ({table_name}: country, postal_code)...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_postal ON {table_name}(country, postal_code);")
        conn.commit()
        print(f"   Done in {time.time() - idx_t0:.1f}s")

    conn.close()
    print(f"\n=== ALL B-TREE INDEXES CREATED SUCCESSFULLY IN {time.time() - t0:.1f} SECONDS ===")

if __name__ == "__main__":
    build_indexes()
