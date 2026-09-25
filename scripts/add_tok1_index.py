import sqlite3
import time

DB_PATH = "D:/hackathon/dataset.db"

NAME_STOPS = {
    "the", "and", "for", "with", "inc", "ltd", "pvt", "llc", "corp", "company", "co",
    "services", "enterprises", "group", "holdings", "private", "limited", "incorporated"
}

def extract_tok1(name_clean: str) -> str:
    if not name_clean:
        return ""
    words = [w for w in name_clean.split() if len(w) >= 3 and w not in NAME_STOPS]
    return words[0] if words else ""

def add_tok1():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -128000;")
    cur = conn.cursor()

    for table_name in ["source1", "source2", "source3"]:
        t0 = time.time()
        print(f"\n--- Adding tok1 to {table_name} ---")
        
        cur.execute(f"PRAGMA table_info({table_name})")
        cols = [r[1] for r in cur.fetchall()]
        if "tok1" not in cols:
            print(f"Adding column tok1 to {table_name}...")
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN tok1 TEXT;")
            conn.commit()

        # Check if already populated
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE tok1 IS NOT NULL")
        already = cur.fetchone()[0]
        if already > 0:
            print(f"{table_name} already has {already:,} tok1 values. Skipping.")
            continue

        cur.execute(f"SELECT entity_id, name_clean FROM {table_name}")
        count = 0
        update_sql = f"UPDATE {table_name} SET tok1 = ? WHERE entity_id = ?"
        
        while True:
            rows = cur.fetchmany(100000)
            if not rows:
                break
            updates = [(extract_tok1(r[1]), r[0]) for r in rows]
            conn.executemany(update_sql, updates)
            conn.commit()
            count += len(updates)
            print(f"Updated {count:,} rows in {table_name} ({time.time() - t0:.1f}s)...")

        print(f"Completed {table_name} in {time.time() - t0:.1f} seconds.")

    print("\n--- Creating Index on tok1 ---")
    for tbl in ["source2", "source3"]:
        t0 = time.time()
        print(f"Creating index idx_{tbl}_tok1...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_tok1 ON {tbl}(country, tok1);")
        conn.commit()
        print(f"Done in {time.time() - t0:.1f}s")

    conn.close()
    print("\n=== TOK1 INDEXING COMPLETE ===")

if __name__ == "__main__":
    add_tok1()
