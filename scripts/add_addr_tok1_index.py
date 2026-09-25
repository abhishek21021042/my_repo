import re
import sqlite3
import time

DB_PATH = "D:/hackathon/dataset.db"

ADDRESS_STOPS = {
    "road", "street", "avenue", "drive", "lane", "near", "opposite", "behind",
    "floor", "building", "block", "sector", "nagar", "colony", "apartment",
    "suite", "unit", "plot", "town", "city", "post", "dist", "district", "state",
    "first", "second", "third", "main", "cross", "phase", "house", "door", "no"
}

RE_WORDS = re.compile(r"\b[a-zA-Z]{4,}\b")

def extract_addr_tok1(addr_clean: str) -> str:
    if not addr_clean:
        return ""
    words = RE_WORDS.findall(addr_clean)
    for w in words:
        wl = w.lower()
        if wl not in ADDRESS_STOPS:
            return wl
    return ""

def add_addr_tok1():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -128000;")
    cur = conn.cursor()

    for table_name in ["source1", "source2", "source3"]:
        t0 = time.time()
        print(f"\n--- Adding addr_tok1 to {table_name} ---")
        
        cur.execute(f"PRAGMA table_info({table_name})")
        cols = [r[1] for r in cur.fetchall()]
        if "addr_tok1" not in cols:
            print(f"Adding column addr_tok1 to {table_name}...")
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN addr_tok1 TEXT;")
            conn.commit()

        # Check if already populated
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE addr_tok1 IS NOT NULL")
        already = cur.fetchone()[0]
        if already > 0:
            print(f"{table_name} already has {already:,} addr_tok1 values. Skipping.")
            continue

        cur.execute(f"SELECT entity_id, address_clean FROM {table_name}")
        count = 0
        update_sql = f"UPDATE {table_name} SET addr_tok1 = ? WHERE entity_id = ?"
        
        while True:
            rows = cur.fetchmany(100000)
            if not rows:
                break
            updates = [(extract_addr_tok1(r[1]), r[0]) for r in rows]
            conn.executemany(update_sql, updates)
            conn.commit()
            count += len(updates)
            print(f"Updated {count:,} rows in {table_name} ({time.time() - t0:.1f}s)...")

        print(f"Completed {table_name} in {time.time() - t0:.1f} seconds.")

    print("\n--- Creating Index on addr_tok1 ---")
    for tbl in ["source2", "source3"]:
        t0 = time.time()
        print(f"Creating index idx_{tbl}_addrtok1...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_addrtok1 ON {tbl}(country, addr_tok1);")
        conn.commit()
        print(f"Done in {time.time() - t0:.1f}s")

    conn.close()
    print("\n=== ADDR_TOK1 INDEXING COMPLETE ===")

if __name__ == "__main__":
    add_addr_tok1()
