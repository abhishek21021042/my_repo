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

def get_address_anchor(addr_clean: str, numbers_str: str) -> str:
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

def add_anchors():
    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -128000;")

    cur = conn.cursor()

    for table_name in ["source1", "source2", "source3"]:
        print(f"\n--- Adding addr_anchor to {table_name} ---")
        t0 = time.time()
        
        # Check if column exists
        cur.execute(f"PRAGMA table_info({table_name})")
        cols = [r[1] for r in cur.fetchall()]
        if "addr_anchor" not in cols:
            print(f"Adding column addr_anchor to {table_name}...")
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN addr_anchor TEXT;")
            conn.commit()

        # Check if already populated
        cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE addr_anchor IS NOT NULL")
        already = cur.fetchone()[0]
        if already > 0:
            print(f"{table_name} already has {already:,} anchors. Skipping update.")
            continue

        # Fetch in chunks and update
        cur.execute(f"SELECT entity_id, address_clean, numbers FROM {table_name}")
        batch = []
        count = 0
        update_sql = f"UPDATE {table_name} SET addr_anchor = ? WHERE entity_id = ?"
        
        while True:
            rows = cur.fetchmany(100000)
            if not rows:
                break
            updates = []
            for r in rows:
                eid, addr, nums = r
                anc = get_address_anchor(addr, nums)
                updates.append((anc, eid))
            
            conn.executemany(update_sql, updates)
            conn.commit()
            count += len(updates)
            print(f"Updated {count:,} rows in {table_name} ({time.time() - t0:.1f}s)...")

        print(f"Completed {table_name} in {time.time() - t0:.1f} seconds.")

    print("\n--- Creating Index on addr_anchor ---")
    for tbl in ["source2", "source3"]:
        t0 = time.time()
        print(f"Creating index idx_{tbl}_anchor...")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{tbl}_anchor ON {tbl}(country, addr_anchor);")
        conn.commit()
        print(f"Done in {time.time() - t0:.1f}s")

    conn.close()
    print("\n=== ADDRESS ANCHOR INDEXING COMPLETE ===")

if __name__ == "__main__":
    add_anchors()
