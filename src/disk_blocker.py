import sqlite3
import unicodedata
import re
from typing import List, Dict, Set, Tuple, Optional
import time

DEFAULT_DB_PATH = "D:/hackathon/dataset.db"

LEET_MAP = str.maketrans({'0': 'o', '1': 'l', '3': 'e', '5': 's', '4': 'a', '@': 'a', '$': 's'})

def strip_accents(s: str) -> str:
    """Strips combining diacritical marks (e.g., Límited -> Limited, Gróup -> Group)."""
    if not s:
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

def to_leet(s: str) -> str:
    """Converts synthetic leetspeak substitutions back to standard letters."""
    if not s:
        return ""
    return str(s).translate(LEET_MAP)

def clean_extra_legal(s: str) -> str:
    """Removes common US professional and corporate suffixes often missed by basic normalizers."""
    if not s:
        return ""
    s_clean = re.sub(r'\b(pc|p\.c\.|pllc|p\.l\.l\.c\.|corp|llc|inc|ltd)\b', '', s, flags=re.I)
    return re.sub(r'[^a-zA-Z0-9]', '', s_clean)

GENERIC_NAME_TOKENS = {
    'enterprises', 'company', 'consulting', 'solutions', 'services', 
    'industries', 'international', 'management', 'development', 'technologies',
    'associates', 'trading', 'group', 'holdings', 'agency', 'global',
    'corporation', 'incorporated', 'limited', 'private', 'system', 'systems'
}

class DiskBTreeBlocker:
    """
    Sub-millisecond multi-channel candidate retrieval backed by SQLite B-Tree indexes:
    - Channel 1: Exact compact name (regular, accent-stripped, leetspeak-decoded)
    - Channel 2: Exact core compact name (legal suffix stripped, accent-stripped, leet-decoded)
    - Channel 3: Exact sorted core name (word-order invariant)
    - Channel 4: Address Spatial Anchor (street number + primary street word)
    - Channel 5: Distinctive Name Word Token (tok1, skipping generic business words)
    - Channel 6: Distinctive Address Word Token (addr_tok1)
    
    Uses per-channel subquery limits to ensure fair representation and prevents
    generic names from crowding out high-precision address anchor matches.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH, max_cands_per_query: int = 150):
        self.db_path = db_path
        self.max_cands_per_query = max_cands_per_query
        self.conn = sqlite3.connect(self.db_path)

    def retrieve_candidates(self, s1_records: List[Dict]) -> List[Dict]:
        cur = self.conn.cursor()
        all_candidates = []

        for s1 in s1_records:
            qid = s1["entity_id"]
            country = s1["country"]
            nc = s1["name_compact"]
            ncc = s1.get("name_core_compact", s1.get("name_no_legal", ""))
            ns = s1["name_sorted"]
            anc = s1.get("addr_anchor", "")
            tok1 = s1.get("tok1", "")
            addr_tok1 = s1.get("addr_tok1", "")

            # Generate synthetic noise invariant variants
            nc_unaccent = strip_accents(nc)
            nc_leet = to_leet(nc_unaccent)
            ncc_unaccent = strip_accents(ncc)
            ncc_leet = to_leet(ncc_unaccent)
            ncc_extra = clean_extra_legal(ncc)

            seen_cands = set()
            for tbl, src_name in [("source2", "S2"), ("source3", "S3")]:
                query = f"""
                    SELECT entity_id, business_name, business_address, 
                           name_clean, name_compact, name_core_compact, name_sorted,
                           address_clean, postal_code, numbers,
                           'exact_compact' as channel
                    FROM (
                        SELECT * FROM {tbl} 
                        WHERE country = ? AND name_compact IN (?, ?, ?)
                        LIMIT 60
                    )
                    
                    UNION
                    
                    SELECT entity_id, business_name, business_address, 
                           name_clean, name_compact, name_core_compact, name_sorted,
                           address_clean, postal_code, numbers,
                           'exact_core' as channel
                    FROM (
                        SELECT * FROM {tbl} 
                        WHERE country = ? AND name_core_compact IN (?, ?, ?, ?)
                        LIMIT 60
                    )
                    
                    UNION
                    
                    SELECT entity_id, business_name, business_address, 
                           name_clean, name_compact, name_core_compact, name_sorted,
                           address_clean, postal_code, numbers,
                           'exact_sorted' as channel
                    FROM (
                        SELECT * FROM {tbl} 
                        WHERE country = ? AND name_sorted = ?
                        LIMIT 60
                    )
                """
                params = [
                    country, nc, nc_unaccent, nc_leet,
                    country, ncc, ncc_unaccent, ncc_leet, ncc_extra,
                    country, ns
                ]

                if anc:
                    query += f"""
                        UNION
                        SELECT entity_id, business_name, business_address, 
                               name_clean, name_compact, name_core_compact, name_sorted,
                               address_clean, postal_code, numbers,
                               'addr_anchor' as channel
                        FROM (
                            SELECT * FROM {tbl} 
                            WHERE country = ? AND addr_anchor = ?
                            LIMIT 50
                        )
                    """
                    params.extend([country, anc])

                if tok1 and len(tok1) >= 4 and tok1.lower() not in GENERIC_NAME_TOKENS:
                    query += f"""
                        UNION
                        SELECT entity_id, business_name, business_address, 
                               name_clean, name_compact, name_core_compact, name_sorted,
                               address_clean, postal_code, numbers,
                               'name_token' as channel
                        FROM (
                            SELECT * FROM {tbl}
                            WHERE country = ? AND tok1 = ?
                            LIMIT 30
                        )
                    """
                    params.extend([country, tok1])

                if addr_tok1 and len(addr_tok1) >= 4:
                    query += f"""
                        UNION
                        SELECT entity_id, business_name, business_address, 
                               name_clean, name_compact, name_core_compact, name_sorted,
                               address_clean, postal_code, numbers,
                               'addr_token' as channel
                        FROM (
                            SELECT * FROM {tbl}
                            WHERE country = ? AND addr_tok1 = ?
                            LIMIT 30
                        )
                    """
                    params.extend([country, addr_tok1])

                cur.execute(query, params)
                matches = cur.fetchall()


                for row in matches:
                    tid = row[0]
                    if tid in seen_cands:
                        continue
                    seen_cands.add(tid)
                    channel = row[10]
                    cand_info = {
                        "entity_id": tid,
                        "business_name": row[1],
                        "business_address": row[2],
                        "name_clean": row[3],
                        "name_compact": row[4],
                        "name_no_legal": row[5],
                        "name_sorted": row[6],
                        "address_clean": row[7],
                        "postal_code": row[8],
                        "numbers": row[9].split(",") if row[9] else []
                    }
                    
                    all_candidates.append({
                        "source1_entity_id": qid,
                        "candidate_entity_id": tid,
                        "candidate_source": src_name,
                        "country": country,
                        "retrieval_channels": channel,
                        "retrieval_score": 10.0 if "exact" in channel else (7.0 if "anchor" in channel else 5.0),
                        "channel_count": 1,
                        "cand_record": cand_info
                    })
                    if len(seen_cands) >= self.max_cands_per_query:
                        break

        return all_candidates

    def close(self):
        self.conn.close()
