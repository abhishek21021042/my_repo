import sqlite3
from typing import List, Dict, Set, Tuple, Optional
import time

DEFAULT_DB_PATH = "D:/hackathon/dataset.db"

class DiskBTreeBlocker:
    """
    Sub-millisecond multi-channel candidate retrieval backed by SQLite B-Tree indexes:
    - Channel 1: Exact compact name (strips whitespace, domain extensions, punctuation)
    - Channel 2: Exact core compact name (legal suffix stripped from anywhere)
    - Channel 3: Exact sorted core name (word-order invariant)
    - Channel 4: Address Spatial Anchor (street number + primary street word)
    - Channel 5: Distinctive Name Word Token (tok1)
    - Channel 6: Distinctive Address Word Token (addr_tok1)
    
    Uses per-channel subquery limits to ensure fair representation and prevents
    generic names from crowding out high-precision address anchor matches.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH, max_cands_per_query: int = 80):
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

            for tbl, src_name in [("source2", "S2"), ("source3", "S3")]:
                query = f"""
                    SELECT entity_id, business_name, business_address, 
                           name_clean, name_compact, name_core_compact, name_sorted,
                           address_clean, postal_code, numbers,
                           'exact_compact' as channel
                    FROM (
                        SELECT * FROM {tbl} 
                        WHERE country = ? AND name_compact = ?
                        LIMIT 20
                    )
                    
                    UNION
                    
                    SELECT entity_id, business_name, business_address, 
                           name_clean, name_compact, name_core_compact, name_sorted,
                           address_clean, postal_code, numbers,
                           'exact_core' as channel
                    FROM (
                        SELECT * FROM {tbl} 
                        WHERE country = ? AND name_core_compact = ?
                        LIMIT 20
                    )
                    
                    UNION
                    
                    SELECT entity_id, business_name, business_address, 
                           name_clean, name_compact, name_core_compact, name_sorted,
                           address_clean, postal_code, numbers,
                           'exact_sorted' as channel
                    FROM (
                        SELECT * FROM {tbl} 
                        WHERE country = ? AND name_sorted = ?
                        LIMIT 20
                    )
                """
                params = [country, nc, country, ncc, country, ns]

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
                            LIMIT 25
                        )
                    """
                    params.extend([country, anc])

                if tok1 and len(tok1) >= 4:
                    query += f"""
                        UNION
                        SELECT entity_id, business_name, business_address, 
                               name_clean, name_compact, name_core_compact, name_sorted,
                               address_clean, postal_code, numbers,
                               'name_token' as channel
                        FROM (
                            SELECT * FROM {tbl}
                            WHERE country = ? AND tok1 = ?
                            LIMIT 15
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
                            LIMIT 15
                        )
                    """
                    params.extend([country, addr_tok1])

                cur.execute(query, params)
                matches = cur.fetchall()

                for row in matches[:self.max_cands_per_query]:
                    tid = row[0]
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

        return all_candidates

    def close(self):
        self.conn.close()
