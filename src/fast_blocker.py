import os
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd

COMMON_ADDRESS_STOPS = {
    "road", "street", "avenue", "drive", "lane", "near", "opposite", "behind",
    "floor", "building", "block", "sector", "nagar", "colony", "apartment",
    "suite", "unit", "plot", "town", "city", "post", "dist", "district", "state"
}

COMMON_NAME_STOPS = {
    "and", "the", "for", "with", "ltd", "inc", "pvt", "limited", "private",
    "corp", "llc", "llp", "company", "co", "services", "enterprises", "group",
    "center", "service"
}

class MultiChannelInvertedBlocker:
    """
    High-recall multi-channel blocker conforming to PRD 9.1:
    - Channel 1: Exact compact name (strips spaces and domains)
    - Channel 2: Exact compact core name (strips legal suffixes anywhere)
    - Channel 3: Exact sorted core name (order-invariant)
    - Channel 4: Distinctive Name word tokens
    - Channel 5: Distinctive Address word tokens (catches transliterated Indic names)
    - Channel 6: Postal code + Address number tokens (spatial anchor)
    - Channel 7: Core name prefix 4-char (catches typo suffixes like 'fcmoinacne')
    """
    def __init__(self, max_candidates_per_query: int = 80):
        self.max_candidates_per_query = max_candidates_per_query

    def build_indexes(self, target_df: pd.DataFrame):
        print(f"Building comprehensive multi-channel inverted indexes on {len(target_df):,} records...")
        
        self.exact_compact = defaultdict(list)
        self.exact_core_compact = defaultdict(list)
        self.exact_sorted = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.postal_num_index = defaultdict(list)
        self.prefix4_index = defaultdict(list)

        target_ids = target_df["entity_id"].values
        compact_names = target_df["name_compact"].values
        core_compacts = target_df["name_compact_core"].values
        sorted_names = target_df["name_sorted"].values
        clean_names = target_df["name_clean"].values
        clean_addresses = target_df["address_clean"].values
        postals = target_df["postal_code"].values
        numbers_list = target_df["numbers"].values

        for i in range(len(target_df)):
            tid = target_ids[i]
            cname = compact_names[i]
            core_c = core_compacts[i]
            sort_n = sorted_names[i]
            clean_n = clean_names[i]
            clean_a = clean_addresses[i]
            post = postals[i]
            nums = numbers_list[i]

            # 1. Exact Name views
            if cname:
                self.exact_compact[cname].append(tid)
            if core_c and core_c != cname:
                self.exact_core_compact[core_c].append(tid)
            if sort_n:
                self.exact_sorted[sort_n].append(tid)

            # 2. Name tokens
            for tok in clean_n.split():
                if len(tok) >= 3 and tok not in COMMON_NAME_STOPS:
                    if len(self.name_token_index[tok]) < 3000:
                        self.name_token_index[tok].append(tid)

            # 3. Address distinctive tokens
            for tok in clean_a.split():
                if len(tok) >= 4 and tok not in COMMON_ADDRESS_STOPS:
                    if len(self.addr_token_index[tok]) < 2500:
                        self.addr_token_index[tok].append(tid)

            # 4. Postal + Number Key
            if post and nums:
                key = f"{post}_{nums[0]}"
                if len(self.postal_num_index[key]) < 300:
                    self.postal_num_index[key].append(tid)

            # 5. Core prefix (4 chars) for distinctive names (len >= 5)
            if core_c and len(core_c) >= 5:
                pref = core_c[:5]
                if len(self.prefix4_index[pref]) < 1000:
                    self.prefix4_index[pref].append(tid)

        print(f"Indexes built: {len(self.exact_compact):,} compact, {len(self.exact_core_compact):,} core compact, {len(self.exact_sorted):,} sorted names.")

    def retrieve(self, query_df: pd.DataFrame, source_name: str) -> List[Dict]:
        print(f"Retrieving candidates for {len(query_df):,} query records from {source_name}...")
        results = []

        q_ids = query_df["entity_id"].values
        q_compacts = query_df["name_compact"].values
        q_core_compacts = query_df["name_compact_core"].values
        q_sorted = query_df["name_sorted"].values
        q_cleans = query_df["name_clean"].values
        q_addresses = query_df["address_clean"].values
        q_postals = query_df["postal_code"].values
        q_numbers = query_df["numbers"].values
        q_countries = query_df["country"].values

        for i in range(len(query_df)):
            qid = q_ids[i]
            qc = q_compacts[i]
            qcore_c = q_core_compacts[i]
            qsort = q_sorted[i]
            qclean = q_cleans[i]
            qaddr = q_addresses[i]
            qpost = q_postals[i]
            qnums = q_numbers[i]
            qcountry = q_countries[i]

            candidate_scores = defaultdict(float)
            candidate_channels = defaultdict(list)

            # Channel 1: Exact compact
            if qc in self.exact_compact:
                for tid in self.exact_compact[qc]:
                    candidate_scores[tid] += 25.0
                    candidate_channels[tid].append("exact_compact")

            # Channel 2: Exact core compact
            if qcore_c in self.exact_core_compact:
                for tid in self.exact_core_compact[qcore_c]:
                    candidate_scores[tid] += 20.0
                    candidate_channels[tid].append("exact_core_compact")

            # Channel 3: Exact sorted core name
            if qsort in self.exact_sorted:
                for tid in self.exact_sorted[qsort]:
                    candidate_scores[tid] += 18.0
                    candidate_channels[tid].append("exact_sorted")

            # Channel 4: Name tokens
            q_name_tokens = [w for w in qclean.split() if len(w) >= 3 and w not in COMMON_NAME_STOPS]
            for tok in q_name_tokens:
                if tok in self.name_token_index:
                    for tid in self.name_token_index[tok]:
                        candidate_scores[tid] += 4.0
                        candidate_channels[tid].append(f"name_{tok}")

            # Channel 5: Address tokens
            q_addr_tokens = [w for w in qaddr.split() if len(w) >= 4 and w not in COMMON_ADDRESS_STOPS]
            for tok in q_addr_tokens:
                if tok in self.addr_token_index:
                    for tid in self.addr_token_index[tok]:
                        candidate_scores[tid] += 3.5
                        candidate_channels[tid].append(f"addr_{tok}")

            # Channel 6: Postal + Number
            if qpost and qnums:
                key = f"{qpost}_{qnums[0]}"
                if key in self.postal_num_index:
                    for tid in self.postal_num_index[key]:
                        candidate_scores[tid] += 6.0
                        candidate_channels[tid].append("postal_num")

            # Channel 7: Core prefix (for typo tolerance)
            if qcore_c and len(qcore_c) >= 5:
                pref = qcore_c[:5]
                if pref in self.prefix4_index:
                    for tid in self.prefix4_index[pref]:
                        candidate_scores[tid] += 2.0
                        candidate_channels[tid].append("prefix5")

            # Keep top candidates per query
            if candidate_scores:
                sorted_cands = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates_per_query]
                for tid, score in sorted_cands:
                    results.append({
                        "source1_entity_id": qid,
                        "candidate_entity_id": tid,
                        "candidate_source": source_name,
                        "country": qcountry,
                        "retrieval_score": score,
                        "retrieval_channels": "|".join(candidate_channels[tid][:5]),
                        "channel_count": len(candidate_channels[tid])
                    })

        print(f"Retrieved {len(results):,} total candidates for {source_name}.")
        return results
