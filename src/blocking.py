import os
import argparse
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.io_utils import read_tsv, write_tsv, parse_matched_ids
from src.normalize import add_normalized_views

class MultiChannelBlocker:
    """
    Multi-channel candidate generator conforming to PRD 9 and Implementation Plan 10.
    Operates per country partition across Source 2 and Source 3.
    """
    def __init__(
        self,
        name_top_k: int = 30,
        combined_top_k: int = 20,
        char_ngram_range: Tuple[int, int] = (3, 4),
        max_features: int = 50000
    ):
        self.name_top_k = name_top_k
        self.combined_top_k = combined_top_k
        self.char_ngram_range = char_ngram_range
        self.max_features = max_features

    def build_indexes(self, target_df: pd.DataFrame):
        """
        Build exact key inverted indexes and TF-IDF matrix for a target source partition.
        """
        print(f"Building exact indexes for {len(target_df):,} target records...")
        # 1. Exact name compact index: compact_name -> list of target_ids
        self.name_compact_idx: Dict[str, List[str]] = {}
        for tid, cname in zip(target_df["entity_id"], target_df["name_compact"]):
            if cname:
                self.name_compact_idx.setdefault(cname, []).append(tid)

        # 2. Postal code index: postal_code -> list of target_ids
        self.postal_idx: Dict[str, List[str]] = {}
        for tid, post in zip(target_df["entity_id"], target_df["postal_code"]):
            if post:
                self.postal_idx.setdefault(post, []).append(tid)

        # 3. TF-IDF vectorizer on business name (char n-grams for typo and noise resilience)
        print("Fitting TF-IDF on target business names...")
        self.name_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            max_features=self.max_features,
            dtype=np.float32
        )
        self.name_target_matrix = self.name_vectorizer.fit_transform(target_df["name_clean"])

        self.target_ids = np.array(target_df["entity_id"])
        print("Target indexes ready.")

    def retrieve_candidates_for_queries(
        self, 
        query_df: pd.DataFrame, 
        source_name: str
    ) -> List[Dict]:
        """
        Retrieves candidates for a query DataFrame of Source 1 entities.
        Returns a list of candidate edge dictionaries.
        """
        print(f"Retrieving {source_name} candidates for {len(query_df):,} queries...")
        q_ids = list(query_df["entity_id"])
        q_compact = list(query_df["name_compact"])
        q_postal = list(query_df["postal_code"])
        q_names = list(query_df["name_clean"])
        q_countries = list(query_df["country"])

        # Transform query names into TF-IDF
        q_matrix = self.name_vectorizer.transform(q_names)

        # Inverted index lookup for exact matches
        candidates_by_q: Dict[str, Dict[str, Dict]] = {qid: {} for qid in q_ids}

        print("Channel 1: Exact compact name matching...")
        for qid, cname in zip(q_ids, q_compact):
            if cname and cname in self.name_compact_idx:
                for tid in self.name_compact_idx[cname]:
                    candidates_by_q[qid][tid] = {
                        "source1_entity_id": qid,
                        "candidate_entity_id": tid,
                        "candidate_source": source_name,
                        "country": query_df.loc[query_df["entity_id"] == qid, "country"].values[0],
                        "exact_name_match": 1,
                        "name_score": 1.0,
                        "channel": "exact_name"
                    }

        # TF-IDF Cosine Similarity via chunked dot product
        print("Channel 2: Name TF-IDF chunked retrieval...")
        chunk_size = 500
        n_queries = len(q_ids)

        for i in range(0, n_queries, chunk_size):
            end_idx = min(i + chunk_size, n_queries)
            sub_q = q_matrix[i:end_idx]
            # sub_q is (chunk_size, vocab), target is (n_target, vocab)
            # Dot product produces (chunk_size, n_target) sparse or dense
            sims = sub_q.dot(self.name_target_matrix.T).toarray()

            for local_idx in range(end_idx - i):
                global_idx = i + local_idx
                qid = q_ids[global_idx]
                row_sims = sims[local_idx]
                
                # Get top_k indices with score > 0.3
                if len(row_sims) > self.name_top_k:
                    top_indices = np.argpartition(row_sims, -self.name_top_k)[-self.name_top_k:]
                    # Sort top indices
                    top_indices = top_indices[np.argsort(-row_sims[top_indices])]
                else:
                    top_indices = np.argsort(-row_sims)

                for rank, tid_idx in enumerate(top_indices):
                    score = float(row_sims[tid_idx])
                    if score < 0.25:
                        continue # Prune completely irrelevant candidates
                    tid = self.target_ids[tid_idx]
                    
                    if tid in candidates_by_q[qid]:
                        cand = candidates_by_q[qid][tid]
                        cand["name_score"] = max(cand.get("name_score", 0.0), score)
                        cand["tfidf_rank"] = rank
                        cand["channel"] = cand["channel"] + "|name_tfidf"
                    else:
                        candidates_by_q[qid][tid] = {
                            "source1_entity_id": qid,
                            "candidate_entity_id": tid,
                            "candidate_source": source_name,
                            "country": q_countries[global_idx],
                            "exact_name_match": 0,
                            "name_score": score,
                            "tfidf_rank": rank,
                            "channel": "name_tfidf"
                        }

        # Flatten into candidate rows
        all_candidates = []
        for qid, cands in candidates_by_q.items():
            all_candidates.extend(cands.values())

        print(f"Retrieved {len(all_candidates):,} candidate pairs for {source_name}.")
        return all_candidates


def evaluate_blocking_recall(
    candidates_df: pd.DataFrame, 
    gt_df: pd.DataFrame, 
    evaluated_s1_ids: Set[str]
) -> Dict[str, float]:
    """
    Evaluates candidate recall against ground truth for evaluated entities.
    PRD Section 9.3:
    Candidate recall = (true positive pairs present in candidates) / (all true positive pairs)
    """
    eval_gt = gt_df[gt_df["source1_entity_id"].isin(evaluated_s1_ids)].copy()
    
    true_pairs = set()
    for _, row in eval_gt.iterrows():
        s1 = row["source1_entity_id"]
        for m in parse_matched_ids(row["matched_entity_ids"]):
            true_pairs.add((s1, m))

    if not true_pairs:
        return {"candidate_recall": 1.0, "total_true_pairs": 0, "captured_true_pairs": 0}

    cand_pairs = set(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    captured = len(true_pairs & cand_pairs)
    recall = captured / len(true_pairs)

    # Check entity level coverage (all matches captured for S1)
    s1_all_captured = 0
    total_non_singleton_entities = 0
    for _, row in eval_gt.iterrows():
        s1 = row["source1_entity_id"]
        matches = parse_matched_ids(row["matched_entity_ids"])
        if matches:
            total_non_singleton_entities += 1
            if all((s1, m) in cand_pairs for m in matches):
                s1_all_captured += 1

    entity_coverage = s1_all_captured / total_non_singleton_entities if total_non_singleton_entities > 0 else 1.0

    return {
        "candidate_recall": float(recall),
        "total_true_pairs": int(len(true_pairs)),
        "captured_true_pairs": int(captured),
        "missed_true_pairs": int(len(true_pairs) - captured),
        "entity_all_matches_coverage": float(entity_coverage),
        "total_candidates_generated": int(len(candidates_df)),
        "avg_candidates_per_s1": float(len(candidates_df) / len(evaluated_s1_ids)) if evaluated_s1_ids else 0
    }
