import re
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

# Leetspeak / typo normalizer map
LEET_TRANS = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "@": "a",
    "$": "s"
})

def normalize_leetspeak(s: str) -> str:
    """Normalize common character substitutions like 'gl0bal' -> 'global'."""
    return s.translate(LEET_TRANS)

def jaro_similarity(s1: str, s2: str) -> float:
    """Fast Jaro string similarity."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    return (
        (matches / len1) +
        (matches / len2) +
        ((matches - transpositions / 2) / matches)
    ) / 3.0

def jaro_winkler(s1: str, s2: str, p: float = 0.1, max_l: int = 4) -> float:
    """Jaro-Winkler string similarity with prefix boost."""
    j = jaro_similarity(s1, s2)
    l = 0
    for i in range(min(len(s1), len(s2), max_l)):
        if s1[i] == s2[i]:
            l += 1
        else:
            break
    return j + (l * p * (1.0 - j))

def token_jaccard(toks1: Set[str], toks2: Set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not toks1 or not toks2:
        return 0.0
    intersection = len(toks1 & toks2)
    union = len(toks1 | toks2)
    return intersection / union if union > 0 else 0.0

def token_containment(toks1: Set[str], toks2: Set[str]) -> float:
    """Token containment: |intersection| / min(|toks1|, |toks2|)."""
    if not toks1 or not toks2:
        return 0.0
    intersection = len(toks1 & toks2)
    min_len = min(len(toks1), len(toks2))
    return intersection / min_len if min_len > 0 else 0.0

def compute_pairwise_features(
    s1_row: Dict,
    cand_row: Dict,
    channel_info: Dict
) -> Dict[str, float]:
    """
    Computes a comprehensive vector of similarity & contradiction features
    for a single candidate pair conforming to PRD 10 and Implementation Plan 12.
    """
    s1_name_clean = s1_row["name_clean"]
    cand_name_clean = cand_row["name_clean"]

    s1_name_core = s1_row["name_no_legal"]
    cand_name_core = cand_row["name_no_legal"]

    s1_name_compact = s1_row["name_compact"]
    cand_name_compact = cand_row["name_compact"]

    s1_addr_clean = s1_row["address_clean"]
    cand_addr_clean = cand_row["address_clean"]

    s1_post = s1_row["postal_code"]
    cand_post = cand_row["postal_code"]

    s1_nums = set(s1_row["numbers"])
    cand_nums = set(cand_row["numbers"])

    # 1. Exact Name Matches
    exact_compact = 1.0 if s1_name_compact and s1_name_compact == cand_name_compact else 0.0
    exact_core = 1.0 if s1_name_core and s1_name_core == cand_name_core else 0.0
    
    # 2. Leetspeak-normalized exact match (e.g. gl0bal -> global)
    s1_leet = normalize_leetspeak(s1_name_compact)
    cand_leet = normalize_leetspeak(cand_name_compact)
    exact_leet = 1.0 if s1_leet and s1_leet == cand_leet else 0.0

    # 3. String & Token Similarities on Name
    name_jw = jaro_winkler(s1_name_clean, cand_name_clean)
    core_jw = jaro_winkler(s1_name_core, cand_name_core)

    s1_name_toks = set(s1_name_clean.split())
    cand_name_toks = set(cand_name_clean.split())
    name_jaccard = token_jaccard(s1_name_toks, cand_name_toks)
    name_containment = token_containment(s1_name_toks, cand_name_toks)

    s1_core_toks = set(s1_name_core.split())
    cand_core_toks = set(cand_name_core.split())
    core_jaccard = token_jaccard(s1_core_toks, cand_core_toks)

    # Name Length difference
    len_diff = abs(len(s1_name_clean) - len(cand_name_clean))
    len_ratio = min(len(s1_name_clean), len(cand_name_clean)) / max(len(s1_name_clean), len(cand_name_clean), 1)

    # 4. Address Similarities
    s1_addr_toks = set(s1_addr_clean.split())
    cand_addr_toks = set(cand_addr_clean.split())
    addr_jaccard = token_jaccard(s1_addr_toks, cand_addr_toks)
    addr_containment = token_containment(s1_addr_toks, cand_addr_toks)
    addr_jw = jaro_winkler(s1_addr_clean[:50], cand_addr_clean[:50])

    # 5. Postal Code Consistency
    if s1_post and cand_post:
        postal_match = 1.0 if s1_post == cand_post else 0.0
        postal_conflict = 1.0 if s1_post != cand_post else 0.0
    else:
        postal_match = 0.0
        postal_conflict = 0.0

    # 6. Address Numbers Consistency
    if s1_nums and cand_nums:
        nums_jaccard = token_jaccard(s1_nums, cand_nums)
        nums_conflict = 1.0 if not (s1_nums & cand_nums) else 0.0
    else:
        nums_jaccard = 0.0
        nums_conflict = 0.0

    # 7. Cross-field interaction
    name_strong_addr_strong = 1.0 if (core_jw >= 0.85 and addr_jaccard >= 0.4) else 0.0
    name_strong_addr_conflict = 1.0 if (core_jw >= 0.9 and postal_conflict == 1.0) else 0.0

    # 8. Retrieval signals
    retrieval_score = float(channel_info.get("retrieval_score", 0.0))
    channel_count = float(channel_info.get("channel_count", 1.0))
    is_s2 = 1.0 if channel_info.get("candidate_source") == "S2" else 0.0

    return {
        "exact_compact": exact_compact,
        "exact_core": exact_core,
        "exact_leet": exact_leet,
        "name_jw": name_jw,
        "core_jw": core_jw,
        "name_jaccard": name_jaccard,
        "name_containment": name_containment,
        "core_jaccard": core_jaccard,
        "len_ratio": len_ratio,
        "addr_jaccard": addr_jaccard,
        "addr_containment": addr_containment,
        "addr_jw": addr_jw,
        "postal_match": postal_match,
        "postal_conflict": postal_conflict,
        "nums_jaccard": nums_jaccard,
        "nums_conflict": nums_conflict,
        "name_strong_addr_strong": name_strong_addr_strong,
        "name_strong_addr_conflict": name_strong_addr_conflict,
        "retrieval_score": retrieval_score,
        "channel_count": channel_count,
        "is_s2": is_s2
    }
