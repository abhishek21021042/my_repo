import os
from typing import Dict, List, Set, Optional
import pandas as pd

def write_submission_outputs(
    s1_entity_ids: List[str],
    predictions: Dict[str, List[str]],
    candidates_dict: Dict[str, List[str]],
    output_dir: str = "output",
    validate: bool = True
) -> Dict[str, str]:
    """
    Writes official competition submission files conforming strictly to PRD Section 20:
    1. output/matching_results.tsv:
       - Header: source1_entity_id\tmatched_entity_ids
       - Every test S1 entity ID exactly once in deterministic order.
       - Matched entity IDs comma-separated, no duplicates, deterministically sorted.
       - No match means empty second field (source1_entity_id\t).
    2. output/candidate_pairs.tsv:
       - Header: source1_entity_id\tcandidate_entity_ids
       - Every test S1 entity ID exactly once.
       - Candidate entity IDs comma-separated, deterministically sorted.
       - Predictions MUST be a strict subset of candidates.
    """
    os.makedirs(output_dir, exist_ok=True)
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # 1. Write matching_results.tsv
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_entity_ids:
            matches = predictions.get(s1_id, [])
            # Deduplicate and sort deterministically
            clean_matches = sorted(list(set(matches)))
            match_str = ",".join(clean_matches)
            f.write(f"{s1_id}\t{match_str}\n")

    # 2. Write candidate_pairs.tsv
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_entity_ids:
            cands = candidates_dict.get(s1_id, [])
            clean_cands = sorted(list(set(cands)))
            cand_str = ",".join(clean_cands)
            f.write(f"{s1_id}\t{cand_str}\n")

    if validate:
        assert_submission_validity(s1_entity_ids, matching_path, candidate_path)

    print(f"Successfully generated and validated submission files:")
    print(f"  - Matching results: {matching_path}")
    print(f"  - Candidate pairs:  {candidate_path}")
    return {"matching_results": matching_path, "candidate_pairs": candidate_path}

def assert_submission_validity(
    expected_s1_ids: List[str],
    matching_file: str,
    candidate_file: str
):
    """
    Executes all pre-submission assertions specified in PRD 20.3:
    - Row count equals S1 count
    - S1 set exactly equal with no duplicates
    - Valid ID prefixes
    - Predictions are a strict subset of candidate pairs
    """
    expected_set = set(expected_s1_ids)
    assert len(expected_set) == len(expected_s1_ids), "Duplicate IDs found in expected S1 list!"

    # Validate matching_results.tsv
    with open(matching_file, "r", encoding="utf-8") as f:
        m_lines = [line.rstrip("\r\n") for line in f if line.strip()]

    assert m_lines[0] == "source1_entity_id\tmatched_entity_ids", f"Invalid matching header: {m_lines[0]}"
    assert len(m_lines) - 1 == len(expected_s1_ids), f"Matching row count mismatch! Expected {len(expected_s1_ids)}, got {len(m_lines) - 1}"

    match_dict = {}
    for line in m_lines[1:]:
        parts = line.split("\t")
        assert len(parts) == 2, f"Line format error (expected 2 tab-separated fields): {line}"
        s1_id, match_str = parts[0], parts[1]
        assert s1_id in expected_set, f"Unknown S1 ID in matching results: {s1_id}"
        assert s1_id not in match_dict, f"Duplicate S1 ID in matching results: {s1_id}"
        m_list = [m for m in match_str.split(",") if m]
        assert len(m_list) == len(set(m_list)), f"Duplicate matches in row: {line}"
        for m in m_list:
            assert m.startswith("S2-") or m.startswith("S3-"), f"Invalid entity prefix in match: {m}"
        match_dict[s1_id] = set(m_list)

    # Validate candidate_pairs.tsv
    with open(candidate_file, "r", encoding="utf-8") as f:
        c_lines = [line.rstrip("\r\n") for line in f if line.strip()]

    assert c_lines[0] == "source1_entity_id\tcandidate_entity_ids", f"Invalid candidate header: {c_lines[0]}"
    assert len(c_lines) - 1 == len(expected_s1_ids), f"Candidate row count mismatch! Expected {len(expected_s1_ids)}, got {len(c_lines) - 1}"

    for line in c_lines[1:]:
        parts = line.split("\t")
        assert len(parts) == 2, f"Line format error in candidate file: {line}"
        s1_id, cand_str = parts[0], parts[1]
        c_list = [c for c in cand_str.split(",") if c]
        c_set = set(c_list)
        assert len(c_list) == len(c_set), f"Duplicate candidates in row: {line}"
        
        # PRD 20.3: Every match MUST be a subset of candidate pairs!
        m_set = match_dict.get(s1_id, set())
        assert m_set.issubset(c_set), f"Integrity Violation: Matches for {s1_id} are not a subset of candidates! Diff: {m_set - c_set}"

    print(f"All PRD 20.3 pre-submission assertions PASSED ({len(expected_s1_ids):,} entities checked)!")
