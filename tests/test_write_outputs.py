import os
import pytest
from src.write_outputs import write_submission_outputs, assert_submission_validity

def test_write_and_validate_submission(tmp_path):
    out_dir = str(tmp_path / "submission_test")
    s1_ids = ["S1-100", "S1-200", "S1-300"]
    
    candidates = {
        "S1-100": ["S2-1", "S2-2", "S3-1"],
        "S1-200": ["S2-5"],
        "S1-300": []  # Singleton with 0 candidates
    }
    
    predictions = {
        "S1-100": ["S2-1", "S3-1"],
        "S1-200": ["S2-5"],
        "S1-300": []  # Singleton
    }
    
    res = write_submission_outputs(s1_ids, predictions, candidates, output_dir=out_dir, validate=True)
    
    assert os.path.exists(res["matching_results"])
    assert os.path.exists(res["candidate_pairs"])
    
    # Verify content
    with open(res["matching_results"], "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-100\tS2-1,S3-1"
    assert lines[2] == "S1-200\tS2-5"
    assert lines[3] == "S1-300\t"  # Empty second field for singleton

def test_subset_violation_caught(tmp_path):
    out_dir = str(tmp_path / "violation_test")
    s1_ids = ["S1-100"]
    candidates = {"S1-100": ["S2-1"]}
    predictions = {"S1-100": ["S2-1", "S3-999"]}  # S3-999 is NOT in candidates!
    
    with pytest.raises(AssertionError, match="Integrity Violation"):
        write_submission_outputs(s1_ids, predictions, candidates, output_dir=out_dir, validate=True)
