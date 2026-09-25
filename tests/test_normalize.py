import pytest
from src.normalize import (
    normalize_text,
    normalize_compact,
    remove_legal_suffixes,
    sort_tokens,
    extract_postal_code,
    extract_numbers
)

def test_normalize_text():
    raw = "  Maure & Williams, Inc. - 100% Quality!  "
    expected = "maure and williams inc 100 quality"
    assert normalize_text(raw) == expected

def test_normalize_compact():
    raw = "Maure-Williams, LLC"
    assert normalize_compact(raw) == "maurewilliamsllc"

def test_remove_legal_suffixes():
    assert remove_legal_suffixes("apple inc") == "apple"
    assert remove_legal_suffixes("tata consultancy services limited") == "tata consultancy"
    assert remove_legal_suffixes("raj investments llp") == "raj investments"
    # Don't strip if only word
    assert remove_legal_suffixes("inc") == "inc"

def test_sort_tokens():
    assert sort_tokens("williams maure colombier") == "colombier maure williams"

def test_extract_postal_code():
    us_addr = "85 Wayne Avenue, Ticonderoga, NY 12883-1234"
    assert extract_postal_code(us_addr, "US") == "12883"

    in_addr = "6(29), C.I.T. Colony, Mylapore, Chennai, Tamil Nadu 600004"
    assert extract_postal_code(in_addr, "India") == "600004"

    fr_addr = "10 Rue de la Paix, 75002 Paris"
    assert extract_postal_code(fr_addr, "France") == "75002"

def test_extract_numbers():
    addr = "6(29), C.I.T. Colony, Plot 102, Floor 3"
    assert extract_numbers(addr) == ["6", "29", "102", "3"]
