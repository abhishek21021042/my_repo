import re
import unicodedata
from typing import List, Tuple, Optional
import pandas as pd

# Standard legal suffixes to filter for core comparison view
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd", "limited",
    "pvt", "private", "llp", "co", "company", "gmbh", "sa", "sarl", "nv",
    "plc", "bv", "ag", "enterprises", "services", "group", "holdings",
    "center", "service", "the"
}

DOMAIN_EXTENSIONS = re.compile(r"\.(com|in|org|net|co|biz|info|io|ai|gov|edu|me)\b", re.IGNORECASE)

RE_WHITESPACE = re.compile(r"\s+")
RE_PUNCT = re.compile(r"[^\w\s]")
RE_ALPHANUMERIC = re.compile(r"[^\w]")
RE_NUMBERS = re.compile(r"\b\d+\b")

# Country-aware postal code patterns
RE_POSTAL_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_POSTAL_IN = re.compile(r"\b([1-9]\d{5})\b")
RE_POSTAL_FR = re.compile(r"\b(\d{5})\b")

def unicode_clean(text: str) -> str:
    """Normalize Unicode to NFKC form and strip."""
    if not text or pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text)).strip()

def strip_domain_extensions(text: str) -> str:
    """Strip web domain endings like .com, .in, .org, .net."""
    return DOMAIN_EXTENSIONS.sub("", text)

def normalize_text(text: str) -> str:
    """
    Standard clean text:
    - Unicode NFKC
    - Lowercase
    - Strip domain extensions (.com, etc.)
    - Replace '&' with 'and'
    - Replace punctuation with spaces
    - Collapse multiple spaces
    """
    t = unicode_clean(text).lower()
    t = strip_domain_extensions(t)
    t = t.replace("&", " and ")
    t = RE_PUNCT.sub(" ", t)
    return RE_WHITESPACE.sub(" ", t).strip()

def normalize_compact(text: str) -> str:
    """
    Alphanumeric-only compact string without any spaces or symbols.
    Also strips domain extensions.
    """
    t = unicode_clean(text).lower()
    t = strip_domain_extensions(t)
    return RE_ALPHANUMERIC.sub("", t)

def remove_legal_suffixes(normalized_name: str) -> str:
    """
    Removes corporate/legal suffixes (from anywhere in the token sequence:
    beginning, middle, or end) to produce a pure core entity name.
    e.g. 'private any farmers limited' -> 'any farmers'
    e.g. 'pvt ensign infra ltd' -> 'ensign infra'
    """
    tokens = normalized_name.split()
    if not tokens:
        return ""
    
    cleaned = [w for w in tokens if w not in LEGAL_SUFFIXES]
    if not cleaned:
        # Don't leave completely empty if string only had a legal suffix
        return normalized_name
    return " ".join(cleaned)

def sort_tokens(normalized_text: str) -> str:
    """
    Sorts space-separated tokens alphabetically.
    Makes comparison word-order invariant.
    """
    tokens = normalized_text.split()
    return " ".join(sorted(tokens))

def extract_numbers(text: str) -> List[str]:
    """
    Extracts ordered list of integer number tokens from text.
    """
    t = unicode_clean(text)
    return RE_NUMBERS.findall(t)

def extract_postal_code(address_text: str, country: Optional[str] = None) -> str:
    """
    Country-aware extraction of postal code.
    Dynamic: handles US, India, France (and unknown country fallbacks).
    """
    t = unicode_clean(address_text)
    c = str(country).strip().upper() if country else ""

    if c == "US":
        matches = RE_POSTAL_US.findall(t)
        return matches[-1] if matches else ""
    elif c == "INDIA":
        matches = RE_POSTAL_IN.findall(t)
        return matches[-1] if matches else ""
    elif c == "FRANCE":
        matches = RE_POSTAL_FR.findall(t)
        return matches[-1] if matches else ""
    else:
        m_in = RE_POSTAL_IN.findall(t)
        if m_in:
            return m_in[-1]
        m_5 = RE_POSTAL_US.findall(t)
        if m_5:
            return m_5[-1]
        return ""

def add_normalized_views(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds multi-view normalized columns to a DataFrame containing
    business_name, business_address, country.
    Preserves original columns untouched!
    """
    df = df.copy()

    # Name views
    df["name_clean"] = df["business_name"].apply(normalize_text)
    df["name_no_legal"] = df["name_clean"].apply(remove_legal_suffixes)
    df["name_compact"] = df["business_name"].apply(normalize_compact)
    df["name_sorted"] = df["name_no_legal"].apply(sort_tokens)
    df["name_compact_core"] = df["name_no_legal"].apply(normalize_compact)

    # Address views
    df["address_clean"] = df["business_address"].apply(normalize_text)
    df["address_compact"] = df["business_address"].apply(normalize_compact)
    
    # Combined view for text retrieval
    df["combined_text"] = df["name_clean"] + " " + df["address_clean"]

    # Numeric & postal views
    df["postal_code"] = [
        extract_postal_code(addr, c)
        for addr, c in zip(df["business_address"], df["country"])
    ]
    df["numbers"] = df["business_address"].apply(extract_numbers)

    return df
