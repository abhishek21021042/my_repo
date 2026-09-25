import os
from typing import Dict, List, Set, Union
import pandas as pd

def read_tsv(path: str) -> pd.DataFrame:
    """
    Safely load a TSV file preserving all fields as strings and preventing
    automatic conversion of strings like 'NA' into nulls.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

def write_tsv(df: pd.DataFrame, path: str) -> None:
    """
    Write DataFrame to a TSV file without index and with tab separator.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)

def parse_matched_ids(raw_str: Union[str, float, None]) -> List[str]:
    """
    Parse a comma-separated string of IDs into a deduplicated list of IDs,
    preserving deterministic order.
    """
    if raw_str is None or pd.isna(raw_str):
        return []
    s = str(raw_str).strip()
    if not s:
        return []
    
    seen = set()
    result = []
    for item in s.split(","):
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result

def format_matched_ids(id_list: Union[List[str], Set[str]]) -> str:
    """
    Format a list or set of IDs into a clean comma-separated string.
    """
    if not id_list:
        return ""
    # Deduplicate while preserving order if list, or sort if set
    if isinstance(id_list, set):
        ordered = sorted(id_list)
    else:
        seen = set()
        ordered = [x for x in id_list if not (x in seen or seen.add(x))]
    return ",".join(ordered)
