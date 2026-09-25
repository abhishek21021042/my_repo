import os
import shutil
import time
import pandas as pd

def main():
    target_file = r"test_data/test_source1 (1).tsv"
    backup_file = r"test_data/test_source1 (1).tsv.bak"

    print("=" * 80)
    print(f"=== SORTING {target_file} BY SERIAL NUMBER (NUMERIC ASCENDING) ===")
    print("=" * 80)
    t0 = time.time()

    # 1. Backup original file if backup doesn't already exist
    if not os.path.exists(backup_file):
        print(f"Creating safe backup: {backup_file} ...")
        shutil.copyfile(target_file, backup_file)
        print("Backup created successfully.")
    else:
        print(f"Backup already exists at: {backup_file}")

    # 2. Read TSV
    print(f"Loading {target_file} into memory...")
    t_read = time.time()
    df = pd.read_csv(target_file, sep="\t", dtype=str)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns in {time.time() - t_read:.2f}s.")
    print(f"Columns: {list(df.columns)}")

    # 3. Extract serial number (numeric part after 'S1-')
    print("Extracting serial number for numerical sorting...")
    # Extract integer ID
    df["serial_no_int"] = df["entity_id"].str.replace("S1-", "", regex=False).astype("int64")

    # 4. Sort by serial number ascending
    print("Sorting rows by serial number ascending...")
    t_sort = time.time()
    df_sorted = df.sort_values(by="serial_no_int", ascending=True).drop(columns=["serial_no_int"])
    print(f"Sorted {len(df_sorted):,} rows in {time.time() - t_sort:.2f}s.")

    # 5. Overwrite the target file cleanly
    print(f"Writing sorted data back to {target_file} ...")
    t_write = time.time()
    df_sorted.to_csv(target_file, sep="\t", index=False, encoding="utf-8")
    print(f"Written in {time.time() - t_write:.2f}s.")

    # 6. Verification
    print("\n--- VERIFICATION CHECKS ---")
    df_verify = pd.read_csv(target_file, sep="\t", nrows=10)
    print("First 10 rows (Ascending Serial No):")
    print(df_verify[["entity_id", "business_name", "country"]])

    print("\nLast 5 rows:")
    df_tail = pd.read_csv(target_file, sep="\t").tail(5)
    print(df_tail[["entity_id", "business_name", "country"]])

    print(f"\nTotal rows verified: {len(df_sorted):,} (Original: {len(df):,})")
    assert len(df_sorted) == len(df), "Row count mismatch!"
    print(f"Total time elapsed: {time.time() - t0:.2f}s.")
    print("=" * 80)

if __name__ == "__main__":
    main()
