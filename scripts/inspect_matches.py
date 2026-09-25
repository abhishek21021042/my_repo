import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids

gt_df = read_tsv('train_ground_truth.tsv')
s1_df = read_tsv('train_source1.tsv')
s1_dict = s1_df.set_index('entity_id').to_dict('index')

target_pairs = []
for _, row in gt_df.head(10).iterrows():
    s1_id = row['source1_entity_id']
    matches = parse_matched_ids(row['matched_entity_ids'])
    if matches:
        target_pairs.append((s1_id, matches))
        if len(target_pairs) == 2:
            break

needed_s2 = {m for _, ms in target_pairs for m in ms if m.startswith('S2-')}
needed_s3 = {m for _, ms in target_pairs for m in ms if m.startswith('S3-')}

print(f'Looking up {len(needed_s2)} S2 IDs and {len(needed_s3)} S3 IDs...')

s2_df = read_tsv('train_source2.tsv')
s2_dict = s2_df[s2_df['entity_id'].isin(needed_s2)].set_index('entity_id').to_dict('index')

s3_df = read_tsv('train_source3.tsv')
s3_dict = s3_df[s3_df['entity_id'].isin(needed_s3)].set_index('entity_id').to_dict('index')

print('\n================ REAL GROUND TRUTH EXAMPLES ================')
for s1_id, matches in target_pairs:
    s1_info = s1_dict[s1_id]
    print(f"\n[SOURCE 1] {s1_id}:")
    print(f"   Name:    {s1_info['business_name']}")
    print(f"   Address: {s1_info['business_address']}")
    print(f"   Country: {s1_info['country']}")
    print("   Matches in Ground Truth:")
    for m in matches:
        if m in s2_dict:
            m_info = s2_dict[m]
            print(f"   -> [SOURCE 2] {m}:")
            print(f"        Name:    {m_info['business_name']}")
            print(f"        Address: {m_info['business_address']}")
        elif m in s3_dict:
            m_info = s3_dict[m]
            print(f"   -> [SOURCE 3] {m}:")
            print(f"        Name:    {m_info['business_name']}")
            print(f"        Address: {m_info['business_address']}")
