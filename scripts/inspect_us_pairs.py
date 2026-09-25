import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids

rapid_s1 = read_tsv('artifacts/splits/rapid_eval_s1.tsv')
india_ids = list(rapid_s1[rapid_s1['country'] == 'India']['source1_entity_id'].iloc[:100])
us_ids = list(rapid_s1[rapid_s1['country'] == 'US']['source1_entity_id'].iloc[:100])
sample_ids = set(india_ids + us_ids)

gt_df = read_tsv('train_ground_truth.tsv')
eval_gt = gt_df[gt_df['source1_entity_id'].isin(sample_ids)]

s1_df = read_tsv('train_source1.tsv')
s1_dict = s1_df[s1_df['entity_id'].isin(sample_ids)].set_index('entity_id').to_dict('index')

print('=== US GROUND TRUTH PAIRS INSPECTION ===')
s2_df = read_tsv('train_source2.tsv')
s3_df = read_tsv('train_source3.tsv')
s2_dict = s2_df.set_index('entity_id').to_dict('index')
s3_dict = s3_df.set_index('entity_id').to_dict('index')

count = 0
for _, row in eval_gt.iterrows():
    s1_id = row['source1_entity_id']
    s1 = s1_dict.get(s1_id)
    if not s1 or s1['country'] != 'US': 
        continue
    matches = parse_matched_ids(row['matched_entity_ids'])
    print(f"\n[S1-US] {s1_id}:")
    print(f"  Name:    {s1['business_name']}")
    print(f"  Address: {s1['business_address']}")
    for m in matches:
        target = s2_dict.get(m) or s3_dict.get(m)
        if target:
            print(f"  -> Match {m}:")
            print(f"     Name:    {target['business_name']}")
            print(f"     Address: {target['business_address']}")
    count += 1
    if count >= 4:
        break
