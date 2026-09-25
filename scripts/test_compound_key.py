import sqlite3
import pandas as pd
from src.io_utils import read_tsv, parse_matched_ids

conn = sqlite3.connect('D:/hackathon/dataset.db')
cur = conn.cursor()

rapid_s1 = read_tsv('artifacts/splits/rapid_eval_s1.tsv')
india_ids = list(rapid_s1[rapid_s1['country'] == 'India']['source1_entity_id'].iloc[:100])
us_ids = list(rapid_s1[rapid_s1['country'] == 'US']['source1_entity_id'].iloc[:100])
sample_ids = set(india_ids + us_ids)

gt_df = read_tsv('train_ground_truth.tsv')
eval_gt_df = gt_df[gt_df['source1_entity_id'].isin(sample_ids)]
gt_dict = {row['source1_entity_id']: set(parse_matched_ids(row['matched_entity_ids'])) for _, row in eval_gt_df.iterrows()}

placeholders = ','.join(['?'] * len(sample_ids))
cur.execute(f'SELECT entity_id, name_clean, numbers FROM source1 WHERE entity_id IN ({placeholders})', list(sample_ids))
s1_info = {r[0]: (r[1], r[2].split(',')[0] if r[2] else '') for r in cur.fetchall()}

print('=== TESTING NAME_TOKEN + NUMBER ANCHOR ===')
captured_new = 0
total_checked = 0
for s1_id, true_matches in list(gt_dict.items())[:30]:
    if not true_matches: continue
    total_checked += len(true_matches)
    n_clean, num = s1_info.get(s1_id, ('', ''))
    words = [w for w in n_clean.split() if len(w) >= 3 and w not in ['the', 'and', 'for', 'with', 'inc', 'ltd', 'pvt', 'llc', 'corp', 'company', 'co']]
    tok1 = words[0] if words else ''
    if not tok1: continue
    
    for m in true_matches:
        tbl = 'source2' if m.startswith('S2-') else 'source3'
        cur.execute(f'SELECT name_clean, numbers FROM {tbl} WHERE entity_id = ?', (m,))
        row = cur.fetchone()
        if row and tok1 in row[0]:
            captured_new += 1

print(f'Captured by distinctive word token: {captured_new}/{total_checked} ({captured_new/total_checked*100:.1f}%)')
conn.close()
