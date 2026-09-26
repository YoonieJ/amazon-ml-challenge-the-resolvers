import time
import pandas as pd

from blocking import _build_keys, _posting_list, load_ground_truth_pairs

DATASET = "../../../dataset/train"


def load(name, nrows=None):
    return pd.read_csv(f"{DATASET}/{name}", sep="\t", dtype=str, nrows=nrows)


s1 = load("train_source1.tsv", 50000)
s2 = load("train_source2.tsv")
s3 = load("train_source3.tsv")
other = pd.concat([s2, s3], ignore_index=True)
other_by_id = other.set_index("entity_id")
s1_by_id = s1.set_index("entity_id")

truth = load_ground_truth_pairs(f"{DATASET}/train_ground_truth.tsv")
truth = truth[truth["source1_entity_id"].isin(s1["entity_id"])]

# raw (uncapped) recall per pass, and combined, to isolate cap effect from
# key-matching effect
s1_keys = _build_keys(s1)
other_keys = _build_keys(other)

found_any = set()
for pass_name in ["name_token", "soundex", "addr_token"]:
    t0 = time.time()
    left = _posting_list(s1_keys, pass_name, max_postings=1000)
    right = _posting_list(other_keys, pass_name, max_postings=1000)
    merged = left.merge(right, on="key", suffixes=("_s1", "_other"))
    pairs = set(zip(merged["entity_id_s1"], merged["entity_id_other"]))
    truth_set = set(zip(truth["source1_entity_id"], truth["candidate_entity_id"]))
    hit = truth_set & pairs
    print(f"{pass_name}: raw pairs={len(pairs):,} recall={len(hit)/len(truth_set):.4f} ({time.time()-t0:.0f}s)")
    found_any |= pairs

truth_set = set(zip(truth["source1_entity_id"], truth["candidate_entity_id"]))
hit_any = truth_set & found_any
print(f"union of 3 passes (uncapped): recall={len(hit_any)/len(truth_set):.4f}")

misses = list(truth_set - found_any)[:10]
print(f"\n{len(truth_set) - len(hit_any)} true matches missed by ALL passes, e.g.:")
for s1_id, other_id in misses:
    r1 = s1_by_id.loc[s1_id]
    r2 = other_by_id.loc[other_id]
    print(f"  S1 {s1_id}: {r1['business_name']!r} | {r1['business_address']!r} | {r1['country']}")
    print(f"  -> {other_id}: {r2['business_name']!r} | {r2['business_address']!r} | {r2['country']}")
