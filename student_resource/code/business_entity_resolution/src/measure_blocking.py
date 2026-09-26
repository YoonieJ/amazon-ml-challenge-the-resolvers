"""Run the blocking module on the real train data and report recall ceiling
and candidate-set size (plan-v1.md build order step 2), before any model
code gets written.
"""
import time

import pandas as pd

from blocking import generate_candidates, load_ground_truth_pairs, recall_ceiling

DATASET = "../../../dataset/train"


def load(name):
    return pd.read_csv(f"{DATASET}/{name}", sep="\t", dtype=str)


def main():
    t0 = time.time()
    s1 = load("train_source1.tsv")
    s2 = load("train_source2.tsv")
    s3 = load("train_source3.tsv")
    other = pd.concat([s2, s3], ignore_index=True)
    print(f"loaded: s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    candidates = generate_candidates(s1, other, top_k=20, max_postings=1000)
    print(f"blocking done in {time.time()-t0:.0f}s")

    n_s1_with_candidates = candidates["source1_entity_id"].nunique()
    per_s1 = candidates.groupby("source1_entity_id").size()
    naive = len(s1) * len(other)
    print(f"candidate pairs: {len(candidates):,}")
    print(f"S1 entities with >=1 candidate: {n_s1_with_candidates:,} / {len(s1):,}")
    print(f"candidates per S1 (mean/median/max): {per_s1.mean():.1f} / {per_s1.median():.0f} / {per_s1.max()}")
    print(f"reduction ratio vs naive cross join: {1 - len(candidates) / naive:.9f}")

    t0 = time.time()
    truth = load_ground_truth_pairs(f"{DATASET}/train_ground_truth.tsv")
    ceiling = recall_ceiling(candidates, truth)
    print(f"recall ceiling: {ceiling:.4f} (of {len(truth):,} true matches, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
