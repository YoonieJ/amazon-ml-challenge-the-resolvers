"""
Exploratory Data Analysis for the Business Entity Resolution Challenge dataset.

Run this from the `student_resource/` directory (the one containing `dataset/`):

    python3 explore_dataset.py

Requires: pandas, matplotlib  (pip install pandas matplotlib)

Outputs:
  - Printed summary tables in the console
  - eda_report.txt          (plain-text summary of everything printed)
  - eda_charts.png          (4-panel figure: row counts, null rates,
                              country distribution, match-count distribution)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("dataset/train")
OUT_DIR = Path(".")

lines = []  # collect text for the report file


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def load(path):
    df = pd.read_csv(path, sep="\t", dtype=str)  # keep everything as string; this is EDA, not modeling
    return df


def main():
    s1 = load(DATA_DIR / "train_source1.tsv")
    s2 = load(DATA_DIR / "train_source2.tsv")
    s3 = load(DATA_DIR / "train_source3.tsv")
    gt = load(DATA_DIR / "train_ground_truth.tsv")

    sources = {"source1": s1, "source2": s2, "source3": s3}

    # ---------- 1. Row counts ----------
    log("=" * 60)
    log("1. ROW COUNTS PER SOURCE")
    log("=" * 60)
    row_counts = {name: len(df) for name, df in sources.items()}
    row_counts["ground_truth"] = len(gt)
    for name, cnt in row_counts.items():
        log(f"  {name:12s}: {cnt:,} rows")
    log()

    # ---------- 2. Null / empty rates ----------
    log("=" * 60)
    log("2. NULL / EMPTY RATES in business_name & business_address")
    log("=" * 60)
    null_rate_records = {}  # for plotting
    for name, df in sources.items():
        log(f"\n  --- {name} ---")
        for col in ["business_name", "business_address"]:
            if col not in df.columns:
                log(f"    {col}: COLUMN NOT FOUND")
                continue
            n_total = len(df)
            n_nan = df[col].isna().sum()
            # also catch empty-string / whitespace-only, common in messy real data
            n_blank = df[col].fillna("").str.strip().eq("").sum()
            nan_rate = n_nan / n_total if n_total else 0
            blank_rate = n_blank / n_total if n_total else 0
            null_rate_records[(name, col)] = blank_rate  # blank_rate includes true NaN
            log(f"    {col:18s}: NaN={n_nan:5d} ({nan_rate:.2%})   "
                f"blank-or-NaN={n_blank:5d} ({blank_rate:.2%})")
    log()

    # ---------- 3. Country distribution ----------
    log("=" * 60)
    log("3. COUNTRY DISTRIBUTION")
    log("=" * 60)
    country_counts = {}
    for name, df in sources.items():
        log(f"\n  --- {name} ---")
        if "country" not in df.columns:
            log("    COLUMN NOT FOUND")
            continue
        vc = df["country"].fillna("<NULL>").value_counts()
        country_counts[name] = vc
        for val, cnt in vc.items():
            log(f"    {val:15s}: {cnt:,} ({cnt/len(df):.2%})")
    log()

    # ---------- 4. Match count distribution ----------
    log("=" * 60)
    log("4. SOURCE 1 MATCH COUNTS (from train_ground_truth.tsv)")
    log("=" * 60)
    if "matched_entity_ids" not in gt.columns:
        log("  COLUMN NOT FOUND in ground truth file")
        n_matches = pd.Series(dtype=int)
    else:
        filled = gt["matched_entity_ids"].fillna("").str.strip()
        n_matches = filled.apply(lambda x: 0 if x == "" else len(x.split(",")))

        n_zero = (n_matches == 0).sum()
        n_one = (n_matches == 1).sum()
        n_multi = (n_matches > 1).sum()
        total = len(gt)

        log(f"  Total Source 1 entities in ground truth: {total:,}")
        log(f"    0 matches (singletons): {n_zero:,} ({n_zero/total:.2%})")
        log(f"    1 match               : {n_one:,} ({n_one/total:.2%})")
        log(f"    >1 matches            : {n_multi:,} ({n_multi/total:.2%})")
        log()
        log("  Full distribution of match counts:")
        dist = n_matches.value_counts().sort_index()
        for k, v in dist.items():
            log(f"    {k} matches: {v:,} entities")

        # sanity check: does every S1 id in source1 appear in ground truth, and vice versa?
        s1_ids = set(s1["entity_id"]) if "entity_id" in s1.columns else set()
        gt_ids = set(gt["source1_entity_id"]) if "source1_entity_id" in gt.columns else set()
        missing_from_gt = s1_ids - gt_ids
        extra_in_gt = gt_ids - s1_ids
        log()
        log(f"  Source1 entities missing from ground truth: {len(missing_from_gt)}")
        log(f"  Ground truth ids not present in source1    : {len(extra_in_gt)}")

    # ================= CHARTS =================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: row counts
    ax = axes[0, 0]
    names = list(row_counts.keys())
    vals = list(row_counts.values())
    ax.bar(names, vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_title("Row Counts per File")
    ax.set_ylabel("Row count")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=20)

    # Panel 2: null/blank rates
    ax = axes[0, 1]
    labels = [f"{name}\n{col}" for (name, col) in null_rate_records.keys()]
    vals = [v * 100 for v in null_rate_records.values()]
    ax.bar(labels, vals, color="#DD8452")
    ax.set_title("Null/Blank Rate: business_name & business_address")
    ax.set_ylabel("% blank or NaN")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    # Panel 3: country distribution (stacked-ish, side by side per source)
    ax = axes[1, 0]
    all_countries = sorted(set().union(*[set(vc.index) for vc in country_counts.values()])) if country_counts else []
    width = 0.25
    x = range(len(all_countries))
    for i, (name, vc) in enumerate(country_counts.items()):
        heights = [vc.get(c, 0) for c in all_countries]
        ax.bar([xi + i * width for xi in x], heights, width=width, label=name)
    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(all_countries, rotation=20)
    ax.set_title("Country Distribution per Source")
    ax.set_ylabel("Row count")
    ax.legend()

    # Panel 4: match count distribution
    ax = axes[1, 1]
    if len(n_matches) > 0:
        dist = n_matches.value_counts().sort_index()
        # cap display at, say, top 10 categories + "10+" bucket for readability
        if len(dist) > 10:
            top = dist.iloc[:10]
            rest = dist.iloc[10:].sum()
            plot_dist = pd.concat([top, pd.Series({"10+": rest})])
        else:
            plot_dist = dist
        ax.bar(plot_dist.index.astype(str), plot_dist.values, color="#64B5CD")
        ax.set_title("Distribution of Match Counts per Source1 Entity")
        ax.set_xlabel("Number of matches")
        ax.set_ylabel("Number of Source1 entities")
    else:
        ax.text(0.5, 0.5, "No ground truth data", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "eda_charts.png", dpi=150)
    log("\nSaved charts to eda_charts.png")

    # ================= write report =================
    (OUT_DIR / "eda_report.txt").write_text("\n".join(lines), encoding="utf-8")
    log("Saved full report to eda_report.txt")


if __name__ == "__main__":
    main()
