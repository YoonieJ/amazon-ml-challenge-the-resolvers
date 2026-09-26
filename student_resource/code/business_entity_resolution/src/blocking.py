"""Blocking / candidate generation (plan-v1.md section 3.2): multi-pass,
indexed lookups, union of candidates across passes so one weak blocking key
doesn't kill recall.

Callers pass a single "other" dataframe that is the concatenation of
source2 + source3 (their entity_ids already carry the S2-/S3- prefix, so
they stay globally unique) -- that way the top-K cap in generate_candidates
applies once per Source-1 entity across both sources, not once per source.
"""
import tempfile

import duckdb
import pandas as pd

from normalize import tokenize

_SOUNDEX_CODES = {c: "1" for c in "BFPV"}
_SOUNDEX_CODES.update({c: "2" for c in "CGJKQSXZ"})
_SOUNDEX_CODES.update({c: "3" for c in "DT"})
_SOUNDEX_CODES["L"] = "4"
_SOUNDEX_CODES.update({c: "5" for c in "MN"})
_SOUNDEX_CODES["R"] = "6"
_SOUNDEX_SILENT = set("HW")

# ponytail: no trigram/near-duplicate pass here. A plain trigram inverted
# index blows up at this scale -- measured ~220M exploded rows for the
# 10.3M-row train source2+source3 name column alone, which pushed a test run
# past 20GB RSS and never finished. The plan calls for MinHash/LSH here
# specifically, not a plain inverted index; add that (e.g. via `datasketch`)
# only if the recall-ceiling measurement on these 3 passes shows a real gap
# that near-duplicate matching would close.
_PASSES = ["name_token", "soundex", "addr_token", "name_bigram", "addr_bigram"]

# bigram passes have far more distinct keys than unigram passes at full
# scale, so they get a lower postings cap to avoid a disk/memory blowup
# while still catching the near-duplicate matches unigrams miss.
_BIGRAM_POSTINGS_DIVISOR = 5


def soundex(word: str) -> str:
    """Classic Soundex phonetic code. Local + deterministic, no external
    phonetic library (plan-v1.md section 8 treats those as out of scope).
    """
    letters = [c for c in word.upper() if c.isalpha()]
    if not letters:
        return ""
    first = letters[0]
    digits, prev = [], _SOUNDEX_CODES.get(first, "")
    for c in letters[1:]:
        if c in _SOUNDEX_SILENT:
            continue  # transparent: doesn't reset prev, unlike a vowel
        code = _SOUNDEX_CODES.get(c, "")  # "" for vowels
        if code and code != prev:
            digits.append(code)
        prev = code
    return (first + "".join(digits) + "000")[:4]


def _prefixed(countries, token_lists):
    """Scope every blocking key to its row's country (plan-v1.md section
    3.2): true matches always share a country, and this keeps a generic
    token like "corporation" from joining across the whole US+India corpus.
    """
    return [
        [f"{country}|{tok}" for tok in tokens]
        for country, tokens in zip(countries, token_lists)
    ]


def _bigrams(token_lists):
    """Adjacent-token-pair keys -- far rarer per-key than single tokens,
    so they survive the max_postings cap that discards common single words.
    """
    return [
        [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
        for toks in token_lists
    ]


def _build_keys(df, name_col="business_name", address_col="business_address"):
    name_tokens = df[name_col].map(tokenize)
    addr_tokens = df[address_col].map(tokenize)
    return pd.DataFrame({
        "entity_id": df["entity_id"],
        "name_token": _prefixed(df["country"], name_tokens),
        "soundex": _prefixed(
            df["country"], name_tokens.map(lambda toks: [c for t in toks if (c := soundex(t))])
        ),
        "addr_token": _prefixed(df["country"], addr_tokens),
        "name_bigram": _prefixed(df["country"], _bigrams(name_tokens)),
        "addr_bigram": _prefixed(df["country"], _bigrams(addr_tokens)),
    })


def _pass_sql(pass_name, max_postings):
    """One blocking pass as SQL: explode a key-list column, drop keys with
    more than `max_postings` postings on either side (hard join-blowup
    safety net), join on the surviving keys, weight each match by 1/df_other
    (IDF: a key held by fewer other-side entities is stronger evidence).
    """
    return f"""
    SELECT l.entity_id_s1, r.entity_id_other, 1.0 / r.df_other AS weight
    FROM (
        SELECT entity_id AS entity_id_s1, key
        FROM (SELECT entity_id, unnest({pass_name}) AS key FROM s1_keys)
        QUALIFY count(*) OVER (PARTITION BY key) <= {max_postings}
    ) l
    JOIN (
        SELECT entity_id AS entity_id_other, key,
               count(*) OVER (PARTITION BY key) AS df_other
        FROM (SELECT entity_id, unnest({pass_name}) AS key FROM other_keys)
        QUALIFY df_other <= {max_postings}
    ) r USING (key)
    """


def generate_candidates(s1_df, other_df, top_k=20, max_postings=20000):
    """Union candidate (S1, other) pairs across every blocking pass, capped
    to the top `top_k` other-side ids per S1 entity by an IDF-weighted
    token-overlap score.

    Processed per-country (every key is already country-scoped, so a
    US key can never match an India key anyway), and each country's
    top_k ranking is finalized before moving to the next country -- this
    keeps every intermediate join and the final combine step small enough
    to avoid the disk/memory blowups seen processing the full combined
    corpus at once.
    """
    s1_keys = _build_keys(s1_df)
    other_keys = _build_keys(other_df)
    countries = sorted(set(s1_df["country"]) | set(other_df["country"]))

    con = duckdb.connect()
    con.execute(f"SET temp_directory='{tempfile.gettempdir()}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='100GB'")
    con.execute("SET max_temp_directory_size='150GiB'")

    with tempfile.TemporaryDirectory() as tmpdir:
        country_results = []
        for country in countries:
            s1_keys_c = s1_keys[s1_df["country"] == country]
            other_keys_c = other_keys[other_df["country"] == country]
            if len(s1_keys_c) == 0 or len(other_keys_c) == 0:
                continue
            con.register("s1_keys", s1_keys_c)
            con.register("other_keys", other_keys_c)
            part_files = []
            for pass_name in _PASSES:
                pass_cap = (
                    max(max_postings // _BIGRAM_POSTINGS_DIVISOR, 1)
                    if pass_name.endswith("_bigram")
                    else max_postings
                )
                out_path = f"{tmpdir}/{country}_{pass_name}.parquet"
                con.execute(f"""
                    COPY (
                        SELECT entity_id_s1, entity_id_other, sum(weight) AS weight
                        FROM ({_pass_sql(pass_name, pass_cap)})
                        GROUP BY entity_id_s1, entity_id_other
                    ) TO '{out_path}' (FORMAT PARQUET)
                """)
                part_files.append(out_path)
            con.unregister("s1_keys")
            con.unregister("other_keys")

            union_query = " UNION ALL ".join(
                f"SELECT * FROM read_parquet('{p}')" for p in part_files
            )
            country_out = f"{tmpdir}/{country}_final.parquet"
            con.execute(f"""
                COPY (
                    WITH pass_pairs AS ({union_query}),
                    scored AS (
                        SELECT entity_id_s1 AS source1_entity_id,
                               entity_id_other AS candidate_entity_id,
                               sum(weight) AS score
                        FROM pass_pairs
                        GROUP BY entity_id_s1, entity_id_other
                    )
                    SELECT source1_entity_id, candidate_entity_id, score
                    FROM scored
                    QUALIFY row_number() OVER (PARTITION BY source1_entity_id ORDER BY score DESC) <= {top_k}
                ) TO '{country_out}' (FORMAT PARQUET)
            """)
            country_results.append(country_out)

        final_union = " UNION ALL ".join(
            f"SELECT * FROM read_parquet(\'{p}\')" for p in country_results
        )
        result = con.execute(f"SELECT * FROM ({final_union})").df()
    return result


def load_ground_truth_pairs(path) -> pd.DataFrame:
    """Explode train_ground_truth.tsv's comma-joined match lists into one
    row per (source1_entity_id, candidate_entity_id) true match. Singletons
    (empty match list) contribute no rows.
    """
    gt = pd.read_csv(path, sep="\t", dtype=str)
    truth = gt.assign(candidate_entity_id=gt["matched_entity_ids"].fillna("").str.split(","))
    truth = truth.explode("candidate_entity_id")
    return truth.loc[truth["candidate_entity_id"] != "", ["source1_entity_id", "candidate_entity_id"]]


def recall_ceiling(candidate_pairs, truth_pairs) -> float:
    """Fraction of true matches in `truth_pairs` that appear in
    `candidate_pairs` -- the upper bound on achievable recall (plan-v1.md
    section 4).
    """
    have = candidate_pairs[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
    merged = truth_pairs.merge(have, how="left", indicator=True)
    return (merged["_merge"] == "both").mean()


def demo() -> None:
    """Sanity checks on tiny synthetic entities (plan-v1.md build order step 2)."""
    assert soundex("Robert") == soundex("Rupert") == "R163"
    assert soundex("") == ""

    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "100 Main St", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Best Traders", "business_address": "5 Oak Rd", "country": "India"},
    ])
    other_df = pd.DataFrame([
        # true match: name + address overlap, same country
        {"entity_id": "S2-1", "business_name": "Acme Corporation", "business_address": "100 Main Street", "country": "US"},
        # same name as S1-1 but wrong country -- must NOT match
        {"entity_id": "S2-2", "business_name": "Acme Corp", "business_address": "999 Nowhere Ave", "country": "India"},
        # true match: partial name overlap + full address overlap
        {"entity_id": "S3-3", "business_name": "Best Trading Co", "business_address": "5 Oak Road", "country": "India"},
        # no overlap with anything
        {"entity_id": "S2-4", "business_name": "Totally Unrelated Biz", "business_address": "1 Zzz Blvd", "country": "US"},
    ])
    result = generate_candidates(s1_df, other_df, top_k=5)
    pairs = set(zip(result["source1_entity_id"], result["candidate_entity_id"]))
    assert pairs == {("S1-1", "S2-1"), ("S1-2", "S3-3")}, pairs

    # top_k capping: the exact match must survive, and a zero-overlap
    # candidate must never appear, however many candidates share partial
    # overlap.
    s1_cap_df = pd.DataFrame([
        {"entity_id": "S1-9", "business_name": "Gamma Corp", "business_address": "1 Elm St", "country": "US"},
    ])
    other_cap_df = pd.DataFrame([
        {"entity_id": "S2-9a", "business_name": "Gamma Corp", "business_address": "1 Elm St", "country": "US"},
        {"entity_id": "S2-9b", "business_name": "Gamma Corp", "business_address": "2 Oak Ave", "country": "US"},
        {"entity_id": "S2-9c", "business_name": "Gamma Traders", "business_address": "9 Zzz Blvd", "country": "US"},
        {"entity_id": "S2-9d", "business_name": "Unrelated Biz", "business_address": "9 Zzz Blvd", "country": "US"},
    ])
    capped = generate_candidates(s1_cap_df, other_cap_df, top_k=2)
    assert len(capped) <= 2
    assert "S2-9a" in set(capped["candidate_entity_id"])
    assert "S2-9d" not in set(capped["candidate_entity_id"])

    truth_pairs = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-2"],
        "candidate_entity_id": ["S2-1", "S3-3", "S2-999"],  # S2-999 is a miss
    })
    ceiling = recall_ceiling(result, truth_pairs)
    assert abs(ceiling - 2 / 3) < 1e-9, ceiling

    print("blocking.py demo: all checks passed")


if __name__ == "__main__":
    demo()
