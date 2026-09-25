# Business Entity Resolution: Plan v1

## 1. What we are building

Given Source 1 (the deduplicated reference set), find every matching record in Source 2 and Source 3 for each Source 1 entity. Output two TSVs: `candidate_pairs.tsv` (blocking output, what the model actually scores) and `matching_results.tsv` (final matches, the only file scored on the leaderboard). Scored with macro F_0.5 per Source 1 entity, so precision matters twice as much as recall and singletons (no match) are worth a full point when predicted correctly.

Hard constraints: no external data or lookup services, final model must be MIT/Apache-2.0 licensed and at most 8B parameters, output must pass `utils/validate_submission.py`.

## 2. What the data actually looks like

Checked directly against the files in `dataset/`, not just the problem statement:

- Train: 2,206,821 Source 1 rows, 5,034,616 Source 2 rows, 5,285,603 Source 3 rows, one ground truth row per Source 1 entity.
- Test: 1,732,544 Source 1 rows, 4,887,273 Source 2 rows, 5,082,316 Source 3 rows. A naive cross join is on the order of 10^13 pairs, so blocking is not optional, it is the whole ballgame.
- Ground truth match-count distribution per Source 1 entity: 123,247 singletons (5.6%), the rest mostly 2-5 matches (peak at 3), tapering to a handful with up to 11 matches. So most entities do have matches, but a meaningful chunk have none, and the model needs to get both right.
- Country in train is only `US` and `India`. Test adds `France` (259,452 of 1,732,544 Source 1 rows, about 15%) with zero training examples. The pipeline must not special-case country values.
- Addresses are frequently incomplete or blank (saw an India record with an empty `business_address` field entirely).
- Names are not consistently Latin script. Saw an India Source 2 record with the business name written entirely in Devanagari (`राम मार्केटिंग प्राइवेट लिमिटेड`) while other India records use Latin-script English. French test names/addresses will add a second non-English language, though still Latin script with accents.

Implication: normalization has to be script-aware and can't assume English tokens. Cross-script name matches (same business, one record in Devanagari, one transliterated to Latin) are a real but probably rare case, better handled by falling back to address/country signal than solved outright.

## 3. Architecture

Four stages, in order:

1. **Normalization** — lowercase, unicode NFKC normalize, strip punctuation, expand a small hand-built list of legal-suffix abbreviations (Corp/Corporation, Ltd/Limited, Pvt/Private, & vs and, Rd/Road, St/Street, etc.), tokenize. No external transliteration services; if we use a deterministic local library like `unidecode` for script folding, document it explicitly in the methodology writeup so it's auditable as "local algorithmic normalization," not an external lookup.
2. **Blocking / candidate generation** — multi-pass, indexed lookups, union of candidates across passes so one weak blocking key doesn't kill recall. Candidate keys:
   - Normalized name token overlap (inverted index: token -> list of entity_ids, take union of postings for a record's tokens, intersect against other same-country-ish sets isn't required since country is open-set, but comparing within country label as one signal is fine since it's just using the given field, not hardcoding logic to it).
   - Phonetic key on the business name (soundex/metaphone-style) to catch typos.
   - Address token overlap, especially any digit sequences (street numbers, PIN/ZIP codes) which are high-precision blocking signals when present.
   - Character n-gram (e.g. trigram) MinHash/LSH as a fallback pass for near-duplicates that share no exact tokens (typos, transliteration).
   Cap candidates per Source 1 entity (e.g. top-K by cheap token-overlap or TF-IDF cosine score) so the pairwise feature stage stays tractable. This capped, final list is exactly `candidate_pairs.tsv`.
3. **Pairwise matching model** — a gradient-boosted tree classifier (LightGBM or XGBoost, both MIT/BSD licensed and nowhere near 8B parameters) trained on labeled pairs from the blocking candidates: positives from ground truth, negatives sampled from candidates that aren't true matches. Features per (S1, candidate) pair:
   - Name: token Jaccard, character n-gram Jaccard/cosine, Levenshtein ratio, Jaro-Winkler, length difference, common-prefix length.
   - Address: token overlap, digit/number-sequence overlap, Levenshtein ratio on normalized address.
   - Country exact-match flag (used as a feature, not a filter, so France just becomes an unseen category the tree splits on naturally).
   - Blocking meta-features: which blocking pass(es) surfaced this candidate, candidate's rank within its S1 group.
   Start with logistic regression as a fast sanity baseline, move to gradient boosting once the feature pipeline is validated.
4. **Grouping and thresholding** — F_0.5 is macro-averaged per Source 1 entity and precision-weighted, so the decision rule isn't a single global probability cutoff by default. Plan: tune a probability threshold on a held-out validation split by directly optimizing macro F_0.5 (not accuracy or AUC), and check whether a per-entity "top match only if above threshold" vs "all matches above threshold" policy scores better, since some S1 entities have multiple true matches and some have none. Empty prediction is the correct call for predicted singletons.

## 4. Validation strategy

No test ground truth, so carve a held-out split from `train_ground_truth.tsv` (stratified by country and by singleton vs. non-singleton so the split isn't skewed) and score it locally using the exact same macro F_0.5 formula as the leaderboard. Track two numbers separately:

- **Blocking recall ceiling**: fraction of true matches that actually appear in `candidate_pairs.tsv` for the validation split. This is the upper bound on achievable recall and the first thing to check if the final score is bad.
- **End-to-end F_0.5**: after the classifier and threshold, on the same split.

## 5. Build order

1. Normalization module + unit checks on a handful of hand-picked noisy examples pulled from the real data (abbreviation, script mismatch, missing address).
2. Blocking module, run on train, measure recall ceiling and candidate-set size (reduction ratio) before writing a single line of model code. If recall ceiling is low, fix blocking before touching the model.
3. Feature extraction over candidate pairs.
4. Train classifier on train split, validate on held-out split, tune threshold for macro F_0.5.
5. Run the full pipeline on test, produce `candidate_pairs.tsv` and `matching_results.tsv`, validate with `utils/validate_submission.py`.
6. Fill in `Documentation_template.md` and assemble the submission zip per the required structure.

## 6. Repo layout

Pipeline code lives under `student_resource/code/business_entity_resolution/src/` to match the required submission structure directly, rather than building it somewhere else and moving it later. `requirements.txt` pins pandas, numpy, and whichever of lightgbm/xgboost we settle on, kept as small as the task allows.

## 7. Open questions to confirm before going further

- Whether a local transliteration library (`unidecode` or similar) counts as acceptable normalization or crosses into the kind of tooling the rules mean to exclude. Treat as excluded by default unless confirmed, and rely on address/country signal for cross-script matches instead.
- Target candidate cap per Source 1 entity (affects both recall ceiling and compute time for feature extraction across ~1.7M-2.2M Source 1 entities). Decide empirically from the recall-ceiling-vs-candidate-set-size curve on train, not up front.
