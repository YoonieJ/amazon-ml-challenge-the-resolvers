# Business Entity Resolution: Plan v1

## 1. What we are building

Given Source 1 (the deduplicated reference set), we need to find every matching record in Source 2 and Source 3 for each Source 1 entity. We output two TSVs: `candidate_pairs.tsv` (blocking output, what our model actually scores) and `matching_results.tsv` (final matches, the only file scored on the leaderboard). We're scored with macro F_0.5 per Source 1 entity, so precision matters twice as much as recall and singletons (no match) are worth a full point when we predict them correctly.

Hard constraints: no external data or lookup services, our final model must be MIT/Apache-2.0 licensed and at most 8B parameters, our output must pass `utils/validate_submission.py`.

## 2. What we found in the data

We checked this directly against the files in `dataset/`, not just the problem statement:

- Train: 2,206,821 Source 1 rows, 5,034,616 Source 2 rows, 5,285,603 Source 3 rows, one ground truth row per Source 1 entity.
- Test: 1,732,544 Source 1 rows, 4,887,273 Source 2 rows, 5,082,316 Source 3 rows. A naive cross join is on the order of 10^13 pairs, so blocking is not optional for us, it is the whole ballgame.
- Ground truth match-count distribution per Source 1 entity: 123,247 singletons (5.6%), the rest mostly 2-5 matches (peak at 3), tapering to a handful with up to 11 matches. So most entities do have matches, but a meaningful chunk have none, and we need to get both right.
- Country in train is only `US` and `India`. Test adds `France` (259,452 of 1,732,544 Source 1 rows, about 15%) with zero training examples. We must not special-case country values anywhere in the pipeline.
- Addresses are frequently incomplete or blank (we found an India record with an empty `business_address` field entirely).
- Names are not consistently Latin script. We found an India Source 2 record with the business name written entirely in Devanagari (`राम मार्केटिंग प्राइवेट लिमिटेड`) while other India records use Latin-script English. French test names/addresses will add a second non-English language, though still Latin script with accents.
- We checked the ground truth for a one-to-one constraint: across all 7,638,365 matched (S2/S3) id occurrences in `train_ground_truth.tsv`, every single one belongs to exactly one Source 1 entity, zero exceptions. So a given S2/S3 record legitimately matches at most one Source 1 business, always. This is a real constraint we can enforce, not an assumption.

Implication: our normalization has to be script-aware and can't assume English tokens. Cross-script name matches (same business, one record in Devanagari, one transliterated to Latin) are a real but probably rare case, we'll handle it by falling back to address/country signal rather than solving it outright.

## 3. Architecture

We're building four stages, in order:

1. **Normalization**: we lowercase, unicode NFKC normalize, strip punctuation, expand a small hand-built list of legal-suffix abbreviations (Corp/Corporation, Ltd/Limited, Pvt/Private, & vs and, Rd/Road, St/Street, etc.), and tokenize. We won't use external transliteration services; if we use a deterministic local library like `unidecode` for script folding, we'll document it explicitly in the methodology writeup so it's auditable as local algorithmic normalization, not an external lookup.
2. **Blocking / candidate generation**: multi-pass, indexed lookups, union of candidates across passes so one weak blocking key doesn't kill our recall. Our candidate keys:
   - Normalized name token overlap (inverted index: token -> list of entity_ids, we take the union of postings for a record's tokens; comparing within the country label as one signal is fine since we're just using the given field, not hardcoding logic to specific values).
   - Phonetic key on the business name (soundex/metaphone-style) to catch typos.
   - Address token overlap, especially any digit sequences (street numbers, PIN/ZIP codes), which are high-precision blocking signals when present.
   - Character n-gram (e.g. trigram) MinHash/LSH as a fallback pass for near-duplicates that share no exact tokens (typos, transliteration).
   We cap candidates per Source 1 entity (e.g. top-K by cheap token-overlap or TF-IDF cosine score) so our pairwise feature stage stays tractable. This capped, final list is exactly `candidate_pairs.tsv`.
3. **Pairwise matching model**: a gradient-boosted tree classifier (LightGBM or XGBoost, both MIT/BSD licensed and nowhere near 8B parameters) trained on labeled pairs from our blocking candidates: positives from ground truth, negatives sampled from candidates that aren't true matches. Features we compute per (S1, candidate) pair:
   - Name: token Jaccard, character n-gram Jaccard/cosine, Levenshtein ratio, Jaro-Winkler, length difference, common-prefix length.
   - Address: token overlap, digit/number-sequence overlap, Levenshtein ratio on normalized address, an explicit "address missing" flag rather than letting a blank string silently score as zero-similarity.
   - Country exact-match flag (used as a feature, not a filter, so France just becomes an unseen category the tree splits on naturally).
   - Blocking meta-features: which blocking pass(es) surfaced this candidate, candidate's rank within its S1 group, block cardinality (how many other candidates share this candidate's block).
   - Mutual nearest neighbor flag: is this candidate also S1's top-scoring option, and is S1 also the candidate's top-scoring option among everything that blocked to it.
   We'll start with logistic regression as a fast sanity baseline, then move to gradient boosting once we've validated the feature pipeline.
4. **Grouping, thresholding, and global assignment**: F_0.5 is macro-averaged per Source 1 entity and precision-weighted, so our decision rule isn't a single global probability cutoff by default. We'll calibrate the classifier's probabilities (isotonic or Platt scaling), then tune a threshold on a held-out validation split by directly optimizing macro F_0.5 (not accuracy or AUC). Because we confirmed every S2/S3 record matches at most one Source 1 entity, we'll enforce that globally as a final assignment step: sort all above-threshold (S1, candidate) pairs by score descending, and once an S2/S3 id is claimed by one S1, we drop it from every other S1's list. This is a precision lever the per-pair classifier can't provide on its own. Empty prediction is the correct call for predicted singletons.

## 4. Validation strategy

We have no test ground truth, so we carve a held-out split from `train_ground_truth.tsv` by Source 1 entity id (not by row, to avoid leakage), stratified by country and by singleton vs. non-singleton, and score it locally using the exact same macro F_0.5 formula as the leaderboard. We track two numbers separately:

- **Blocking recall ceiling**: fraction of true matches that actually appear in `candidate_pairs.tsv` for the validation split. This is our upper bound on achievable recall and the first thing we check if the final score is bad.
- **End-to-end F_0.5**: after the classifier, threshold, and global assignment, on the same split.

We'll also run a second, deliberate stress test: hold out one entire country (say, all of India) from training and validate only on it, to approximate what generalizing to France (zero training coverage) will actually look like. This tells us which features are secretly country-specific versus genuinely script/format agnostic.

## 5. Build order

1. Normalization module, with a handful of hand-picked noisy examples pulled from the real data (abbreviation, script mismatch, missing address) as a sanity check.
2. Blocking module, run on train, measure recall ceiling and candidate-set size (reduction ratio) before we write a single line of model code. If recall ceiling is low, we fix blocking before touching the model.
3. Feature extraction over candidate pairs.
4. Baseline logistic regression to sanity-check the features, then the gradient-boosted classifier, trained on the group-aware train split, validated on the held-out split.
5. Calibrate probabilities, tune the threshold for macro F_0.5, apply the global one-to-one assignment.
6. Country-holdout stress test, targeted error analysis (bucket false positives/negatives by country and by missing-address), and feature adjustments based on what we actually see, not guesses.
7. Run the full pipeline on test, produce `candidate_pairs.tsv` and `matching_results.tsv`, validate with `utils/validate_submission.py`.
8. Upload to the leaderboard, compare our public score against our local validation score. A large gap tells us our validation split doesn't represent the real test distribution, most likely around France.
9. Iterate on whichever stage is actually capping the score, blocking recall or classifier precision, not both at once.
10. Fill in `Documentation_template.md` and assemble the submission zip per the required structure.

## 6. Where we can push F_0.5 further

Roughly ranked by payoff versus effort:

1. **Enforce the one-to-one constraint globally** (already folded into section 3 above). Since we confirmed no S2/S3 record legitimately matches two different S1 entities, resolving conflicts with a global greedy/Hungarian-style assignment directly removes a class of false positive our per-pair classifier can't see. Given the precision-heavy metric, this is likely our single highest-leverage move.
2. **Mutual nearest neighbor as a feature**, feeding into the assignment above rather than only applied after.
3. **The country-holdout stress test** described in section 4, specifically to catch features that quietly depend on US/India-specific formatting (PIN code shape, ZIP format) instead of generalizing.
4. **Cross-script names.** A local, deterministic transliteration feature (pending the open question below) plus leaning harder on address/PIN exact-match when name similarity is unreliable.
5. **Probability calibration** before thresholding and before the global assignment step, since uncalibrated GBM scores can rank correctly but be badly scaled.
6. **Error-analysis-driven iteration**: pull actual false positives/negatives from validation, bucket by country and by missing-address, and let what we see there drive the next feature, not blind hyperparameter tuning.
7. **Address-missing as an explicit feature** so the model can learn to weight name signal more heavily specifically when address is absent.

We're deliberately skipping deep learning or embedding/transformer models for the matching step. Gradient boosting on engineered features is the standard, well-understood approach for this problem shape, it's easier for us to debug, it trivially satisfies the license/parameter constraint, and we think the one-to-one constraint and mutual nearest neighbor checks above will move the score more than a fancier model would. We'll revisit only if the GBM plateaus below our blocking recall ceiling.

## 7. Repo layout

We're putting pipeline code under `student_resource/code/business_entity_resolution/src/` to match the required submission structure directly, rather than building it somewhere else and moving it later. Our `requirements.txt` pins pandas, numpy, and whichever of lightgbm/xgboost we settle on, kept as small as the task allows.

## 8. Open questions we need to confirm before going further

- Whether a local transliteration library (`unidecode` or similar) counts as acceptable normalization or crosses into the kind of tooling the rules mean to exclude. We're treating it as excluded by default unless confirmed, and relying on address/country signal for cross-script matches instead.
- Our target candidate cap per Source 1 entity (affects both recall ceiling and compute time for feature extraction across ~1.7M-2.2M Source 1 entities). We'll decide this empirically from the recall-ceiling-vs-candidate-set-size curve on train, not up front.
