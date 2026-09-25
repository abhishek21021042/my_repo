# Business Entity Resolution — Detailed implementation plan

## 1. Purpose

This document converts the precision-first PRD into an executable plan for a four-person team. The system must link every Source 1 business to zero, one, or many records in Source 2 and Source 3, using only the supplied challenge data.

The competition score is macro-averaged `F_0.5`, so false matches are more damaging than missed matches. The implementation therefore follows one rule:

> Retrieve broadly, score carefully, and abstain whenever the evidence is weak.

A perfect score cannot be guaranteed because noisy names and incomplete addresses can be genuinely ambiguous. The practical objective is to maximize validation `F_0.5`, protect precision, and keep the full pipeline reproducible.

---

## 2. Fixed challenge requirements

- Input and output files are tab-separated; always read and write with an explicit tab separator.
- Source 1 is deduplicated and is the reference side of every comparison.
- Each Source 1 record may have zero, one, or multiple matches in Source 2 and Source 3.
- Training countries are US and India; test also contains France.
- Country handling must be dynamic. Never restrict code to a fixed `{US, India}` vocabulary.
- Every test Source 1 ID must appear exactly once in both output files.
- `matching_results.tsv` is scored.
- `candidate_pairs.tsv` must contain the exact final candidate set sent to the matching model.
- Every predicted match must also be present in that entity's candidate list.
- Predicted IDs may only be valid test IDs from Source 2 or Source 3.
- Final models must be no larger than 8B parameters and have an allowed MIT or Apache 2.0 license.
- External business databases, geocoding, search APIs, or internet-based enrichment are prohibited.

## 3. Deliverables

1. A reproducible training pipeline.
2. A reproducible test inference pipeline.
3. `output/matching_results.tsv`.
4. `output/candidate_pairs.tsv`.
5. Entity-level validation and threshold-search reports.
6. Error-analysis slices for singleton, multi-match, country, and difficult negative cases.
7. A final submission archive with code, pinned dependencies, outputs, and methodology documentation.

---

## 4. Success criteria and engineering gates

The numbers below are internal go/no-go targets, not claims about achieved performance.

| Gate | Provisional target | Reason |
|---|---:|---|
| Candidate recall | >= 99.5% | Protect recall ceiling |
| Positive pair coverage | >= 99.5% | All validation positives |
| Output S1 coverage | 100% | Submission requirement |
| Invalid referenced IDs | 0 | Submission requirement |
| Match outside candidates | 0 | Pipeline consistency |
| Duplicate list IDs | 0 | Submission requirement |
| Reproducible rerun | Exact | Auditability |

The primary model-selection metric is validation macro `F_0.5`. Pairwise ROC-AUC is diagnostic only and must not decide the winning system.

Secondary metrics:

- Entity-level precision, recall, and `F_0.5`.
- Pair-level precision and recall.
- Singleton accuracy.
- Candidate recall by source, country, and positive-set size.
- Mean, p95, and maximum candidate count per Source 1 record.
- Runtime and peak memory for each pipeline stage.

---

## 5. End-to-end architecture

```text
TSV files
  -> schema and ID validation
  -> raw-field preservation
  -> multi-view normalization
  -> dynamic same-country indexes
  -> multi-channel candidate retrieval
  -> candidate union, deduplication, and safe pruning
  -> pair feature generation
  -> calibrated pair classifier
  -> hard-conflict checks and entity-level decision rules
  -> abstention or zero-to-many matches
  -> exact candidate and match TSV serialization
  -> official validator
```

The system has two deliberately different operating points:

- **Blocking stage:** optimized for recall. It may produce extra pairs.
- **Decision stage:** optimized for precision. It accepts only well-supported pairs.

No match is created outside the final candidate set. The candidate set is frozen before model inference and serialized from the same in-memory object used for scoring.

## 6. Recommended repository structure

```text
business_entity_resolution/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   └── precision.yaml
├── src/
│   ├── io_utils.py
│   ├── validate_data.py
│   ├── normalize.py
│   ├── parse_fields.py
│   ├── split.py
│   ├── blocking.py
│   ├── build_pairs.py
│   ├── features.py
│   ├── train_matcher.py
│   ├── calibrate.py
│   ├── decide.py
│   ├── score.py
│   ├── infer.py
│   └── write_outputs.py
├── tests/
│   ├── test_normalize.py
│   ├── test_blocking.py
│   ├── test_score.py
│   └── test_outputs.py
├── artifacts/
│   ├── models/
│   ├── vectorizers/
│   ├── thresholds/
│   └── reports/
└── output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

Generated artifacts should include a configuration snapshot, random seed, dependency versions, feature list, split IDs, model file checksum, and threshold file. Raw data must remain unchanged.

---

## 7. Phase 0 — Data audit and contracts

### 7.1 Load safely

```python
import pandas as pd

def read_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
```

Preserve all fields as strings. Create normalized columns beside raw columns; never overwrite source values.

### 7.2 Validate every source file

For each file, assert:

- Exact required columns are present.
- `entity_id` is non-empty and unique.
- Prefix agrees with the file: `S1-`, `S2-`, or `S3-`.
- Country is retained as a string, including unseen labels.
- Empty names and addresses are counted, not silently discarded.
- Duplicate rows and duplicate normalized records are reported.
- Leading/trailing whitespace and tab/newline corruption are measured.

For ground truth, assert:

- Every `source1_entity_id` exists in training Source 1.
- Every listed match exists in training Source 2 or Source 3.
- No duplicate match ID occurs within a row.
- Empty match lists remain true empty lists.
- Training Source 1 coverage is complete or the missing rows are explicitly reported.

### 7.3 Produce an audit report

Record row counts, country counts, missing-field rates, positive matches per Source 1, singleton rate, source-wise positive counts, and exact/near-duplicate rates. These values must come from the actual files; do not hard-code them.

---

## 8. Phase 1 — Leakage-safe validation

Split by Source 1 entity, never by generated pair. Otherwise, pairs from the same reference entity can leak into both training and validation.

### 8.1 Recommended evaluation design

- Create five deterministic folds of Source 1 IDs.
- Stratify approximately by country, singleton versus non-singleton, and match-count bucket: `0`, `1`, `2–3`, `4+`.
- Build training pairs using only the training-fold entities.
- Build validation candidates against Source 2 and Source 3 without using validation labels.
- Compute final predictions and `F_0.5` at the entity level.
- Keep one fixed fold for rapid iteration; confirm finalists across all folds.

France has no labeled training examples. Do not simulate performance claims for it. Validate country-agnostic normalization and retrieval by hiding one known country during an additional robustness experiment, training on the other country, and testing on the held-out country. This is a stress test, not a substitute for real France labels.

### 8.2 Exact entity-level scorer

For each Source 1 entity:

- If truth and prediction are both empty, score `1.0`.
- If exactly one is empty, score `0.0`.
- Otherwise calculate precision and recall from ID sets, then:

```text
F_0.5 = 1.25 * precision * recall / (0.25 * precision + recall)
```

Macro-average across all Source 1 entities. Add unit tests for empty/empty, empty/non-empty, exact match, partial match, false merge, duplicate input, and multi-match cases.

---

## 9. Phase 2 — Multi-view normalization

Aggressive normalization can accidentally collapse distinct businesses. Keep several representations and let features compare them separately.

### 9.1 Shared transformations

For both name and address:

- Unicode normalize and case-fold.
- Standardize whitespace.
- Replace punctuation with spaces while preserving a second compact alphanumeric view.
- Normalize `&` and `and` into a comparable token.
- Retain digits in a dedicated numeric-token view.
- Generate token, character, and sorted-token representations.
- Preserve the raw text for later features and debugging.

### 9.2 Business-name views

Create at least:

- `name_raw_clean`: conservative cleanup only.
- `name_tokens`: tokenized normalized text.
- `name_no_legal`: legal suffixes removed from a comparison view.
- `name_compact`: letters and digits without spaces.
- `name_sorted`: tokens sorted for word-order variation.
- `name_char`: Unicode normalized text for character n-grams.

Legal suffix removal must be feature-only; never delete the original evidence. Learn observed suffix patterns from training data, while keeping a small reviewed normalization map for obvious variants such as `ltd`/`limited`, `pvt`/`private`, `inc`/`incorporated`, and `llc`.

### 9.3 Address views

Create at least:

- `address_raw_clean`.
- `address_tokens`.
- `address_compact`.
- `address_numbers`: ordered numeric tokens.
- `postal_code`: extracted where pattern confidence is high.
- `state_region`: conservative extraction.
- `city_tokens`: conservative extraction or token subset.
- `unit_tokens`: unit, flat, suite, floor, building, and plot evidence.

Abbreviation expansion should create an alternate view instead of replacing the original. Character n-grams remain important for transliteration, typos, and unseen languages.

### 9.4 Country handling

Create indexes by the country strings observed at runtime:

```python
for country_value in sorted(source1["country"].unique()):
    build_country_partition(country_value)
```

No list of allowed countries belongs in code. Missing country values should enter a separate missing-country partition and require stricter downstream evidence.

---

## 10. Phase 3 — High-recall candidate generation

Generate candidates separately for Source 1-to-Source 2 and Source 1-to-Source 3. Retrieve within the same dynamically discovered country partition by default. Union several independent channels so one type of corruption does not destroy recall.

### 10.1 Candidate channels

1. **Exact normalized keys**
   - Exact `name_compact`.
   - Exact conservative name plus a strong address component.
   - Exact address compact form when sufficiently long and non-generic.

2. **Name character TF-IDF**
   - Word-boundary-aware character n-grams, typically lengths 2–5 or 3–5.
   - Retrieve top candidates independently from Source 2 and Source 3.
   - Fit vectorizers on supplied challenge text only.

3. **Address character TF-IDF**
   - Character n-grams on normalized addresses.
   - Useful for punctuation, ordering, transliteration, and abbreviation noise.

4. **Combined text retrieval**
   - Concatenate weighted name and address views.
   - Prevent a long address from completely drowning the business name.

5. **Rare-token blocks**
   - Shared rare name tokens.
   - Shared rare address tokens.
   - Shared postal code or strong numeric sequence plus partial name evidence.

6. **Numeric-address rescue**
   - Same postal code, building number, or distinctive ordered number pattern.
   - Require at least weak name support to avoid broad apartment-building collisions.

### 10.2 Candidate union and pruning

For every candidate, retain retrieval provenance:

- Which channels retrieved it.
- Rank and score per channel.
- Number of supporting channels.
- Source side: S2 or S3.

Deduplicate the union by `(source1_entity_id, candidate_entity_id)`. Safe pruning may use rank, retrieval score, and minimum evidence, but it must be tuned only after measuring positive coverage.

Start with generous caps, for example top 50 per retrieval channel per target source. Reduce only if candidate recall remains above the chosen gate. These are starting values, not final thresholds.

### 10.3 Blocking evaluation

Report candidate recall as:

```text
number of labeled positive pairs present in candidate set
--------------------------------------------------------
number of labeled positive pairs in ground truth
```

Also report the percentage of Source 1 entities for which all true matches were retrieved. Slice results by country, S2 versus S3, missing name, missing address, singleton status, and match-count bucket.

### 10.4 Candidate artifact invariant

Create the model input pairs first, freeze them, and generate `candidate_pairs.tsv` from that frozen frame. The classifier must score exactly those rows. This avoids a mismatch between audit output and actual inference.

---

## 11. Phase 4 — Pair labels and hard-negative mining

### 11.1 Labels

- Positive pairs come directly from ground truth.
- Negative pairs come from the candidate generator but are not listed as matches for that Source 1 entity.
- Never label an arbitrary cross-country Cartesian product; such negatives are too easy and teach little.

### 11.2 Negative mixture

Use a controlled mixture:

- **Hard name negatives:** similar names, conflicting addresses.
- **Hard address negatives:** similar addresses, conflicting names.
- **Near-duplicate negatives:** high combined retrieval score but not labeled.
- **Same-building negatives:** shared address components with different unit/business evidence.
- **Random in-block negatives:** a smaller background sample.

Cap negatives per Source 1 record so entities with huge candidate lists do not dominate. Use sample weights to preserve difficult negatives and to reflect precision-heavy costs.

### 11.3 Iterative mining

After the first model:

1. Score all training candidates out-of-fold.
2. Collect high-scoring false positives.
3. Categorize the failure pattern.
4. Add those pairs with stronger weight.
5. Retrain and re-evaluate on untouched folds.

Do not mine from the same predictions used to report final validation results without a fresh out-of-fold pass.

---

## 12. Phase 5 — Feature engineering

Use transparent pair features first. They train quickly, expose errors, and support strict decision rules.

### 12.1 Name features

- Character TF-IDF cosine on multiple views.
- Word TF-IDF cosine.
- Token Jaccard and weighted Jaccard.
- Token-set and token-sort similarity.
- Normalized edit similarity.
- Jaro-Winkler similarity.
- Exact compact-name flag.
- Prefix and acronym agreement.
- Legal-suffix agreement/conflict.
- Shared rare-token count.
- Name-length ratio and token-count difference.

### 12.2 Address features

- Character and word TF-IDF cosine.
- Token Jaccard and containment.
- Ordered numeric-token agreement.
- Building or house-number agreement.
- Postal-code exact match, mismatch, and missingness.
- State/region agreement, conflict, and missingness.
- City-token overlap.
- Unit/flat/suite agreement or conflict.
- Address-length ratio.
- Shared rare-token count.

### 12.3 Cross-field and retrieval features

- Country equality and missingness.
- Name-strong/address-weak interaction.
- Address-strong/name-weak interaction.
- Minimum and geometric mean of major name/address scores.
- Retrieval rank and score from every channel.
- Count of channels that retrieved the candidate.
- Candidate source indicator: S2 or S3.
- Missing-name and missing-address patterns.

Never encode candidate IDs or row order as predictive features.

### 12.4 Feature reliability

Each extracted component needs both a value and a presence flag. A missing postal code is not a mismatch. Conflicts should only fire when both sides contain confidently extracted values.

---

## 13. Phase 6 — Matching model

### 13.1 Primary model

Use a gradient-boosted tree classifier over engineered features as the first production model. LightGBM is a suitable implementation after the team records the exact package and license in the final audit.

Training requirements:

- Train with entity-grouped folds.
- Use deterministic seeds and recorded parameters.
- Monitor validation log loss and entity-level `F_0.5`.
- Use early stopping.
- Address class imbalance with sampling and/or weights.
- Save fold models and out-of-fold probabilities.

### 13.2 Optional text reranker

A cross-encoder may rerank only the most competitive candidates if it improves out-of-fold `F_0.5` enough to justify latency. Before adoption, verify the exact model checkpoint's license is MIT or Apache 2.0 and its size is within 8B parameters. Do not infer eligibility from the library license alone.

Safe sequence:

1. Baseline engineered-feature model.
2. Add out-of-fold text-model score as one feature.
3. Recalibrate final probabilities.
4. Keep the reranker only if it improves multiple folds and difficult slices.

No external lookup or externally retrieved business data may enter the text model.

### 13.3 Probability calibration

Raw tree probabilities may not be calibrated. Fit Platt scaling or isotonic calibration on out-of-fold predictions only. Compare expected calibration error and, more importantly, threshold stability across folds.

Calibration does not replace threshold search; it makes confidence bands more interpretable.

---

## 14. Phase 7 — Precision-first decision policy

The model outputs pair scores. A separate decision layer converts those scores into zero-to-many matches.

### 14.1 Hard conflicts

Apply a veto only when a field is present and reliably extracted on both records. Candidate vetoes include:

- Different non-empty country labels.
- Conflicting high-confidence postal codes with no exceptional name/address evidence.
- Conflicting primary building numbers when the remaining address is otherwise similar.
- Strongly different names combined with weak address evidence.

Every veto must be validated by checking how many true positive pairs it would remove. A rule that removes meaningful positives is not “hard” enough to ship.

### 14.2 Evidence tiers

Classify candidates by available evidence:

- **Tier A:** strong name and strong address.
- **Tier B:** very strong name, partial address.
- **Tier C:** strong address, partial name.
- **Tier D:** only one usable field.
- **Tier E:** conflicting or extremely sparse evidence.

Tune a separate threshold for each tier only when every tier has enough validation examples. Otherwise use one global threshold plus a stricter sparse-record threshold.

### 14.3 Threshold search

Search thresholds using out-of-fold probabilities and the exact macro `F_0.5` scorer.

Recommended process:

1. Coarse global search from `0.50` to `0.99`.
2. Fine search around the best region.
3. Test stricter thresholds for singletons and sparse evidence.
4. Evaluate fold variance and country/source slices.
5. Select the simplest threshold policy within a small margin of the best mean score.

The threshold is an experimentally selected parameter, not a fixed `0.8` rule.

### 14.4 Margin and ambiguity controls

For each Source 1 record, examine the top scores:

- Accept candidates above the applicable threshold.
- For borderline top candidates, require a minimum margin over the next contradictory candidate.
- If two candidates are near-tied but mutually inconsistent, abstain from the uncertain candidate rather than forcing a merge.
- Permit multiple matches when each pair independently has strong evidence; do not force exactly one match.

### 14.5 Singleton policy

Return an empty list when no candidate clears all checks. Correct abstention on a singleton earns full entity credit; an unsupported match scores zero for that entity.

Use a dedicated “any match?” diagnostic model only if it improves out-of-fold macro `F_0.5`. Otherwise, the calibrated maximum pair score plus evidence rules is easier to audit.

### 14.6 France policy

France records must flow through the same dynamic pipeline. Use language-agnostic character features and numeric evidence. Do not lower the threshold merely because the country is unseen. If calibration appears uncertain, use the validated sparse/unseen-country conservative threshold and abstain on weak evidence.

---

## 15. Phase 8 — Output generation

Create one row for every test Source 1 ID, preserving deterministic order.

`matching_results.tsv` columns:

```text
source1_entity_id	matched_entity_ids
```

`candidate_pairs.tsv` columns:

```text
source1_entity_id	candidate_entity_ids
```

Serialization rules:

- Use tabs between columns.
- Use commas between IDs, without spaces or quoting.
- Write an empty second field for no matches/candidates.
- Remove duplicate IDs while preserving deterministic rank order.
- Assert all output IDs exist in the applicable test source file.
- Assert each predicted match is in that row's candidate list.
- Assert exact Source 1 coverage and uniqueness.
- Run the supplied `utils/validate_submission.py` before every upload.

---

## 16. Four-part team division

Each member owns one production boundary and its tests. Shared schemas and artifact formats are frozen early to avoid integration drift.

### Part 1 — Member 1: Data, normalization, and blocking

**Responsibilities**

- Implement TSV loaders and schema checks.
- Build raw-preserving normalized views.
- Implement S1-to-S2 and S1-to-S3 retrieval channels.
- Measure candidate recall and candidate volume.
- Freeze and serialize the exact candidate model-input frame.

**Deliverables**

- `validate_data.py`, `normalize.py`, `parse_fields.py`, `blocking.py`.
- Candidate parquet/TSV with retrieval provenance.
- Blocking-recall report and missed-positive examples.
- Unit tests for Unicode, empty fields, IDs, and country partitions.

**Acceptance gate**

- Candidate recall reaches the agreed gate across folds and key slices.
- France/unseen country labels run without code changes.
- Candidate output contains only valid S2/S3 IDs.

### Part 2 — Member 2: Features and matching model

**Responsibilities**

- Generate labeled pairs from frozen candidates.
- Build pairwise name, address, numeric, and retrieval features.
- Implement hard-negative sampling and iterative mining.
- Train fold models and produce out-of-fold probabilities.
- Evaluate the optional text reranker only after the baseline is stable.

**Deliverables**

- `build_pairs.py`, `features.py`, `train_matcher.py`.
- Feature dictionary with definitions and missingness behavior.
- Saved fold models, parameters, and out-of-fold scores.
- Feature ablation and false-positive report.

**Acceptance gate**

- Training is reproducible from a config and seed.
- No IDs or labels leak into model features.
- Out-of-fold predictions cover every validation candidate exactly once.

### Part 3 — Member 3: Scoring, calibration, and precision tuning

**Responsibilities**

- Implement the exact entity-level macro `F_0.5` scorer.
- Calibrate model probabilities using out-of-fold data.
- Search global, evidence-tier, and singleton thresholds.
- Design and audit hard conflicts, score margins, and abstention rules.
- Run country, source, match-count, and missing-field error slices.

**Deliverables**

- `score.py`, `calibrate.py`, `decide.py`.
- Threshold JSON/YAML with provenance.
- Cross-fold score report and confidence intervals or fold spread.
- Decision log explaining every rule retained or rejected.

**Acceptance gate**

- Unit tests cover metric edge cases.
- Selected policy is stable across folds.
- Every hard conflict has a measured positive-recall impact.

### Part 4 — Member 4: Integration, inference, and submission

**Responsibilities**

- Build the end-to-end command-line pipeline.
- Pin dependencies and package model artifacts.
- Implement deterministic output writing and all structural assertions.
- Run performance profiling and optimize memory/runtime.
- Produce final outputs, README, methodology document, and zip archive.

**Deliverables**

- `infer.py`, `write_outputs.py`, configs, tests, and run scripts.
- Both final TSV files.
- Reproduction README and pinned environment.
- Validator log and submission archive.

**Acceptance gate**

- Clean environment reproduces outputs from source data.
- Official validator prints `PASS`.
- Archive structure exactly matches the challenge specification.

---

## 17. Interfaces between team members

| Producer | Consumer | Frozen interface |
|---|---|---|
| Member 1 | Members 2, 4 | Candidate schema |
| Member 2 | Member 3 | OOF score schema |
| Member 3 | Member 4 | Decision config |
| Member 4 | Whole team | Run contract |

### 17.1 Candidate schema

Minimum columns:

```text
source1_entity_id
candidate_entity_id
candidate_source
country
retrieval_channels
name_retrieval_score
address_retrieval_score
combined_retrieval_score
best_retrieval_rank
```

### 17.2 Scored-pair schema

Minimum columns:

```text
source1_entity_id
candidate_entity_id
fold
label
raw_score
calibrated_score
evidence_tier
conflict_flags
```

### 17.3 Decision configuration

Store, under version control:

- Global and tier thresholds.
- Margin rule.
- Hard-conflict switches.
- Feature/model artifact version.
- Validation run ID and fold scores.

No threshold should live only inside a notebook.

---

## 18. Five-day execution schedule

### Day 1 — Contracts and baseline

- **Member 1:** Audit files, create normalization v1, implement exact and name TF-IDF blocks.
- **Member 2:** Define pair schema, baseline similarity features, and training harness.
- **Member 3:** Implement exact metric and fixed split generation; unit-test edge cases.
- **Member 4:** Create repository, configuration system, CLI skeleton, and CI smoke tests.

**End-of-day gate:** Baseline candidates, model, entity score, and output writer run on a small deterministic subset.

### Day 2 — Candidate recall and strong baseline

- **Member 1:** Add address, combined-text, rare-token, and numeric rescue channels.
- **Member 2:** Add full feature set and first hard-negative mixture.
- **Member 3:** Produce first out-of-fold threshold curve and singleton analysis.
- **Member 4:** Integrate cached stages and memory/runtime logging.

**End-of-day gate:** Candidate-recall report exists; first full-fold `F_0.5` is reproducible.

### Day 3 — Precision hardening

- **Member 1:** Analyze missed positives; tune caps without unsafe early filters.
- **Member 2:** Mine high-score false positives; run feature ablations.
- **Member 3:** Test calibration, evidence tiers, conflict rules, and score margins.
- **Member 4:** Run end-to-end inference rehearsal and output assertions.

**End-of-day gate:** False-positive categories are measured, and the best precision-first policy beats the baseline on multiple folds.

### Day 4 — Robustness and finalists

- **Member 1:** Finalize unseen-country-safe blocking and normalization tests.
- **Member 2:** Evaluate finalist ensemble or optional reranker if licensed and useful.
- **Member 3:** Confirm finalists across all folds and stress-test country holdout.
- **Member 4:** Profile full data, pin dependencies, and prepare packaging scripts.

**End-of-day gate:** One primary and one fallback configuration are frozen.

### Day 5 — Final training and submission

- Retrain the chosen model on all labeled training entities.
- Fit final vectorizers and calibration artifacts using the approved procedure.
- Run test inference once using the frozen configuration.
- Generate both TSV files from the same frozen candidate frame.
- Run structural assertions and the official validator.
- Manually inspect random empty, single-match, and multi-match rows.
- Build and test the final zip in a clean environment.
- Upload only the validated `matching_results.tsv` to the leaderboard.

**End-of-day gate:** Reproduction succeeds, validator passes, and final artifacts are archived with checksums.

---

## 19. Experiment plan

Change one major component at a time. Every experiment record must include split version, seed, candidate version, feature version, model parameters, decision policy, overall macro `F_0.5`, fold scores, precision, recall, singleton accuracy, candidate recall, runtime, and notes.

Run experiments in this order:

1. Exact-key baseline.
2. Name TF-IDF retrieval plus basic string features.
3. Multi-channel blocking.
4. Full name and address features.
5. Hard-negative mixture.
6. Out-of-fold calibration.
7. Global threshold optimization.
8. Evidence-aware threshold or margin policy.
9. Carefully audited conflict rules.
10. Optional fold ensemble.
11. Optional licensed text reranker.

### Promotion rule

Promote a change only when:

- Mean macro `F_0.5` improves or remains statistically/operationally equivalent with lower complexity.
- Precision does not collapse on singletons or sparse records.
- Candidate recall does not fall below the blocking gate.
- Improvement is not isolated to one fold.
- Runtime remains inside the competition inference limit.

Public leaderboard movement is supporting evidence, not the sole selection criterion. Repeated leaderboard tuning can overfit the public subset.

---

## 20. Compute and storage plan

- Store immutable raw TSV files separately from generated artifacts.
- Cache normalized tables, sparse matrices, candidate pairs, and pair features.
- Use sparse TF-IDF matrices and chunked nearest-neighbor retrieval.
- Partition work by country and target source where possible.
- Avoid full S1-by-S2 or S1-by-S3 Cartesian joins.
- Write large intermediate tables in a columnar format with explicit schemas.
- Record peak RAM, wall-clock time, and candidate counts for capacity planning.
- Use GPU only if a validated reranker needs it; the baseline retrieval and tree model can be CPU-first.
- Keep periodic model and report checkpoints in team-controlled challenge storage.

Before a costly run, execute the same command on a small slice and verify schema, cache keys, and output paths.

## 21. Reproducible command flow

The exact flags can differ, but the final README should expose one linear workflow:

```bash
python -m src.validate_data --data-dir dataset
python -m src.split --train-dir dataset/train --config configs/precision.yaml
python -m src.blocking --split train --config configs/precision.yaml
python -m src.build_pairs --split train --config configs/precision.yaml
python -m src.train_matcher --config configs/precision.yaml
python -m src.calibrate --config configs/precision.yaml
python -m src.score --split validation --config configs/precision.yaml
python -m src.infer --test-dir dataset/test --config configs/precision.yaml
python -m src.write_outputs --config configs/precision.yaml
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Every command must fail loudly on missing files, schema mismatches, stale artifact versions, or configuration incompatibility.

---

## 22. Precision-focused error analysis

Review false positives before false negatives because the metric emphasizes precision.

For each false positive, capture:

- Raw and normalized name/address on both sides.
- Candidate retrieval channels and ranks.
- Top feature values.
- Model and calibrated scores.
- Applied threshold, margin, and conflict rules.
- Error category.

Suggested categories:

- Same generic name, different business.
- Same building, different business.
- Same chain or branch confusion.
- Unit, floor, or house-number conflict.
- Postal or region conflict.
- Legal suffix caused over-normalization.
- Address missing on one side.
- Name missing on one side.
- Transliteration or language issue.
- Near-tied competing candidates.
- Possible annotation ambiguity.

Convert repeated categories into features or carefully measured rules. Do not create one-off ID-specific exceptions.

## 23. Risk register

### Risk: Blocking misses true matches

**Mitigation:** Union independent retrieval channels, keep generous initial caps, inspect every missed validation positive, and track all-match entity coverage.

### Risk: Over-normalization creates false merges

**Mitigation:** Preserve raw and alternate views, use legal-suffix removal only as a feature, and require cross-field support.

### Risk: Pair classifier looks strong but entity score is weak

**Mitigation:** Select models and thresholds using exact macro `F_0.5`, including singleton rows.

### Risk: Validation leakage

**Mitigation:** Split Source 1 entities before candidate pair creation; generate out-of-fold predictions for calibration and mining.

### Risk: France distribution shift

**Mitigation:** Dynamic country partitions, Unicode character features, no fixed country one-hot vocabulary, country-holdout stress test, conservative abstention.

### Risk: Candidate file differs from model input

**Mitigation:** Freeze one candidate frame, score it directly, and serialize the audit file from that same object.

### Risk: Public leaderboard overfitting

**Mitigation:** Prefer cross-fold stability, log every submission, and limit changes based solely on public score.

### Risk: License non-compliance

**Mitigation:** Record exact library/model checkpoint, version, source, parameter count, and license before adoption. Reject ambiguous checkpoints.

### Risk: Runtime or memory failure

**Mitigation:** Sparse matrices, partitioned retrieval, cached stages, chunked inference, and full-scale rehearsal before the final run.

### Risk: Invalid submission format

**Mitigation:** Automated structural assertions plus the supplied validator on every generated output.

---

## 24. Final release checklist

### Data and model

- [ ] Raw files remain unchanged.
- [ ] Train/test schemas and ID prefixes pass checks.
- [ ] Validation splits are saved and entity-grouped.
- [ ] Candidate recall and volume reports are saved.
- [ ] Final features contain no label, ID, or row-order leakage.
- [ ] Thresholds come from out-of-fold entity-level scoring.
- [ ] Model and checkpoint licenses are verified and recorded.
- [ ] Final model size is within the 8B limit.
- [ ] No prohibited external lookup or enrichment is used.

### Output

- [ ] Both files are tab-separated with exact headers.
- [ ] Every test Source 1 ID appears exactly once.
- [ ] Empty predictions are written as empty fields.
- [ ] Every listed ID exists in test Source 2 or Source 3.
- [ ] No duplicate IDs occur within a list.
- [ ] Every final match is present in the candidate list.
- [ ] `candidate_pairs.tsv` reflects the exact model input.
- [ ] The supplied validator prints `PASS`.

### Reproducibility and package

- [ ] Dependencies are pinned.
- [ ] Seeds and configuration are saved.
- [ ] README reproduces data-to-output execution.
- [ ] Clean-environment smoke test succeeds.
- [ ] Output checksums are recorded.
- [ ] Final zip has the required directory structure.
- [ ] Methodology document describes blocking, model, features, thresholds, and validation.

---

## 25. Definition of done

The implementation is complete only when a clean run can regenerate both output files from the provided data, the exact entity-level validation report supports the selected precision-first policy, all required IDs and relationships pass structural checks, the supplied validator passes, and the submission package contains reproducible code plus documented model-license compliance.
