# PRD: Precision-first business entity resolution system

**Project:** Amazon AI/ML Challenge — Business Entity Resolution  
**Document type:** Product requirements document  
**Strategy:** High-recall candidate generation + conservative, calibrated matching  
**Primary metric:** Macro F<sub>0.5</sub>  
**Team size:** 4 members

---

## 1. Executive summary

Humein Source 1 ke har deduplicated business record ke liye Source 2 aur Source 3 mein zero, one, ya multiple matching records identify karne hain. Records ke beech common identifier nahi hai; matching noisy `business_name`, `business_address`, aur open-set `country` fields se hogi.

Competition ka metric macro F<sub>0.5</sub> hai, jisme precision ko recall se zyada importance milti hai. Isliye system ka design principle hoga:

> **Candidates miss na hon, lekin final match tabhi nikle jab evidence strong ho. Doubtful case mein empty prediction safer hai.**

100% accuracy guarantee nahi ki ja sakti, kyunki data ambiguous, incomplete aur noisy ho sakta hai. Product goal guaranteed perfection nahi, balki unseen test set par reproducible tareeke se maximum possible precision aur macro F<sub>0.5</sub> achieve karna hai.

Recommended system:

1. Country-safe, multi-channel blocking se high-recall candidate set.
2. Name, address, numeric aur geographic-text consistency features.
3. Hard-negative-trained LightGBM pair classifier as primary model.
4. Optional licensed multilingual cross-encoder as reranker or agreement signal.
5. Out-of-fold probability calibration and segment-specific thresholds.
6. Hard contradiction vetoes, score-margin checks and singleton-first abstention.
7. Strict output validation and reproducible final package.

---

## 2. Problem definition

### 2.1 Input

Teen tab-separated source files milte hain:

- **Source 1:** Deduplicated reference entities.
- **Source 2:** Noisy records containing possible Source 1 matches.
- **Source 3:** Noisy records containing possible Source 1 matches.

Har source record mein:

- `entity_id`
- `business_name`
- `business_address`
- `country`

`entity_id` prefix source batata hai: `S1-`, `S2-`, ya `S3-`.

### 2.2 Required prediction

Har Source 1 entity ke liye Source 2 aur Source 3 ke saare matching IDs predict karne hain. Valid cardinality:

- zero matches;
- exactly one match;
- multiple matches.

Model ko top-1 match force nahi karna hai. Agar reliable evidence nahi hai to `matched_entity_ids` empty rehna chahiye.

### 2.3 Required outputs

1. **`output/matching_results.tsv`** — leaderboard-scored final predictions.
2. **`output/candidate_pairs.tsv`** — final candidate set actually sent to the matcher.

Dono files mein har test Source 1 ID ki exactly one row honi chahiye. Final matched IDs candidate list ka subset hone chahiye.

### 2.4 Important data condition

Training mein US aur India hain; test mein France bhi hai. `country` ko `{US, India}` tak hard-code ya one-hot constrain nahi karna hai. Unknown country label ko generic country partition aur language-agnostic normalization ke through handle karna hoga.

### 2.5 Immediate data-integrity gate

Shared snippets mein Source 1 IDs aur ground-truth Source 1 IDs overlap nahi kar rahe. Full files milte hi pipeline sabse pehle ye verify kare:

- ground truth ke saare `source1_entity_id` training Source 1 mein exist karte hain;
- ground-truth matches training Source 2/3 mein exist karte hain;
- prefixes aur file membership consistent hain.

Agar overlap zero ya unexpectedly low ho, training start nahi karni; likely snippets different samples/splits se aaye hain.

---

## 3. Goals and non-goals

### 3.1 Primary goals

- **Maximize macro F<sub>0.5</sub>:** thresholding aur model selection isi metric par honge.
- **Protect precision:** false merge ko aggressively avoid karna.
- **Preserve blocking recall:** true match matcher tak pahunchna chahiye.
- **Identify singletons:** no-match entity ko empty output dena.
- **Generalize to France:** unseen country ko valid test segment ki tarah process karna.
- **Reproducibility:** one-command pipeline se dono TSV files regenerate hon.
- **Auditability:** every prediction ke feature scores, model score aur decision reason log hon.

### 3.2 Secondary goals

- CPU-first baseline jo quickly iterate ho sake.
- Optional GPU reranker only when validation gain clear ho.
- Memory-safe sparse retrieval for large data.
- Segment-wise diagnostics for country, source and missingness.

### 3.3 Non-goals

- External geocoding, web search, company databases, government registries or commercial ER APIs use karna.
- Har S1 ke liye compulsory match predict karna.
- Raw nearest-neighbour score ko directly final prediction banana.
- Public leaderboard ke chhote changes ko blindly optimize karna.
- Global one-to-one matching enforce karna, jab tak training data us constraint ko prove na kare.

---

## 4. Success metrics and acceptance targets

Targets validation objectives hain, achieved results nahi. Final values out-of-fold experiments se confirm hongi.

| Metric | Target | Priority |
|---|---:|---|
| Macro F<sub>0.5</sub> | Best OOF | Primary |
| Final precision | Maximize | Critical |
| Candidate recall | ≥ 99.5% | Critical |
| Output validity | 100% pass | Mandatory |
| S1 row coverage | 100% | Mandatory |
| Final-in-candidate rate | 100% | Mandatory |

### 4.1 Metric definition

Per Source 1 entity:

`F0.5 = (1.25 × precision × recall) / (0.25 × precision + recall)`

Then all Source 1 entities ka macro average liya jayega.

Edge cases evaluator mein explicitly implement hon:

- true empty + predicted empty → `1.0`;
- true non-empty + predicted empty → `0.0`;
- true empty + predicted non-empty → `0.0`;
- otherwise normal set-based precision, recall and F<sub>0.5</sub>.

### 4.2 Why precision-first

F<sub>0.5</sub> false positives ko strongly punish karta hai. Singleton par ek incorrect link poora entity score `0` kar deta hai. Isliye final stage ka objective maximum pair recall nahi, balki **well-calibrated high-confidence links** hai.

### 4.3 Model-selection rule

Final configuration select hogi:

1. Highest repeated out-of-fold macro F<sub>0.5</sub>.
2. Near-tie mein higher precision.
3. Phir lower fold variance.
4. Phir simpler and faster pipeline.

Public leaderboard movement ko sirf weak signal maana jayega; private leaderboard generalization ke liye OOF stability primary evidence hogi.

---

## 5. Proposed architecture

```text
TSV ingestion
  --> schema and ID validation
  --> raw + normalized field views
  --> multi-channel blocking by country
  --> candidate union and controlled pruning
  --> pairwise feature generation
  --> primary pair classifier
  --> optional neural agreement score
  --> calibrated segment thresholds
  --> hard vetoes + margin rules
  --> entity-level aggregation / abstention
  --> matching_results.tsv + candidate_pairs.tsv
  --> official submission validator
```

System intentionally two-stage hoga:

- **Blocking layer:** Recall-oriented. Broad but computationally bounded.
- **Decision layer:** Precision-oriented. Conservative and allowed to abstain.

Blocking threshold ko final matching threshold ke saath mix nahi karna hai. Candidate generation ka score probability nahi hota.

---

## 6. Data ingestion and quality checks

### 6.1 Parser requirements

- Always `sep="\t"`.
- IDs and text fields ko strings ke roop mein read karein.
- Empty strings ko preserve/normalize consistently karein.
- Original raw columns untouched rakhein.
- Duplicate rows aur duplicate IDs separately report karein.
- Unicode text ko UTF-8 mein process karein.

### 6.2 Mandatory validation checks

- Required columns exactly available.
- `S1-`, `S2-`, `S3-` prefix correct file mein.
- Every training ground-truth S1 ID exists.
- Every matched training ID exists in S2/S3.
- Ground-truth lists comma-separated, deduplicated.
- Country null and unique-label counts.
- Name/address missingness by source and country.
- Duplicate normalized names and addresses.
- One S2/S3 ID kitne S1 labels mein appear karta hai.

**Important:** Source 1 deduplicated hai, lekin statement global one-to-one constraint explicitly define nahi karta. Isliye S2/S3 ko ek hi S1 se force-map karna default rule nahi hoga. Training evidence milne par conflict feature ya post-processing experiment kiya ja sakta hai.

### 6.3 Data profiling report

Per source and country:

- record count;
- missing-name and missing-address rate;
- text-length quantiles;
- token-count quantiles;
- postal code, state and city extraction coverage;
- exact/near duplicate rates;
- matches-per-S1 distribution;
- singleton fraction;
- positive name/address similarity distributions.

Ye report normalization, blocking `top_k`, negative sampling aur thresholds decide karne ka basis hogi.

---

## 7. Text normalization design

Har field ke multiple views banenge. Ek aggressive normalized value par depend nahi karna, kyunki over-normalization distinct businesses ko collapse kar sakta hai.

### 7.1 Shared normalization

- Unicode NFKC normalization.
- Lowercase / case-fold.
- Whitespace collapse.
- Punctuation-separated and punctuation-preserved views.
- Accent-folded retrieval view plus original Unicode view.
- `&` and `and` ka alternate token view.
- Repeated punctuation removal.
- Alphanumeric tokenization.
- Digits preserve karna; house numbers delete nahi karne.

### 7.2 Business-name views

- `name_raw_clean`: minimally cleaned.
- `name_core`: verified legal suffixes down-weighted/removed.
- `name_tokens_sorted`: order-insensitive comparison.
- `name_char`: compact character string for typo similarity.
- `name_rare_tokens`: high-IDF tokens only.

Legal suffixes informative ho sakte hain; unhe raw representation se kabhi delete nahi karna. Core view extra signal hai, replacement nahi.

### 7.3 Address views

- `address_raw_clean`: minimally cleaned.
- `address_tokens`: normalized token list.
- `address_char`: character n-gram view.
- `address_numbers`: ordered numeric tokens.
- `postal_code`: country-aware extractor where format is known.
- `admin_tokens`: state/region/city tokens learned only from provided data.
- `unit_tokens`: apartment, floor, suite, plot, house references.
- `landmark_tokens`: near/opp/behind-type context retained.

Numbers are high-value but context-sensitive. `12` versus `120` strong contradiction ho sakta hai; `unit 2` missing hona contradiction nahi. House, unit, floor and postal tokens ko separate feature families mein compare karna hai.

### 7.4 Country handling

- Raw country string normalize karke equality feature banayein.
- Blocking primarily same-country within karein.
- Unknown labels generic path se process hon.
- Country ko fixed train-only one-hot vocabulary na banayein.
- Country mismatch candidate ko default reject karein, unless official data analysis cross-country positives prove kare.

### 7.5 Transliteration policy

Retrieval ke liye original Unicode aur accent-folded/romanized views ka union use kiya ja sakta hai. Final decision mein transliteration similarity akela sufficient evidence nahi hoga. Koi external lookup or geocoder allowed nahi.

---

## 8. Label construction and validation split

### 8.1 Positive pairs

Ground truth mein har `(S1, matched S2/S3)` pair positive hai.

### 8.2 Negative pairs

Random negatives alone easy honge aur model ko unrealistic confidence denge. Training negatives ka mix:

- **Hard blocking negatives:** high name/address retrieval score but not in truth.
- **Same-name negatives:** similar name, conflicting address.
- **Same-address negatives:** similar address, conflicting core name.
- **Numeric-conflict negatives:** strong text overlap with house/postal mismatch.
- **Near-threshold negatives:** prior model score high but label false.
- **Small random-negative sample:** broad calibration ke liye.

Unlabeled pair ko negative tabhi treat karein jab ground truth complete maana gaya ho. Statement all matches provide karta hai, phir bhi integrity checks ke baad hi assumption lock hogi.

### 8.3 Split design

- Split unit **Source 1 entity** hoga, pair nahi.
- Saare pairs of one S1 same fold mein rahenge.
- Country, source mix, singleton status aur match-count distribution ko folds mein balance karein.
- Recommended: 5-fold grouped cross-validation; time limited ho to 3-fold.
- Thresholds only out-of-fold predictions par tune hon.

Pair-random split prohibited hai because same entity patterns train aur validation mein leak ho sakte hain.

### 8.4 Leakage control

- TF-IDF vocabulary/IDF fold training data par fit ho.
- Learned abbreviations and token statistics fold training part se aayen.
- Calibration fold-held-out scores par ho.
- Full-train model sirf configuration lock hone ke baad train ho.

---

## 9. Candidate generation / blocking

Blocking ka goal approximately all true links retain karna hai, not final precision.

### 9.1 Candidate channels

Har S1 record ke liye S2 aur S3 separately search hon, same-country partition ke andar:

1. **Name character TF-IDF:** character 2–5 grams; typo and punctuation robust.
2. **Address character TF-IDF:** character 3–5 grams; reordered/noisy addresses ke liye.
3. **Combined field TF-IDF:** weighted name + address representation.
4. **Name token retrieval:** word n-grams / BM25-style sparse score.
5. **Rare-token index:** uncommon business/address token overlap.
6. **Postal/ZIP block:** matching postal code + loose name similarity.
7. **Numeric-address block:** shared important numbers + name evidence.
8. **Exact normalized keys:** exact core name or compact address keys.

Each channel ke top candidates ka union final recall increase karega. Initial `top_k` per channel profiling se set hoga; baseline experiment 20–50 per source/channel se start kar sakta hai.

### 9.2 Candidate pruning

Union bahut large ho to only safe pruning:

- definite country mismatch;
- zero meaningful name overlap **and** zero address evidence;
- impossible explicit postal/state conflict with weak name evidence;
- low score across every retrieval channel.

Pruning threshold validation candidate recall se gated hoga. Final `candidate_pairs.tsv` exactly post-pruning candidates contain karega jo model inference mein score hue.

### 9.3 Blocking evaluation

- **Candidate recall:** true positive pairs present / all true pairs.
- **Entity coverage:** at least one true candidate wale matched S1 entities.
- **Reduction ratio:** candidates / all possible cross-source pairs.
- Recall by country, target source, text-missingness and match count.

Candidate recall macro F<sub>0.5</sub> ka ceiling hai. If recall target miss hota hai, matcher tuning se pehle blocking fix hogi.

### 9.4 Scale strategy

- Sparse TF-IDF matrices.
- Chunked nearest-neighbour retrieval.
- S2/S3 and countries independently indexed.
- Stable deterministic tie-breaking by `entity_id`.
- Cached normalized tables and candidate edges.
- Full Cartesian product memory mein materialize nahi karna.

---

## 10. Pairwise feature engineering

### 10.1 Name features

- Character TF-IDF cosine.
- Word TF-IDF cosine.
- Jaro-Winkler similarity.
- Normalized Levenshtein ratio.
- Token-set and token-sort ratios.
- Token Jaccard and containment.
- Core-name exact match.
- First/last informative token agreement.
- Acronym agreement.
- Rare-token overlap count and weighted overlap.
- Length ratio and unmatched-token count.
- Legal suffix compatibility.

### 10.2 Address features

- Character and word TF-IDF cosine.
- Token Jaccard / containment.
- Postal-code exact match and contradiction.
- House/plot number agreement and contradiction.
- Unit/floor number agreement.
- City/state token agreement.
- Numeric-token Jaccard.
- Longest common token subsequence.
- Address length and missingness pair.
- Landmark-token overlap.

### 10.3 Cross-field and metadata features

- Country equality.
- Candidate source: S2 versus S3.
- Name strong + address weak interaction.
- Address strong + name weak interaction.
- Core name exact + numeric consistency.
- Retrieval rank and score from each blocker.
- Number of blocking channels retrieving the pair.
- Score gaps from top candidate.
- Number of similar candidates for the S1.
- Field completeness on both records.

### 10.4 Feature safety principle

A single fuzzy name score final match approve nahi karega. Strong decision ideally independent evidence combine kare:

- name identity;
- location/address consistency;
- numeric component consistency;
- retrieval/model agreement.

Missing evidence aur contradictory evidence different hain. Missing postal code neutral ho sakta hai; two explicit different postal codes negative signal hain.

---

## 11. Model strategy

### 11.1 Baseline

Rule-based score + logistic regression establish karega ki features and evaluator correct hain. Baseline intentionally simple aur interpretable hoga.

### 11.2 Primary model

**LightGBM gradient-boosted trees** on engineered pair features recommended primary model hai because:

- mixed nonlinear similarities efficiently combine karta hai;
- missing values naturally handle karta hai;
- CPU training/inference fast hai;
- feature importance and error analysis easy hai;
- strict high-confidence boundary tune ki ja sakti hai.

Class imbalance handling:

- hard-negative-rich batches;
- controlled negative downsampling;
- sample weights;
- validation probability calibration.

Class weights metric threshold ka substitute nahi hain; final choice macro F<sub>0.5</sub> par hogi.

### 11.3 Optional neural model

Provided text only use karke a verified MIT/Apache-licensed multilingual text encoder or cross-encoder, under the team's 4B preference and official 8B ceiling, optional experiment hoga.

Best use:

- top candidate reranking;
- an extra semantic similarity feature;
- conservative agreement gate for borderline pairs.

Neural score primary model ko replace tabhi karega jab repeated OOF macro F<sub>0.5</sub> and precision improve kare. Model card, exact license, parameter count and checkpoint hash final documentation mein record honge.

### 11.4 Ensemble policy

Precision-first ensemble options:

- calibrated weighted blend;
- LightGBM accept + neural veto;
- both models above thresholds;
- auto-accept only when rule evidence and model agree.

`AND` gating precision badha sakta hai but recall reduce karega; selection OOF metric se hogi.

---

## 12. Maximum-precision decision policy

Ye section proposed solution ka central differentiator hai.

### 12.1 Never force a match

Har S1 ke liye zero predictions valid hain. Top candidate weak ho to empty output. “Closest” ka matlab “same business” nahi hota.

### 12.2 Probability calibration

Raw tree scores reliable probabilities nahi hote. Out-of-fold scores par isotonic regression ya Platt calibration compare karein. Calibration leakage-free honi chahiye.

### 12.3 Segment-specific thresholds

One global threshold baseline hoga. Uske baad sufficient validation support hone par thresholds separately tune karein:

- country label;
- target source S2/S3;
- name/address completeness;
- exact postal available or missing;
- top-score ambiguity band.

Small France-like unseen segment ke liye training-country-specific threshold blindly use na karein. Generic conservative fallback threshold use ho.

### 12.4 Three-zone policy

- **Auto-accept:** calibrated score ≥ high threshold and no hard contradiction.
- **Gray zone:** medium score; test inference mein reject/abstain unless independent rules strongly confirm.
- **Reject:** below threshold or veto triggered.

Threshold values hard-code nahi kiye jayenge; OOF grid search se optimize honge. Search fine-grained hona chahiye, especially high-score region mein.

### 12.5 Hard contradiction vetoes

Potential vetoes sirf training evidence se validated hone ke baad enable hon:

- explicit country mismatch;
- strong core-name incompatibility;
- both present but conflicting high-confidence postal codes;
- both present but incompatible US states;
- strong house/plot number conflict with no alternate address explanation;
- suspicious generic-name match with weak address evidence.

Veto missing field par trigger nahi hoga. Raw address string inequality veto nahi hai.

### 12.6 Independent-evidence rule

Borderline pair approve karne ke liye at least two independent evidence families required:

- strong name + compatible address;
- exact core name + matching postal/numeric tokens;
- strong address + distinctive name overlap;
- primary model + neural model agreement, with no contradiction.

### 12.7 Margin and ambiguity rule

For each S1 and target source:

- top score;
- second-best score;
- top–second margin;
- number of near-tied candidates

track karein. Low margin indicates ambiguity. High absolute score but near-tie case ko higher threshold ya abstention mile. This prevents common-name false merges.

### 12.8 Multi-match policy

Har candidate independently pass kare, lekin same S1 ke accepted matches ke beech consistency diagnostics run hon. Multi-match count prior ko hard cap na banayein. Unusually large prediction list warning generate kare, automatic truncation nahi unless validation supports it.

### 12.9 Precision ladder

High-confidence deterministic patterns ko separate acceptance tier diya ja sakta hai:

1. exact core name + exact strong address key;
2. near-exact distinctive name + exact postal/house number;
3. high calibrated model score + no veto;
4. model ensemble agreement in gray zone;
5. otherwise empty.

Rule order and thresholds OOF-tested hon; intuitive rule bina measured evidence deploy nahi hoga.

### 12.10 Threshold stability

Threshold choose karte waqt single best grid point ke bajay plateau prefer karein. Agar tiny threshold shift score collapse karta hai, configuration fragile hai. Fold-wise precision, recall and F<sub>0.5</sub> variation inspect karein.

---

## 13. Singleton detection

Singletons direct score contribute karte hain. Dedicated strategy:

- no candidate above threshold → empty;
- top score low → empty;
- top score medium with low margin → empty;
- name-only match with absent/conflicting address → usually empty;
- generic business name + weak location → empty;
- strong candidate but hard contradiction → empty.

Optional entity-level singleton model features:

- maximum calibrated pair probability;
- top–second score margin;
- number of retrieval channels agreeing;
- best name/address scores;
- candidate count;
- field missingness;
- top candidate contradiction flags.

Entity-level model deploy tabhi ho jab direct thresholding se consistent OOF gain de.

---

## 14. France and unseen-country robustness

France test-only hai, so supervised country-specific memorization risky hai.

Requirements:

- country label treated as normalized string, not closed category;
- character n-grams and token overlap remain primary;
- accents ke original and folded views;
- numeric and postal components preserved;
- generic legal-suffix handling, not only US/India vocabulary;
- unseen-country fallback thresholds conservative;
- no France records dropped from output;
- every France S1 gets exactly one result row, empty allowed.

### 14.1 Pseudo-unseen validation

Country generalization simulate karne ke liye:

1. Train only US, validate India as unseen.
2. Train only India, validate US as unseen.
3. Compare generic versus country-specific normalization.
4. Select components that transfer rather than memorize.

Distribution identical nahi hogi, but failure modes reveal honge.

---

## 15. Error-analysis framework

Every OOF false positive and false negative bucket ho:

- same/similar name, different address;
- different name, same address;
- missing address;
- numeric conflict;
- transliteration/accent issue;
- word-order variation;
- legal-suffix variation;
- generic business name;
- singleton false merge;
- multi-match partial recovery;
- blocking miss;
- threshold miss;
- model ranking error.

Priority order:

1. Singleton false positives.
2. High-confidence false positives.
3. Blocking false negatives.
4. Multi-match omissions.
5. Long-tail normalization issues.

Each proposed rule must include before/after OOF precision, recall and macro F<sub>0.5</sub>. Anecdotal fixes without aggregate evaluation merge nahi honge.

---

## 16. Experiment plan

### 16.1 Baselines

- Exact normalized name and address rules.
- Name-only TF-IDF nearest neighbour.
- Combined name/address logistic regression.
- LightGBM with initial features.

### 16.2 Blocking ablations

- Name only.
- Address only.
- Combined only.
- Multi-channel union.
- Rare-token and numeric channels added.
- Candidate `top_k` sweep.

### 16.3 Matcher ablations

- Remove numeric features.
- Remove retrieval rank features.
- Random versus hard negatives.
- Global versus calibrated scores.
- Global versus segment thresholds.
- With/without vetoes.
- With/without optional neural score.

### 16.4 Decision ablations

- Forced top-1 versus abstention.
- Threshold only versus threshold + margin.
- Threshold + hard vetoes.
- Pair-only versus entity-level singleton model.

### 16.5 Experiment log schema

Each run records:

- run ID and code commit;
- dataset fingerprint;
- fold seed;
- normalization version;
- blocker configuration;
- feature version;
- model parameters;
- calibration method;
- thresholds;
- pair precision/recall;
- macro F<sub>0.5</sub>;
- candidate recall;
- runtime and memory;
- artifact paths.

---

## 17. Work division for 4 team members

Parallel work possible hai, lekin interfaces day one par freeze honi chahiye.

### Part 1 — Data, normalization and blocking

**Owner:** Member 1  
**Goal:** True pairs matcher tak reliably pahunchana.

Responsibilities:

- data loader and schema validation;
- profiling report;
- reusable normalization library;
- country/source partitioned sparse indexes;
- multi-channel candidate generation;
- candidate recall and reduction metrics;
- final candidate table contract;
- candidate cache for other members.

Outputs:

- `normalized_source*.parquet` or equivalent;
- `train_candidates.tsv/parquet`;
- `test_candidates.tsv/parquet`;
- blocking report by segment;
- `candidate_pairs.tsv` writer.

Acceptance criteria:

- no data-integrity errors;
- target candidate recall met on every fold or misses documented;
- every final candidate valid S2/S3 ID;
- deterministic results for same seed/config;
- scalable without full Cartesian join.

Dependencies supplied to others:

- stable normalization functions;
- candidate pair schema;
- retrieval scores/ranks.

### Part 2 — Features and matching model

**Owner:** Member 2  
**Goal:** Candidate pairs ko accurate match probabilities dena.

Responsibilities:

- pair feature library;
- positive and hard-negative dataset;
- baseline logistic model;
- LightGBM training and inference;
- optional neural reranker experiment;
- model/card license verification;
- feature importance and error slices.

Outputs:

- `pair_features` module;
- trained fold models;
- OOF raw scores;
- test raw scores;
- feature dictionary;
- model metadata and license record.

Acceptance criteria:

- no S1 leakage across folds;
- reproducible training;
- hard negatives included;
- score distribution and high-confidence errors reported;
- inference fits compute budget.

Dependencies:

- consumes Part 1 candidates;
- supplies OOF/test pair scores to Part 3.

### Part 3 — Precision tuning, calibration and validation

**Owner:** Member 3  
**Goal:** Raw pair scores ko optimal final decisions mein convert karna.

Responsibilities:

- official macro F<sub>0.5</sub> evaluator;
- calibration comparison;
- threshold grid search;
- segment thresholds with minimum-support rules;
- hard veto and margin evaluation;
- singleton policy;
- OOF error analysis;
- ablation and stability dashboard.

Outputs:

- calibrated OOF/test scores;
- locked decision-policy config;
- threshold report;
- false-positive/false-negative buckets;
- fold-wise score summary.

Acceptance criteria:

- edge cases exactly handled;
- chosen policy beats forced top-1 baseline;
- precision/F<sub>0.5</sub> gain repeated across folds;
- no threshold tuned on test labels or public leaderboard subsets;
- fallback policy exists for unseen country.

Dependencies:

- consumes Part 2 scores;
- supplies final decision config to Part 4.

### Part 4 — Integration, QA and submission

**Owner:** Member 4  
**Goal:** Reproducible end-to-end system aur valid submission deliver karna.

Responsibilities:

- config-driven pipeline orchestration;
- train/inference commands;
- entity-level aggregation;
- exact TSV formatting;
- official validator execution;
- package structure and pinned dependencies;
- run logs, seeds and checksums;
- final methodology document and zip.

Outputs:

- `matching_results.tsv`;
- `candidate_pairs.tsv`;
- runnable `src/` pipeline;
- `README.md`;
- pinned `requirements.txt`;
- completed methodology document;
- final submission archive.

Acceptance criteria:

- every S1 exactly once;
- empty matches correctly serialized;
- no duplicates or invalid IDs;
- final links subset of candidate links;
- validator prints `PASS`;
- clean environment can reproduce outputs;
- package contains no external lookup or unlicensed model.

### 17.1 Shared interface contract

Candidate-pair internal table minimum fields:

- `source1_entity_id`
- `candidate_entity_id`
- `candidate_source`
- `country`
- retrieval scores per channel
- retrieval ranks per channel
- channel-count

Scored-pair table adds:

- raw model score;
- calibrated probability;
- contradiction flags;
- decision threshold;
- accepted boolean;
- decision reason.

No member silently changes column names. Schema change requires all four owners to update against one versioned contract.

### 17.2 Merge order

1. Part 1 freezes data and candidate interfaces.
2. Part 2 produces OOF pair scores.
3. Part 3 freezes decision policy.
4. Part 4 performs clean end-to-end run.
5. All members inspect high-confidence false positives and final package.

---

## 18. Recommended repository structure

```text
code/business_entity_resolution/
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   └── final.yaml
├── src/
│   ├── data_io.py
│   ├── validate_data.py
│   ├── normalize.py
│   ├── blocking.py
│   ├── build_pairs.py
│   ├── features.py
│   ├── train_matcher.py
│   ├── calibrate.py
│   ├── decide.py
│   ├── evaluate.py
│   ├── infer.py
│   └── write_submission.py
├── tests/
│   ├── test_normalize.py
│   ├── test_metric.py
│   ├── test_blocking.py
│   └── test_submission.py
└── models/
    └── model_metadata.json

output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

Large intermediate files and datasets final zip mein include na karein unless rules require them.

---

## 19. End-to-end execution contract

```bash
python -m src.validate_data --data-dir dataset
python -m src.train_matcher --config configs/final.yaml
python -m src.infer --config configs/final.yaml --split test
python -m src.write_submission --config configs/final.yaml
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Exact command implementation repository ke hisaab se adjust ho sakta hai, lekin README mein one reproducible path mandatory hai.

### 19.1 Determinism

- random seeds fixed;
- package versions pinned;
- stable sorting before output;
- model/config hash logged;
- input file fingerprint logged;
- threshold config stored with model.

---

## 20. Output-generation rules

### 20.1 `matching_results.tsv`

- Header exactly: `source1_entity_id`, `matched_entity_ids`.
- Single tab between columns.
- Every test S1 exactly once.
- Match list comma-separated, no duplicates.
- Only valid test S2/S3 IDs.
- No match means empty second field.
- Deterministic ID ordering.

### 20.2 `candidate_pairs.tsv`

- Header exactly: `source1_entity_id`, `candidate_entity_ids`.
- Every test S1 exactly once.
- Contains exact final candidate set scored by matcher.
- Final matched IDs must be subset.
- Empty list valid when no candidates.

### 20.3 Pre-submission assertions

- row count equals test Source 1 count;
- S1 set exactly equal;
- no duplicate S1 rows;
- no invalid prefixes;
- all referenced IDs exist;
- no duplicate IDs inside lists;
- all matches included in candidates;
- official validator passes.

---

## 21. QA and test plan

### 21.1 Unit tests

- TSV parsing with commas inside addresses.
- Empty match serialization.
- Unicode and accent normalization.
- Legal suffix handling.
- Number extraction.
- F<sub>0.5</sub> singleton edge cases.
- Candidate subset assertion.
- Deterministic ordering.

### 21.2 Data tests

- schema and dtype checks;
- ID prefix/file checks;
- referential integrity;
- missingness drift;
- country-label drift;
- candidate-count explosion;
- no silent record drop.

### 21.3 Model tests

- OOF only metrics;
- fold entity isolation;
- calibration reliability plots;
- threshold stability;
- segment precision;
- high-confidence false-positive review;
- unseen-country simulation.

### 21.4 Regression tests

A fixed small fixture should cover:

- exact match;
- abbreviation;
- typo;
- reordered address;
- same name/different address;
- missing address;
- singleton;
- multiple matches;
- unseen country label.

Fixture synthetic/test-only ho; competition score claim ke liye use nahi hoga.

---

## 22. Risks and mitigations

### 22.1 Over-aggressive normalization

**Risk:** Distinct names collapse.  
**Mitigation:** Raw, minimal and aggressive views parallel rakho; exact key alone final match na banaye.

### 22.2 Blocking misses

**Risk:** True pair matcher ko nahi milta.  
**Mitigation:** Multi-channel union, recall slices, false-negative inspection, controlled larger `top_k`.

### 22.3 Easy-negative bias

**Risk:** Validation pair accuracy high but real precision poor.  
**Mitigation:** Candidate-derived hard negatives and full entity-level evaluation.

### 22.4 Threshold overfitting

**Risk:** One fold/public board par best, private par weak.  
**Mitigation:** Repeated grouped folds, calibration, stable plateau selection.

### 22.5 Singleton false merges

**Risk:** Empty-truth entity score becomes zero.  
**Mitigation:** No forced match, high thresholds, margin and contradiction rules.

### 22.6 France distribution shift

**Risk:** train-only country logic fails.  
**Mitigation:** open-set labels, multilingual character features, pseudo-unseen validation, conservative fallback.

### 22.7 Format rejection

**Risk:** Good model but invalid submission.  
**Mitigation:** assertions plus official validator in final pipeline.

### 22.8 License or fair-play violation

**Risk:** Disqualification.  
**Mitigation:** provided data only; model license, size and provenance documented before use.

---

## 23. Milestones and exit criteria

### Milestone A — Data ready

Exit when:

- integrity checks pass;
- mismatch in shared snippets is explained by correct full files;
- profiling report complete;
- normalized views versioned.

### Milestone B — Blocking ready

Exit when:

- OOF candidate recall target is met or trade-off documented;
- candidate volume fits inference budget;
- `candidate_pairs.tsv` semantics match actual inference set.

### Milestone C — Matcher ready

Exit when:

- hard-negative model beats baselines;
- calibration and thresholds use OOF predictions;
- segment and singleton metrics available;
- optional neural model retained only if measurable gain exists.

### Milestone D — Submission ready

Exit when:

- clean end-to-end rerun succeeds;
- both output files pass all assertions and official validator;
- final model/license constraints verified;
- archive structure and reproduction steps verified.

---

## 24. Definition of done

Project complete tab maana jayega jab:

- all test Source 1 records have exactly one output row;
- final decisions use a locked, OOF-selected threshold policy;
- weak/ambiguous candidates are allowed to abstain;
- blocking recall and matcher precision are separately measured;
- candidate file is exactly the model-scored candidate set;
- final matches are a subset of candidates;
- France follows generic open-set flow;
- no external lookup is used;
- selected model satisfies license and parameter limits;
- package is reproducible and validator passes;
- methodology records normalization, blocking, features, model, calibration and thresholds.

---

## 25. Recommended first implementation

Fastest strong starting point:

1. Build raw/minimal/core normalized text views.
2. Generate same-country candidates from name-char, address-char and combined TF-IDF channels.
3. Confirm ≥99.5% OOF candidate recall before aggressive pruning.
4. Train LightGBM on positives plus candidate-derived hard negatives.
5. Add numeric/postal contradiction features.
6. Calibrate OOF scores.
7. Grid-search high thresholds and top-score margin for macro F<sub>0.5</sub>.
8. Predict empty on uncertain cases; never force top-1.
9. Add optional multilingual reranker only if OOF precision and F<sub>0.5</sub> improve.
10. Run strict output checks and official validator.

This sequence should deliver a competitive CPU-first baseline quickly, while leaving neural reranking as a measured upgrade rather than a dependency.

---

## 26. Final precision checklist

Before locking a submission, answer “yes” to each:

- Did blocking retain nearly every known positive in OOF?
- Were negatives sampled from realistic candidates?
- Is every score reported from entity-grouped OOF validation?
- Were probabilities calibrated?
- Was the final threshold selected on macro F<sub>0.5</sub>, not pair accuracy?
- Are singletons allowed to remain empty?
- Are low-margin top candidates rejected or escalated?
- Are explicit contradictions vetoed only when both values are present?
- Does unseen-country data use a generic conservative policy?
- Does every final match exist in the candidate list?
- Do both TSVs contain every test S1 exactly once?
- Does the official validator return `PASS`?
- Are model license and parameter count documented?
- Is external lookup completely absent?

If any answer is “no,” submission should not be treated as final.

---

## 27. Assumptions requiring confirmation from full data

- Ground truth lists are complete for training.
- Same-country blocking is valid for all positives.
- The provided Source 1 and ground-truth snippets came from different subsets, explaining their non-overlapping IDs.
- Dataset size permits sparse TF-IDF retrieval under available memory.
- Country-specific veto rules have enough support to validate.

Any failed assumption should change the relevant component through measured OOF experiments, not ad-hoc test-time rules.

---

## Source basis

This PRD is grounded in the provided **Business Entity Resolution Challenge** problem statement and the Source 1 / ground-truth snippets shared in the conversation. No external business lookup or external dataset is proposed.
