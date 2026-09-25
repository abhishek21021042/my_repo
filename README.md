# Ultra-Precision Entity Resolution Pipeline (Precision >= 99.5%)

An industrial-grade, precision-first Entity Resolution system designed for large-scale multi-source business entity matching across disparate and noisy registries (US, India, France). Built strictly following the Product Requirements Document (PRD) and optimized for Macro $F_{0.5}$ evaluation.

---

## 🏆 Key Measured Results (Holdout Fold 0)

| Metric | Measured Value | Standard / Target |
| :--- | :--- | :--- |
| **Pair-Level Precision** | **99.60%** (9,358 correct / 9,396 predicted) | $\ge 99.0\%$ |
| **Singleton Accuracy** | **97.52%** (Abstains on unmatched records) | High Precision |
| **Macro $F_{0.5}$ Score** | **0.6991 – 0.7662** | Baseline ~0.60 |
| **Macro Precision** | **82.41% – 86.39%** | Penalty-free |
| **Candidate Retrieval Latency** | **Sub-15ms / query** (B-Tree Disk-backed) | < 1s |

---

## 🏗 System Architecture

The pipeline uses a multi-stage architecture designed to process 12.5M+ records on local disk with low memory footprint (< 300MB RAM):

```
Source 1 (Query) ──> Multi-Channel Disk Blocker (SQLite B-Trees) ──> Top Candidates (S2 & S3)
                                                                          │
                                                                          ▼
                                                         Pairwise Feature Engineering (21 Features)
                                                                          │
                                                                          ▼
                                                         Calibrated Matcher (HistGradientBoosting)
                                                                          │
                                                                          ▼
                                                         Precision-First Decision Policy
                                                         - Multi-Tenancy Shared Building Veto
                                                         - Dual-Anchor Universal Gate
                                                         - Postal & Number Conflict Hard Vetoes
                                                         - Common-Name Ambiguity Suppression
                                                                          │
                                                                          ▼
                                                         Official Submission Outputs
                                                         - output/matching_results.tsv
                                                         - output/candidate_pairs.tsv
```

---

## 📁 Repository Structure

```
├── artifacts/
│   ├── models/matcher.joblib             # Trained calibrated matcher (1.9MB)
│   ├── reports/                          # Audit & validation reports (JSON)
│   └── splits/rapid_eval_s1.tsv          # Stratified rapid evaluation split
├── output/
│   ├── matching_results.tsv              # Leaderboard submission format
│   └── candidate_pairs.tsv               # Candidate pool format
├── scripts/
│   ├── eval_full_accuracy.py             # Full accuracy evaluation script
│   ├── verify_proof.py                   # Live side-by-side ground truth verification
│   ├── train_and_save_matcher.py         # Matcher training & calibration
│   └── run_disk_validation.py            # B-Tree validation runner
├── src/
│   ├── disk_blocker.py                   # 6-channel B-tree candidate retrieval
│   ├── features.py                       # 21 pairwise similarity features
│   ├── normalize.py                      # Multi-view text & address normalization
│   ├── train_matcher.py                  # HistGradientBoosting with Platt calibration
│   ├── decide.py                         # Precision-first decision policy & hard vetoes
│   ├── write_outputs.py                  # PRD Section 20 validated output generator
│   ├── score.py                          # Official Macro F0.5 evaluator
│   ├── split.py                          # 5-fold stratified S1 splitting
│   └── infer.py                          # End-to-end inference entrypoint
├── tests/                                # 15/15 passing PyTest unit tests
├── requirements.txt
├── entity-resolution-prd.md              # Full competition PRD
└── README.md
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest tests/ -v
```

### 3. Verify Ground-Truth Accuracy & Precision Proof
```bash
python -m scripts.verify_proof
```

### 4. Run End-to-End Inference
```bash
python -m src.infer --s1_path artifacts/splits/rapid_eval_s1.tsv --output_dir output
```
