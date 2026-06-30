# Geographic Generalization in Traffic Severity Prediction

A machine learning project investigating **geographic generalization** in traffic accident severity prediction.

Rather than optimizing performance on random train/test splits, this project evaluates whether a model trained on multiple U.S. cities can accurately predict accident severity in a **completely unseen city** while remaining robust under later temporal distribution shift.

---

## Motivation

Many accident prediction studies evaluate models using random train/test splits, allowing location-specific patterns to leak into both training and evaluation.

While this often produces impressive metrics, it does not answer an important deployment question:

> **Can a model trained in one set of cities generalize to a city it has never seen before?**

This project addresses that question using a geographically disjoint evaluation protocol designed to measure real-world robustness rather than in-sample accuracy.

---

## Key Features

- Geographic holdout evaluation
- Leakage-resistant feature engineering
- Group-based cross validation
- Temporal robustness evaluation
- Threshold optimization for deployment
- Interpretable baseline and gradient-boosted models

---

## Evaluation Strategy

The evaluation intentionally mimics a real deployment scenario.

```text
Training Cities
        │
        ▼
 Feature Engineering
        │
        ▼
 GroupKFold Validation
        │
        ▼
 Model Selection
        │
        ▼
 Boston Holdout Test
        │
        ▼
 Temporal Robustness Test
```

Boston is **never observed during training** and serves exclusively as the final evaluation city.

The selected model is then evaluated on later Boston data (2019–2023) to measure robustness under temporal distribution shift.

---

## Methodology

### Data Processing

- Removed approximately **102,000** duplicate accident records.
- Audited missing values, label stability, and temporal drift.
- Restricted model development to **2016–2018** after identifying structural changes in severity labels.
- Engineered cyclical temporal features and weekend indicators.
- Derived a generalized `Speed_Class` feature from road characteristics.
- Retained weather and infrastructure variables expected to generalize across cities.

### Leakage Prevention

To encourage geographic transferability, the model excludes:

- Street names
- GPS coordinates
- High-cardinality location identifiers
- Post-accident variables

Evaluation uses **GroupKFold by city**, preventing observations from the same city from appearing in both training and validation folds.

---

## Models

Two complementary models were evaluated:

- Logistic Regression (interpretable baseline)
- XGBoost (non-linear benchmark)

Performance was measured using:

- ROC-AUC
- PR-AUC
- F1-score
- Recall

The final decision threshold was optimized to better reflect deployment class prevalence.

---

## Results

| Evaluation | ROC-AUC | PR-AUC | F1 | Recall |
|---|---:|---:|---:|---:|
| Boston Holdout (2016–2018) | **0.801** | **0.688** | **0.721** | **0.776** |
| Boston Temporal Robustness (2019–2023) | **0.790** | — | — | **0.730** |

The selected model maintained strong predictive performance when transferred to a geographically unseen city while remaining relatively stable under later temporal distribution shift.

---

## Project Structure

```text
├── data/
│   ├── raw/
│   └── preprocessed/
├── models/
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_modeling.ipynb
└── scripts/
    └── preprocess.py
```

---

## Reproducing the Project

### Preprocess the dataset

```bash
python scripts/preprocess.py
```

### Optional preprocessing

```bash
python scripts/preprocess.py --post --boston
```

### Train and evaluate

Use the notebooks:

- `01_data_audit.ipynb`
- `02_modeling.ipynb`

to reproduce the complete feature engineering, model training, and evaluation pipeline.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

