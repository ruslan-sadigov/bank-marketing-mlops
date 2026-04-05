# Technical Documentation — Bank Marketing Campaign Prediction

## 1. Executive Summary

This project predicts whether a bank customer will subscribe to a term deposit following a marketing campaign. The system is trained on the UCI Bank Marketing dataset and deployed as a FastAPI backend with a Streamlit frontend, containerized via Docker Compose.

**Key characteristics:**
- **Task**: Binary classification
- **Target**: `y` ∈ {no, yes} (encoded as `y_numeric` ∈ {0, 1})
- **Model**: XGBoost classifier (tuned via Optuna)
- **Performance**: ~84% accuracy, ~0.92 ROC-AUC, ~0.61 recall on positive class
- **Production features**: 15 (9 categorical, 6 numerical) — `duration` excluded

---

## 2. Dataset

### 2.1 Source
UCI Bank Marketing Dataset — `bank-full.csv` (45,211 records, 17 columns).

### 2.2 Class Distribution
| Class | Count | Percentage |
|---|---|---|
| No (0) | 39,922 | 88.30% |
| Yes (1) | 5,289 | 11.70% |

Severe class imbalance (~7.5:1 ratio) addressed via XGBoost's `scale_pos_weight` parameter.

### 2.3 Feature Summary

**Numerical (6):** `age`, `balance`, `day`, `campaign`, `pdays`, `previous`

**Categorical (9):** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

**Excluded feature:** `duration` — while strongly correlated with the target (Spearman r = 0.34), it is only known *after* a call is made, leading to target leakage in a pre-call prediction scenario. The production model is trained without it to reflect realistic deployment conditions.

---

## 3. Exploratory Data Analysis

### 3.1 Categorical Associations (Cramér's V with target)

| Feature | Cramér's V |
|---|---|
| poutcome | 0.312 |
| month | 0.260 |
| contact | 0.151 |
| housing | 0.139 |
| job | 0.136 |
| education | 0.073 |
| loan | 0.068 |
| marital | 0.066 |
| default | 0.022 |

**Key finding:** `poutcome` (previous campaign outcome) and `month` are the strongest categorical predictors. `default` has negligible association.

### 3.2 Numerical Correlations (Spearman)

| Feature | Correlation | Notes |
|---|---|---|
| duration | 0.343 | Strongest — but excluded (see §2.3) |
| pdays | -0.019 | Days since last contact |
| previous | ~0.09 | Number of previous contacts |
| balance | 0.053 | Average yearly balance |
| day | -0.028 | Day of month |
| age | -0.009 | Customer age |

### 3.3 Observations from Notebook

- `pdays == -1` indicates customers never previously contacted (~82% of dataset)
- `poutcome = "unknown"` is highly correlated with `pdays = -1`
- `contact = "unknown"` is common and semantically similar to `cellular`
- Outlier in `previous` column (value 275) investigated but kept in production

---

## 4. Modeling Approach

### 4.1 Candidate Models Evaluated

Four models were trained and compared in the notebook:

| # | Model | Duration Feature | Status |
|---|---|---|---|
| 1 | CatBoost | No | Baseline |
| 2 | XGBoost | No | **Production** |
| 3 | CatBoost | Yes | Reference |
| 4 | XGBoost | Yes | Reference |

**Decision:** XGBoost without duration chosen for production. Duration-based models have target leakage; CatBoost did not outperform XGBoost in this context.

### 4.2 Preprocessing Pipeline

Implemented as a scikit-learn `Pipeline` with two stages:

```
Pipeline:
  ├── preprocessor: ColumnTransformer
  │     ├── encoder: OrdinalEncoder
  │     │     - handle_unknown="use_encoded_value"
  │     │     - unknown_value=-1
  │     │     - applied to: 9 categorical features
  │     └── remainder: passthrough
  │           - applied to: 6 numerical features
  └── model: XGBClassifier
```

**Why OrdinalEncoder over OneHotEncoder?** XGBoost handles ordinal-encoded categorical features efficiently through its tree-splitting algorithm, avoiding the dimensionality explosion of one-hot encoding.

**Handling unknown categories:** `unknown_value=-1` allows the model to gracefully handle categories not seen during training — critical for production robustness.

### 4.3 Train/Test Split

- **Strategy:** Stratified 80/20 split
- **Random state:** 42 (reproducibility)
- **Train:** 36,168 samples
- **Test:** 9,043 samples (7,985 "no", 1,058 "yes")

Split performed once in `src/data/make_dataset.py` and saved to three files:

| File | Contains `y_numeric` | Purpose |
|---|---|---|
| `bank_train.csv` | Yes | Model training and evaluation |
| `bank_test.csv` | Yes | Model evaluation and visualization |
| `bank_test_sample.csv` | No | Frontend upload simulation (realistic production input) |

### 4.4 Hyperparameter Tuning

- **Framework:** Optuna
- **Trials:** 50
- **Cross-validation:** StratifiedKFold (3 splits)
- **Objective:** Maximize F1-score on positive class

**Search space:**

| Parameter | Range |
|---|---|
| n_estimators | 100–1000 |
| learning_rate | 0.01–0.3 (log scale) |
| max_depth | 3–12 |
| reg_lambda | 1–10 |
| subsample | 0.5–1.0 |
| colsample_bytree | 0.5–1.0 |
| min_child_weight | 1–10 |

**Final hyperparameters:**

```python
XGBClassifier(
    n_estimators=425,
    learning_rate=0.0122498,
    max_depth=12,
    reg_lambda=8.681178,
    subsample=0.788238,
    colsample_bytree=0.596907,
    min_child_weight=9,
    scale_pos_weight=<computed>,
    random_state=42,
    eval_metric="logloss",
)
```

`scale_pos_weight` is computed dynamically from the training set: `(y_train == 0).sum() / (y_train == 1).sum()` ≈ 7.5.

---

## 5. Model Performance

### 5.1 Classification Report (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No (0) | 0.944 | 0.871 | 0.906 | 7,985 |
| Yes (1) | 0.385 | 0.612 | 0.473 | 1,058 |
| **Accuracy** | | | **0.840** | 9,043 |
| Macro Avg | 0.665 | 0.741 | 0.689 | 9,043 |
| Weighted Avg | 0.879 | 0.840 | 0.855 | 9,043 |

**ROC-AUC:** ~0.92

### 5.2 Interpretation

- **High recall on positive class (0.61)** — the model catches most customers who will subscribe, which is business-critical (missing a subscriber costs more than a wasted call)
- **Lower precision on positive class (0.39)** — ~61% of predicted subscribers won't actually subscribe, but this is an acceptable tradeoff given the class imbalance
- **ROC-AUC of 0.92** — strong ranking ability regardless of threshold choice
- Decision threshold could be adjusted upward in production to favor precision if the business prefers fewer false positives

### 5.3 Feature Importance

XGBoost's built-in `feature_importances_` (mean gain across all trees) is extracted from the trained pipeline and plotted in `reports/figures/feature_importance.png`. Feature names follow the ColumnTransformer output order: categorical features first (OrdinalEncoder), then numerical (passthrough).

**Expected ranking (consistent with EDA):**

| Rank | Feature | Basis |
|---|---|---|
| High | `poutcome` | Strongest categorical predictor (Cramér's V = 0.312) |
| High | `month` | Second strongest categorical (Cramér's V = 0.260) |
| High | `pdays` | Days since last contact — high signal for repeat campaigns |
| Mid | `contact` | Cramér's V = 0.151 |
| Mid | `housing` | Cramér's V = 0.139 |
| Mid | `balance` | Spearman r = 0.053 |
| Low | `default` | Near-zero association (Cramér's V = 0.022) |

> The authoritative ranking is in `feature_importance.png` — EDA associations are computed on raw data, while XGBoost importance reflects learned split gain within the full pipeline.

### 5.4 Model Artifacts

Serialized pipeline saved as `backend/models/bank_xgb.pkl.gz` using `gzip` + `pickletools.optimize()` for minimal file size. Entire `Pipeline` object (preprocessor + model) is pickled together — inference code loads one artifact and calls `.predict()`.

---

## 6. System Architecture

### 6.1 Service Topology

```
┌─────────────┐      HTTP/JSON      ┌─────────────┐
│   User      │◄───────────────────►│  Streamlit  │
│ (Browser)   │                     │  Frontend   │
└─────────────┘                     │  :8501      │
                                    └──────┬──────┘
                                           │
                                           │ POST /predict
                                           │ (multipart)
                                           ▼
                                    ┌─────────────┐
                                    │  FastAPI    │
                                    │  Backend    │
                                    │  :8000      │
                                    └──────┬──────┘
                                           │
                                           ▼
                                    ┌─────────────┐
                                    │ XGBoost     │
                                    │ Pipeline    │
                                    │ (.pkl.gz)   │
                                    └─────────────┘
```

Both services run on a shared Docker bridge network (`app_network`) and communicate via service names.

### 6.2 Backend (`backend/app.py`)

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns service status + UTC/Baku timestamps |
| POST | `/predict` | Accepts CSV/Excel upload, returns predictions |

**Request flow for `/predict`:**
1. Validate file extension (`.csv`, `.xlsx`, `.xls`)
2. Read file content into bytes
3. Call `predict_model.main(file_content, filename)`
4. Pipeline loads data → extracts `ALL_FEATURES` → predicts → converts to Python `int` list
5. Return JSON with predictions, count, and processing time

**Response format:**
```json
{
  "status": "success",
  "message": "Predictions generated successfully",
  "data": {
    "predictions": [0, 1, 0, 1],
    "probabilities": [0.1428, 0.8732, 0.2341, 0.7651],
    "num_predictions": 4,
    "processing_time_seconds": 0.234
  }
}
```

**CORS:** Configured to allow all origins (development-friendly; should be restricted in production).

### 6.3 Frontend (`frontend/app.py`)

**Key components:**
- File uploader (sidebar)
- Backend API URL configuration (sidebar)
- Decision threshold slider (sidebar, 0.0–1.0, default 0.5)
- Dataset overview metrics (rows, columns, size)
- Data preview (first 100 rows)
- Optional statistical summary (`df.describe()`)
- Column validation against `REQUIRED_COLUMNS` before API call
- Prediction results table with probability scores and "No"/"Yes" labels
- Subscription rate metric (recalculates live when threshold changes)
- CSV download of predictions merged with original data

**Session state management:**
- `file_key` — tuple of (filename, size) for change detection
- `file_bytes` — raw bytes for API submission
- `df` — parsed DataFrame for preview
- `results` — API response

### 6.4 ML Pipeline Code Structure

```
backend/src/
├── data/
│   └── make_dataset.py       # Generates train/test CSVs from raw data
├── features/
│   └── build_features.py     # Feature column definitions (constants)
├── models/
│   ├── train_model.py        # Training pipeline
│   └── predict_model.py      # Inference function (called from API)
└── visualization/
    └── visualize.py          # Evaluation plots and reports
```

---

## 7. Deployment

### 7.1 Containerization

Two Dockerfiles, one per service. Both use `ghcr.io/astral-sh/uv:bookworm-slim` as the base image for fast UV-based dependency installation.

**Key build steps (backend):**
1. Install system dependencies (`clang`, `git`)
2. Copy backend source
3. Run `uv sync --no-dev` (production dependencies only)
4. Launch `uvicorn` on port 8000

### 7.2 Docker Compose

Two services (`backend`, `frontend`) share:
- A bridge network (`app_network`) enabling service-name-based communication
- Named volumes for virtual environments (isolated from host OS)
- Bind mounts for source code (live reload in development)

### 7.3 CI (GitHub Actions)

**Workflow:** `.github/workflows/ci-build.yaml`

**Jobs:**

| Job | Trigger | Runner | Purpose |
|---|---|---|---|
| `ci` | Push + PR to main | `ubuntu-latest` | Run Ruff linting |

Ruff is invoked via `uvx` (no project install needed) and targets `backend/` and `frontend/` directories, picking up lint rules from `backend/pyproject.toml`.

---

## 8. Reproducibility

To regenerate the model from scratch:

```bash
cd backend

# 1. Regenerate train/test split
uv run python -m src.data.make_dataset

# 2. Train the model
uv run python -m src.models.train_model

# 3. Regenerate evaluation reports and figures
uv run python -m src.visualization.visualize
```

Random seed `42` is used throughout (train/test split, XGBoost model) for deterministic results.

Generated evaluation artifacts (saved to `backend/reports/figures/`):
- `classification_report.png` — precision/recall/F1 heatmap
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`
- `class_distribution.png`
- `categorical_associations.png` (Cramér's V)
- `numeric_correlations.png` (Spearman)

---

## 9. Limitations & Future Work

**Current limitations:**
- `duration` excluded prevents using a strong signal when it *is* available (post-call scenarios)
- No drift detection — model assumes inference-time data distribution matches training
- No model versioning — retraining overwrites the single `.pkl.gz` artifact
- Decision threshold configurable via frontend slider but not persisted across sessions

**Possible improvements:**
- Track feature importance drift between retraining runs
- Add a separate model for post-call scenarios that *does* include `duration`
- Implement MLflow or similar for experiment tracking and model versioning
