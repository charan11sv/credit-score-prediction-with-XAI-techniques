# ~~~~~~~~"Each notebooks in each folder have their own set of documentations in '.md' format that can guide you with those notebooks."~~~~~~~~
# ~~~~~~~~"You may not find that many visualizations in the notebooks as these are presented as is when i worked, focussing on conducting max no of experiments. You may refer to a small sample project i built to demonstarte the visualization, MLOPS skills. For this and others you can rely on the documentation files to understand my work. "~~~~~~~~
# credit-score-prediction-with-XAI-techniques

Predict a continuous **credit score `y`** from **304 numeric features** (`x001 … x304`) with rigorous **missing-data analysis**, **MICE** imputation, a strong **LightGBM** baseline, **feature selection (RFE)**, a **Neural Monotonic Additive Model (NMAM)** for domain-aligned interpretability, and **Explainable AI (XAI)** (primarily **SHAP / TreeSHAP**, plus intrinsic diagnostics and recommended extensions).

> **Goal.** Build a trustworthy, high-accuracy credit-score regressor that  
> 1) treats missingness correctly at scale,  
> 2) establishes a regularized, high-performing tree-ensemble baseline,  
> 3) demonstrates monotonic constraints where domain rules apply, and  
> 4) provides transparent explanations for both global and individual predictions.

---

## Quick Summary (held-out 20k validation)

| Model      |    MAE   |   RMSE   |    R²    |
|-----------:|---------:|---------:|---------:|
| **LightGBM** | **17.1429** | **24.6641** | **0.9570** |
| NMAM       |  32.4728 |  43.7233 |  0.8647  |

**Why this matters:** LightGBM delivers excellent accuracy on numeric tabular data after sound imputation. The NMAM showcases **monotonic constraints** for policy/compliance—even if it underperforms here without further tuning and normalization.

---

## Contents

- [Project structure](#project-structure)  
- [Data & schema](#data--schema)  
- [Problem framing & metrics](#problem-framing--metrics)  
- [Missing data analysis: MCAR/MAR/MNAR](#missing-data-analysis-mcarmarmnar)  
  - [MCAR heuristic (“Little’s test” style)](#mcar-heuristic-littles-test-style)  
  - [MAR checks](#mar-checks)  
  - [MNAR checks](#mnar-checks)  
  - [Decision: MICE over PMM](#decision-mice-over-pmm)  
- [Imputation pipeline (MICE)](#imputation-pipeline-mice)  
- [Leakage controls & data hygiene](#leakage-controls--data-hygiene)  
- [Outlier policy](#outlier-policy)  
- [Train / validation protocol](#train--validation-protocol)  
- [Baseline model — LightGBM](#baseline-model--lightgbm)  
  - [Hyperparameters & rationale](#hyperparameters--rationale)  
  - [Training logs (excerpt)](#training-logs-excerpt)  
  - [Validation results](#validation-results)  
  - [Built-in importance vs SHAP](#built-in-importance-vs-shap)  
- [Feature selection — RFE](#feature-selection--rfe)  
- [Monotonic model — Neural Monotonic Additive Model (NMAM)](#monotonic-model--neural-monotonic-additive-model-nmam)  
  - [Architecture & constraints](#architecture--constraints)  
  - [Training setup](#training-setup)  
  - [Results](#results)  
  - [Interpreting NMAM partials](#interpreting-nmam-partials)  
- [Explainability — SHAP / TreeSHAP & other XAI](#explainability--shap--treeshap--other-xai)  
  - [Global explanations](#global-explanations)  
  - [Local explanations](#local-explanations)  
  - [Robustness, stability & caveats](#robustness-stability--caveats)  
  - [Recommended XAI extensions](#recommended-xai-extensions)  
- [Why LightGBM leads, when NMAM matters](#why-lightgbm-leads-when-nmam-matters)  
- [Practical guidance & recipes](#practical-guidance--recipes)  
- [Error analysis, calibration & governance](#error-analysis-calibration--governance)  
- [Reproducibility & versions](#reproducibility--versions)  
- [Appendix — selected raw outputs](#appendix--selected-raw-outputs)  
- [(Optional) How to reproduce locally](#optional-how-to-reproduce-locally)

---

## Project structure

Single primary notebook:

1. **`credit-score (1).ipynb`**  
   End-to-end sequence: schema checks → **missingness analysis** → **MICE** imputation → **LightGBM** baseline (+ param sweeps) → **RFE (100 features)** → **Monotonic neural model (NMAM)** → **XAI (SHAP / TreeSHAP)**.

Artifacts produced during runs: console logs, printed metrics, and (when rendered) SHAP plots.

---

## Data & schema

- **Features:** strictly numeric `x001 … x304`  
- **Target:** `y` (continuous credit score)  
- **Shapes (from outputs):**  
  - **Train:** `(80,000, 305)` → 304 features + `y`  
  - **Validation (holdout):** `(20,000, 305)` (same columns)  
- **Dtypes:** approximately `float64 ≈ 41`, `int64 ≈ 264` (i.e., all numeric)

**Design choice:** Numeric-only design simplifies imputation and avoids encoding pitfalls; tree models excel here.

---

## Problem framing & metrics

**Task:** Supervised regression to predict `y` (credit score).

**Primary metrics (reported on held-out validation):**
- **MAE** — robust and interpretable in score units.  
- **RMSE** — penalizes large errors more strongly (captures tail risk).  
- **R²** — variance explained; quick comparative measure.

> **Why MAE & RMSE together?** MAE is less sensitive to outliers; RMSE highlights large-error regimes. Reporting both captures different risk views.

---

## Missing data analysis: MCAR/MAR/MNAR

A structured analysis informs the imputation strategy.

### MCAR heuristic (“Little’s test” style)

- **Approach:** Chi-square tests between **missingness indicators** (0/1 per feature) and other variables (or binned versions).  
- **Finding:** Many features (e.g., `x045`, `x057`, `x058`, `x098`, `x148`) show **p ≈ 0** → **NOT MCAR** (missingness depends on observed data).

> **Theory (MCAR).** Under **MCAR**, missingness is independent of both observed and unobserved values. True MCAR is rare; failing this check motivates **model-based imputation** that conditions on observed features.

### MAR checks

- **Approach 1:** Chi-square associations between a feature’s missingness and **binned** versions of other features.  
- **Approach 2:** **Spearman** correlations between missingness indicators and other numeric features.  
- **Outcome:** Many columns flagged **“✅ MAR Detected.”**

> **Theory (MAR).** Under **MAR**, missingness depends only on **observed** variables—ideal for multivariate imputation methods such as **MICE**.

### MNAR checks

- **Approach:** For columns with missingness, compare distributions on **observed subsets** via **KS-test** and **t-test** (heuristics that may hint at MNAR).  
- **Outcome:** Several columns (e.g., `x044`, `x148`, `x155`, `x162`, `x222`, `x223`, …) reported **“NOT MNAR”** (no strong evidence).

> **Caution (MNAR).** **MNAR** depends on the **unobserved** value and cannot be conclusively diagnosed from observed data alone. Lack of evidence ≠ evidence of lack; domain review still recommended.

### Decision: MICE over PMM

- Given high, widespread missingness and MAR-consistent signals, we chose **MICE** (IterativeImputer) rather than PMM.  
- **Rationale:** MICE leverages multivariate linear relations (with shrinkage), typically outperforming univariate or kNN-style imputers in high-dimensional numeric tables.

---

## Imputation pipeline (MICE)

- **Library:** `sklearn.impute.IterativeImputer` (a MICE implementation) with default **BayesianRidge** regressors.  
- **Protocol:**  
  1) **Fit** the imputer on **train only** (prevents leakage)  
  2) **Transform** the held-out validation with the same fitted imputer  
- **Convergence:** Iterative cycles run; “early stopping not reached” indicates conservative tolerances—acceptable here.

**Why BayesianRidge?** L2-style shrinkage stabilizes per-feature regressions under multicollinearity and many predictors.

---

## Leakage controls & data hygiene

- **No target leakage into imputation:** Imputer is fit on train; validation is only transformed.  
- **No label-dependent preprocessing on validation:** All fitting steps (imputer, feature selection, models) performed on **train only**.  
- **Consistent schema:** Column order/dtypes aligned across train/val.  
- **Reproducibility:** `random_state` fixed for LightGBM & RFE (NMAM determinism can vary unless deep-learning seeds + deterministic flags are enforced).

---

## Outlier policy

- **No explicit outlier removal** in this version.  
- **Reasoning:** Large samples + tree ensembles often handle heavy tails via splits; explicit trimming risks discarding informative extremes. Revisit if residual diagnostics flag leverage points.

---

## Train / validation protocol

- **Train file:** `CreditScore_train.csv` (80k rows)  
- **Validation file:** `CreditScore_test.csv` (20k rows)  
- After MICE:
  - `X_train = imputed_mice.drop('y', axis=1)`, `y_train = imputed_mice['y']`  
  - `X_val   = test_imputed_mice.drop('y', axis=1)`, `Y_val   = test_imputed_mice['y']`  
- **Shapes:** Train `(80,000, 304)`, Validation `(20,000, 304)`

---

## Baseline model — LightGBM

### Hyperparameters & rationale

    # Core settings used in the strong baseline
    num_leaves = 128        # rich leaf capacity; pairs well with many trees
    max_depth  = -1         # let trees grow as needed; regularize elsewhere
    learning_rate = 0.005   # small LR + many trees = smooth convergence
    n_estimators  = 10000   # with early stopping to prevent overfit

    # Regularization & efficiency
    subsample = 0.7
    feature_fraction = 0.6
    colsample_bytree = 0.6
    bagging_fraction = 0.7
    bagging_freq = 5
    lambda_l1 = 0.2
    lambda_l2 = 1.2
    early_stopping_rounds = 100

- **Small LR + many estimators** → stable, high-quality fit; early stopping controls overfitting.  
- **Row/feature subsampling & bagging** → lower variance and training cost.  
- **L1/L2** regularization → improved generalization.

### Training logs (excerpt)

    [100]  train l1: 66.26  rmse: 78.05   val l1: 66.78  rmse: 78.58
    [500]  train l1: 23.14  rmse: 30.03   val l1: 23.97  rmse: 31.34
    [1000] train l1: 17.49  rmse: 24.23   val l1: 18.98  rmse: 26.70
    [1500] train l1: 15.99  rmse: 22.24   val l1: 18.14  rmse: 25.77
    [2000] train l1: 15.02  rmse: 20.85   val l1: 17.77  rmse: 25.33
    [3000] train l1: 13.61  rmse: 18.84   val l1: 17.42  rmse: 24.92
    [3600] train l1: 12.92  rmse: 17.87   val l1: 17.29  rmse: 24.78
    [4200] train l1: 12.31  rmse: 17.02   val l1: 17.19  rmse: 24.67
    [4700] train l1: 11.84  rmse: 16.37   val l1: 17.12  rmse: 24.60

### Validation results

| Model      |    MAE   |   RMSE   |    R²    |
|-----------:|---------:|---------:|---------:|
| **LightGBM** | **17.1429** | **24.6641** | **0.9570** |

### Built-in importance vs SHAP

- **LightGBM importances** (gain/split) are fast sanity checks but can be **biased** by cardinality/correlation.  
- **SHAP (TreeSHAP)** gives **consistent**, locally accurate attributions; we rely on **SHAP** for explanations and treat built-ins as supporting diagnostics.

---

## Feature selection — RFE

- **Method:** `RFE` with `LGBMRegressor(n_estimators=500, random_state=42)`  
- **Goal:** retain **100 features** (`step=5`)  
- **Status:** Selected feature list printed (truncated in saved output). `X_train_selected` / `X_val_selected` constructed.

> **Planned follow-up:** Retrain LightGBM on the 100-feature subset; compare latency and accuracy vs full model. RFE often preserves accuracy while improving speed/model simplicity.

---

## Monotonic model — Neural Monotonic Additive Model (NMAM)

### Architecture & constraints

- **Motivation:** Some drivers should be **monotone** (e.g., higher income must not reduce score).  
- **Monotone features chosen:** `['x235', 'x236']`  
- **Design:**  
  - **MonotonicLinear** layers with **softplus**-constrained non-negative weights → enforce **non-decreasing** effect per constrained feature.  
  - Small **additive subnet** per monotone feature (learns flexible monotone shapes).  
  - **Non-monotone MLP** for other features.  
  - Concatenate → **linear head** → prediction.

> **Functional view:**  
> *score* ≈ Σᵢ **fᵢ**(monotone_featureᵢ) + **g**(other_features), with each **fᵢ** non-decreasing.

### Training setup

- **Optimizer:** AdamW (`lr=0.005`, `weight_decay=1e-5`)  
- **Scheduler:** CosineAnnealingLR (`T_max=200`)  
- **Loss:** MSE  
- **Batch size:** 1024  
- **Epochs:** 200  
- **Inputs:** MICE-imputed features (no StandardScaler applied in the recorded run)

**Training (excerpt):**

    Epoch [0/200],   Avg Loss: 192.7512
    Epoch [50/200],  Avg Loss: 38.2802
    Epoch [100/200], Avg Loss: 35.2240
    Epoch [150/200], Avg Loss: 33.0897
    Test Loss: 31.9768   # MSE

### Results

| Model |    MAE   |   RMSE   |    R²    |
|------:|---------:|---------:|---------:|
| NMAM  |  32.4728 |  43.7233 |  0.8647  |

> **Note:** Underperforms LightGBM here (likely due to missing feature scaling & limited depth). Demonstrates **monotonic constraints** valuable for compliance/interpretability.

### Interpreting NMAM partials

- The **monotone subnets** yield **per-feature partial response curves** (one per constrained feature).  
- Curves are **shape-constrained** (non-decreasing), aiding **policy checks** (e.g., “does increasing this driver always help or plateau?”).  
- These provide **face validity** for model-risk documentation and stakeholder review.

---

## Explainability — SHAP / TreeSHAP & other XAI

Primary explanations are from **SHAP / TreeSHAP** for the LightGBM model; we also rely on **intrinsic** diagnostics and provide recommended extensions.

### Global explanations

- **SHAP summary beeswarm** (on a 1,000-row validation sample):  
  - Ranks features by mean |SHAP| (**global importance**).  
  - **Color gradient** encodes feature value → shows **directionality** (whether high values push predictions up/down).  
- **SHAP bar plot (global):**  
  - Aggregates mean |SHAP| per feature → **stable ranking** of drivers.

**How to read:** If a feature shows mostly positive SHAP for higher values, higher values tend to **increase** predicted score; mostly negative implies **downward** pressure.

### Local explanations

- **SHAP waterfall / force plots** (single instance):  
  - Start from the **expected value** (dataset baseline) and add **feature contributions** to reach the prediction.  
  - Great for **case-level audit**, **exception/edge review**, and **challenge testing**.

**Intrinsic diagnostics (complementary):**
- **Gain / split count** importances (from LightGBM) — quick sanity checks.  
- **Per-tree path inspection** (optional) — understand specific decision routes for a given instance.

### Robustness, stability & caveats

- **Correlated features:** SHAP distributes credit among correlated features; interpret clusters of correlated features **collectively**.  
- **Sampling variance:** Beeswarm on 1,000 rows speeds plotting; for stability, aggregate across multiple random samples or CV folds.  
- **Monotone NMAM:** While SHAP is applied to LightGBM, NMAM’s **monotone partials** are directly interpretable and complement tree explanations.

### Recommended XAI extensions

- **Permutation importance** (model-agnostic robustness check).  
- **PDP/ICE** (Partial Dependence / Individual Conditional Expectation) to visualize average vs individual effects.  
- **ALE** (Accumulated Local Effects) to reduce correlation bias in marginal-effect estimates.  
- **SHAP interaction values** to surface pairwise interactions (costly but insightful).  
- **Gradient-based attributions** (e.g., saliency/Integrated Gradients) for NMAM if adopting deeper non-monotone trunks.

---

## Why LightGBM leads, when NMAM matters

**Why LightGBM wins here**
- Captures **non-linearities** & **interactions** without manual feature engineering.  
- Robust under **imputation noise** and heavy tails; **regularization + subsampling** curb overfit.  
- Excellent bias-variance trade-off for large, numeric-only tabular data.

**When NMAM matters**
- **Compliance** and **policy alignment:** enforce monotonicity for specific features; simplify stakeholder narratives.  
- **Interpretability:** monotone partials provide clear, regulation-friendly behavior.  
- **Stability:** monotone constraints reduce spurious oscillations near boundaries.

---

## Practical guidance & recipes

**Imputation (MICE)**
- Keep MICE under MAR-like patterns. If runtime grows, **RFE → MICE** (impute on a reduced feature set first).

**LightGBM tuning**
- Sweep around baseline:  
  - `num_leaves ∈ {64, 96, 128, 192}`  
  - `feature_fraction ∈ [0.5, 0.8]`, `subsample ∈ [0.6, 0.9]`  
  - `learning_rate ∈ {0.003, 0.005, 0.01}` with **early stopping**  
- Use **Optuna/Hyperopt** for efficient search; optimize **MAE**, monitor **RMSE**.

**RFE usage**
- Target **100–150** features; compare accuracy and **inference latency**. Prefer the smaller set if metrics stay within tolerance.

**NMAM hygiene**
- **Standardize** inputs (e.g., `StandardScaler`) before training.  
- Consider **Huber** or **MAE** loss if large errors are outlier-driven.  
- Add light **dropout** in the non-monotone trunk; explore **LR warm-up** or cosine restarts.

**XAI reporting bundle (recommended artifacts)**
- **Beeswarm** (global) + **bar** (global) + **3–5 waterfalls** (local) exported as images.  
- **Top-N SHAP table** (CSV) with mean |SHAP| and rank.  
- **NMAM partial curves** for each monotone feature (with axis units and, optionally, CI via bootstraps).

---

## Error analysis, calibration & governance

- **Residual diagnostics:**  
  - Error by **predicted score deciles** (mean absolute residual per decile).  
  - Check **heteroskedasticity**: RMSE/MAE by decile or by key drivers.  
- **Segment stability:**  
  - Error slices by **time**, **cohort/segment**, or **region** (if available).  
- **Uncertainty (optional):**  
  - Bootstrap **MAE/RMSE** on the holdout to compute **95% CIs** — helpful for governance sign-off.  
- **Score calibration (optional):**  
  - If downstream needs calibrated intervals, fit **isotonic** or **Platt** models over residuals or prediction intervals.  
- **Documentation:**  
  - Maintain **data lineage**, **assumptions**, **limitations**, and a **model change log** for subsequent versions.

---

## Reproducibility & versions

**Environment (from notebook prints):**
- LightGBM **4.5.0**  
- scikit-learn **1.2.2**  
- NumPy **1.26.4**  
- Pandas **2.2.3**  
- Python **3.10.12**

**Order of operations**  
1) Schema & EDA → 2) Missingness analysis → 3) **MICE fit on train**, transform validation →  
4) **LightGBM** baseline (+ sweeps) → 5) **RFE(100)** (prep) → 6) **NMAM** train/eval → 7) **XAI (SHAP / TreeSHAP)**.

**Randomness controls**  
- Fixed `random_state` for LightGBM & RFE; for deep learning, set seeds and deterministic flags if exact reproducibility is required.

---

## Appendix — selected raw outputs

- **Columns present:** `x001 … x304, y`  
- **Train shape:** `(80,000, 305)`  
- **Validation shape:** `(20,000, 305)`  
- **Missingness examples:** `x002: 17185`, `x003: 17185`, `x004: 17181`, `x005: 4867`, `x148: 33470` (others similar)  
- **LightGBM (held-out):** `MAE=17.1429`, `RMSE=24.6641`, `R²=0.9570`  
- **NMAM (held-out):** `MAE=32.4728`, `RMSE=43.7233`, `R²=0.8647`

---

## (Optional) How to reproduce locally

> The notebook performs all steps; these bullets outline the sequence.

1. Load `CreditScore_train.csv`, `CreditScore_test.csv`.  
2. Run missingness diagnostics (MCAR/MAR/MNAR heuristics).  
3. Fit **IterativeImputer (MICE)** on train; transform both train and validation.  
4. Train **LightGBM** with the baseline params and early stopping; record MAE/RMSE/R².  
5. (Optional) **RFE(100)** on train → retrain LightGBM on reduced set.  
6. Train **NMAM** on the imputed train; evaluate on validation.  
7. Compute **SHAP (TreeSHAP)** on the LightGBM model (sample ~1,000 rows for global plots); export beeswarm, bar, and waterfall plots.  
8. Export a **top-N SHAP CSV** and **NMAM partial plots**.
