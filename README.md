# Breast Cancer Classification with Neural Networks: A Reproducible Baseline

## Abstract

This repository presents a concise, **reproducible** baseline for binary breast cancer classification using a shallow neural network. The pipeline demonstrates end-to-end practice suitable for clinical ML prototyping: deterministic data partitioning, leakage-safe preprocessing, training with internal validation, held-out test evaluation, and deployment-ready single-sample inference. On the scikit-learn breast cancer dataset (569 samples, 30 features), the model achieves **93.9%** test accuracy (loss **0.169**), with a **best validation accuracy of 97.8%** (final validation loss **0.122**).

---

## Data & Problem

* **Task:** Binary classification (malignant vs. benign).
* **Dataset:** `sklearn.datasets.load_breast_cancer` (569 samples; 30 standardized numeric features).
* **Split:** **80/20** train/test, **stratified**, fixed random seed.

---

## Methods

* **Preprocessing:**

  * Train/test split with `train_test_split(stratify=y, random_state=42)`.
  * `StandardScaler` **fit on training data only**, reapplied to validation/test/inference to prevent leakage.
* **Model:**

  * Keras Sequential MLP: `Dense(20, ReLU) → Dense(1, Sigmoid)`.
  * Loss: binary cross-entropy; Optimizer: Adam; Metric: accuracy; Epochs: 10 (baseline setting).
* **Validation protocol:**

  * Keras `validation_split=0.1` on the training fold.
  * Learning curves (accuracy/loss) monitored to detect under/over-fitting.
* **Reproducibility:**

  * Seeds set for NumPy, Python `random`, and TensorFlow; deterministic split.

---

## Results

**Held-out Test Set**

* **Accuracy:** **93.9%**
* **Loss:** **0.169**

**Validation (from training fold)**

* **Best accuracy:** **97.8%**
* **Final loss:** **0.122**
* **Trend:** validation accuracy improved from **71.7% → 95.7%**, peaking at **97.8%**.

> Notes: Minor variations are expected across environments; the repository controls common randomness sources and uses a deterministic split.

---

## Inference Protocol (Deployment Hygiene)

* Persist the **fitted** `StandardScaler` from training.
* For any new sample: reshape → apply the same scaler → predict probability → map to class label (consistent with training targets).

---

## Reproducing the Experiments

```bash
# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install numpy pandas scikit-learn tensorflow matplotlib jupyter

# Run the notebook
jupyter notebook DL_Project_1_Breast_Cancer_Classification_with_NN.ipynb
```

---

## What This Demonstrates (Skills & Practice)

* **End-to-end ML engineering:** data ingestion → leakage-safe preprocessing → training/validation → held-out testing → single-sample inference.
* **Validation rigor:** internal validation split with learning-curve diagnostics, plus fixed seeds for reproducibility.
* **Deployment awareness:** consistent preprocessing at inference, correct probability-to-label mapping, and a clean path to packaging (scaler + model).

---

## Limitations & Future Work

* **Metrics:** add AUC-ROC/AUPRC, confusion matrix, precision/recall/F1, and **calibration** (ECE, Brier score).
* **Validation:** upgrade to **StratifiedKFold** or nested CV; report CIs via bootstrapping.
* **Regularization/Tuning:** explore width/depth, L2/Dropout, early stopping, LR schedules, class-imbalance handling.
* **Explainability & Robustness:** SHAP/permutation importance (on scaled features), sensitivity to noise/feature ablation.
* **Packaging:** persist scaler (`joblib`) and model (Keras `SavedModel`); provide a minimal **FastAPI** endpoint for inference.

---

## Citation / Acknowledgement

* Dataset accessed via **scikit-learn** (`load_breast_cancer`).
* Please cite this repository if you build upon the code or results.

---

If you’d like, I can add a **Figures** section (ROC, PR, calibration, confusion matrix) and a minimal **FastAPI** app in `api/` that loads the saved scaler/model and serves `/predict`.
