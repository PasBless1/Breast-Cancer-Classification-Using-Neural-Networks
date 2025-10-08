# Breast Cancer Classification with Neural Networks: A Reproducible Baseline

## Project Overview
This project implements a deep learning pipeline to classify breast tumors as malignant or benign using the Wisconsin Breast Cancer Diagnostic dataset. The goal is to build an accurate, robust, and clinically relevant predictive system that can aid in the early detection of breast cancer using artificial neural networks.

## Motivation
Early diagnosis of breast cancer significantly increases the chances of successful treatment and patient survival. Traditional diagnostic methods are limited by human interpretation, which can lead to misclassification, especially when tumors have subtle or heterogeneous features. By automating the diagnostic process with neural networks trained on rich digital pathology data, this project aims to provide reliable, explainable, and scalable breast cancer screening.

## Dataset
Source: UCI Wisconsin Breast Cancer Diagnostic Dataset (via scikit-learn)

Records: 569 patients

Features: 30 numerical attributes extracted from digitized images of fine needle aspirate (FNA) of breast masses (e.g., mean radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension across mean, error, and worst-case values)

Target Classes: 0 (Malignant), 1 (Benign)

Class Balance: 212 malignant, 357 benign

All features are continuous, and there are no missing values, making this dataset ideal for training deep learning models for medical imaging tasks

## Methodology
**Data Preprocessing**

Converted and explored the dataset using pandas and numpy.

Standardized all features using StandardScaler to ensure robust and stable neural network training.

Employed an 80/20 stratified split for training and testing to maintain the class distribution in both sets.

**Model Architecture**

Sequential Feedforward Neural Network built using TensorFlow and Keras.

Input Layer: Flattened vector of 30 features.

Hidden Layer: Dense layer with 20 neurons and ReLU activation.

Output Layer: Dense layer with 2 neurons and sigmoid activation for binary classification.

Optimization: Adam optimizer and sparse categorical crossentropy loss function.

Training: 10 epochs with a 10% validation split to monitor overfitting and learning progress.

**Evaluation**

Accuracy on Test Data: 93.86%

Loss convergence and validation accuracy tracked over epochs, showing consistent improvement and minimal overfitting.

The predictive system uses probability-based outputs, with the argmax function mapping predictions to the relevant class label (malignant or benign).

**Results**
Achieved high test accuracy (93.86%), demonstrating the effectiveness of the neural network in distinguishing between malignant and benign cases.

The standardized feature pipeline and straightforward model architecture make this approach highly reproducible and extensible to other medical imaging datasets.

Provides interpretable decision support for clinicians, with rapid and automated assessment of digital pathology inputs.

## Discussion
**Strengths**
Utilizes well-curated, real-world dataset from digitized clinical data.

Achieves high accuracy with a relatively simple neural network structure.

End-to-end automated workflow for data standardization, training, evaluation, and prediction.

Easily extendable to multiclass or more complex architectures (e.g., deeper networks, convolutional models) for future research.

**Limitations**
Relatively small, single-institution dataset might limit generalizability for broader populations.

Current architecture does not provide uncertainty estimation or model explanation for each decision.

Results should be validated on external, more diverse datasets for regulatory or clinical deployment.

## Conclusion
This project demonstrates that accurate and robust breast cancer detection is achievable through careful feature engineering, data standardization, and streamlined neural network modeling. The pipeline presented offers a blueprint for digital diagnostic tools in medical imaging and can serve as a foundation for future research in clinical machine learning and digital pathology.


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


