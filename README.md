# Predicting Mortality in Heart Failure Patients

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ANN-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## What This Project Is About

Heart failure affects over 64 million people worldwide. A 2020 study in *BMC Medical Informatics* showed that machine learning models can predict patient survival from just two clinical markers — serum creatinine and ejection fraction. That finding is the motivation behind this project.

Using clinical records from 299 heart failure patients, this project builds and compares two machine learning models — a **Support Vector Machine (SVM)** and an **Artificial Neural Network (ANN)** — to predict the likelihood of death during the follow-up period. The goal is to understand whether routine blood work and cardiac measurements alone can distinguish survivors from non-survivors.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `heart_failure_clinical_records_dataset.csv` | 299 | Clinical records of heart failure patients with 12 features and a binary death outcome |

**Features:**

| Feature | Description |
|---------|-------------|
| `age` | Age of patient (years) |
| `creatinine_phosphokinase` | CPK enzyme level in blood (mcg/L) |
| `ejection_fraction` | % of blood pumped out per heartbeat |
| `platelets` | Platelet count (kiloplatelets/mL) |
| `serum_creatinine` | Serum creatinine level (mg/dL) |
| `serum_sodium` | Serum sodium level (mEq/L) |
| `time` | Follow-up period (days) |
| `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking` | Binary clinical/demographic flags |
| `DEATH_EVENT` | **Target** — 1: died · 0: survived |

**Source:** [Heart Failure Clinical Records — Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

---

## Pipeline

### 1. Exploratory Data Analysis
- Class distribution of `DEATH_EVENT` — visualised with a labelled count plot
- Full feature correlation heatmap (20×20) with annotation
- Age distribution grouped by death outcome
- **Swarm + Boxen plots** for 7 continuous features (`age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium`, `time`) — one plot per feature, split by `DEATH_EVENT`

### 2. Preprocessing
- Feature matrix `X`: all columns except `DEATH_EVENT`
- Target vector `y`: `DEATH_EVENT`
- **StandardScaler** applied to normalise all 12 features to zero mean and unit variance
- 70/30 train-test split using `train_test_split`
- Scaled feature distributions validated with a Boxen plot

### 3. Model 1 — Support Vector Machine (SVM)
- `sklearn.svm.SVC` with default RBF kernel
- Trained on scaled feature matrix
- Evaluated with full `classification_report` (precision, recall, F1 per class)

### 4. Model 2 — Artificial Neural Network (ANN)
Architecture:
```
Dense(16, activation='relu', input_dim=12)
Dense(8, activation='relu')
Dropout(0.25)
Dense(8, activation='relu')
Dense(1, activation='sigmoid')
```
- Optimizer: **Adam** | Loss: **Binary Crossentropy** | Metric: Accuracy
- Training: up to **100 epochs**, batch size 25, 25% validation split
- **Early stopping:** `patience=20`, `min_delta=0.001`, `restore_best_weights=True`
- Training and validation loss + accuracy curves plotted post-training
- Predictions thresholded at 0.5 for binary output
- Evaluated with `classification_report`

---

## Results

Both models are evaluated on the same held-out 30% test set.

| Model | Approach | Key Config |
|-------|----------|------------|
| SVM | Classical ML | RBF kernel, StandardScaler input |
| ANN | Deep Learning | 3 hidden layers, Dropout, Early Stopping |

---

## Tech Stack

```
Python 3 · pandas · numpy · matplotlib · seaborn · scikit-learn · tensorflow/keras · Jupyter Notebook
```

---

## Project Structure

```
heart-failure-mortality-prediction/
├── Predicting_Mortality_in_heart_failure_patients.ipynb  # Full notebook
├── heart_failure_clinical_records_dataset.csv            # Dataset
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/RIDA-IYENGAR/heart-failure-mortality-prediction.git
cd heart-failure-mortality-prediction
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook Predicting_Mortality_in_heart_failure_patients.ipynb
```

---

## Potential Next Steps

- Hyperparameter tuning for SVM (kernel type, C, gamma)
- Benchmark against Random Forest, XGBoost, and LightGBM
- SHAP or LIME for feature importance — identify which clinical markers drive prediction
- Cross-validation for more robust performance estimates on this small dataset (n=299)
- Reproduce the original paper's finding: test whether serum creatinine + ejection fraction alone match full-feature performance

---

## Reference

Chicco, D. & Jurman, G. (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making, 20(16).

---

## About

**Rida Iyengar** · Biotechnology Engineering, PES University  
[GitHub](https://github.com/RIDA-IYENGAR)

