# 📊 Data Mining Project – Vrije Universiteit Amsterdam

This repository contains all code, data, and outputs for the **group assignments** of the *Data Mining* course at **Vrije Universiteit Amsterdam**.

---
## 📁 Folder Structure

```text
├── raw_data/              # All input datasets
├── src/                   # Source code for Assignment 1
│   ├── classification/    # Classification and RNN regression models
│   ├── regression/        # Random Forest regression
│   └── data/              # Data preprocessing scripts
├── 2/                     # Code and analysis for Assignment 2
│   └── eda/               # Exploratory Data Analysis and feature engineering
├── fig/                   # Output plots and visualizations
├── presentation.ipynb     # EDA notebook (Assignment 1)
└── README.md              # Project description and instructions
---
```

## 📦 Dataset Overview

All datasets are stored in the `raw_data/` folder.

### 📍 Assignment 1 Datasets:
- `cleaned_data_daily_summary_mood_imputed_sliding_window.csv`  
  ➤ Used for regression with sliding window (RNN, Decision Tree)

- `mood_classified_sliding_window.csv`  
  ➤ Used for binary classification with RNN and Random Forest

- `mood_classified.csv`  
  ➤ Classification without sliding window

- `cleaned_data_daily_summary_mood_imputed.csv`  
  ➤ Regression without sliding window

---

## 🧠 Algorithms Implemented

### 🔷 Assignment 1 – Mood Prediction from Smartphone Usage
Located in `src/`:

- **Classification:**
  - Random Forest
  - RNN (TensorFlow & PyTorch)

- **Regression:**
  - Decision Tree
  - Random Forest
  - RNN (regression)

- Includes hyperparameter tuning scripts and evaluation code

---

### 🔶 Assignment 2 – Hotel Recommendation System

Located in `2/`:

- `2/eda/` contains:
  - Exploratory data analysis
  - Feature engineering
  - Data preprocessing insights
  - Bias detection and handling (e.g., for `srch_query_affinity_score`)

- Model implementation (in subfolders):
  - LightGBM with LambdaRank objective
  - CatBoost
  - KNN (benchmark)
  - Evaluation metrics: **NDCG@k**, **Hit@1**, and score visualizations
  - Feature impact analysis and cold-start handling strategy

---

## 🧰 Tools and Libraries

All code is written in **Python**, using:

- `scikit-learn`
- `LightGBM`, `CatBoost`
- `TensorFlow`, `PyTorch`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 📬 Contact

For questions or contributions, feel free to open an issue or contact any group member directly.



