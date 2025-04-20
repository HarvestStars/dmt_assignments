# 📊 Data Mining Project – Vrije Universiteit Amsterdam

This repository contains the code and data for the **first group assignment** of the *Data Mining* course at **Vrije Universiteit Amsterdam**. The assignment uses an advanced dataset featuring smartphone usage data from **27 participants**.

---

## 📁 Folder Structure

- `raw_data/` – All datasets used for the project  
- `src/` – Source code organized into subfolders:
  - `classification/` – Classification and RNN regression models
  - `regression/` – Random Forest regression
  - `data/` – Data cleaning scripts
- `fig/` – Output plots and visualizations
- `presentation.ipynb` – EDA summary notebook

---

## 📦 Dataset Overview

All datasets are located in the `raw_data/` folder:

- `cleaned_data_daily_summary_mood_imputed_sliding_window.csv`  
  ➤ Used for **regression algorithms** (RNN and Decision Tree)  
  ➤ Preprocessed with:
  - Sliding window of size 5
  - Missing values imputed using variable means

- `mood_classified_sliding_window.csv`  
  ➤ Used for **classification algorithms** (RNN and Random Forest)  
  ➤ Target mood variable converted into **binary classes (0 and 1)**

- `mood_classified.csv`  
  ➤ Classification dataset **without** sliding window

- `cleaned_data_daily_summary_mood_imputed.csv`  
  ➤ Regression dataset **without** sliding window

---

## 🧠 Algorithms Implemented

All source code is stored in `src/`.

### 🔷 Classification & RNN Regression (in `src/classification/`):
- `RNN_trial.py` – RNN for regression
- `RNN_trial_torch.py` – PyTorch version of RNN regression
- `decision_tree.py` – Decision Tree for regression
- `tree_tuning.py` – Grid Search for Decision Tree hyperparameters
- `RandomForest.py` – Random Forest classifier
- `tuning_paras.py` – Hyperparameter tuning for Random Forest
- `RNN_classification.py` – RNN for binary classification

### 🔶 Regression (in `src/regression/`):
- `RandomForest.py` – Random Forest regressor

---

## 🧰 Packages and Tools

All code is written in **Python**, using the following libraries:

- [`scikit-learn`](https://scikit-learn.org/)
- [`TensorFlow`](https://www.tensorflow.org/)
- [`PyTorch`](https://pytorch.org/)
- `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 📈 Output & Results

- Visual outputs and performance plots are saved in the `fig/` folder
- Exploratory Data Analysis (EDA) can be found in:
  - [`presentation.ipynb`](./presentation.ipynb)

---

## 📬 Contact

For questions or feedback, feel free to open an issue or contact a group member.



