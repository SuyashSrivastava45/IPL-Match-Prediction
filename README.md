# 🏏 IPL 2026 Match Outcome Prediction  
### 🚀 Kaggle Competition Solution using LightGBM

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-orange?style=for-the-badge&logo=kaggle)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-red?style=for-the-badge)

---

# 📌 Overview

This project presents a **machine learning solution for the IPL 2026 Match Forecast Kaggle Competition**.

The objective is to predict **probability distributions** for multiple scoring outcomes instead of predicting a single winner.  
The model focuses on generating **well-calibrated probabilities** optimized for **Log Loss evaluation**.

---

# 🎯 Competition Objective

For each IPL match, predict probabilities for the following outcomes:

| Class | Description |
|------|-------------|
| 🔵 A_small | Team A scores a lower total |
| 🔵 A_big | Team A scores a higher total |
| 🔴 B_small | Team B scores a lower total |
| 🔴 B_big | Team B scores a higher total |

The final submission must satisfy:

```python
A_small + A_big + B_small + B_big = 1
```

---

# ⚙️ Workflow

## 🔹 Data Processing
- Cleaned raw IPL ball-by-ball dataset
- Converted innings-level data into match-level statistics
- Aggregated total runs for each match

---

## 🔹 Target Engineering
Used **quantile-based binning** to create balanced multiclass targets:

- Class 0 → Very Low Score
- Class 1 → Low Score
- Class 2 → High Score
- Class 3 → Very High Score

This helped reduce class imbalance and improve probability learning.

---

# 🧠 Feature Engineering

Generated lightweight statistical features:

- Total match runs
- Mean-centered runs
- Standardized scoring metrics

These features provide baseline match scoring behavior for multiclass prediction.

---

# 🤖 Model

## ✅ LightGBM Multiclass Classifier

The project uses **LightGBM** because of:

- Fast training speed
- Efficient handling of tabular data
- Strong multiclass performance
- Better probability estimation

### Model Configuration

```python
LGBMClassifier(
    n_estimators=200,
    learning_rate=0.15,
    max_depth=3,
    num_leaves=15,
    objective='multiclass',
    num_class=4
)
```

---

# 🔥 Probability Calibration

Raw predictions from boosting models were highly overconfident.

To improve probability spread and Log Loss performance, **probability smoothing** was applied:

```python
probs = probs ** 0.25
probs = probs / probs.sum(axis=1, keepdims=True)
```

This significantly improved prediction calibration.

---

# 📊 Submission Format

The final submission file contains:

| match_id | A_small | A_big | B_small | B_big |
|----------|----------|--------|----------|--------|

✔ All values are probabilities  
✔ All rows are normalized  
✔ Fully competition-compatible

---

# 📁 Project Structure

```bash
📦 IPL-Match-Prediction
 ┣ 📜 final_lgbm_notebook.ipynb
 ┣ 📜 submission.csv
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

# 🚀 Results

✅ Successfully generated valid Kaggle submissions  
✅ Balanced probability distributions  
✅ Stable multiclass baseline model  
✅ Competition-ready pipeline

---

# 🧩 Key Learnings

- Probability calibration is critical in Log Loss competitions
- Balanced target engineering improves multiclass learning
- Even simple statistical features can produce competitive baseline results

---

# 🔮 Future Improvements

- Team-wise historical performance
- Venue & pitch conditions
- Player form analysis
- Ensemble models (XGBoost + LightGBM)
- Advanced feature engineering

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- LightGBM
- Scikit-learn

---

# 🏆 Kaggle Competition

### IPL 2026 Match Forecast Challenge

This repository contains my complete solution pipeline for the competition, including:
- Data preprocessing
- Feature engineering
- Model training
- Probability calibration
- Submission generation

---

# 👨‍💻 Author

## Suyash Prakash Srivastava
B.Tech Student | Machine Learning Enthusiast

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
