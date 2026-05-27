# 🏏 IPL 2026 Match Outcome Prediction  
### 🚀 Kaggle Competition Solution using LightGBM

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green?style=for-the-badge)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-orange?style=for-the-badge&logo=kaggle)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-red?style=for-the-badge)

---

# 📌 Overview

This project presents a **machine learning solution for the IPL 2026 Match Forecast Kaggle Competition**.

The objective is to predict **probability distributions** for multiple scoring outcomes rather than predicting a single winner.  
The model is optimized for **Log Loss evaluation**, focusing on balanced and calibrated probabilities.

---

# 🎯 Competition Objective

For each IPL match, predict probabilities for the following outcomes:

| Class | Description |
|------|-------------|
| 🔵 A_small | Team A scores a lower total |
| 🔵 A_big | Team A scores a higher total |
| 🔴 B_small | Team B scores a lower total |
| 🔴 B_big | Team B scores a higher total |

---

# ⚙️ Workflow

## 🔹 Data Processing
- Cleaned IPL ball-by-ball dataset
- Converted innings-level data into match-level statistics
- Aggregated total match runs

---

## 🔹 Target Engineering
Used **quantile-based binning** to create balanced multiclass targets:

- Class 0 → Very Low Score
- Class 1 → Low Score
- Class 2 → High Score
- Class 3 → Very High Score

---

# 🧠 Feature Engineering

Generated lightweight statistical features:

- Total match runs
- Mean-centered runs
- Standardized scoring metrics

---

# 🤖 Model

## ✅ LightGBM Multiclass Classifier

The project uses **LightGBM** for:

- Fast training
- Efficient tabular learning
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

Raw boosting predictions were highly overconfident.

To improve probability distribution and Log Loss performance:

```python
probs = probs ** 0.25
probs = probs / probs.sum(axis=1, keepdims=True)
```

This significantly improved prediction calibration.

---

# 📊 Submission Format

Final submission contains:

| match_id | A_small | A_big | B_small | B_big |
|----------|----------|--------|----------|--------|

✔ All values are probabilities  
✔ Each row sums to 1  
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

# 🚀 Installation & Setup

## 🔹 Clone Repository

```bash
git clone https://github.com/your-username/IPL-Match-Prediction.git
cd IPL-Match-Prediction
```

---

## 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔹 Run Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
final_lgbm_notebook.ipynb
```

Run all cells to generate:

```bash
submission.csv
```

---

# 📦 Requirements

```txt
numpy==1.26.4
pandas==2.2.2
lightgbm==4.3.0
scikit-learn==1.4.2
scipy==1.13.1
joblib==1.4.2
threadpoolctl==3.5.0
python-dateutil==2.9.0.post0
pytz==2024.1
tzdata==2024.1
```

---

# 🏆 Results

✅ Valid Kaggle submission  
✅ Balanced multiclass probability outputs  
✅ Stable baseline forecasting pipeline  
✅ Competition-ready workflow

---

# 🧩 Key Learnings

- Probability calibration is critical for Log Loss competitions
- Balanced target engineering improves multiclass prediction
- Even lightweight statistical features can provide strong baseline performance

---

# 🔮 Future Improvements

- Team-level historical performance
- Venue & pitch analysis
- Player form tracking
- Ensemble models (XGBoost + LightGBM)
- Advanced cricket-specific features

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

This repository contains:
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
