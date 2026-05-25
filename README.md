# 🏏 IPL 2026 Match Outcome Prediction  
### 🚀 LightGBM | Probability Calibration | Kaggle Project

---

## 📌 Overview  

This project builds a **machine learning model to forecast IPL 2026 match outcomes** using historical ball-by-ball data.

Instead of predicting a single winner, the model outputs a **probability distribution over multiple scoring scenarios**, making it suitable for **log-loss based competitions**.

---

## 🎯 Problem Statement  

Predict the probability of each outcome:

- 🔵 `A_small` → Team A scores low  
- 🔵 `A_big` → Team A scores high  
- 🔴 `B_small` → Team B scores low  
- 🔴 `B_big` → Team B scores high  

👉 Each prediction must be a **valid probability distribution (sum = 1)**

---

## ⚙️ Approach  

### 🔹 Data Processing  
- Converted **ball-by-ball data → match-level total runs**
- Cleaned inconsistent columns  
- Aggregated match-wise statistics  

---

### 🔹 Target Engineering  
- Used **quantile-based binning** to create balanced classes:
  - Class 0 → Very Low Score  
  - Class 1 → Low Score  
  - Class 2 → High Score  
  - Class 3 → Very High Score  

---

### 🔹 Feature Engineering  
Simple but effective statistical features:

- Total runs  
- Mean-centered runs  
- Standardized score  

---

## 🤖 Model  

- ⚡ **LightGBM (Multiclass Classification)**
- Tuned for:
  - Reduced overfitting  
  - Better generalization  
- Applied:
  - Subsampling  
  - Regularization (L1 & L2)  

---

## 🔥 Key Technique: Probability Calibration  

Raw model predictions were **overconfident**.

To fix this:

```python
probs = probs ** 0.25
probs = probs / probs.sum(axis=1, keepdims=True)
