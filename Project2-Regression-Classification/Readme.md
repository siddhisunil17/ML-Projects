# 📊 Project 2 — Classification and Regression

This repository contains the solution for **Programming Assignment 2** of *CSE574: Introduction to Machine Learning* at the University at Buffalo.  
The project explores both **classification** (LDA, QDA) and **regression** techniques (OLS, Ridge, Non-linear Regression).

---

## 🧾 Contents

| File               | Description                                                 |
|--------------------|-------------------------------------------------------------|
| `script.py`        | All function implementations for Problems 1–5               |
| `video.mp4`        | Video explanation with visualizations and commentary        |
| `CSE574_Project2.pdf` | Official problem statement for the assignment           |
| `sample.pickle`    | 2D dataset for LDA/QDA experiments                          |
| `diabetes.pickle`  | Real-world medical dataset for regression tasks             |
| `requirements.txt` | Dependencies required to run the project                    |

---

## 📚 Assignment Breakdown

### 🔹 Problem 1: Gaussian Discriminators
- Implemented **LDA** and **QDA**.
- Compared decision boundaries and classification accuracies.
- Visualized results on 2D sample data.

### 🔹 Problem 2: Linear Regression (OLS)
- Built OLS regression with and without intercept.
- Computed **Mean Squared Error (MSE)** on train and test datasets.

### 🔹 Problem 3: Ridge Regression
- Added L2 regularization to OLS.
- Explored impact of **λ (lambda)** from 0 to 1.
- Visualized error trends to determine optimal λ.

### 🔹 Problem 4: Ridge via Gradient Descent
- Used `scipy.optimize.minimize` with custom objective function.
- Validated gradient-based weights against closed-form solution.

### 🔹 Problem 5: Non-linear Regression
- Applied polynomial mapping (p = 0 to 6).
- Compared performance under λ = 0 and λ = optimal.
- Analyzed test error behavior with increasing model complexity.

---

## 📈 Results Summary

- **Regularization** reduces overfitting and improves generalization.
- **Gradient descent** achieves comparable results to closed-form Ridge.
- **Non-linear regression** improves accuracy with optimal complexity (best p).
- **QDA** outperforms **LDA** in modeling non-linear class boundaries.

---

## ▶️ Running the Code

### ✅ Install dependencies:

pip install -r requirements.txt

###  ▶️ Run the script:

python script.py

