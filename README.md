# 🎬 Credit Card Default Prediction – EDA & ML Project

An **end-to-end Exploratory Data Analysis (EDA) and Machine Learning project** using **Python, Scikit-learn, and Streamlit** to analyze customer financial behavior and predict **credit card default risk**.

---

## 🧩 Tech Stack & Tools

![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)
![Language](https://img.shields.io/badge/Language-Python-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Library](https://img.shields.io/badge/Library-Pandas-purple)
![Visualization](https://img.shields.io/badge/Visualization-Matplotlib-blue)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Project Overview

This project focuses on analyzing **credit card customer data** to understand the factors that lead to **payment default** and building a **machine learning model** to predict default risk.

The workflow combines:
- In-depth **EDA**
- Feature engineering & preprocessing
- Logistic Regression modeling
- Real-time **Streamlit deployment**

🚀 **Live App:**  
https://credit-card-default-payment-predictor-082007.streamlit.app/

---

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was conducted to understand data distributions, identify patterns, and determine the most influential features contributing to credit card default risk.

---

## 🔑 Key Insights

### ⭐ Payment History  
Customers with frequent **payment delays** are significantly more likely to default.

### ⭐ Credit Limit  
Lower credit limits are associated with higher default rates.

### ⭐ Bill vs Payment Amount  
High outstanding balances with low repayments increase default risk.

### ⭐ Demographics  
Demographic features have moderate influence; financial behavior dominates predictions.

### ⭐ Class Imbalance  
The dataset is imbalanced, reflecting real-world financial data.

---

## 🧠 Machine Learning Approach

- Feature scaling using **StandardScaler**
- Binary classification with **Logistic Regression**
- Model persistence using `joblib`
- Consistent preprocessing during inference

---

## 🧭 How to Use

### ▶️ Use the Web App
1. Open the live application:  
   👉 https://credit-card-default-payment-predictor-082007.streamlit.app/
2. Enter customer demographic and financial details in the input form.
3. Click **Predict** to view:
   - Default prediction (Yes / No)
   - Probability score

---

### ▶️ Run Locally

Clone the repository:
```bash

git clone https://github.com/your-username/credit-card-default-predictor.git
cd Credit_Card_Default_Predictor
pip install -r requirements.txt
streamlit run appp.py



