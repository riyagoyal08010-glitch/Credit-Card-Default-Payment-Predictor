# 💳 Credit Card Default Payment Predictor (ML + Streamlit)

![Python](https://img.shields.io/badge/Language-Python-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📊 Machine Learning–Powered Credit Risk Prediction System

An interactive **Streamlit web application** that predicts whether a credit card customer is likely to **default on payment**, using demographic and financial data.

This project demonstrates an **end-to-end Machine Learning pipeline**, including data preprocessing, feature scaling, model training, and real-time deployment.

🚀 **Live Demo:**  
👉 https://credit-card-default-payment-predictor-082007.streamlit.app/

---

## 🚀 Features

- 📈 Predicts credit card default risk in real time  
- 🧠 Logistic Regression–based ML model  
- 🔄 Feature scaling using StandardScaler  
- ⚡ Instant predictions via Streamlit UI  
- 🎨 Clean UI with custom CSS styling  
- 🌐 Deployed and publicly accessible  

---

## 🧠 Machine Learning Workflow

1. Data preprocessing & feature selection  
2. Feature scaling using `StandardScaler`  
3. Model training using Logistic Regression  
4. Model serialization with `joblib`  
5. Streamlit-based web deployment  

---

## 🗂️ Project Structure

CREDIT_CARD_DEFAULT_PREDICTOR/
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   └── model_features.pkl
│
├── app.py
├── Credit_Card_Default.ipynb
├── requirements.txt
├── README.md
└── LICENSE

---

## 🛠️ Tech Stack

- 🐍 Python  
- 📊 Pandas & NumPy  
- 🤖 Scikit-learn  
- 🎈 Streamlit  
- 💾 Joblib  

---

## ⚙️ Installation & Local Setup

Clone the repository:
```bash
git clone https://github.com/your-username/credit-card-default-payment-predictor.git
pip install -r requirements.txt
streamlit run appp.py

## 📘 Project Explanation

This project focuses on solving a **credit risk assessment problem**, where the goal is to predict whether a credit card customer is likely to **default on their payment** in the future.

The application uses a **Machine Learning classification model** trained on historical customer data containing demographic and financial attributes. These features are processed and fed into a **Logistic Regression model**, which outputs the probability of default.

To ensure accurate predictions:
- Input features are scaled using **StandardScaler**## 📊 Exploratory Data Analysis (EDA) Insights

Exploratory Data Analysis was performed to understand the dataset, identify patterns, and determine which features contribute most to credit card default risk.

### 🔍 Key Observations

- **Class Imbalance**  
  The dataset shows an imbalance between defaulters and non-defaulters, with non-default cases being significantly higher. This is typical in financial risk datasets and was considered during model evaluation.

- **Payment History is Highly Predictive**  
  Features related to past payment behavior (such as delayed payments) have a strong correlation with default risk. Customers with frequent payment delays are more likely to default.

- **Credit Limit Impact**  
  Customers with **lower credit limits** tend to have a higher probability of default compared to those with higher limits.

- **Bill Amount vs Payment Amount**  
  Higher outstanding bill amounts combined with lower repayment amounts increase the likelihood of default.

- **Demographic Trends**  
  Certain demographic attributes (such as age group and education level) show moderate influence, but financial behavior features dominate prediction performance.

- **Feature Distribution & Outliers**  
  Several numerical features are right-skewed and contain outliers. Feature scaling was applied to normalize distributions and improve model stability.

### 📈 Conclusion from EDA

EDA revealed that **financial behavior features** are more influential than demographic features in predicting credit card default. These insights guided feature selection, scaling, and model choice, leading to better predictive performance.


- The same preprocessing steps used during training are reused during inference
- The trained model, scaler, and feature list are saved and loaded using `joblib`

The trained model is deployed using **Streamlit**, allowing users to interactively enter customer details and receive real-time predictions through a clean web interface.

This project demonstrates:
- Practical understanding of **Machine Learning fundamentals**
- Implementation of a **complete ML pipeline**
- Ability to deploy ML models as real-world applications
- Experience with **model persistence, inference, and UI integration**

Overall, this project bridges the gap between **model development** and **production-level deployment**, making it suitable for real-world ML use cases and portfolio demonstration.




