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
│ ├── logistic_regression_model.pkl
│ ├── scaler.pkl
│ └── model_features.pkl
│
├── appp.py
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


