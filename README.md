# 🏦 Bank Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a bank customer is **likely to churn** based on their demographic, financial, and account-related information.

The project uses an **Artificial Neural Network (ANN)** built with TensorFlow/Keras and provides a web-based prediction interface using Flask.

🔗 **Live Demo:** https://bank-customer-churn-prediction-rwfm.onrender.com/

---

## 📌 Project Overview

Customer churn is a major challenge for banks and financial institutions. Losing existing customers can be significantly more expensive than retaining them.

This project aims to predict the probability of a customer leaving the bank by analyzing information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Credit Card ownership
- Active membership status
- Estimated Salary

The trained ANN model processes the customer's information and returns:

- **Churn Probability**
- **Churn Prediction**

---

## 🎯 Project Objectives

The main objectives of this project are:

- Perform data preprocessing and feature engineering.
- Encode categorical variables appropriately.
- Scale numerical features.
- Build and train an Artificial Neural Network.
- Evaluate the model's performance.
- Save the trained model and preprocessing objects.
- Build a Flask-based web application.
- Deploy the application online.
- Provide an easy-to-use interface for customer churn prediction.

---

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Categorical Encoding
     │
     ├── Geography → One-Hot Encoding
     │
     └── Gender → Label Encoding
     │
     ▼
Feature Scaling
     │
     ▼
Train / Test Split
     │
     ▼
Artificial Neural Network
     │
     ▼
Model Evaluation
     │
     ▼
Saved Model (.h5)
     │
     ▼
Flask Web Application
     │
     ▼
Customer Input
     │
     ▼
Preprocessing
     │
     ▼
ANN Prediction
     │
     ▼
Churn Probability