# 🏦 Bank Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a bank customer is likely to **leave the bank (churn)** based on their demographic, financial, and account information.

The project uses an **Artificial Neural Network (ANN)** built with TensorFlow/Keras and provides an interactive web interface using **Streamlit**.

## 🚀 Live Demo

🔗 **Try the application:**  
https://bank-customer-churn-prediction-brefuxpsjaxbaytacpshny.streamlit.app/
---

## 📌 Project Overview

Customer churn is an important problem for banks because losing existing customers can have a significant impact on revenue.

This project uses historical customer data to train an Artificial Neural Network that estimates the probability that a customer will leave the bank.

The application allows users to enter customer information and receive:

- Churn probability
- Stay probability
- Churn/Stay prediction
- Visual probability indicator
- Explanation of the prediction

---

## 🧠 Machine Learning Model

The prediction model is an **Artificial Neural Network (ANN)** implemented using:

- TensorFlow
- Keras

The model receives processed customer information and outputs a probability between `0` and `1`.

### Prediction Logic

```text
Probability >= 0.50
        ↓
Likely to Churn

Probability < 0.50
        ↓
Likely to Stay