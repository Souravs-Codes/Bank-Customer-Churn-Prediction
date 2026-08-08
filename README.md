🏦 Bank Customer Churn Prediction

<p align="center">

<b>{=html}An end-to-end Deep Learning project for predicting bankcustomer churn</b>{=html}<br>{=html} Built with TensorFlow/Keras anddeployed with Streamlit

</p>

<p align="center">

<a href="https://bank-customer-churn-prediction-brefuxpsjaxbaytacpshny.streamlit.app/">{=html}<img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo">{=html}</a>{=html}<a href="https://github.com/Souravs-Codes/Bank-Customer-Churn-Prediction">{=html}<img src="https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">{=html}</a>{=html}

</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white">{=html}<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white">{=html}<img src="https://img.shields.io/badge/Keras-3.x-D00000?style=flat-square&logo=keras&logoColor=white">{=html}<img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">{=html}<img src="https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">{=html}

</p>

🌐 Live Application

🚀 Open the Bank Customer Churn Predictor

Enter a customer's information and the application estimates theprobability that the customer may leave the bank.

Note: The prediction is intended for educational and demonstrationpurposes and should not be used as the sole basis for real-worldbanking decisions.

📌 Overview

Customer churn is one of the major challenges faced by banks andfinancial institutions. When customers leave, organizations can loserecurring revenue and valuable long-term relationships.

This project develops a binary classification model using anArtificial Neural Network (ANN) to estimate whether a bank customer islikely to churn based on demographic, financial, and account-relatedinformation.

The trained model is integrated into an interactive Streamlit webapplication, turning the machine learning model into a usableprediction system.

What the application provides

👤 Customer information input

🧹 Consistent preprocessing with the training pipeline

🔤 Gender label encoding

🌍 Geography one-hot encoding

📏 Feature scaling

🧠 Artificial Neural Network inference

📊 Churn probability

🟢 Stay probability

🚨 Churn/Stay classification

🎨 Interactive visual prediction interface

☁️ Cloud deployment through Streamlit Community Cloud

🎯 Problem Statement

Given a customer's demographic, financial, and account information,predict whether the customer is likely to leave the bank.

The model performs binary classification:

0 → Customer is predicted to stay
1 → Customer is predicted to churn

The neural network produces a probability between 0 and 1.

For the application:

Probability >= 0.50
        ↓
Likely to Churn

Probability < 0.50
        ↓
Likely to Stay

The threshold can be changed depending on the business objective.

🧠 Machine Learning Approach

The project follows a complete machine learning workflow:

Raw Customer Data
       │
       ▼
Data Cleaning & Exploration
       │
       ▼
Feature Selection
       │
       ▼
Categorical Encoding
       │
       ├── Gender → Label Encoding
       │
       └── Geography → One-Hot Encoding
       │
       ▼
Feature Scaling
       │
       ▼
Train / Validation / Test Workflow
       │
       ▼
Artificial Neural Network
       │
       ▼
Model Evaluation
       │
       ▼
Save Trained Model
       │
       ▼
Streamlit Application
       │
       ▼
Customer Churn Prediction

📋 Input Features

The application uses the following features:

Feature             Description

CreditScore       Customer's credit scoreGeography         Customer's geographical locationGender            Customer's genderAge               Customer's ageTenure            Number of years the customer has been with the bankBalance           Customer's account balanceNumOfProducts     Number of bank products used by the customerHasCrCard         Whether the customer has a credit cardIsActiveMember    Whether the customer is an active bank memberEstimatedSalary   Customer's estimated salary

🔄 Data Preprocessing

The preprocessing pipeline is important because the model expects thesame feature representation used during training.

1. Gender Encoding

The categorical Gender feature is transformed using a fittedLabelEncoder.

2. Geography Encoding

The Geography feature is transformed using a fitted OneHotEncoder.

This converts categorical locations into numerical columns suitable forthe neural network.

3. Feature Scaling

The resulting feature vector is transformed using the fitted scalerstored in:

scaler.pkl

Using the same fitted preprocessing objects during inference preventsinconsistencies between training and prediction.

🧬 Artificial Neural Network

The final prediction model is an Artificial Neural Network implementedwith TensorFlow/Keras.

The trained model is stored as:

best_model.h5

At inference time, the application:

Receives customer information.

Encodes categorical values.

Applies the saved scaler.

Passes the processed features to the ANN.

Obtains a churn probability.

Converts the probability into a churn/stay prediction.

📊 Prediction Output

The application presents the model output in a user-friendly form.

Churn Probability

Example:

Churn Probability: 73.42%

Stay Probability

The complementary probability:

Stay Probability = 100% - Churn Probability

Example:

Stay Probability: 26.58%

Final Prediction

🚨 Likely to Churn

or

🟢 Likely to Stay

🖥️ Application

The Streamlit application provides a graphical interface for interactingwith the trained model.

Main workflow

Enter Customer Details
          ↓
Click "Predict Customer Churn"
          ↓
Preprocess Input
          ↓
ANN Prediction
          ↓
Calculate Probability
          ↓
Display Result

🗂️ Project Structure

Bank-Customer-Churn-Prediction/
│
├── Datasets/
│   └── Churn_Modelling.csv
│
├── logs/
│   └── fit/
│
├── best_model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── OHE_encoder_geography.pkl
│
├── experiments.ipynb
├── Prediction.ipynb
│
├── streamlit_app.py
│
├── app.py
├── templates/
│   ├── index.html
│   └── predict.html
│
├── Procfile
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md

📁 Important Files

File                          Purpose

streamlit_app.py            Main Streamlit applicationbest_model.h5               Trained ANN modelscaler.pkl                  Saved feature scalerlabel_encoder_gender.pkl    Saved Gender encoderOHE_encoder_geography.pkl   Saved Geography encoderexperiments.ipynb           Model development and experimentationPrediction.ipynb            Prediction/testing workflowapp.py                      Flask-based version of the applicationtemplates/                  HTML templates used by the Flask versionrequirements.txt            Python dependenciesProcfile                    Process configuration for Flask deploymentDatasets/                   Dataset files

🛠️ Tech Stack

Programming

Python

Data Processing

Pandas

NumPy

Machine Learning

Scikit-learn

Deep Learning

TensorFlow

Keras

Visualization / Analysis

Matplotlib

TensorBoard

Web Application

Streamlit

Flask

Development

Jupyter Notebook

Visual Studio Code

Version Control

Git

GitHub

Deployment

Streamlit Community Cloud

⚙️ Installation

1. Clone the repository

git clone https://github.com/Souravs-Codes/Bank-Customer-Churn-Prediction.git

2. Navigate into the project

cd Bank-Customer-Churn-Prediction

3. Create a virtual environment

Windows:

python -m venv venv

4. Activate the environment

PowerShell:

.\venv\Scripts\Activate.ps1

Command Prompt:

venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt

▶️ Run the Streamlit Application

Start the application with:

streamlit run streamlit_app.py

The application will normally be available at:

http://localhost:8501

🧪 Run the Flask Version

The repository also contains a Flask implementation.

The Flask application can be started with:

python app.py

For production-style execution:

gunicorn app:app

The Streamlit version is the primary deployed interface for thisproject.

☁️ Deployment

The Streamlit application is deployed through Streamlit CommunityCloud.

Deployment flow:

GitHub Repository
       │
       ▼
Streamlit Community Cloud
       │
       ▼
Install requirements.txt
       │
       ▼
Load best_model.h5
       │
       ▼
Load preprocessing objects
       │
       ▼
Run streamlit_app.py
       │
       ▼
Public Web Application

Deployment Requirements

The repository must contain:

streamlit_app.py

requirements.txt

best_model.h5

scaler.pkl

label_encoder_gender.pkl

OHE_encoder_geography.pkl

The Python and TensorFlow versions should also be mutually compatiblewith the wheels available on the deployment platform.

📈 Model Evaluation

Model evaluation was performed during the experimentation stage usingthe project notebooks.

The repository intentionally does not hard-code an accuracy value inthis README unless it is taken directly from the final experiment. Thisavoids presenting an unverified performance number.

For the exact experimental results, refer to:

experiments.ipynb

Recommended metrics for evaluating this binary classification probleminclude:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

For churn prediction, accuracy alone is not enough, especially whenthe classes are imbalanced. Recall, precision, F1-score, and ROC-AUCshould also be considered.

🔍 Why Churn Probability Matters

A binary prediction alone can hide useful information.

For example:

Customer A → 51% churn probability
Customer B → 95% churn probability

Both could be classified as "Likely to Churn" using a 0.50 threshold,but Customer B represents a much stronger predicted risk.

A probability-based system therefore gives the user more information forpotential risk prioritization.

💡 Possible Business Use Case

A bank could use a churn prediction system to identify customers who mayrequire additional attention.

For example:

High Risk
   ↓
95% churn probability
   ↓
Priority retention action

Medium Risk
   ↓
65% churn probability
   ↓
Review customer engagement

Low Risk
   ↓
20% churn probability
   ↓
Normal customer management

In a real banking system, these decisions should use additionalbusiness rules, customer history, fairness checks, privacy controls,and human oversight.

🚀 Future Improvements

Possible improvements include:

Hyperparameter tuning

Cross-validation

Class imbalance handling

ROC-AUC visualization

Confusion matrix in the web application

SHAP-based model explainability

Feature importance / sensitivity analysis

Customer risk categories

Batch prediction from CSV

Prediction history

Database integration

Model monitoring

Automated model retraining

Better input validation

Authentication for production deployment

Automated CI/CD pipeline

⚠️ Limitations

This project has several limitations:

Dataset limitations

The model is dependent on the historical dataset used for training.Real-world customer behavior may differ from the training distribution.

Model limitations

An ANN can identify patterns in historical data, but it does notestablish causal relationships between customer characteristics andchurn.

Business limitations

A churn prediction should not automatically trigger financial orcustomer-service decisions without appropriate validation and businessrules.

Deployment limitations

The deployed application is intended primarily as a demonstration andportfolio project rather than a production banking system.

🔐 Privacy & Security

Do not enter real customer personally identifiable information (PII)into the public demo.

The application is designed for demonstration using non-sensitiveexample data.

A production implementation would require appropriate:

Data protection

Authentication

Authorization

Encryption

Audit logging

Access control

Regulatory compliance

🎓 What This Project Demonstrates

This project demonstrates an end-to-end workflow from experimentation todeployment:

Data Analysis
      ↓
Feature Engineering
      ↓
Data Preprocessing
      ↓
Artificial Neural Network
      ↓
Model Evaluation
      ↓
Model Serialization
      ↓
Prediction Pipeline
      ↓
Web Application
      ↓
Cloud Deployment

Skills demonstrated

Python programming

Data preprocessing

Feature engineering

Categorical encoding

Feature scaling

Neural network development

TensorFlow/Keras

Model serialization

Prediction pipelines

Streamlit development

Flask development

Git/GitHub

Cloud deployment

Basic ML system design

👨‍💻 Author

Sourav Mukherjee

B.Tech CSE (AI & ML)

GitHub:https://github.com/Souravs-Codes

🔗 Project Links

Repository:https://github.com/Souravs-Codes/Bank-Customer-Churn-Prediction

Live Demo:https://bank-customer-churn-prediction-brefuxpsjaxbaytacpshny.streamlit.app/

📜 Disclaimer

This project is developed for educational, learning, and portfoliopurposes.

The predictions generated by this application should not be consideredprofessional financial, banking, or customer-retention advice. Aproduction system would require extensive validation, monitoring,security controls, fairness evaluation, and domain-specific review.

<p align="center">

<b>{=html}Built with Python, TensorFlow, Keras and Streamlit🚀</b>{=html}

</p>