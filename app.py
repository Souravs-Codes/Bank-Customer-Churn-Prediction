import pickle
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = 'best_model.h5'
SCALER_PATH = 'scaler.pkl'
OHE_PATH = 'OHE_encoder_geography.pkl'
ENCODER_PATH = 'label_encoder_gender.pkl'

# Load model
model = load_model(MODEL_PATH)

# Load scaler
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Load One-Hot Encoder
with open(OHE_PATH, 'rb') as f:
    ohe = pickle.load(f)

# Load Gender Label Encoder
with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)


def preprocess_input(data):

    input_df = pd.DataFrame([data])

    # Encode Gender
    input_df['Gender'] = encoder.transform(input_df['Gender'])

    # One-hot encode Geography
    geo_encoded = ohe.transform(
        [[data['Geography']]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=ohe.get_feature_names_out(['Geography'])
    )

    # Remove original Geography and add encoded columns
    input_df = pd.concat(
        [
            input_df.drop('Geography', axis=1),
            geo_encoded_df
        ],
        axis=1
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    return input_scaled


# -------------------------
# Landing Page
# -------------------------
@app.route('/')
def index():

    return render_template('index.html')


# -------------------------
# Prediction Page
# -------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    result = None

    if request.method == 'POST':

        data = {
            'CreditScore': int(request.form['CreditScore']),
            'Geography': request.form['Geography'],
            'Gender': request.form['Gender'],
            'Age': int(request.form['Age']),
            'Tenure': int(request.form['Tenure']),
            'Balance': float(request.form['Balance']),
            'NumOfProducts': int(request.form['NumOfProducts']),
            'HasCrCard': int(request.form['HasCrCard']),
            'IsActiveMember': int(request.form['IsActiveMember']),
            'EstimatedSalary': float(request.form['EstimatedSalary'])
        }

        # Preprocess
        scaled_input = preprocess_input(data)

        # Prediction
        probability = model.predict(
            scaled_input,
            verbose=0
        )[0][0]

        probability = float(probability)

        result = {
            'probability': f"{probability * 100:.2f}%",
            'label': (
                'Likely to Churn'
                if probability > 0.5
                else 'Unlikely to Churn'
            )
        }

    return render_template(
        'predict.html',
        result=result
    )


if __name__ == '__main__':
    app.run(debug=True)