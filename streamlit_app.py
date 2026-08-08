import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.html("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(37, 99, 235, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(124, 58, 237, 0.16),
            transparent 30%
        ),
        #070b14;
}


/* Main container */

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid rgba(148, 163, 184, 0.12);
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 25px;
}

.sidebar-section {
    margin-top: 25px;
}

.sidebar-heading {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.sidebar-text {
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.7;
}


/* Hero */

.hero {
    padding: 55px;
    border-radius: 28px;

    background:
        linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.96),
            rgba(15, 23, 42, 0.94)
        );

    border: 1px solid rgba(148, 163, 184, 0.18);

    box-shadow:
        0 25px 70px rgba(0, 0, 0, 0.35);

    margin-bottom: 30px;
}


.hero-badge {
    display: inline-block;

    padding: 8px 16px;

    border-radius: 999px;

    background: rgba(59, 130, 246, 0.12);

    border: 1px solid rgba(59, 130, 246, 0.35);

    color: #60a5fa;

    font-size: 13px;

    font-weight: 700;

    letter-spacing: 0.5px;

    margin-bottom: 20px;
}


.hero h1 {
    font-size: 52px;
    line-height: 1.05;

    margin: 0;

    color: #f8fafc;

    font-weight: 800;
}


.hero h1 span {
    color: #60a5fa;
}


.hero p {
    max-width: 780px;

    margin-top: 20px;

    font-size: 18px;

    line-height: 1.7;

    color: #94a3b8;
}


/* KPI cards */

.kpi-container {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 16px;

    margin-bottom: 35px;
}


.kpi {
    padding: 24px;

    border-radius: 18px;

    background:
        rgba(15, 23, 42, 0.85);

    border:
        1px solid rgba(148, 163, 184, 0.15);

    box-shadow:
        0 10px 30px rgba(0,0,0,0.15);
}


.kpi-number {
    font-size: 30px;

    font-weight: 800;

    color: #60a5fa;
}


.kpi-label {
    margin-top: 5px;

    color: #94a3b8;

    font-size: 14px;
}


/* Section heading */

.section-title {
    font-size: 27px;

    font-weight: 800;

    color: #f8fafc;

    margin-top: 25px;

    margin-bottom: 10px;
}


.section-subtitle {
    color: #94a3b8;

    margin-bottom: 25px;
}


/* Prediction button */

div.stButton > button {
    width: 100%;

    height: 55px;

    border-radius: 14px;

    border: none;

    font-size: 17px;

    font-weight: 750;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    box-shadow:
        0 10px 25px rgba(37,99,235,0.25);
}


div.stButton > button:hover {
    border: none;

    color: white;

    background:
        linear-gradient(
            135deg,
            #1d4ed8,
            #6d28d9
        );
}


/* Result */

.result-safe {
    padding: 35px;

    border-radius: 24px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.15),
            rgba(6,78,59,0.25)
        );

    border:
        1px solid rgba(16,185,129,0.35);

    margin-top: 30px;
}


.result-danger {
    padding: 35px;

    border-radius: 24px;

    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.15),
            rgba(127,29,29,0.25)
        );

    border:
        1px solid rgba(239,68,68,0.35);

    margin-top: 30px;
}


.result-icon {
    font-size: 55px;
}


.result-title {
    font-size: 30px;

    font-weight: 800;

    margin-top: 10px;

    color: #f8fafc;
}


.result-text {
    color: #cbd5e1;

    font-size: 16px;

    margin-top: 10px;
}


/* Footer */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 13px;

    margin-top: 50px;

    padding-top: 20px;

    border-top:
        1px solid rgba(148,163,184,0.1);
}


/* Mobile */

@media (max-width: 900px) {

    .kpi-container {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .hero h1 {
        font-size: 40px;
    }

}

</style>
""")


# ============================================================
# LOAD MODEL AND PREPROCESSORS
# ============================================================

@st.cache_resource
def load_prediction_objects():

    model = load_model("best_model.h5", compile=False)

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("OHE_encoder_geography.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return (
        model,
        label_encoder_gender,
        onehot_encoder_geo,
        scaler
    )


# ============================================================
# LOAD OBJECTS
# ============================================================

try:

    (
        model,
        label_encoder_gender,
        onehot_encoder_geo,
        scaler
    ) = load_prediction_objects()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "The prediction model could not be loaded."
    )

    st.exception(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html("""
    <div class="sidebar-title">
        🏦 Churn Predictor
    </div>
    """)

    st.html("""
    <div class="sidebar-section">

        <div class="sidebar-heading">
            About
        </div>

        <div class="sidebar-text">
            This application predicts whether a bank
            customer is likely to leave the bank using
            an Artificial Neural Network.
        </div>

    </div>
    """)

    st.html("""
    <div class="sidebar-section">

        <div class="sidebar-heading">
            Model
        </div>

        <div class="sidebar-text">
            🧠 Artificial Neural Network
        </div>

    </div>
    """)

    st.html("""
    <div class="sidebar-section">

        <div class="sidebar-heading">
            Pipeline
        </div>

        <div class="sidebar-text">
            Customer Data<br>
            ↓<br>
            Encoding<br>
            ↓<br>
            Scaling<br>
            ↓<br>
            Neural Network<br>
            ↓<br>
            Churn Probability
        </div>

    </div>
    """)


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">

    <div class="hero-badge">
        🧠 ARTIFICIAL NEURAL NETWORK
        • CUSTOMER ANALYTICS
    </div>

    <h1>
        Predict Customer <span>Churn</span>
    </h1>

    <p>
        Identify customers who may be at risk of leaving
        a bank using an Artificial Neural Network trained
        on customer demographic, financial and account
        information.
    </p>

</div>
""")


# ============================================================
# KPI CARDS
# ============================================================

st.html("""
<div class="kpi-container">

    <div class="kpi">

        <div class="kpi-number">
            10
        </div>

        <div class="kpi-label">
            Customer Features
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-number">
            ANN
        </div>

        <div class="kpi-label">
            Prediction Model
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-number">
            AI
        </div>

        <div class="kpi-label">
            Intelligent Prediction
        </div>

    </div>


    <div class="kpi">

        <div class="kpi-number">
            1
        </div>

        <div class="kpi-label">
            Customer per Prediction
        </div>

    </div>

</div>
""")


# ============================================================
# INPUT SECTION
# ============================================================

st.html("""
<div class="section-title">
    Customer Information
</div>

<div class="section-subtitle">
    Enter the customer's details below to estimate
    their probability of leaving the bank.
</div>
""")


# ============================================================
# FORM
# ============================================================

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            step=1
        )

        geography = st.selectbox(
            "Geography",
            [
                "France",
                "Germany",
                "Spain"
            ]
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

        tenure = st.number_input(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5,
            step=1
        )


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=75000.0,
            step=1000.0
        )

        num_of_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=1,
            step=1
        )

        has_cr_card = st.selectbox(
            "Has Credit Card?",
            [
                "Yes",
                "No"
            ]
        )

        is_active_member = st.selectbox(
            "Is Active Member?",
            [
                "Yes",
                "No"
            ]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=100000.0,
            step=1000.0
        )


    st.write("")

    submitted = st.form_submit_button(
        "🔮 Predict Customer Churn"
    )


# ============================================================
# PREDICTION
# ============================================================

# ============================================================
# PREDICTION
# ============================================================

if submitted:

    if not model_loaded:
        st.error(
            "Model files could not be loaded. "
            "Check that all .h5 and .pkl files are in "
            "the same directory as streamlit_app.py."
        )
        st.stop()

    try:

        # ========================================================
        # CREATE DATA EXACTLY LIKE FLASK
        # ========================================================

        data = {
            'CreditScore': int(credit_score),
            'Geography': geography,
            'Gender': gender,
            'Age': int(age),
            'Tenure': int(tenure),
            'Balance': float(balance),
            'NumOfProducts': int(num_of_products),
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': float(estimated_salary)
        }

        # ========================================================
        # PREPROCESSING
        # SAME LOGIC AS FLASK
        # ========================================================

        input_df = pd.DataFrame([data])

        # --------------------------------------------------------
        # Encode Gender
        # --------------------------------------------------------

        input_df['Gender'] = label_encoder_gender.transform(
            input_df['Gender']
        )

        # --------------------------------------------------------
        # One-Hot Encode Geography
        # --------------------------------------------------------

        geography_df = pd.DataFrame({
            'Geography': [data['Geography']]
        })

        geo_encoded = onehot_encoder_geo.transform(
            geography_df
        )

        if hasattr(geo_encoded, "toarray"):
            geo_encoded = geo_encoded.toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(
                ['Geography']
            )
        )

        # --------------------------------------------------------
        # Remove original Geography
        # Add encoded Geography columns at END
        # --------------------------------------------------------

        input_df = pd.concat(
            [
                input_df.drop('Geography', axis=1),
                geo_encoded_df
            ],
            axis=1
        )

        # ========================================================
        # DEBUG - SHOW EXACT FEATURES
        # ========================================================

        st.write("### Features sent to model")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        # ========================================================
        # SCALE
        # ========================================================

        input_scaled = scaler.transform(input_df)

        # ========================================================
        # PREDICTION
        # ========================================================

        prediction_output = model.predict(
            input_scaled,
            verbose=0
        )

        # Extract probability
        probability = float(
            np.asarray(prediction_output).reshape(-1)[0]
        )

        # Safety check
        probability = max(
            0.0,
            min(1.0, probability)
        )

        # ========================================================
        # CLASSIFICATION
        # ========================================================

        prediction = 1 if probability >= 0.5 else 0

        churn_percentage = probability * 100
        stay_percentage = (1 - probability) * 100

        # ========================================================
        # RESULT
        # ========================================================

        st.divider()

        st.html("""
        <div class="section-title">
            Prediction Result
        </div>
        """)

        if prediction == 1:

            st.html(f"""
            <div class="result-danger">

                <div class="result-icon">
                    ⚠️
                </div>

                <div class="result-title">
                    Customer is likely to churn
                </div>

                <div class="result-text">
                    The model estimates a
                    <strong>{churn_percentage:.2f}%</strong>
                    probability that this customer will
                    leave the bank.
                </div>

            </div>
            """)

            st.warning(
                "This customer may require retention "
                "strategies such as personalized offers, "
                "better support or targeted engagement."
            )

        else:

            st.html(f"""
            <div class="result-safe">

                <div class="result-icon">
                    ✅
                </div>

                <div class="result-title">
                    Customer is likely to stay
                </div>

                <div class="result-text">
                    The model estimates a
                    <strong>{churn_percentage:.2f}%</strong>
                    probability that this customer will
                    leave the bank.
                </div>

            </div>
            """)

            st.success(
                "This customer currently appears to have "
                "a lower risk of churn."
            )

        # ========================================================
        # PROBABILITY
        # ========================================================

        st.write("")

        st.subheader("Churn Probability")

        st.progress(probability)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Stay Probability",
                f"{stay_percentage:.2f}%"
            )

        with col2:
            st.metric(
                "Churn Probability",
                f"{churn_percentage:.2f}%"
            )

        with col3:
            st.metric(
                "Decision Threshold",
                "50%"
            )

        # ========================================================
        # DEBUG INFORMATION
        # ========================================================

        with st.expander("🔍 Prediction Details"):

            st.write(
                "Raw model output:",
                prediction_output
            )

            st.write(
                "Probability:",
                probability
            )

            st.write(
                "Prediction:",
                prediction
            )

            st.write(
                "Feature order:"
            )

            st.code(
                "\n".join(input_df.columns.tolist())
            )

    except Exception as e:

        st.error("Prediction failed.")
        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    🏦 Bank Customer Churn Prediction

    <br><br>

    Powered by
    Artificial Neural Networks • TensorFlow • Streamlit

</div>
""")