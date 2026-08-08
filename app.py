import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
from textwrap import dedent


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes AI | Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(59, 130, 246, 0.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(124, 58, 237, 0.10),
            transparent 28%
        ),
        #080c14;
    color: #f8fafc;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0d1422 0%,
        #080d17 100%
    );

    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {
    padding: 38px 42px;
    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.22),
            rgba(124,58,237,0.20)
        );

    border: 1px solid rgba(96,165,250,0.20);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.25);

    margin-bottom: 34px;
}

.badge {
    display: inline-block;

    padding: 7px 14px;

    border-radius: 50px;

    background: rgba(59,130,246,0.14);

    border: 1px solid rgba(96,165,250,0.30);

    color: #60a5fa;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.5px;

    margin-bottom: 15px;
}

.hero-title {
    font-size: 44px;

    font-weight: 800;

    letter-spacing: -1.8px;

    line-height: 1.15;

    color: #f8fafc;
}

.hero-subtitle {
    margin-top: 12px;

    max-width: 850px;

    color: #aab6c8;

    font-size: 16px;

    line-height: 1.7;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 23px;

    font-weight: 750;

    color: #f8fafc;

    margin-top: 24px;

    margin-bottom: 18px;
}


/* =========================================================
   INPUT LABELS
   ========================================================= */

label {
    font-weight: 600 !important;
    color: #e5e7eb !important;
}


/* =========================================================
   INPUT BOXES
   ========================================================= */

div[data-baseweb="input"] {
    background: #171c28 !important;

    border-radius: 12px !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    min-height: 48px;
}

div[data-baseweb="input"]:focus-within {
    border-color: #3b82f6 !important;

    box-shadow:
        0 0 0 2px rgba(59,130,246,0.15);
}


/* =========================================================
   SELECT / NUMBER INPUT
   ========================================================= */

input {
    color: #f8fafc !important;
}


/* =========================================================
   PREDICT BUTTON
   ========================================================= */

.stButton > button {
    width: 100%;

    min-height: 54px;

    border-radius: 14px;

    border: 0;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    font-size: 16px;

    font-weight: 700;

    box-shadow:
        0 12px 30px rgba(37,99,235,0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 16px 38px rgba(37,99,235,0.35);

    color: white;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

.metric-box {
    background:
        linear-gradient(
            145deg,
            rgba(20,27,41,0.96),
            rgba(14,20,32,0.96)
        );

    border: 1px solid rgba(255,255,255,0.09);

    border-radius: 18px;

    padding: 22px;

    min-height: 115px;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.18);
}

.metric-title {
    color: #94a3b8;

    font-size: 12px;

    font-weight: 700;

    letter-spacing: 0.8px;

    text-transform: uppercase;

    margin-bottom: 9px;
}

.metric-value {
    color: #f8fafc;

    font-size: 25px;

    font-weight: 800;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    margin-top: 22px;

    padding: 34px;

    border-radius: 24px;

    text-align: center;

    background:
        linear-gradient(
            145deg,
            rgba(18,27,43,0.98),
            rgba(12,18,30,0.98)
        );

    border: 1px solid rgba(255,255,255,0.09);

    box-shadow:
        0 20px 55px rgba(0,0,0,0.25);
}

.result-icon {
    font-size: 48px;

    margin-bottom: 8px;
}

.risk-number {
    font-size: 56px;

    line-height: 1;

    font-weight: 800;

    margin: 10px 0;
}

.low-risk {
    color: #34d399;
}

.high-risk {
    color: #f87171;
}

.result-title {
    font-size: 23px;

    font-weight: 800;

    margin-top: 10px;
}

.result-description {
    color: #94a3b8;

    max-width: 650px;

    margin: 10px auto 0;

    line-height: 1.6;
}


/* =========================================================
   INFORMATION CARD
   ========================================================= */

.info-card {
    padding: 20px;

    border-radius: 18px;

    background:
        rgba(20,27,41,0.85);

    border: 1px solid rgba(255,255,255,0.08);
}


/* =========================================================
   DISCLAIMER
   ========================================================= */

.disclaimer {
    margin-top: 28px;

    padding: 17px 20px;

    border-radius: 15px;

    background:
        rgba(245,158,11,0.07);

    border:
        1px solid rgba(245,158,11,0.18);

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.6;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    margin-top: 50px;

    padding-top: 22px;

    border-top:
        1px solid rgba(255,255,255,0.08);

    text-align: center;

    color: #64748b;

    font-size: 12px;
}


/* =========================================================
   PROGRESS BAR
   ========================================================= */

div[data-testid="stProgressBar"] > div {
    border-radius: 20px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .hero {
        padding: 28px 24px;
    }

    .hero-title {
        font-size: 31px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .risk-number {
        font-size: 45px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "Model",
    "diabetes_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "Model",
    "scaler.pkl"
)


try:

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

except FileNotFoundError:

    st.error(
        "❌ Model files were not found. "
        "Please check the Model folder."
    )

    st.code(
        """
Model/
├── diabetes_model.pkl
└── scaler.pkl
        """
    )

    st.stop()

except Exception as e:

    st.error(
        "❌ Unable to load the machine learning model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🩺 Diabetes AI</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        dedent(
            """
            <div class="info-card">

                <div style="
                    font-size:17px;
                    font-weight:700;
                    margin-bottom:8px;
                ">
                    AI Risk Assessment
                </div>

                <div style="
                    color:#94a3b8;
                    font-size:13px;
                    line-height:1.6;
                ">
                    Machine Learning powered diabetes
                    risk estimation.
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.markdown("**Algorithm**")
    st.caption("Random Forest Classifier")

    st.markdown("**Model Accuracy**")
    st.caption("82%")

    st.markdown("**Input Features**")
    st.caption("8 clinical parameters")

    st.markdown("---")

    st.markdown("### 📌 Features")

    st.markdown(
        """
        - 🤰 Pregnancy history
        - 🧪 Glucose level
        - ❤️ Blood pressure
        - 📏 Skin thickness
        - 💉 Insulin level
        - ⚖️ BMI
        - 🧬 Diabetes pedigree
        - 🎂 Age
        """
    )

    st.markdown("---")

    st.caption("Diabetes AI Risk Prediction")
    st.caption("Machine Learning Project")


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    dedent(
        """
        <div class="hero">

            <div class="badge">
                ● AI POWERED HEALTH ANALYTICS
            </div>

            <div class="hero-title">
                🩺 Diabetes Risk Prediction
            </div>

            <div class="hero-subtitle">
                Analyze clinical parameters using a Machine
                Learning model to estimate the probability
                of diabetes risk.
            </div>

        </div>
        """
    ),
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📋 Patient Information
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")


# =========================================================
# LEFT INPUTS
# =========================================================

with col1:

    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "🧪 Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    bp = st.number_input(
        "❤️ Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=80.0,
        step=1.0
    )

    skin = st.number_input(
        "📏 Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


# =========================================================
# RIGHT INPUTS
# =========================================================

with col2:

    insulin = st.number_input(
        "💉 Insulin Level",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# =========================================================
# FEATURE OVERVIEW
# =========================================================

st.markdown(
    """
    <div class="section-title">
        📊 Clinical Data Overview
    </div>
    """,
    unsafe_allow_html=True
)


features = [
    pregnancies,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
]

labels = [
    "Pregnancy",
    "Glucose",
    "Blood Pressure",
    "Skin",
    "Insulin",
    "BMI",
    "DPF",
    "Age"
]


fig = plt.figure(figsize=(12, 4))

plt.bar(
    labels,
    features
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.15
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍  Analyze Diabetes Risk"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT
    # -----------------------------------------------------

    input_data = np.array(
        [[
            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age
        ]]
    )


    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    try:

        scaled_data = scaler.transform(
            input_data
        )

    except Exception as e:

        st.error(
            "❌ Error while scaling input data."
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    try:

        prediction = model.predict(
            scaled_data
        )

        probability = model.predict_proba(
            scaled_data
        )

    except Exception as e:

        st.error(
            "❌ Error while making prediction."
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # RISK PROBABILITY
    # -----------------------------------------------------

    risk_prob = (
        probability[0][1] * 100
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.markdown(
        """
        <div class="section-title">
            🎯 Prediction Result
        </div>
        """,
        unsafe_allow_html=True
    )


    metric1, metric2, metric3 = st.columns(
        3,
        gap="medium"
    )


    # -----------------------------------------------------
    # RISK PROBABILITY
    # -----------------------------------------------------

    with metric1:

        st.markdown(
            dedent(
                f"""
                <div class="metric-box">

                    <div class="metric-title">
                        Risk Probability
                    </div>

                    <div class="metric-value">
                        {risk_prob:.2f}%
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with metric2:

        prediction_text = (
            "HIGH RISK"
            if prediction[0] == 1
            else "LOW RISK"
        )

        st.markdown(
            dedent(
                f"""
                <div class="metric-box">

                    <div class="metric-title">
                        Prediction
                    </div>

                    <div class="metric-value">
                        {prediction_text}
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    with metric3:

        st.markdown(
            dedent(
                """
                <div class="metric-box">

                    <div class="metric-title">
                        Model
                    </div>

                    <div class="metric-value">
                        Random Forest
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


    # =====================================================
    # MAIN RESULT CARD
    # =====================================================

    if prediction[0] == 1:

        st.markdown(
            dedent(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        ⚠️
                    </div>

                    <div class="risk-number high-risk">
                        {risk_prob:.1f}%
                    </div>

                    <div class="result-title">
                        Higher Diabetes Risk
                    </div>

                    <div class="result-description">
                        The machine learning model estimates
                        an elevated diabetes risk based on
                        the provided clinical parameters.
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            dedent(
                f"""
                <div class="result-card">

                    <div class="result-icon">
                        ✅
                    </div>

                    <div class="risk-number low-risk">
                        {risk_prob:.1f}%
                    </div>

                    <div class="result-title">
                        Lower Diabetes Risk
                    </div>

                    <div class="result-description">
                        The machine learning model estimates
                        a lower diabetes risk based on
                        the provided clinical parameters.
                    </div>

                </div>
                """
            ),
            unsafe_allow_html=True
        )


    # =====================================================
    # RISK PROGRESS
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.progress(
        min(max(risk_prob / 100, 0), 1),
        text=f"Estimated Diabetes Risk: {risk_prob:.2f}%"
    )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.markdown(
        """
        <div class="section-title">
            📌 Model Performance
        </div>
        """,
        unsafe_allow_html=True
    )


    # IMPORTANT:
    # These are demonstration values.
    # Replace with your real test-set values.

    y_true = [
        0, 1, 0, 1, 0,
        1, 0, 0, 1, 1
    ]

    y_pred = [
        0, 1, 0, 1, 0,
        0, 0, 0, 1, 1
    ]


    cm = confusion_matrix(
        y_true,
        y_pred
    )


    fig2 = plt.figure(
        figsize=(6, 4)
    )

    plt.imshow(cm)

    plt.title(
        "Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    plt.xticks(
        [0, 1],
        ["No Diabetes", "Diabetes"]
    )

    plt.yticks(
        [0, 1],
        ["No Diabetes", "Diabetes"]
    )


    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold"
            )


    plt.tight_layout()

    st.pyplot(
        fig2,
        use_container_width=False
    )

    plt.close(fig2)


    # =====================================================
    # PDF REPORT
    # =====================================================

    st.markdown(
        """
        <div class="section-title">
            📄 Prediction Report
        </div>
        """,
        unsafe_allow_html=True
    )


    pdf_buffer = BytesIO()


    doc = SimpleDocTemplate(
        pdf_buffer
    )


    styles = getSampleStyleSheet()

    elements = []


    elements.append(
        Paragraph(
            "<b>Diabetes Risk Prediction Report</b>",
            styles["Title"]
        )
    )


    elements.append(
        Spacer(
            1,
            0.4 * inch
        )
    )


    # Patient data

    elements.append(
        Paragraph(
            f"Pregnancies: {pregnancies}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Glucose Level: {glucose}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Blood Pressure: {bp}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Skin Thickness: {skin}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Insulin Level: {insulin}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"BMI: {bmi}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Diabetes Pedigree Function: {dpf}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Age: {age}",
            styles["Normal"]
        )
    )


    elements.append(
        Spacer(
            1,
            0.3 * inch
        )
    )


    # Prediction

    elements.append(
        Paragraph(
            f"Prediction: "
            f"{'High Risk' if prediction[0] == 1 else 'Low Risk'}",
            styles["Normal"]
        )
    )


    elements.append(
        Paragraph(
            f"Risk Probability: {risk_prob:.2f}%",
            styles["Normal"]
        )
    )


    doc.build(
        elements
    )


    pdf_buffer.seek(0)


    st.download_button(
        label="📥 Download Prediction Report",
        data=pdf_buffer,
        file_name="Diabetes_Prediction_Report.pdf",
        mime="application/pdf"
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown(
    dedent(
        """
        <div class="disclaimer">

            ⚠️ <b>Important:</b>
            This application is an educational Machine Learning
            project and should not be considered a medical diagnosis.
            Please consult a qualified healthcare professional
            for medical advice.

        </div>
        """
    ),
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    dedent(
        """
        <div class="footer">

            🩺 Diabetes AI Risk Prediction
            &nbsp; • &nbsp;
            Machine Learning Healthcare Project

        </div>
        """
    ),
    unsafe_allow_html=True
)
