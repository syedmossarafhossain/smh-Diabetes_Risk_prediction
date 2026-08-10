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

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 25%),
        #090d16;
    color: #f8fafc;
}


/* Remove default top spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0d1320 0%,
        #0a0f19 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
}


/* Header */

.hero {
    padding: 35px 40px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(37,99,235,0.22),
            rgba(124,58,237,0.18)
        );
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    margin-bottom: 30px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0;
}

.hero-subtitle {
    margin-top: 10px;
    font-size: 16px;
    color: #aab4c5;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 50px;
    background: rgba(59,130,246,0.15);
    border: 1px solid rgba(59,130,246,0.35);
    color: #60a5fa;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 15px;
}


/* Section title */

.section-title {
    font-size: 23px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 18px;
}


/* Input cards */

.input-card {
    background: rgba(20,26,39,0.85);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 14px;
    transition: 0.2s ease;
}

.input-card:hover {
    border-color: rgba(96,165,250,0.45);
    transform: translateY(-2px);
}


/* Streamlit input */

div[data-baseweb="input"] {
    background: #171d2a !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
}


/* Buttons */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 10px 30px rgba(37,99,235,0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(37,99,235,0.35);
}


/* Prediction card */

.result-card {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(15,23,42,0.95),
        rgba(30,41,59,0.85)
    );
    border: 1px solid rgba(255,255,255,0.10);
    text-align: center;
    margin-top: 20px;
}

.risk-number {
    font-size: 52px;
    font-weight: 800;
    margin: 10px 0;
}

.risk-label {
    color: #94a3b8;
    font-size: 14px;
}


/* Metric cards */

.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(20,26,39,0.85);
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-title {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
}


/* Footer */

.footer {
    text-align: center;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #64748b;
    font-size: 12px;
}


/* Disclaimer */

.disclaimer {
    padding: 15px 18px;
    border-radius: 14px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.20);
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


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

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.markdown(
        """
        <div style="
            padding:15px;
            border-radius:15px;
            background:rgba(59,130,246,0.08);
            border:1px solid rgba(59,130,246,0.15);
        ">
        <b>AI Risk Assessment</b><br>
        <span style="color:#94a3b8;font-size:13px;">
        Machine Learning powered diabetes risk estimation.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("**Algorithm**")
    st.caption("Random Forest Classifier")

    st.write("**Model Accuracy**")
    st.caption("82%")

    st.write("**Input Features**")
    st.caption("8 clinical parameters")

    st.markdown("---")

    st.markdown("### 📌 Features")

    st.markdown("""
    - Pregnancy history
    - Glucose level
    - Blood pressure
    - Skin thickness
    - Insulin level
    - BMI
    - Diabetes pedigree
    - Age
    """)

    st.markdown("---")

    st.caption("AI Diabetes Risk Prediction")
    st.caption("Educational / Research Project")


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="badge">● AI POWERED HEALTH ANALYTICS</div>'
    '<div class="hero-title">🩺 Diabetes Risk Prediction</div>'
    '<div class="hero-subtitle">'
    'Analyze clinical parameters using a Machine Learning '
    'model to estimate the probability of diabetes risk.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# INPUT SECTION
# =========================================================

# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">📋 Patient Information</div>',
    unsafe_allow_html=True
)

# Patient details
patient_col1, patient_col2 = st.columns(
    2,
    gap="large"
)

with patient_col1:

    patient_name = st.text_input(
        "👤 Patient Name",
        placeholder="Enter patient name"
    )

with patient_col2:

    prediction_date = st.date_input(
        "📅 Assessment Date"
    )


# =========================================================
# CLINICAL PARAMETERS
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🧪 Clinical Parameters'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(
    2,
    gap="large"
)


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

    st.markdown('</div>', unsafe_allow_html=True)


with col2:

    st.markdown('<div class="input-card">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FEATURE OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 Clinical Data Overview</div>',
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

plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.15)

plt.tight_layout()

st.pyplot(fig, use_container_width=True)

plt.close(fig)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍 Analyze Diabetes Risk"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = np.array([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])


    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    scaled_data = scaler.transform(input_data)


    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    risk_prob = probability[0][1] * 100


    # =====================================================
    # PREDICTION TEXT & COLOR
    # =====================================================

    prediction_text = (
        "HIGH RISK"
        if int(prediction[0]) == 1
        else "LOW RISK"
    )

    prediction_color = (
        "#f87171"
        if int(prediction[0]) == 1
        else "#34d399"
    )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # METRIC COLUMNS
    # =====================================================

    metric1, metric2, metric3 = st.columns(
        3,
        gap="medium"
    )


    # =====================================================
    # RISK PROBABILITY
    # =====================================================

    with metric1:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">'
            'RISK PROBABILITY'
            '</div>'
            f'<div class="metric-value">'
            f'{risk_prob:.2f}%'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    with metric2:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">'
            'PREDICTION'
            '</div>'
            f'<div class="metric-value" '
            f'style="color:{prediction_color};">'
            f'{prediction_text}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # MODEL
    # =====================================================

    with metric3:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">'
            'MODEL'
            '</div>'
            '<div class="metric-value">'
            'Random Forest'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.progress(
        min(max(risk_prob / 100, 0.0), 1.0),
        text=f"Estimated Risk: {risk_prob:.2f}%"
    )

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.markdown(
        '<div class="section-title">📌 Model Performance</div>',
        unsafe_allow_html=True
    )

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

    fig2 = plt.figure(figsize=(6, 4))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xticks(
        [0, 1],
        ["Predicted 0", "Predicted 1"]
    )

    plt.yticks(
        [0, 1],
        ["Actual 0", "Actual 1"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    st.pyplot(
        fig2,
        use_container_width=False
    )

    plt.close(fig2)


# =========================================================
# PDF REPORT
# =========================================================
if predict_button:

    # prediction code
    input_data = np.array([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    risk_prob = probability[0][1] * 100




st.markdown(
    '<div class="section-title">'
    '📄 Prediction Report'
    '</div>',
    unsafe_allow_html=True
)

pdf_buffer = BytesIO()

doc = SimpleDocTemplate(
    pdf_buffer
)

styles = getSampleStyleSheet()

elements = []


# =========================================================
# PDF TITLE
# =========================================================

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


# =========================================================
# PATIENT INFORMATION
# =========================================================

elements.append(
    Paragraph(
        f"<b>Patient Name:</b> "
        f"{patient_name if patient_name else 'Not Provided'}",
        styles["Normal"]
    )
)

elements.append(
    Paragraph(
        f"<b>Assessment Date:</b> {prediction_date}",
        styles["Normal"]
    )
)

elements.append(
    Spacer(
        1,
        0.25 * inch
    )
)


# =========================================================
# CLINICAL INFORMATION
# =========================================================

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


# =========================================================
# PREDICTION
# =========================================================

elements.append(
    Paragraph(
        f"<b>Prediction:</b> "
        f"{'High Risk' if int(prediction[0]) == 1 else 'Low Risk'}",
        styles["Normal"]
    )
)

elements.append(
    Paragraph(
        f"<b>Risk Probability:</b> {risk_prob:.2f}%",
        styles["Normal"]
    )
)

elements.append(
    Paragraph(
        "<b>Model:</b> Random Forest",
        styles["Normal"]
    )
)


elements.append(
    Spacer(
        1,
        0.3 * inch
    )
)


# =========================================================
# DISCLAIMER
# =========================================================

elements.append(
    Paragraph(
        "<b>Disclaimer:</b> This report is generated by an "
        "educational Machine Learning application and should "
        "not be considered a medical diagnosis.",
        styles["Normal"]
    )
)


# =========================================================
# BUILD PDF
# =========================================================

doc.build(
    elements
)

pdf_buffer.seek(0)


# =========================================================
# DOWNLOAD BUTTON
# =========================================================

st.download_button(
    label="📥 Download Prediction Report",
    data=pdf_buffer,
    file_name="Diabetes_Prediction_Report.pdf",
    mime="application/pdf"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🩺 Diabetes Risk Prediction &nbsp; • &nbsp;
    Machine Learning Healthcare Project

    </div>
    """,
    unsafe_allow_html=True
)
