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
    position: relative;
    overflow: hidden;

    margin-top: 50px;
    margin-bottom: 45px;

    padding: 42px 48px;

    min-height: 260px;

    border-radius: 28px;

    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(124, 58, 237, 0.28),
            transparent 32%
        ),
        radial-gradient(
            circle at 15% 80%,
            rgba(37, 99, 235, 0.20),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.98),
            rgba(17, 24, 39, 0.94)
        );

    border: 1px solid rgba(148, 163, 184, 0.14);

    box-shadow:
        0 25px 80px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);

    backdrop-filter: blur(20px);
}


/* Decorative glow */

.hero::before {
    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    right: -90px;
    top: -120px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.30),
            transparent 70%
        );

    filter: blur(10px);

    pointer-events: none;
}


/* Decorative grid */

.hero::after {
    content: "";

    position: absolute;

    inset: 0;

    background-image:
        linear-gradient(
            rgba(255,255,255,0.025) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255,255,255,0.025) 1px,
            transparent 1px
        );

    background-size: 32px 32px;

    mask-image: linear-gradient(
        to bottom right,
        black,
        transparent 70%
    );

    pointer-events: none;
}


/* Hero content */

.hero-content {
    position: relative;
    z-index: 2;
}


/* Badge */

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 8px 14px;

    border-radius: 999px;

    background:
        rgba(59, 130, 246, 0.10);

    border: 1px solid
        rgba(96, 165, 250, 0.25);

    color: #93c5fd;

    font-size: 12px;
    font-weight: 700;

    letter-spacing: 0.4px;

    margin-bottom: 18px;

    box-shadow:
        0 8px 25px rgba(37, 99, 235, 0.12);
}


/* Badge dot */

.badge-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #60a5fa;

    box-shadow:
        0 0 12px rgba(96, 165, 250, 0.8);
}


/* Hero title */

.hero-title {
    margin: 0;

    max-width: 760px;

    font-size: clamp(34px, 4vw, 54px);

    line-height: 1.05;

    font-weight: 800;

    letter-spacing: -2px;

    color: #f8fafc;
}


/* Gradient title */

.hero-title span {
    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #818cf8,
            #c084fc
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    background-clip: text;
}


/* Subtitle */

.hero-subtitle {
    margin-top: 18px;

    max-width: 680px;

    font-size: 16px;

    line-height: 1.7;

    color: #94a3b8;
}


/* Hero bottom information */

.hero-meta {
    display: flex;

    align-items: center;

    gap: 12px;

    margin-top: 26px;

    flex-wrap: wrap;
}


/* Meta cards */

.hero-meta-item {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 9px 13px;

    border-radius: 10px;

    background: rgba(255,255,255,0.035);

    border: 1px solid
        rgba(255,255,255,0.07);

    color: #cbd5e1;

    font-size: 12px;

    font-weight: 500;
}


/* Right visual */

.hero-visual {
    position: absolute;

    z-index: 2;

    right: 48px;
    top: 50%;

    transform: translateY(-50%);

    width: 210px;
    height: 210px;

    display: flex;

    align-items: center;
    justify-content: center;
}


/* AI pulse circle */

.hero-orb {
    width: 150px;
    height: 150px;

    border-radius: 50%;

    display: flex;

    align-items: center;
    justify-content: center;

    background:
        radial-gradient(
            circle at 35% 30%,
            rgba(96,165,250,0.35),
            rgba(124,58,237,0.12) 45%,
            rgba(15,23,42,0.9) 70%
        );

    border: 1px solid
        rgba(129,140,248,0.35);

    box-shadow:
        0 0 50px rgba(99,102,241,0.22),
        inset 0 0 35px rgba(96,165,250,0.08);

    animation: heroPulse 4s ease-in-out infinite;
}


/* Orb icon */

.hero-orb-icon {
    font-size: 58px;

    filter:
        drop-shadow(
            0 0 18px rgba(96,165,250,0.45)
        );
}


/* Orb animation */

@keyframes heroPulse {

    0%, 100% {
        transform: scale(1);
        box-shadow:
            0 0 50px rgba(99,102,241,0.22),
            inset 0 0 35px rgba(96,165,250,0.08);
    }

    50% {
        transform: scale(1.05);
        box-shadow:
            0 0 75px rgba(99,102,241,0.35),
            inset 0 0 45px rgba(96,165,250,0.12);
    }
}


/* Responsive */

@media (max-width: 900px) {

    .hero {
        padding: 34px 30px;
    }

    .hero-visual {
        opacity: 0.25;
        right: 15px;
    }

    .hero-title {
        max-width: 650px;
    }
}


@media (max-width: 600px) {

    .hero {
        margin-top: 20px;
        padding: 28px 22px;
        min-height: 300px;
    }

    .hero-title {
        font-size: 34px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .hero-visual {
        display: none;
    }

    .hero-meta {
        gap: 8px;
    }
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
    
    '<div class="hero-content">'
    
    '<div class="badge">'
    '<span class="badge-dot"></span>'
    'AI-POWERED HEALTH ANALYTICS'
    '</div>'
    
    '<div class="hero-title">'
    'Diabetes Risk '
    '<span>Prediction</span>'
    '</div>'
    
    '<div class="hero-subtitle">'
    'Analyze clinical health parameters with a '
    'machine learning model and receive an '
    'instant diabetes risk assessment.'
    '</div>'
    
    '<div class="hero-meta">'
    
    '<div class="hero-meta-item">'
    '🤖 Random Forest'
    '</div>'
    
    '<div class="hero-meta-item">'
    '⚡ Instant Prediction'
    '</div>'
    
    '<div class="hero-meta-item">'
    '🔒 Secure Analysis'
    '</div>'
    
    '</div>'
    
    '</div>'
    
    
    '<div class="hero-visual">'
    
    '<div class="hero-orb">'
    
    '<div class="hero-orb-icon">'
    '🧬'
    '</div>'
    
    '</div>'
    
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

    # -----------------------------------------------------
    # PREDICTION TEXT
    # -----------------------------------------------------

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
    # PDF REPORT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📄 Prediction Report'
        '</div>',
        unsafe_allow_html=True
    )

    # Create PDF buffer
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer
    )

    styles = getSampleStyleSheet()

    elements = []

    # =====================================================
    # PDF TITLE
    # =====================================================

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

    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

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

    # =====================================================
    # CLINICAL INFORMATION
    # =====================================================

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

    # =====================================================
    # PREDICTION
    # =====================================================

    elements.append(
        Paragraph(
            f"<b>Prediction:</b> "
            f"{prediction_text}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Risk Probability:</b> "
            f"{risk_prob:.2f}%",
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

    # =====================================================
    # DISCLAIMER
    # =====================================================

    elements.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an "
            "educational Machine Learning application and should "
            "not be considered a medical diagnosis.",
            styles["Normal"]
        )
    )

    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(
        elements
    )

    pdf_buffer.seek(0)

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

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
