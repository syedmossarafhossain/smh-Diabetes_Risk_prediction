import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction App")
st.write("This app predicts diabetes risk using Machine Learning.")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "Model", "diabetes_model.pkl")
scaler_path = os.path.join(BASE_DIR, "Model", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.sidebar.header("Model Information")
st.sidebar.write("Algorithm: Random Forest Classifier")
st.sidebar.write("Accuracy: 0.82")  # Replace with your actual accuracy

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0)
    bp = st.number_input("Blood Pressure", min_value=0.0)
    skin = st.number_input("Skin Thickness", min_value=0.0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

# ---------------- INTERACTIVE FEATURE CHART ----------------
st.subheader("📊 Input Feature Overview")

features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
labels = ["Preg", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]

fig = plt.figure()
plt.bar(labels, features)
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    risk_prob = probability[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("High Risk of Diabetes ⚠️")
    else:
        st.success("Low Risk of Diabetes ✅")

    st.info(f"Risk Probability: {risk_prob:.2f}%")

    # ---------------- CONFUSION MATRIX VISUALIZATION ----------------
    st.subheader("📌 Model Confusion Matrix")

    # Dummy example values (replace with real if stored)
    y_true = [0,1,0,1,0,1,0,0,1,1]
    y_pred = [0,1,0,1,0,0,0,0,1,1]

    cm = confusion_matrix(y_true, y_pred)

    fig2 = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xticks([0,1])
    plt.yticks([0,1])
    st.pyplot(fig2)

    # ---------------- PDF REPORT GENERATION ----------------
    pdf_path = "prediction_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Diabetes Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))

    elements.append(Paragraph(f"Pregnancies: {pregnancies}", styles["Normal"]))
    elements.append(Paragraph(f"Glucose: {glucose}", styles["Normal"]))
    elements.append(Paragraph(f"Blood Pressure: {bp}", styles["Normal"]))
    elements.append(Paragraph(f"BMI: {bmi}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"Prediction: {'High Risk' if prediction[0]==1 else 'Low Risk'}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Probability: {risk_prob:.2f}%", styles["Normal"]))

    doc.build(elements)

    with open(pdf_path, "rb") as file:
        st.download_button(
            label="📥 Download Prediction Report",
            data=file,
            file_name="Diabetes_Prediction_Report.pdf",
            mime="application/pdf"

        )
