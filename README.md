# 🩺 Diabetes Risk Prediction App

## 📌 Overview

The **Diabetes Risk Prediction App** is a Machine Learning web application that predicts whether a person is at **high or low risk of diabetes** based on medical attributes.
The model is trained using the **Pima Indians Diabetes Dataset** and deployed using **Streamlit** for an interactive web interface.

---

## 🚀 Live Demo

🔗 **Live App:**
https://smh-diabetes-risk-prediction.streamlit.app/

---

## 🧠 Machine Learning Model

**Algorithm Used:** Random Forest Classifier


**Accuracy Achieved:** ~77–82%


**Dataset:** Pima Indians Diabetes Dataset


### Features Used

* Pregnancies
* Glucose Level
* Blood Pressure
* Skin Thickness
* Insulin Level
* BMI
* Diabetes Pedigree Function
* Age

---

## ✨ Features of the App

✔ Interactive user interface


✔ Real-time diabetes risk prediction


✔ Probability score display


✔ Feature visualization (bar chart)


✔ Confusion matrix visualization


✔ Downloadable **PDF prediction 
report**


✔ Clean and responsive UI

---

## 📊 How It Works

1. User enters medical details.
2. Data is scaled using **StandardScaler**.
3. The trained **Random Forest model** predicts the outcome.
4. The app displays:

* Risk category (High / Low)
* Prediction probability
* Visualization charts

Users can also **download a PDF report** of the prediction.

---

## 📂 Project Structure

```
Diabetes_Risk_prediction/  
│  
├── Model/  
│   ├── diabetes_model.pkl  
│   └── scaler.pkl  
│  
├── app.py  
├── modeltraining.py  
├── requirements.txt  
└── README.md  
```

---

## ⚙️ Installation & Setup

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt  
```

### 3️⃣ Run the App

```bash
streamlit run app.py  
```

---

## 📈 Model Training

To retrain the model:

```bash
python modeltraining.py  
```

This will:

* Train the Random Forest model
* Save the model as **diabetes_model.pkl**
* Save the scaler as **scaler.pkl**

---

## 🛠 Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* ReportLab

---

## 🎯 Future Improvements

* Deploy with authentication
* Add SHAP feature importance visualization
* Store prediction history in database
* Improve model with hyperparameter tuning
* Convert to full-stack app (FastAPI + React)

---

## 👩‍💻 Author

**Syed Mossaraf Hossain**


Bachelor of Technology in Computer Science and Engineering (B.Tech CSE)


Aspiring Data Scientist & ML Enthusiast

---

⭐ If you like this project, consider giving it a **star on GitHub!**


