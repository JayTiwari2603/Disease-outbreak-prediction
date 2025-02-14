import streamlit as st
import pickle
import numpy as np

# Set page configuration (Must be the first command)
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Load trained models
@st.cache_data
def load_models():
    with open("diabetes_model.sav", "rb") as f:
        diabetes_model = pickle.load(f)
    with open("heartmodel.sav", "rb") as f:
        heart_model = pickle.load(f)
    with open("parkinsons_model.pkl", "rb") as f:
        parkinsons_model = pickle.load(f)
    
    return diabetes_model, heart_model, parkinsons_model

diabetes_model, heart_model, parkinsons_model = load_models()

# Sidebar Navigation
st.sidebar.title("Disease Prediction System")
selected = st.sidebar.radio("Select Disease Prediction", ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"])

# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction")
    
    # Input fields
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure Value", min_value=0)
    skin_thickness = st.number_input("Skin Thickness Value", min_value=0)
    insulin = st.number_input("Insulin Level", min_value=0)
    bmi = st.number_input("BMI Value", min_value=0.0, format="%.2f")
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function Value", min_value=0.0, format="%.2f")
    age = st.number_input("Age of the Person", min_value=0, step=1)

    if st.button("Diabetes Test Result"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        prediction = diabetes_model.predict(input_data)[0]
        result = "✅ No Diabetes" if prediction == 0 else "❌ Likely Diabetes"
        st.success(result)

# Heart Disease Prediction
elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction")
    
    # Input fields (Example)
    age = st.number_input("Age", min_value=1, step=1)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type", min_value=0, step=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol Level", min_value=0)
    fbs = st.number_input("Fasting Blood Sugar", min_value=0)
    restecg = st.number_input("Resting ECG", min_value=0)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
    exang = st.number_input("Exercise Induced Angina", min_value=0)
    oldpeak = st.number_input("ST Depression", min_value=0.0, format="%.2f")
    slope = st.number_input("Slope of Peak Exercise", min_value=0)
    ca = st.number_input("Number of Major Vessels", min_value=0)
    thal = st.number_input("Thalassemia", min_value=0)

    if st.button("Heart Disease Test Result"):
        input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = heart_model.predict(input_data)[0]
        result = "✅ No Heart Disease" if prediction == 0 else "❌ Likely Heart Disease"
        st.success(result)

# Parkinson's Prediction
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Prediction")
    
    # Example Inputs (Modify as per dataset)
    fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, format="%.2f")
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, format="%.2f")
    flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, format="%.2f")
    jitter_percent = st.number_input("Jitter(%)", min_value=0.0, format="%.5f")
    shimmer = st.number_input("Shimmer", min_value=0.0, format="%.5f")

    if st.button("Parkinson's Test Result"):
        input_data = np.array([[fo, fhi, flo, jitter_percent, shimmer]])
        prediction = parkinsons_model.predict(input_data)[0]
        result = "✅ No Parkinson’s Disease" if prediction == 0 else "❌ Likely Parkinson’s Disease"
        st.success(result)
