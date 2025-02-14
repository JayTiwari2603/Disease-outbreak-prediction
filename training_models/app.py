import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import joblib
import numpy as np

# ✅ This must be the first Streamlit command
st.set_page_config(page_title='Disease Prediction System', layout='wide', page_icon='🩺')

# Function to load the model
def load_model(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)  
    except (pickle.UnpicklingError, EOFError):
        return joblib.load(path)  
    except Exception as e:
        st.error(f"❌ Error loading model `{path}`: {e}")
        return None

# Load the models
diabetes_model = load_model("diabetes_model.pkl")
heart_disease_model = load_model("heart_disease_model.pkl")
parkinsons_model = load_model("parkinsons_model.pkl")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"],
        icons=["activity", "heart", "person"],
        menu_icon="hospital",
        default_index=0
    )

# **Diabetes Prediction Page**
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction 🩸")
    
    # Input fields
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=30)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    # Prediction
    if st.button("Diabetes Test Result"):
        if diabetes_model:
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = diabetes_model.predict(input_data)[0]
            result = "✅ No Diabetes" if prediction == 0 else "❌ Likely Diabetes"
            st.success(result)

# **Heart Disease Prediction Page**
elif selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction ❤️")
    
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.number_input("Chest Pain Type", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    restecg = st.number_input("Resting ECG Result", min_value=0, max_value=2, value=1)
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.number_input("Slope of Peak Exercise ST Segment", min_value=0, max_value=2, value=1)
    ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)
    thal = st.number_input("Thalassemia Type", min_value=0, max_value=3, value=1)

    if st.button("Heart Disease Test Result"):
        if heart_disease_model:
            input_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, 1 if fbs == "Yes" else 0, 
                                    restecg, thalach, 1 if exang == "Yes" else 0, oldpeak, slope, ca, thal]])
            prediction = heart_disease_model.predict(input_data)[0]
            result = "✅ No Heart Disease" if prediction == 0 else "❌ High Risk of Heart Disease"
            st.success(result)

# **Parkinson's Disease Prediction Page**
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction 🧠")
    
    fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=500.0, value=150.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=600.0, value=200.0)
    flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=500.0, value=100.0)
    jitter = st.number_input("Jitter(%)", min_value=0.0, max_value=1.0, value=0.01)
    shimmer = st.number_input("Shimmer", min_value=0.0, max_value=1.0, value=0.02)
    hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0)
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.4)
    dfa = st.number_input("DFA", min_value=0.0, max_value=2.0, value=0.6)
    spread1 = st.number_input("Spread1", min_value=-10.0, max_value=10.0, value=-4.0)
    spread2 = st.number_input("Spread2", min_value=-10.0, max_value=10.0, value=2.0)
    d2 = st.number_input("D2", min_value=0.0, max_value=10.0, value=2.5)
    ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.3)

    if st.button("Parkinson's Test Result"):
        if parkinsons_model:
            input_data = np.array([[fo, fhi, flo, jitter, shimmer, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
            prediction = parkinsons_model.predict(input_data)[0]
            result = "✅ No Parkinson's Disease" if prediction == 0 else "❌ Likely Parkinson's Disease"
            st.success(result)
