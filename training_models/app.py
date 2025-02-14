import os
import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title='Disease Prediction System', layout='wide', page_icon='🩺')

# Function to load models safely
def load_model(path):
    if not os.path.exists(path):
        st.warning(f"⚠️ Model file not found: `{path}`. Please upload the model file.")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Error loading model `{path}`: {e}")
        return None

# Define model paths
MODEL_DIR = "training_models"
model_paths = {
    "Diabetes": os.path.join(MODEL_DIR, "diabetes_model.sav"),
    "Heart Disease": os.path.join(MODEL_DIR, "heartmodel.sav"),
    "Parkinson's": os.path.join(MODEL_DIR, "parkinsons_model.pkl")
}

# Load models
models = {disease: load_model(path) for disease, path in model_paths.items()}

# Sidebar Menu
with st.sidebar:
    selected = option_menu('Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Function to validate and convert user input
def validate_input(inputs):
    try:
        return [float(x.strip()) if x.strip() else 0 for x in inputs]
    except ValueError:
        st.error('⚠️ Please enter only numerical values.')
        return None

# -------------------------------
# Diabetes Prediction Section
# -------------------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')
    
    features = ['Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
    user_inputs = [st.text_input(f) for f in features]
    
    if st.button('Diabetes Test Result'):
        user_input = validate_input(user_inputs)
        if user_input and models["Diabetes"]:
            diab_prediction = models["Diabetes"].predict([user_input])
            result = '🚨 The person is diabetic' if diab_prediction[0] == 1 else '✅ The person is not diabetic'
            st.success(result)

# -------------------------------
# Heart Disease Prediction Section
# -------------------------------
elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')
    
    heart_features = ['Age', 'Sex', 'Chest Pain Type (cp)', 'Resting Blood Pressure (trestbps)', 'Cholesterol (chol)', 'Fasting Blood Sugar (fbs)', 'Resting ECG (restecg)', 'Max Heart Rate (thalach)', 'Exercise Induced Angina (exang)', 'Oldpeak', 'Slope', 'Ca', 'Thal']
    user_inputs = [st.text_input(f) for f in heart_features]
    
    if st.button('Heart Disease Test Result'):
        user_input = validate_input(user_inputs)
        if user_input and len(user_input) == 13 and models["Heart Disease"]:
            heart_prediction = models["Heart Disease"].predict([user_input])
            result = '🚨 The person is likely to have heart disease' if heart_prediction[0] == 1 else '✅ The person is not likely to have heart disease'
            st.success(result)

# -------------------------------
# Parkinson's Disease Prediction Section
# -------------------------------
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")
    
    parkinsons_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    user_inputs = [st.text_input(f) for f in parkinsons_features]
    
    if st.button("Parkinson's Test Result"):
        user_input = validate_input(user_inputs)
        if user_input and len(user_input) == 22 and models["Parkinson's"]:
            parkinsons_prediction = models["Parkinson's"].predict([user_input])
            result = '🚨 The person is likely to have Parkinson's disease' if parkinsons_prediction[0] == 1 else '✅ The person is not likely to have Parkinson's disease'
            st.success(result)
