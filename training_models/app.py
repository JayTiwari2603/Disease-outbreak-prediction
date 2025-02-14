import os
import pickle
import joblib
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title='Disease Prediction System', layout='wide', page_icon='🩺')

# Function to load models safely
def load_model(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)  # Try pickle first
    except (pickle.UnpicklingError, EOFError):
        return joblib.load(path)  # Try joblib as fallback
    except Exception as e:
        st.error(f"❌ Error loading model {path}: {e}")
        return None

# Define model paths
model_paths = {
    "Diabetes": r"C:\Users\hp\Desktop\DISEASE OUTBREAKS\training_models\diabetes_model.sav",
    "Heart Disease": r"C:\Users\hp\Desktop\DISEASE OUTBREAKS\training_models\heartmodel.sav",
    "Parkinson's": r"C:\Users\hp\Desktop\DISEASE OUTBREAKS\training_models\parkinsons_model.pkl"
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

# -------------------------------
# Diabetes Prediction Section
# -------------------------------
if selected == 'Diabetes Prediction':
    st.title('🩸 Diabetes Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age = st.text_input('Age of the person')

    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies or 0), float(Glucose or 0), float(BloodPressure or 0), 
                          float(SkinThickness or 0), float(Insulin or 0), float(BMI or 0), 
                          float(DiabetesPedigreeFunction or 0), float(Age or 0)]

            diab_prediction = models["Diabetes"].predict([user_input])
            result = '🚨 The person is diabetic' if diab_prediction[0] == 1 else '✅ The person is not diabetic'
            st.success(result)

        except ValueError:
            st.error('⚠️ Please enter valid numerical values')

# -------------------------------
# Heart Disease Prediction Section
# -------------------------------
elif selected == 'Heart Disease Prediction':
    st.title('❤️ Heart Disease Prediction using ML')

    heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                      'restecg', 'thalach', 'exang', 'oldpeak', 
                      'slope', 'ca', 'thal']

    user_inputs = []

    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(heart_features):
        with [col1, col2, col3][i % 3]:  
            user_inputs.append(st.text_input(feature))

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(x.strip()) if x.strip() else 0 for x in user_inputs]

            if len(user_input) != 13:
                st.error(f"❌ Incorrect input count! Expected 13, got {len(user_input)}")
            else:
                heart_prediction = models["Heart Disease"].predict([user_input])
                result = '🚨 The person is likely to have heart disease' if heart_prediction[0] == 1 else '✅ The person is not likely to have heart disease'
                st.success(result)

        except ValueError:
            st.error('⚠️ Please enter only numerical values.')

# -------------------------------
# Parkinson's Disease Prediction Section
# -------------------------------
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction using ML")

    parkinsons_features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
        'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
        'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
        'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
    ]

    user_inputs = []

    col1, col2, col3 = st.columns(3)
    for i, feature in enumerate(parkinsons_features):
        with [col1, col2, col3][i % 3]:  
            user_inputs.append(st.text_input(feature))

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(x.strip() or 0) for x in user_inputs]

            if len(user_input) != 22:
                st.error(f"❌ Incorrect input count! Expected 22, got {len(user_input)}")
            else:
                parkinsons_prediction = models["Parkinson's"].predict([user_input])
                result = '🚨 The person is likely to have Parkinson’s disease' if parkinsons_prediction[0] == 1 else '✅ The person is not likely to have Parkinson’s disease'
                st.success(result)

        except ValueError:
            st.error('⚠️ Please enter only numerical values.')
