import os      # to interact with file system("operating System")
import pickle   # for loading pre-trained models
import streamlit as st   # web app
from streamlit_option_menu import option_menu   # for sidebar menu

# Set the page configuration
st.set_page_config(page_title='Prediction of Disease Outbreaks',
                   layout='wide',  # allow us to use the full page
                   page_icon='doctor')  # gives an emoji for the web page

# Set file paths for models (make it dynamic using os.path)
diabetes_model_path = r'C:\Users\hp\Desktop\DISEASE OUTBREAKS\training_models\diabetes_model.sav'
heart_disease_model_path =  r'C:\Users\hp\Desktop\DISEASE OUTBREAKS\training_models\heartmodel.sav'
parkinsons_model_path = r'C:\Users\hp\Desktop\DISEASE OUTBREAKS\training_models\parkinsons_model.pkl'

# Load the models
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))
import joblib
parkinsons_model = joblib.load(parkinsons_model_path)

# Sidebar menu
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreak System', 
                           ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
                           menu_icon='hospital-fill', 
                           icons=['activity', 'heart', 'person'], 
                           default_index=0)

# Based on the selected option, display corresponding content
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    # Add widgets to collect input from user (for example, age, BMI, etc.)
    # Predict using diabetes_model and display the result
    
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    # Add widgets for heart disease input
    # Predict using heart_disease_model and display the result
    
elif selected == "Parkinson's Prediction":
    st.title("Parkinson's Prediction")
    # Add widgets for Parkinson's input
    # Predict using parkinsons_model and display the result

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    col1,col2,col3 = st.columns(3)
    with col1:
        Pregnancies= st.text_input ('Number of Pregnancies')
    with col2:
        Glucose= st.text_input('Glucose level')
    with col3:
        BloodPressure= st.text_input('Blood Pressure Value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness Value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI =st.text_input('BMI Value')
    with col1:
        DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
    with col2:
        Age=st.text_input('Age of the person')

diab_diagnosis = ''
if st.button('Diabetes Test Result'):
    user_input = [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    user_input= [float(x)for x in user_input]
    diab_prediction= diabetes_model.predict([user_input])
    if diab_prediction[0]==1:
        diab_diagnosis='The person is diabetic'
    else:
        diab_diagnosis='The person is not diabetic'
st.success(diab_diagnosis)

if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    col1,col2,col3 = st.columns(3)
    with col1:
        Age= st.text_input ('Age of the person')
    with col2:
        Sex= st.text_input('Sex')
    with col3:
        cp= st.text_input('cp Value')
    with col1:
        Trestbps = st.text_input('Value of trestbps')
    with col2:
        Chol = st.text_input('chol Level')
    with col3:
        fbs =st.text_input('fbs Value')
    with col1:
        restecg=st.text_input('restecg Value')
    with col2:
        Thalach=st.text_input('Thalach Value')
    with col3:
        exang=st.text_input('exang Value')
    with col1:
        Oldpeak=st.text_input('oldpeak Value')
    with col2:
        slope=st.text_input('slope Value')
    with col3:
        ca=st.text_input('ca Value')
    with col1:
        thal=st.text_input('thal Value')
    

    