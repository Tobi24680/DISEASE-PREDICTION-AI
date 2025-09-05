import streamlit as s
import joblib as j
import numpy as np
import pandas as pd


diabetes = j.load("diabetes_model.pkl")
heart = j.load("heart_model.pkl")
liver = j.load("liver_model.pkl")

s.markdown(
    """
    <h1 style="text-align:center; color:#1A5276; font-family:Trebuchet MS, sans-serif; font-size:42px;">
        🩺 Disease Prediction AI
    </h1>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = s.columns(3)
with col1:
        s.success("🧬 **Diabetes Prediction**\n\nCheck your risk of diabetes.")
with col2:
        s.error("❤️ **Heart Disease**\n\nPredict possibility of heart issues.")
with col3:
        s.warning("🧪 **Liver Disease**\n\nAnalyze liver health indicators.")


s.sidebar.markdown("<h2 style='color:black;'> Disease Prediction</h2>", unsafe_allow_html=True)
s.sidebar.write("Choose below disease to check prediction:")

option = s.sidebar.selectbox(
    "Select..",
    ["Home", "DIABETES", "HEART", "LIVER"],
    index=0
)


if option == "Home":
    s.markdown("<h1 style='text-align:center; color:black;'>🏠 Welcome to Disease Prediction AI</h1>", unsafe_allow_html=True)

    s.write("""
    This AI will uses Machine Learning to predict the chances of Diabetes,  Heart Disease, and  Liver Disease. 
    

    ### 📌 How to Use:
    1. Select a disease from the sidebar.  
    2. Fill in your medical details in the form.  
    3. Click Submit to see the prediction.  

    ### ⚠️ Disclaimer:  
    - Always consult a doctor for professional medical advice.  
    """)


elif option == "DIABETES":
    s.header("Diabetes Prediction")

    pregnancies = s.number_input("Pregnancies", min_value=0, step=1)
    glucose = s.number_input("Glucose", min_value=1, step=1)
    blood_pressure = s.number_input("Blood Pressure", min_value=1, step=1)
    skin_thickness = s.number_input("Skin Thickness", min_value=1, step=1)
    insulin = s.number_input("Insulin", min_value=1, step=1)
    bmi = s.number_input("BMI", min_value=1.0, step=0.1, format="%.1f")
    dpf = s.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, format="%.2f")
    age = s.number_input("Age", min_value=1, step=1)

    if s.button("Submit"):
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]], columns=[
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ])
        prediction = diabetes.predict(input_data)[0]

        if prediction == 1:
            s.error("⚠️ The person is likely to have Diabetes.")
        else:
            s.success("✅ The person is not likely to have Diabetes.")


elif option == "HEART":
    s.header("Heart Disease Prediction")

    age = s.number_input("Age", min_value=1, step=1)
    sex = s.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = s.number_input("Chest Pain Type", min_value=0, max_value=3, step=1)
    trestbps = s.number_input("Resting Blood Pressure", min_value=1, step=1)
    chol = s.number_input("Cholesterol", min_value=1, step=1)
    fbs = s.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
    restecg = s.number_input("Resting ECG Results (0-2)", min_value=0, max_value=2, step=1)
    thalach = s.number_input("Thalach", min_value=1, step=1)
    exang = s.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
    oldpeak = s.number_input("Oldpeak", min_value=0.0, step=0.1, format="%.1f")
    slope = s.number_input("Slope", min_value=0, max_value=2, step=1)
    ca = s.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, step=1)
    thal = s.number_input("Thal", min_value=0, max_value=3, step=1)

    if s.button("Submit"):
        input_data = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal
        ]], columns=[
            "age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"
        ])
        prediction = heart.predict(input_data)[0]

        if prediction == 1:
            s.error("⚠️ The person is likely to have Heart Disease.")
        else:
            s.success("✅ The person is not likely to have Heart Disease.")

elif option == "LIVER":
    s.header(" Liver Disease Prediction")

    age = s.number_input("Age", min_value=1, step=1)
    gender = s.selectbox("Gender", ["Male", "Female"])
    gender_val = 1 if gender == "Male" else 0

    total_bilirubin = s.number_input("Total Bilirubin", min_value=0.0, step=0.1)
    direct_bilirubin = s.number_input("Direct Bilirubin", min_value=0.0, step=0.1)
    alkaline_phosphotase = s.number_input("Alkaline Phosphotase", min_value=1, step=1)
    alamine_aminotransferase = s.number_input("Alamine Aminotransferase", min_value=1, step=1)
    aspartate_aminotransferase = s.number_input("Aspartate Aminotransferase", min_value=1, step=1)
    total_proteins = s.number_input("Total Proteins", min_value=0.0, step=0.1)
    albumin = s.number_input("Albumin", min_value=0.0, step=0.1)
    agr = s.number_input("Albumin and Globulin Ratio", min_value=0.0, step=0.1)

    if s.button("Submit"):
        input_data = pd.DataFrame([[
            age, gender_val,total_bilirubin, direct_bilirubin,
            alkaline_phosphotase,alamine_aminotransferase,
            aspartate_aminotransferase,total_proteins,albumin,agr
        ]], columns=[
            "Age","Gender","Total_Bilirubin","Direct_Bilirubin",
            "Alkaline_Phosphotase","Alamine_Aminotransferase",
            "Aspartate_Aminotransferase","Total_Protiens",
            "Albumin","Albumin_and_Globulin_Ratio"
        ])
        prediction = liver.predict(input_data)[0]

        if prediction == 1:
            s.error("⚠️ The person is likely to have Liver Disease.")
        else:
            s.success("✅ The person is not likely to have Liver Disease.")



s.markdown("---", unsafe_allow_html=True)
s.markdown(
    """
    <div style="text-align:center; padding:15px; color:#2E86C1;">
        <h4>🤖 AI for Health • Smarter Predictions, Better Decisions</h4>
        <p> Powered by Artificial Intelligence | Built By SHREERAM M K </p>
    </div>
    """,
    unsafe_allow_html=True
)

