import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="AI Disease Predictor", layout="wide")

# Sidebar toggle for theme
selected_theme = st.sidebar.radio("🌗 Select Theme", ["Dark", "Light"])

# Theme-specific CSS
if selected_theme == "Dark":
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e1117;
        color: #d0d0d0;
        font-family: 'Segoe UI', sans-serif;
    }
    input, select {
        background-color: #1e1e1e !important;
        color: #d0d0d0 !important;
    }
    .stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        height: 42px;
    }
    .stNumberInput input {
        background-color: #1e1e1e !important;
    }
    .result-box {
        background-color: #1f2937;
        padding: 1.5em;
        border-radius: 12px;
        margin-top: 1.5em;
        text-align: center;
    }
    .result-box h3 {
        color: #4CAF50;
    }
    div[data-testid="stForm"] {
        border: 2px solid red;
        padding: 20px;
        border-radius: 10px;
        background-color: #1f2937;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

else:  # Light theme
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #ffffff;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    input, select {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stButton > button {
        background-color: #1976d2 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: 600;
        height: 42px;
    }
    .stNumberInput input {
        background-color: #ffffff !important;
    }
    .result-box {
        background-color: #e3f2fd;
        padding: 1.5em;
        border-radius: 12px;
        margin-top: 1.5em;
        text-align: center;
    }
    .result-box h3 {
        color: #1976d2;
    }
    div[data-testid="stForm"] {
        border: 2px solid red;
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🩺 AI Disease Prediction Tool")
st.markdown("This AI Disease Prediction Model is an intelligent healthcare support tool that utilizes machine learning techniques to predict potential diseases based on a range of user-input health indicators such as age, gender, BMI (Body Mass Index), blood pressure, sugar level, cholesterol levels, smoking habits, and family medical history. By analyzing these parameters, the model—trained on a comprehensive dataset and supported by encoded categorical variables—can identify and classify the most probable diseases a patient might have. It presents the top predicted disease along with a percentage-based confidence score, and also displays the next two most likely conditions. This enables users to better understand their health risks and encourages proactive care. While it is not a substitute for clinical diagnosis or treatment, the model serves as a valuable assistant for preliminary screening, education, and promoting healthy lifestyle choices by offering recommendations based on the predictions.")
st.subheader("📋 Add Parameters")

# Input Form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)

    with col2:
        bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
        sugar = st.number_input("Sugar Level (mg/dL)", 50, 300, 100)
        smoking = st.selectbox("Smoking Status", encoders["Smoking"].classes_)

    with col3:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 180)
        fam = st.selectbox("Family History", encoders["FamilyHistory"].classes_)

    submitted = st.form_submit_button("🔍 Predict Disease")

# On form submission
if submitted:
    input_dict = {
        "Age": age,
        "Gender": encoders["Gender"].transform([gender])[0],
        "BMI": bmi,
        "BP": bp,
        "Sugar": sugar,
        "Cholesterol": chol,
        "Smoking": encoders["Smoking"].transform([smoking])[0],
        "FamilyHistory": encoders["FamilyHistory"].transform([fam])[0]
    }
    X = pd.DataFrame([input_dict])

    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]
    disease_classes = encoders["Disease"].inverse_transform(model.classes_)

    predicted_disease = encoders["Disease"].inverse_transform([prediction])[0]
    accuracy = prediction_proba[prediction] * 100

    top3_idx = np.argsort(prediction_proba)[::-1][:3]
    top3_diseases = disease_classes[top3_idx]
    top3_probs = prediction_proba[top3_idx] * 100

    st.markdown(f"""
    <div class='result-box'>
        <h3>🎯 Predicted Disease: {predicted_disease}</h3>
        <p><strong>Accuracy:</strong> {accuracy:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    fig = px.bar(x=top3_diseases, y=top3_probs,
                 labels={'x': 'Disease', 'y': 'Chances (%)'},
                 color=top3_diseases,
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title="Chances(%)", height=350, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 💡 Recommendation")
    if predicted_disease == "No Disease":
        st.success("You're doing great! Stay healthy 🧘‍♂️")
    else:
        st.warning(f"Consult a physician regarding **{predicted_disease}**.")
        st.markdown("""
        - 🥦 Maintain a balanced diet  
        - 🚶‍♀️ Stay physically active  
        - 🛌 Get 7–8 hours of sleep  
        - 🚭 Quit smoking/alcohol (if applicable)  
        - 🧪 Monitor BP, Sugar, Cholesterol regularly  
        """)

