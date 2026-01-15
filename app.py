import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Health Risk Predictor", layout="centered")

st.title("🩺 AI Health Risk Prediction System")
st.write("Predict health risk using ML and get AI-based explanation")

# -------------------------
# API KEY
# -------------------------
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY not found in Streamlit Secrets")
    st.stop()

# -------------------------
# SAMPLE DATA & MODEL
# -------------------------
data = {
    "age": [22, 35, 45, 60],
    "bmi": [19, 28, 32, 30],
    "smoking": [0, 1, 1, 0],
    "risk": [0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["age", "bmi", "smoking"]]
y = df["risk"]

ml_model = LogisticRegression()
ml_model.fit(X, y)

# -------------------------
# USER INPUT
# -------------------------
st.subheader("Enter Your Details")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
smoking = st.selectbox("Smoking Habit", ["No", "Yes"])

smoking_val = 1 if smoking == "Yes" else 0

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict Health Risk"):
    prediction = ml_model.predict([[age, bmi, smoking_val]])
    risk = "High Risk" if prediction[0] == 1 else "Low Risk"

    st.success(f"Predicted Health Risk: **{risk}**")

    # -------------------------
    # LLM SETUP
    # -------------------------
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=api_key
    )

    health_prompt = PromptTemplate(
        input_variables=["age", "bmi", "smoking", "risk"],
        template="""
You are an AI health assistant.

User details:
Age: {age}
BMI: {bmi}
Smoking habit: {smoking}
Predicted health risk: {risk}

Explain the risk in simple words.
Give exactly 3 lifestyle improvement tips.
Add a disclaimer that this is not medical advice.
"""
    )

    health_chain = health_prompt | llm

    response = health_chain.invoke({
        "age": age,
        "bmi": bmi,
        "smoking": smoking,
        "risk": risk
    })

    st.subheader("🤖 AI Health Advice")
    st.write(response.content)
