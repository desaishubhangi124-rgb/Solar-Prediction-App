import streamlit as st
import pandas as pd
import pickle

# load model
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("⚡ Solar Recommendation System")
st.markdown("💡 ML-based Monthly Savings Prediction")

st.divider()

# ---------------- INPUT (WITH DEFAULT VALUES) ----------------
load_kw = st.number_input("⚙️ Load (kW)", value=5.0)
monthly_kwh = st.number_input("🔋 Monthly kWh", value=300.0)
yearly_kwh = st.number_input("📈 Yearly kWh", value=3600.0)
tariff_rate = st.number_input("💰 Tariff Rate", value=8.0)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Savings"):

    input_data = {col: 0 for col in columns}

    if "Load_kW" in input_data:
        input_data["Load_kW"] = load_kw
    if "Monthly_KWh" in input_data:
        input_data["Monthly_KWh"] = monthly_kwh
    if "Yearly_KWh" in input_data:
        input_data["Yearly_KWh"] = yearly_kwh
    if "Tariff_Rate" in input_data:
        input_data["Tariff_Rate"] = tariff_rate

    input_df = pd.DataFrame([input_data])
    input_df = input_df[columns]

    prediction = model.predict(input_df)[0]

    st.success(f"💡 Estimated Monthly Savings: ₹ {prediction:.2f}")
    st.balloons()

st.markdown("---")
st.markdown("⚡ Built with Streamlit + Machine Learning")
