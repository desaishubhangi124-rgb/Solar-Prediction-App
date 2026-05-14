import streamlit as st
import pandas as pd
import pickle

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("⚡ Solar Recommendation System")
st.markdown("💡 AI-based Monthly Savings Prediction")

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Energy Details")

load_kw = st.number_input("⚙️ Load (kW)")
monthly_kwh = st.number_input("🔋 Monthly kWh")
yearly_kwh = st.number_input("📈 Yearly kWh")
tariff_rate = st.number_input("💰 Tariff Rate")

st.divider()

# ---------------- VALIDATION + PREDICTION ----------------
if st.button("🚀 Predict Savings"):

    if load_kw == 0 or monthly_kwh == 0 or yearly_kwh == 0 or tariff_rate == 0:
        st.warning("⚠️ Please enter all values before prediction!")

    else:
        input_data = {
            "Load_kW": load_kw,
            "Monthly_KWh": monthly_kwh,
            "Yearly_KWh": yearly_kwh,
            "Tariff_Rate": tariff_rate
        }

        input_df = pd.DataFrame([input_data])

        # match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        st.success(f"💡 Estimated Monthly Savings: ₹ {prediction:.2f}")
        st.balloons()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🔧 Built with Streamlit | ⚡ ML Solar Prediction System")
