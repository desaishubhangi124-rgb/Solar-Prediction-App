import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("⚡ Solar Recommendation System")
st.markdown("💡 AI-based monthly savings prediction system")

st.divider()

# ---------------- IMAGE UPLOAD ----------------
st.subheader("🖼️ Upload Image")

uploaded_file = st.file_uploader("📁 Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image Preview", use_container_width=True)

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Energy Details")

load_kw = st.number_input("⚙️ Load (kW)", value=5.0, step=0.5)
monthly_kwh = st.number_input("🔋 Monthly kWh", value=300.0, step=10.0)
yearly_kwh = st.number_input("📈 Yearly kWh", value=3600.0, step=50.0)
tariff_rate = st.number_input("💰 Tariff Rate", value=8.0, step=0.5)

st.divider()

# ---------------- PREPARE INPUT ----------------
input_data = {
    "Load_kW": load_kw,
    "Monthly_KWh": monthly_kwh,
    "Yearly_KWh": yearly_kwh,
    "Tariff_Rate": tariff_rate
}

input_df = pd.DataFrame([input_data])

# match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Savings"):
    prediction = model.predict(input_df)[0]

    st.success(f"💡 Estimated Monthly Savings: ₹ {prediction:.2f}")

    st.balloons()

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🔧 Built with Streamlit | ⚡ Machine Learning Powered Solar System")