import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("⚡ Solar Recommendation System")
st.markdown("💡 AI-based Monthly Savings Prediction")

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Energy Details")

load_kw = st.number_input("⚙️ Load (kW)", value=None)
monthly_kwh = st.number_input("🔋 Monthly kWh", value=None)
yearly_kwh = st.number_input("📈 Yearly kWh", value=None)
tariff_rate = st.number_input("💰 Tariff Rate", value=None)

st.divider()

# ---------------- IMAGE (OPTIONAL) ----------------
st.subheader("🖼️ Upload Image (Optional)")

uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

st.divider()

# ---------------- VALIDATION ----------------
if st.button("🚀 Predict Savings"):

    if load_kw is None or monthly_kwh is None or yearly_kwh is None or tariff_rate is None:
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
