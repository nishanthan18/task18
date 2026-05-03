import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction")

# Check model exists
if not os.path.exists("model.pkl"):
    st.error("❌ model.pkl not found. Run train_model.py and upload it.")
    st.stop()

# Load model
model = joblib.load("model.pkl")

st.subheader("Enter House Details")

area = st.number_input("Area (sq ft)", 500, 5000, 2000)
bedrooms = st.number_input("Bedrooms", 1, 6, 3)
bathrooms = st.number_input("Bathrooms", 1, 4, 2)
floors = st.number_input("Floors", 1, 3, 1)
age = st.number_input("Age of House", 0, 50, 5)
location = st.slider("Location Score", 1, 10, 5)

if st.button("Predict Price"):
    try:
        data = np.array([[area, bedrooms, bathrooms, floors, age, location]])
        prediction = model.predict(data)

        st.success(f"💰 Estimated Price: ₹ {int(prediction[0]):,}")

    except Exception as e:
        st.error("⚠️ Prediction failed. Check input or model.")