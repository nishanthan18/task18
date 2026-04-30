import streamlit as st
import joblib
import numpy as np

st.title("🏠 House Price Prediction")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

area = st.number_input("Area", 500, 5000, 2000)
bedrooms = st.number_input("Bedrooms", 1, 6, 3)
bathrooms = st.number_input("Bathrooms", 1, 4, 2)
floors = st.number_input("Floors", 1, 3, 1)
age = st.number_input("Age", 0, 50, 5)
location = st.slider("Location Score", 1, 10, 5)

if st.button("Predict"):
    data = np.array([[area, bedrooms, bathrooms, floors, age, location]])
    data = scaler.transform(data)
    prediction = model.predict(data)
    st.success(f"Predicted Price: ₹ {int(prediction[0]):,}")