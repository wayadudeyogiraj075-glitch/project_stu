import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("ML Model Prediction App")

st.write("Enter input values to get prediction:")

# Example: change these inputs based on your model features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# Convert input to array
input_data = np.array([[feature1, feature2, feature3]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")
