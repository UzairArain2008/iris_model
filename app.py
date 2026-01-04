import streamlit as st
import pickle
import numpy as np

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")

st.title("🌸 Iris Flower Prediction App")
st.write("Enter flower measurements to predict the species.")

# User inputs
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌼 Predicted Species: **{species[prediction]}**")
