import streamlit as st
import joblib
import numpy as np

#Load model
model = joblib.load('model.pkl')

# Page configuration
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# Custom title
st.title("🎓 Student Placement Prediction")
st.write("Predict whether a student will get placed based on IQ and CGPA.")

st.divider()

# Input section
st.subheader("Enter Student Details")

IQ = st.number_input(
    "Enter Student IQ",
    min_value=50,
    max_value=200,
    value=100
)

CGPA = st.number_input(
    "Enter Student CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

# Prediction button
if st.button("Predict Placement", type="primary"):

    # Prepare input
    input_data = np.array([[IQ, CGPA]])

    # Prediction
    prediction = model.predict(input_data)

    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("🎉 Congratulations! Student is likely to get placed.")

    else:
        st.error("Unfortunately, student is unlikely to get placed.")

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]

        st.write("### Prediction Probability")

        st.write(f"Placement Probability: {probability[1]*100:.2f}%")
        st.progress(float(probability[1]))

        st.write(f"Non-Placement Probability: {probability[0]*100:.2f}%")

st.divider()

st.caption("Machine Learning Project | Student Placement Prediction")
