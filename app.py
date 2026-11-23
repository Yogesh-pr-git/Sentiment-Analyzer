import streamlit as st
import joblib
import pandas as pd

#Load Model
model = joblib.load('model.pkl')

#Build Web application
st.title("🤖AI-Powered Sentiment Analyzer System")
st.write("Enter Text to Predict Sentiment")

if model is None:
    st.error(f"Model file '{model}' not found! Please run the serialization cell in your Colab notebook first to save the model.")
else:
    user_input = st.text_area("Enter Your Review Here:",placeholder="Type Something like: 'This product is absolutely amazing!")

    if st.button("Analyze Sentiment",type="primary"):
        if user_input.strip() == "":
            st.warning("Please Enter Some text to Analyze")
        else:
            #Prediction
            prediction = model.predict([user_input])[0]
            probability = model.predict_proba([user_input])[0]
            classes = model.classes_

            #Map Result colors
            color_map = {
                "Positive":"green",
                "Negative":"red",
                "Neutral" :"orange"
            }
            text_color = color_map.get(prediction,"blue")

            #Display main prediction Result
            st.markdown(f'### Sentiment:{text_color}[{prediction}]')

            #display probability breakdown
            st.write("** Confidence Breakdown **")
            for cls,prob in zip(classes, probability):
                st.write(f" - {cls}: {prob * 100:.2f}%")
                st.progress(float(prob))







