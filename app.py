import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("job_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake Job Posting Detector")

st.write("Paste a job description below to check if it's fake or real.")

user_input = st.text_area("Job Description")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a job description.")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("Fake Job Posting")
            st.write("Be careful: This job may contain scam patterns.")
        else:
            st.success("Real Job Posting")
            st.write("This job appears legitimate based on text analysis.")

st.write("Note: This prediction is based on a machine learning model and may not always be accurate.")
