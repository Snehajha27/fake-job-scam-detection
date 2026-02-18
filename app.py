import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page Configuration
st.set_page_config(page_title="Fake Job Scam Detector", layout="centered")

# Title Section
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Fake Job & Internship Scam Detection System</h1>", unsafe_allow_html=True)

st.markdown("### JSPM University - TYBCA Final Year Project")
st.markdown("**Team Members:** Sneha Jha | Sujit | Ashutosh")

st.markdown("---")

# Load Dataset
data = pd.read_csv("dataset.csv")

X = data['text']
y = data['label']

# Train Model
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# User Input
st.subheader("Enter Job / Internship Message Below:")
msg = st.text_area("")

if st.button("Analyze Message"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        vect = vectorizer.transform([msg])
        prediction = model.predict(vect)
        probability = model.predict_proba(vect)

        fake_prob = round(probability[0][1] * 100, 2)
        genuine_prob = round(probability[0][0] * 100, 2)

        st.markdown("---")

        if prediction[0] == 1:
            st.error(f"⚠️ Fake Job Offer Detected! (Confidence: {fake_prob}%)")
        else:
            st.success(f"✅ Genuine Job Offer (Confidence: {genuine_prob}%)")

        st.info(f"Fake Probability: {fake_prob}% | Genuine Probability: {genuine_prob}%")

st.markdown("---")
st.caption("Developed using Machine Learning (Naive Bayes) & NLP | Streamlit Deployment")

