import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Fake Job Scam Detector", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Fake Job & Internship Scam Detection System</h1>", unsafe_allow_html=True)

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

# Scam Keywords List
scam_keywords = ["fee", "registration", "OTP", "urgent", "limited seats", "payment", "click here", "guaranteed"]

# URL Detection Function
def contains_suspicious_url(text):
    urls = re.findall(r'(https?://\S+)', text)
    suspicious = []
    for url in urls:
        if not any(domain in url for domain in ["tcs.com", "infosys.com", "wipro.com"]):
            suspicious.append(url)
    return suspicious

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

        # Prediction Result
        if prediction[0] == 1:
            st.error(f"⚠️ Fake Job Offer Detected! (Confidence: {fake_prob}%)")
        else:
            st.success(f"✅ Genuine Job Offer (Confidence: {genuine_prob}%)")

        # URL Detection
        suspicious_urls = contains_suspicious_url(msg)
        if suspicious_urls:
            st.warning("⚠️ Suspicious URL Detected:")
            for url in suspicious_urls:
                st.write(url)

        # Keyword Highlighting
        detected_keywords = [word for word in scam_keywords if word.lower() in msg.lower()]
        if detected_keywords:
            st.info(f"⚠️ Scam Related Keywords Found: {', '.join(detected_keywords)}")

        st.markdown("---")
        st.info(f"Fake Probability: {fake_prob}% | Genuine Probability: {genuine_prob}%")

# Awareness Section
st.markdown("---")
st.subheader("⚠️ Scam Awareness Tips")

st.markdown("""
- Never pay registration or interview fees.
- Do not share OTP or personal banking details.
- Verify company email domain.
- Avoid urgent payment requests.
- Check company website authenticity.
""")

st.markdown("---")
st.caption("Developed using Machine Learning (Naive Bayes), NLP & Streamlit | 2026 TYBCA Project")


