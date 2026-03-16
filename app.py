import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="JobShield AI", layout="wide")

st.title("🛡 JobShield AI")
st.subheader("Fake Job, Email & URL Detection System")

# Load dataset
data = pd.read_csv("dataset.csv")

X = data["text"]
y = data["label"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model training
model = MultinomialNB()
model.fit(X_vec, y)

menu = st.sidebar.selectbox(
    "Navigation",
    ["Home","Job Message Detection","Email Checker","URL Checker"]
)

# ---------------- HOME ----------------

if menu == "Home":

    st.write("### Welcome to JobShield AI")

    st.write("""
    This system detects:
    
    • Fake job messages  
    • Suspicious emails  
    • Scam URLs
    
    Using Machine Learning.
    """)

# ---------------- JOB MESSAGE ----------------

elif menu == "Job Message Detection":

    st.header("📩 Fake Job Message Detector")

    message = st.text_area("Paste Job Message")

    if st.button("Analyze Message"):

        vec = vectorizer.transform([message])
        prediction = model.predict(vec)[0]

        if prediction == "Fake":
            st.error("⚠ This looks like a FAKE job message")
        else:
            st.success("✅ This job message looks genuine")

# ---------------- EMAIL CHECKER ----------------

elif menu == "Email Checker":

    st.header("📧 Email Scam Detector")

    email = st.text_input("Enter Email")

    if st.button("Check Email"):

        if "gmail.com" in email or "yahoo.com" in email:
            st.warning("⚠ Free email domains are often used in scams")

        elif "hr" in email or "job" in email:
            st.info("ℹ Job related email detected")

        else:
            st.success("✅ Email looks normal")

# ---------------- URL CHECKER ----------------

elif menu == "URL Checker":

    st.header("🌐 URL Scam Detector")

    url = st.text_input("Paste URL")

    if st.button("Check URL"):

        if "xyz" in url or "free" in url:
            st.error("⚠ Suspicious website detected")

        elif "https" in url:
            st.success("✅ Secure website")

        else:
            st.warning("⚠ Website may not be secure")

st.sidebar.markdown("---")
st.sidebar.write("Developed by TYBCA Students")
