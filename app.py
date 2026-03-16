import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="JobShield AI",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- LOGIN SYSTEM ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():

    st.markdown(
        """
        <h1 style='text-align:center;'>🛡️ JobShield AI</h1>
        <h3 style='text-align:center;'>Fake Job Detection System</h3>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

# ---------------- LOAD MODEL ----------------

@st.cache_data
def load_model():

    data = pd.read_csv("dataset.csv")

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return vectorizer, model, data

# ---------------- MAIN APP ----------------

def main_app():

    vectorizer, model, data = load_model()

    st.sidebar.title("Navigation")

    menu = st.sidebar.radio(
        "Go to",
        [
            "Dashboard",
            "Fake Job Detector",
            "Email Checker",
            "URL Checker",
            "About Project"
        ]
    )

# ---------------- DASHBOARD ----------------

    if menu == "Dashboard":

        st.title("📊 Project Dashboard")

        total = len(data)
        fake = len(data[data.label == "Fake"])
        real = len(data[data.label == "Real"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Messages", total)
        col2.metric("Fake Jobs", fake)
        col3.metric("Real Jobs", real)

        st.write("### Sample Dataset")
        st.dataframe(data)

# ---------------- JOB DETECTOR ----------------

    elif menu == "Fake Job Detector":

        st.title("🧠 Fake Job Message Detection")

        message = st.text_area("Paste Job Message")

        if st.button("Analyze Message"):

            vec = vectorizer.transform([message])
            prediction = model.predict(vec)[0]

            if prediction == "Fake":
                st.error("⚠ This appears to be a FAKE job message")
            else:
                st.success("✅ This job message looks genuine")

# ---------------- EMAIL CHECKER ----------------

    elif menu == "Email Checker":

        st.title("📧 Email Scam Checker")

        email = st.text_input("Enter Email Address")

        if st.button("Check Email"):

            if "gmail.com" in email or "yahoo.com" in email:
                st.warning("⚠ Free email domain – may be suspicious")

            elif "hr" in email or "job" in email:
                st.info("ℹ Job related email detected")

            else:
                st.success("✅ Email looks safe")

# ---------------- URL CHECKER ----------------

    elif menu == "URL Checker":

        st.title("🌐 URL Scam Detection")

        url = st.text_input("Paste Website URL")

        if st.button("Analyze URL"):

            if "xyz" in url or "free" in url:
                st.error("⚠ Suspicious URL detected")

            elif "https" in url:
                st.success("✅ Secure website")

            else:
                st.warning("⚠ Website may not be secure")

# ---------------- ABOUT ----------------

    elif menu == "About Project":

        st.title("📘 About Project")

        st.write("""
        **JobShield AI** is a machine learning based system that detects fake job
        offers and internship scams.

        Features:
        • Fake Job Message Detection  
        • Email Scam Detection  
        • URL Phishing Detection  

        Technology Used:
        - Python
        - Streamlit
        - TF-IDF
        - Naive Bayes Machine Learning
        """)

# ---------------- RUN APP ----------------

if st.session_state.logged_in:
    main_app()
else:
    login()
