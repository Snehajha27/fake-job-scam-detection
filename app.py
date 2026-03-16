import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(page_title="JobShield AI", page_icon="🛡️", layout="wide")

# ---------------- CUSTOM COLORS ----------------

st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.big-title {
    font-size:40px !important;
    color:#4ade80;
    text-align:center;
    font-weight:bold;
}

.sub-title {
    text-align:center;
    color:#cbd5f5;
}

.card {
    background-color:#1e293b;
    padding:20px;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SESSION ----------------

if "login" not in st.session_state:
    st.session_state.login = False


# ---------------- LOGIN PAGE ----------------

def login_page():

    st.markdown('<p class="big-title">🛡️ JobShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Fake Job Detection System</p>', unsafe_allow_html=True)

    st.write("")

    with st.container():

        st.markdown('<div class="card">', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if username == "admin" and password == "1234":
                st.session_state.login = True
                st.success("Login Successful")
            else:
                st.error("Invalid Username or Password")

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

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
        "Go To",
        ["Dashboard", "Fake Job Detector", "Email Checker", "URL Checker", "About"]
    )

# ---------------- DASHBOARD ----------------

    if menu == "Dashboard":

        st.markdown("## 📊 Dashboard")

        total = len(data)
        fake = len(data[data.label == "Fake"])
        real = len(data[data.label == "Real"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Messages", total)
        col2.metric("Fake Jobs", fake)
        col3.metric("Real Jobs", real)

        st.write("### Dataset Preview")
        st.dataframe(data)

# ---------------- JOB DETECTOR ----------------

    elif menu == "Fake Job Detector":

        st.markdown("## 🧠 Fake Job Message Detector")

        message = st.text_area("Paste Job Message Here")

        if st.button("Analyze"):

            vec = vectorizer.transform([message])
            prediction = model.predict(vec)[0]

            if prediction == "Fake":
                st.error("⚠ This message looks like a FAKE job offer")
            else:
                st.success("✅ This job message seems genuine")

# ---------------- EMAIL CHECKER ----------------

    elif menu == "Email Checker":

        st.markdown("## 📧 Email Scam Checker")

        email = st.text_input("Enter Email Address")

        if st.button("Check Email"):

            if "gmail.com" in email or "yahoo.com" in email:
                st.warning("⚠ Free email domain – could be suspicious")

            elif "hr" in email or "job" in email:
                st.info("ℹ Job related email detected")

            else:
                st.success("✅ Email looks normal")

# ---------------- URL CHECKER ----------------

    elif menu == "URL Checker":

        st.markdown("## 🌐 URL Scam Detector")

        url = st.text_input("Enter Website URL")

        if st.button("Analyze URL"):

            if "xyz" in url or "free" in url:
                st.error("⚠ Suspicious URL detected")

            elif "https" in url:
                st.success("✅ Secure website")

            else:
                st.warning("⚠ Website might not be secure")

# ---------------- ABOUT ----------------

    elif menu == "About":

        st.markdown("## 📘 About Project")

        st.write("""
        **JobShield AI** is a machine learning based system developed to detect fake job offers.

        Features:
        • Fake Job Message Detection  
        • Email Scam Checker  
        • URL Scam Detection  

        Technology Used:
        - Python
        - Streamlit
        - TF-IDF
        - Naive Bayes
        """)


# ---------------- RUN APP ----------------

if st.session_state.login:
    main_app()
else:
    login_page()
