import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="JobShield AI", page_icon="🛡️", layout="wide")

# ---------- STYLING ----------

st.markdown("""
<style>

.stApp {
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

.title {
text-align:center;
font-size:45px;
font-weight:bold;
color:#38bdf8;
}

.subtitle{
text-align:center;
color:#cbd5f5;
}

.card{
background:#1e293b;
padding:25px;
border-radius:15px;
box-shadow:0 4px 15px rgba(0,0,0,0.4);
}

.metric{
font-size:25px;
font-weight:bold;
color:#4ade80;
}

</style>
""", unsafe_allow_html=True)

# ---------- LOGIN SESSION ----------

if "login" not in st.session_state:
    st.session_state.login = False

# ---------- LOGIN PAGE ----------

def login():

    st.markdown('<p class="title">🛡 JobShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fake Job Detection System</p>', unsafe_allow_html=True)

    st.write("")

    col1,col2,col3 = st.columns([1,2,1])

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if username == "admin" and password == "1234":
                st.session_state.login = True
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Credentials")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------

def load_model():

    data = pd.read_csv("dataset.csv")

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer()

    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return vectorizer, model, data

# ---------- MAIN APP ----------

def app():

    vectorizer, model, data = load_model()

    st.sidebar.title("🧭 Navigation")

    menu = st.sidebar.radio(
        "Select Page",
        ["Dashboard","Fake Job Detector","Email Checker","URL Checker","About"]
    )

# ---------- DASHBOARD ----------

    if menu == "Dashboard":

        st.markdown("## 📊 Dashboard")

        total=len(data)
        fake=len(data[data.label=="Fake"])
        real=len(data[data.label=="Real"])

        c1,c2,c3=st.columns(3)

        c1.metric("Total Messages",total)
        c2.metric("Fake Jobs",fake)
        c3.metric("Real Jobs",real)

        st.write("### Dataset Preview")
        st.dataframe(data)

# ---------- JOB DETECTOR ----------

    elif menu == "Fake Job Detector":

        st.markdown("## 🧠 Fake Job Message Detection")

        message = st.text_area("Paste Job Message")

        if st.button("Analyze Message"):

            vec = vectorizer.transform([message])

            result = model.predict(vec)[0]

            if result=="Fake":
                st.error("⚠ This looks like a FAKE job message")
            else:
                st.success("✅ This job message looks genuine")

# ---------- EMAIL CHECK ----------

    elif menu == "Email Checker":

        st.markdown("## 📧 Email Scam Checker")

        email = st.text_input("Enter Email Address")

        if st.button("Check Email"):

            if "gmail.com" in email or "yahoo.com" in email:
                st.warning("⚠ Free email domain detected")

            elif "hr" in email or "job" in email:
                st.info("ℹ Job related email")

            else:
                st.success("✅ Email seems safe")

# ---------- URL CHECK ----------

    elif menu == "URL Checker":

        st.markdown("## 🌐 URL Scam Detection")

        url = st.text_input("Paste URL")

        if st.button("Analyze URL"):

            if "xyz" in url or "free" in url:
                st.error("⚠ Suspicious website")

            elif "https" in url:
                st.success("✅ Secure website")

            else:
                st.warning("⚠ Website may not be secure")

# ---------- ABOUT ----------

    elif menu == "About":

        st.markdown("## 📘 About Project")

        st.write("""
JobShield AI is a Machine Learning based system that detects fake job messages, phishing emails and scam URLs.

Technologies Used
• Python  
• Streamlit  
• TF-IDF  
• Naive Bayes Machine Learning
""")

# ---------- RUN ----------

if st.session_state.login:
    app()
else:
    login()
