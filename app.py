import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page config
st.set_page_config(page_title="JobShield AI", page_icon="🛡️", layout="wide")

# ---------- STYLING ----------

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

.title{
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

</style>
""", unsafe_allow_html=True)

# ---------- LOGIN SESSION ----------

if "login" not in st.session_state:
    st.session_state.login = False


# ---------- LOGIN PAGE ----------

def login():

    st.markdown('<p class="title">🛡 JobShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fake Job Detection System</p>', unsafe_allow_html=True)

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


# ---------- LOAD DATA & MODEL ----------

def load_model():

    data = pd.read_csv("dataset.csv")

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return vectorizer, model, data


# ---------- MAIN APPLICATION ----------

def app():

    vectorizer, model, data = load_model()

    st.sidebar.title("Navigation")

    menu = st.sidebar.radio(
        "Go to",
        ["Dashboard","Fake Job Detector","Email Checker","URL Checker","About"]
    )

# ---------- DASHBOARD ----------

    if menu == "Dashboard":

        st.markdown("## 📊 Project Dashboard")

        total=len(data)
        fake=len(data[data.label=="Fake"])
        real=len(data[data.label=="Real"])

        col1,col2,col3=st.columns(3)

        col1.metric("Total Messages",total)
        col2.metric("Fake Jobs",fake)
        col3.metric("Real Jobs",real)

        st.write("### Dataset Distribution")

        chart_data=pd.DataFrame({
            "Type":["Fake","Real"],
            "Count":[fake,real]
        })

        st.bar_chart(chart_data.set_index("Type"))

        st.write("### Dataset Preview")
        st.dataframe(data)

# ---------- FAKE JOB DETECTOR ----------

    elif menu == "Fake Job Detector":

        st.markdown("## 🧠 Fake Job Message Detection")

        message = st.text_area("Paste Job Message")

        if st.button("Analyze Message"):

            vec = vectorizer.transform([message])

            prediction = model.predict(vec)[0]

            probability = model.predict_proba(vec).max()*100

            if prediction=="Fake":

                st.error(f"⚠ This looks like a FAKE job message")

                st.progress(int(probability))

                st.write(f"Confidence: **{probability:.2f}%**")

            else:

                st.success("✅ This job message looks genuine")

                st.progress(int(probability))

                st.write(f"Confidence: **{probability:.2f}%**")

# ---------- EMAIL CHECKER ----------

    elif menu == "Email Checker":

        st.markdown("## 📧 Email Scam Checker")

        email = st.text_input("Enter Email Address")

        if st.button("Check Email"):

            if "gmail.com" in email or "yahoo.com" in email:
                st.warning("⚠ Free email domain detected (may be suspicious)")

            elif "hr" in email or "job" in email:
                st.info("ℹ Job related email")

            else:
                st.success("✅ Email appears safe")

# ---------- URL CHECKER ----------

    elif menu == "URL Checker":

        st.markdown("## 🌐 URL Scam Detection")

        url = st.text_input("Paste Website URL")

        if st.button("Analyze URL"):

            if "xyz" in url or "free" in url:
                st.error("⚠ Suspicious website detected")

            elif "https" in url:
                st.success("✅ Secure website")

            else:
                st.warning("⚠ Website may not be secure")

# ---------- ABOUT ----------

    elif menu == "About":

        st.markdown("## 📘 About Project")

        st.write("""
**JobShield AI** is a Machine Learning based system that detects fake job offers and internship scams.

Features:
• Fake Job Message Detection  
• Email Scam Checker  
• URL Phishing Detection  

Technology Used:
- Python
- Streamlit
- TF-IDF
- Naive Bayes Machine Learning
""")


# ---------- RUN ----------

if st.session_state.login:
    app()
else:
    login()
