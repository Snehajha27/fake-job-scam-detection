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

# ---------------- LIGHT UI STYLE ----------------

st.markdown("""
<style>

.stApp{
background-color:#f8fafc;
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#2563eb;
}

.subtitle{
text-align:center;
color:#475569;
}

.card{
background:white;
padding:25px;
border-radius:12px;
box-shadow:0 4px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SESSION ----------------

if "login" not in st.session_state:
    st.session_state.login=False

# ---------------- LOGIN PAGE (CLIENT SIDE) ----------------

def login():

    st.markdown('<p class="title">🛡 JobShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fake Job Detection System</p>', unsafe_allow_html=True)

    col1,col2,col3=st.columns([1,2,1])

    with col2:

        st.markdown('<div class="card">',unsafe_allow_html=True)

        username=st.text_input("Username")
        password=st.text_input("Password",type="password")

        if st.button("Login"):

            if username=="admin" and password=="1234":

                st.session_state.login=True
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Credentials")

        st.markdown('</div>',unsafe_allow_html=True)

# ---------------- SERVER SIDE MODEL ----------------

def load_model():

    data=pd.read_csv("dataset.csv")

    X=data["text"]
    y=data["label"]

    vectorizer=TfidfVectorizer()

    X_vec=vectorizer.fit_transform(X)

    model=MultinomialNB()
    model.fit(X_vec,y)

    return vectorizer,model


# ---------------- MAIN APPLICATION ----------------

def app():

    vectorizer,model=load_model()

    st.sidebar.title("Navigation")

    menu=st.sidebar.radio(
        "Select Module",
        ["Dashboard","Fake Job Detector","Email Checker","URL Checker","System Architecture"]
    )

# ---------------- DASHBOARD ----------------

    if menu=="Dashboard":

        st.markdown("## Project Overview")

        col1,col2,col3=st.columns(3)

        col1.info("🧠 Machine Learning Model")
        col2.info("📧 Email Scam Detection")
        col3.info("🌐 URL Scam Detection")

        st.write("""
This system detects **fake job offers and internship scams** using
Natural Language Processing and Machine Learning.

The application follows a **Client–Server Architecture**.
""")

# ---------------- FAKE JOB DETECTOR ----------------

    elif menu=="Fake Job Detector":

        st.markdown("## Fake Job Message Detection")

        message=st.text_area("Enter Job Message")

        if st.button("Analyze Message"):

            vec=vectorizer.transform([message])

            result=model.predict(vec)[0]

            probability=model.predict_proba(vec).max()*100

            if result=="Fake":

                st.error("⚠ Fake Job Detected")

                st.progress(int(probability))

                st.write(f"Confidence: {probability:.2f}%")

            else:

                st.success("✅ Genuine Job Message")

                st.progress(int(probability))

                st.write(f"Confidence: {probability:.2f}%")

# ---------------- EMAIL CHECKER ----------------

    elif menu=="Email Checker":

        st.markdown("## Email Scam Checker")

        email=st.text_input("Enter Email Address")

        if st.button("Check Email"):

            if "gmail.com" in email or "yahoo.com" in email:

                st.warning("Free email domain detected")

            elif "hr" in email or "job" in email:

                st.info("Job related email detected")

            else:

                st.success("Email looks safe")

# ---------------- URL CHECKER ----------------

    elif menu=="URL Checker":

        st.markdown("## URL Scam Detection")

        url=st.text_input("Enter Website URL")

        if st.button("Analyze URL"):

            if "xyz" in url or "free" in url:

                st.error("Suspicious Website")

            elif "https" in url:

                st.success("Secure Website")

            else:

                st.warning("Website may not be secure")

# ---------------- SYSTEM ARCHITECTURE ----------------

    elif menu=="System Architecture":

        st.markdown("## Client – Server Architecture")

        st.write("""
CLIENT SIDE

• Login Interface  
• User enters job message / email / URL  
• Sends request to server  

SERVER SIDE

• Machine Learning Model  
• Text preprocessing  
• Fake job classification  
• Returns prediction result  

The server processes the request and sends the result back to the client interface.
""")


# ---------------- RUN APP ----------------

if st.session_state.login:
    app()
else:
    login()
