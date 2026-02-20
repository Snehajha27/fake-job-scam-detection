import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time

st.set_page_config(
    page_title="JobShield AI",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(-45deg, #141E30, #243B55, #1f4037, #99f2c8);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stButton>button {
    background: linear-gradient(to right, #00C9A7, #92FE9D);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
}
div[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h1 style='text-align:center;'>🔐 JobShield AI Login</h1>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "jspm123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.image("logo.png", width=150)
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Menu", ["Dashboard", "Analyze", "Chatbot"])
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- LOAD DATA ----------------
data = pd.read_csv("dataset.csv")
X = data["text"]
y = data["label"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

accuracy = round(accuracy_score(y, model.predict(X_vectorized)) * 100, 2)

fake_count = sum(y == 1)
genuine_count = sum(y == 0)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Dashboard</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{accuracy}%")
    col2.metric("Fake Messages", fake_count)
    col3.metric("Genuine Messages", genuine_count)

    colA, colB = st.columns(2)

    # Bar Chart
    fig1, ax1 = plt.subplots()
    ax1.bar(["Fake", "Genuine"], [fake_count, genuine_count])
    colA.pyplot(fig1)

    # Pie Chart
    fig2, ax2 = plt.subplots()
    ax2.pie([fake_count, genuine_count], labels=["Fake", "Genuine"], autopct='%1.1f%%')
    colB.pyplot(fig2)

# ---------------- ANALYZE ----------------
elif menu == "Analyze":
    st.markdown("<h1 style='text-align:center;'>🔍 Analyze Job Message</h1>", unsafe_allow_html=True)

    msg = st.text_area("Enter Job / Internship Message")

    if st.button("Analyze"):
        with st.spinner("Analyzing message..."):
            time.sleep(1)

        if msg.strip() == "":
            st.warning("Please enter a message.")
        else:
            vect = vectorizer.transform([msg])
            prediction = model.predict(vect)
            prob = model.predict_proba(vect)

            fake_prob = round(prob[0][1] * 100, 2)
            genuine_prob = round(prob[0][0] * 100, 2)

            if prediction[0] == 1:
                st.error(f"⚠️ Fake Job Detected ({fake_prob}%)")
            else:
                st.success(f"✅ Genuine Job ({genuine_prob}%)")

# ---------------- CHATBOT ----------------
elif menu == "Chatbot":
    st.markdown("<h1 style='text-align:center;'>🤖 AI Safety Assistant</h1>", unsafe_allow_html=True)

    question = st.text_input("Ask about job safety")

    if question:
        q = question.lower()

        if "fee" in q:
            st.info("Never pay registration or interview fees.")
        elif "otp" in q:
            st.info("Do not share OTP with anyone.")
        elif "bank" in q:
            st.info("Never share banking details.")
        else:
            st.info("Always verify job offers from official company websites.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>🚀 JobShield AI | Developed by Sneha Jha | TYBCA 2026</center>", unsafe_allow_html=True)
