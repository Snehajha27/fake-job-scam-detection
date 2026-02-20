import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Fake Job Scam Detector", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}
.stButton>button {
    background-color: #00C9A7;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.stTextInput>div>div>input, .stTextArea textarea {
    border-radius: 10px;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h1 style='text-align:center;'>🔐 Admin Login</h1>", unsafe_allow_html=True)
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
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dashboard", "Analyze Message", "Chatbot"])
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

predictions = model.predict(X_vectorized)
accuracy = round(accuracy_score(y, predictions) * 100, 2)

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.markdown("<h1 style='text-align:center;'>📊 Project Dashboard</h1>", unsafe_allow_html=True)
    st.success(f"Model Accuracy: {accuracy}%")

    fake_count = sum(y == 1)
    genuine_count = sum(y == 0)

    col1, col2 = st.columns(2)
    col1.metric("Fake Messages", fake_count)
    col2.metric("Genuine Messages", genuine_count)

    fig, ax = plt.subplots()
    ax.bar(["Fake", "Genuine"], [fake_count, genuine_count])
    ax.set_facecolor("#243B55")
    fig.patch.set_facecolor("#243B55")
    st.pyplot(fig)

# ---------------- ANALYZE ----------------
elif menu == "Analyze Message":
    st.markdown("<h1 style='text-align:center;'>🔍 Analyze Job Message</h1>", unsafe_allow_html=True)

    msg = st.text_area("Enter Job / Internship Message")

    if st.button("Analyze"):
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
    st.markdown("<h1 style='text-align:center;'>🤖 Job Safety Chatbot</h1>", unsafe_allow_html=True)

    question = st.text_input("Ask something about job safety")

    if question:
        q = question.lower()

        if "fee" in q:
            st.info("Never pay registration or interview fees.")
        elif "otp" in q:
            st.info("Do not share OTP with anyone.")
        elif "bank" in q:
            st.info("Never share banking details.")
        elif "domain" in q:
            st.info("Verify company domain carefully.")
        else:
            st.info("Always verify job offers from official websites.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center>Developed by Sneha Jha | Sujit | Ashutosh | TYBCA 2026</center>", unsafe_allow_html=True)
