import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Fake Job Scam Detector", layout="centered")

# ---------- LOGIN SYSTEM ----------
def login():
    st.title("🔐 Login Panel")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "jspm123":
            st.session_state.logged_in = True
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------- MAIN DASHBOARD ----------
st.title("Fake Job & Internship Scam Detection System")
st.markdown("### JSPM University - TYBCA Final Year Project")
st.markdown("**Team Members:** Sneha Jha | Sujit | Ashutosh")
st.markdown("---")

# ---------- DATASET UPLOAD ----------
st.subheader("📂 Upload Dataset (Admin)")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("dataset.csv")

X = data['text']
y = data['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# ---------- MODEL METRICS ----------
predictions = model.predict(X_vectorized)
accuracy = round(accuracy_score(y, predictions) * 100, 2)

st.success(f"Model Accuracy: {accuracy}%")

cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.markdown("---")

# ---------- MESSAGE ANALYSIS ----------
st.subheader("Analyze Job Message")
msg = st.text_area("Enter Job / Internship Message")

if st.button("Analyze"):
    vect = vectorizer.transform([msg])
    prediction = model.predict(vect)
    prob = model.predict_proba(vect)

    fake_prob = round(prob[0][1] * 100, 2)
    genuine_prob = round(prob[0][0] * 100, 2)

    if prediction[0] == 1:
        st.error(f"⚠️ Fake Job Offer Detected ({fake_prob}%)")
    else:
        st.success(f"✅ Genuine Job Offer ({genuine_prob}%)")

# ---------- SIMPLE CHATBOT ----------
st.markdown("---")
st.subheader("🤖 Job Safety Chatbot")

user_question = st.text_input("Ask about job safety...")

if user_question:
    if "fee" in user_question.lower():
        st.write("Never pay registration fees for jobs.")
    elif "otp" in user_question.lower():
        st.write("Do not share OTP with anyone.")
    elif "domain" in user_question.lower():
        st.write("Verify official company website domain.")
    else:
        st.write("Always verify job offers from official company website.")

st.markdown("---")
st.caption("Developed using ML, NLP & Streamlit | TYBCA 2026")
