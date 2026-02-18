import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Fake Job Scam Detector", layout="centered")

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login Panel")
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

# ---------------- MAIN DASHBOARD ----------------
st.title("Fake Job & Internship Scam Detection System")
st.markdown("### JSPM University - TYBCA Final Year Project")
st.markdown("**Team Members:** Sneha Jha | Sujit | Ashutosh")
st.markdown("---")

# ---------------- LOAD DATA ----------------
try:
    data = pd.read_csv("dataset.csv")
except:
    st.error("dataset.csv not found. Please upload it.")
    st.stop()

# Validate dataset columns
if "text" not in data.columns or "label" not in data.columns:
    st.error("Dataset must contain 'text' and 'label' columns.")
    st.stop()

X = data["text"]
y = data["label"]

# ---------------- TRAIN MODEL ----------------
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# ---------------- MODEL METRICS ----------------
predictions = model.predict(X_vectorized)
accuracy = round(accuracy_score(y, predictions) * 100, 2)

st.success(f"Model Accuracy: {accuracy}%")

st.subheader("Confusion Matrix")

cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots()
ax.imshow(cm, cmap="Blues")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Genuine", "Fake"])
ax.set_yticklabels(["Genuine", "Fake"])

for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

st.pyplot(fig)

st.markdown("---")

# ---------------- MESSAGE ANALYSIS ----------------
st.subheader("Analyze Job / Internship Message")

msg = st.text_area("Enter Message Here")

if st.button("Analyze"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        vect = vectorizer.transform([msg])
        prediction = model.predict(vect)
        prob = model.predict_proba(vect)

        fake_prob = round(prob[0][1] * 100, 2)
        genuine_prob = round(prob[0][0] * 100, 2)

        if pre

