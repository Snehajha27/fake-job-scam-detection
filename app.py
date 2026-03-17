import streamlit as st
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Scam Detector", layout="centered")

# -------------------------
# TRAIN SIMPLE ML MODEL
# -------------------------
data = [
    ("Earn money fast from home without experience", "fake"),
    ("Pay registration fee to get job", "fake"),
    ("Limited seats apply now", "fake"),
    ("Join our company with official email support", "real"),
    ("Software internship with stipend and interview process", "real"),
    ("Apply through company website only", "real"),
]

texts, labels = zip(*data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# -------------------------
# ANALYSIS FUNCTION
# -------------------------
def analyze(text):
    score = 0
    reasons = []

    # ML Prediction
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)[0]
    confidence = max(model.predict_proba(X_input)[0])

    # Rule checks
    if "fee" in text.lower():
        score += 2
        reasons.append("🚨 Payment request detected")

    emails = re.findall(r"\S+@\S+", text)
    for e in emails:
        if "gmail" in e or "yahoo" in e:
            score += 1
            reasons.append(f"⚠️ Personal email: {e}")

    urls = re.findall(r"https?://\S+", text)
    for u in urls:
        domain = urlparse(u).netloc
        if "linkedin" not in domain:
            score += 1
            reasons.append(f"⚠️ Suspicious URL: {domain}")

    # Final decision
    if prediction == "fake" or score >= 3:
        final = "FAKE"
    else:
        final = "REAL"

    return final, confidence, reasons


# -------------------------
# UI
# -------------------------
st.title("🤖 AI Scam Detection Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Enter job message / email / URL")

if st.button("Analyze 🚀"):
    if user_input:
        st.session_state.chat.append(("user", user_input))

        result, confidence, reasons = analyze(user_input)

        bot_msg = f"""
Result: {result}  
Confidence: {round(confidence*100,2)}%
"""
        st.session_state.chat.append(("bot", bot_msg))

        for r in reasons:
            st.session_state.chat.append(("bot", r))

# Display chat
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")

st.write("---")
st.caption("AI-Based Fake Job Detection System | ML + NLP")
