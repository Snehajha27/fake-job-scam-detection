import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Scam Detector", layout="wide")

# -------------------------
# LOAD DATASET
# -------------------------
data = pd.read_csv("data.csv")

# Train ML model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
model = MultinomialNB()
model.fit(X, data["label"])

# -------------------------
# ANALYSIS FUNCTION
# -------------------------
def analyze(text):
    score = 0
    reasons = []

    X_input = vectorizer.transform([text])
    pred = model.predict(X_input)[0]
    prob = max(model.predict_proba(X_input)[0])

    if "fee" in text.lower():
        score += 2
        reasons.append("🚨 Payment request")

    emails = re.findall(r"\S+@\S+", text)
    for e in emails:
        if "gmail" in e:
            score += 1
            reasons.append(f"⚠️ Personal email: {e}")

    urls = re.findall(r"https?://\S+", text)
    for u in urls:
        domain = urlparse(u).netloc
        if "linkedin" not in domain:
            score += 1
            reasons.append(f"⚠️ Suspicious URL: {domain}")

    if pred == "fake" or score >= 3:
        final = "FAKE"
    else:
        final = "REAL"

    return final, prob, reasons

# -------------------------
# TABS UI
# -------------------------
tab1, tab2, tab3 = st.tabs(["🤖 Detector", "📊 Dashboard", "📁 Upload Dataset"])

# -------------------------
# TAB 1: CHATBOT DETECTOR
# -------------------------
with tab1:
    st.title("🤖 AI Scam Detection Chatbot")

    user_input = st.text_input("Enter job message")

    if st.button("Analyze"):
        if user_input:
            result, prob, reasons = analyze(user_input)

            st.subheader(f"Result: {result}")
            st.write(f"Confidence: {round(prob*100,2)}%")

            st.progress(prob)

            if reasons:
                st.write("### Reasons:")
                for r in reasons:
                    st.write("-", r)

# -------------------------
# TAB 2: DASHBOARD
# -------------------------
with tab2:
    st.title("📊 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Label Distribution")
        st.bar_chart(data["label"].value_counts())

    with col2:
        st.subheader("Text Length Analysis")
        data["length"] = data["text"].apply(len)
        st.line_chart(data["length"])

    st.subheader("Dataset Preview")
    st.dataframe(data)

# -------------------------
# TAB 3: UPLOAD DATASET
# -------------------------
with tab3:
    st.title("📁 Upload Custom Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write("Preview:", new_data.head())

        X_new = vectorizer.fit_transform(new_data["text"])
        model.fit(X_new, new_data["label"])

        st.success("Model retrained successfully!")

st.write("---")
st.caption("Final Year Project | AI Fake Job Detection System 🚀")
