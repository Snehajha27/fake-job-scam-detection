import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake Job Scam Detector", layout="centered")

# Dark Theme Styling
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: #00FFAA;'>Fake Job & Internship Scam Detection System</h1>", unsafe_allow_html=True)

st.markdown("### JSPM University - TYBCA Final Year Project")
st.markdown("**Team Members:** Sneha Jha | Sujit | Ashutosh")

st.markdown("---")

# Load Dataset
data = pd.read_csv("dataset.csv")
X = data['text']
y = data['label']

# Train Model
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# Calculate Accuracy
predictions = model.predict(X_vectorized)
accuracy = round(accuracy_score(y, predictions) * 100, 2)

st.success(f"Model Accuracy: {accuracy}%")

st.markdown("---")

# Show Dataset Graph
fake_count = sum(y == 1)
genuine_count = sum(y == 0)

fig, ax = plt.subplots()
ax.bar(["Fake", "Genuine"], [fake_count, genuine_count])
st.pyplot(fig)

st.markdown("---")

# Scam Keywords
scam_keywords = ["fee", "registration", "OTP", "urgent", "limited seats", "payment", "click here", "guaranteed"]

def contains_suspicious_url(text):
    urls = re.findall(r'(https?://\S+)', text)
    suspicious = []
    for url in urls:
        if not any(domain in url for domain in ["tcs.com", "infosys.com", "wipro.com"]):
            suspicious.append(url)
    return suspicious

# User Input
st.subheader("Enter Job / Internship Message Below:")
msg = st.text_area("")

if st.button("Analyze Message"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        vect = vectorizer.transform([msg])
        prediction = model.predict(vect)
        probability = model.predict_proba(vect)

        fake_prob = round(probability[0][1] * 100, 2)
        genuine_prob = round(probability[0][0] * 100, 2)

        if prediction[0] == 1:
            result_text = f"Fake Job Offer Detected! (Confidence: {fake_prob}%)"
            st.error(result_text)
        else:
            result_text = f"Genuine Job Offer (Confidence: {genuine_prob}%)"
            st.success(result_text)

        suspicious_urls = contains_suspicious_url(msg)
        if suspicious_urls:
            st.warning("Suspicious URL Detected:")
            for url in suspicious_urls:
                st.write(url)

        detected_keywords = [word for word in scam_keywords if word.lower() in msg.lower()]
        if detected_keywords:
            st.info(f"Scam Keywords Found: {', '.join(detected_keywords)}")

        # Download Report
        st.download_button(
            label="Download Analysis Report",
            data=result_text,
            file_name="analysis_report.txt",
            mime="text/plain"
        )

st.markdown("---")

st.subheader("⚠️ Scam Awareness Tips")
st.markdown("""
- Never pay registration or interview fees.
- Do not share OTP or banking details.
- Verify official company domain.
- Avoid urgent payment requests.
- Cross-check job offers on official websites.
""")

st.markdown("---")
st.caption("Developed using Machine Learning, NLP & Streamlit | TYBCA 2026")
