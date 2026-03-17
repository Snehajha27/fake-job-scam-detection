import streamlit as st
import re
from urllib.parse import urlparse

st.set_page_config(page_title="AI Fake Job Detector", layout="centered")

# -----------------------------
# Detection Logic
# -----------------------------
def analyze_text(text):
    score = 0
    reasons = []

    suspicious_keywords = [
        "earn money fast", "no experience", "limited seats",
        "registration fee", "pay now", "guaranteed job",
        "instant joining", "work from home", "whatsapp only"
    ]

    for word in suspicious_keywords:
        if word in text.lower():
            score += 1
            reasons.append(f"⚠️ Suspicious phrase: {word}")

    # Email detection
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    for email in emails:
        if any(x in email for x in ["gmail", "yahoo", "hotmail"]):
            score += 1
            reasons.append(f"⚠️ Personal email used: {email}")

    # URL detection
    urls = re.findall(r"https?://\S+", text)
    for url in urls:
        domain = urlparse(url).netloc
        if not any(x in domain for x in ["linkedin", "naukri", "indeed"]):
            score += 1
            reasons.append(f"⚠️ Unverified URL: {domain}")

    # Payment check
    if "fee" in text.lower() or "payment" in text.lower():
        score += 2
        reasons.append("🚨 Payment request detected")

    # Decision
    if score >= 4:
        result = "FAKE"
    elif score >= 2:
        result = "SUSPICIOUS"
    else:
        result = "REAL"

    return result, score, reasons, emails, urls


# -----------------------------
# UI Design (Chatbot Style)
# -----------------------------
st.title("🤖 AI Fake Job & Scam Detector")

st.markdown("### 💬 Chat with Detector")

user_input = st.text_area("Paste job message / email / URL")

if st.button("Analyze Now 🚀"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result, score, reasons, emails, urls = analyze_text(user_input)

        # Chat UI simulation
        st.markdown("#### 🤖 Bot Response:")

        if result == "FAKE":
            st.error(f"⚠️ High Risk: {result}")
        elif result == "SUSPICIOUS":
            st.warning(f"⚠️ Medium Risk: {result}")
        else:
            st.success(f"✅ Low Risk: {result}")

        # Score meter
        st.progress(min(score / 6, 1.0))

        st.write(f"### 🔢 Risk Score: {score}/6")

        # Reasons
        if reasons:
            st.write("### 📌 Why?")
            for r in reasons:
                st.write("-", r)

        # Extracted Emails
        if emails:
            st.write("### 📧 Emails Found:")
            for e in emails:
                st.code(e)

        # Extracted URLs
        if urls:
            st.write("### 🌐 URLs Found:")
            for u in urls:
                st.code(u)

st.write("---")
st.caption("Final Year Project | AI-based Fake Job Detection System")
