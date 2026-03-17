import streamlit as st
import re
import time
from urllib.parse import urlparse

st.set_page_config(page_title="AI Scam Detector", layout="centered")

# -------------------------
# Custom Styling (Dark UI)
# -------------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
.chat-box {
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.user {background-color: #1f2937; text-align: right;}
.bot {background-color: #111827;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Detection Logic
# -------------------------
def analyze(text):
    score = 0
    reasons = []

    keywords = [
        "earn money fast", "no experience", "limited seats",
        "registration fee", "pay now", "guaranteed job",
        "instant joining", "whatsapp only"
    ]

    for k in keywords:
        if k in text.lower():
            score += 1
            reasons.append(f"⚠️ Suspicious phrase: {k}")

    # Emails
    emails = re.findall(r"\S+@\S+", text)
    for e in emails:
        if any(x in e for x in ["gmail", "yahoo"]):
            score += 1
            reasons.append(f"⚠️ Personal email: {e}")

    # URLs
    urls = re.findall(r"https?://\S+", text)
    for u in urls:
        domain = urlparse(u).netloc
        if not any(x in domain for x in ["linkedin", "indeed", "naukri"]):
            score += 1
            reasons.append(f"⚠️ Unknown site: {domain}")

    if "fee" in text.lower():
        score += 2
        reasons.append("🚨 Payment requested")

    # Decision
    if score >= 4:
        return "FAKE", score, reasons
    elif score >= 2:
        return "SUSPICIOUS", score, reasons
    else:
        return "REAL", score, reasons


# -------------------------
# Chat System
# -------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

st.title("🤖 AI Fake Job & Scam Detector")

user_input = st.text_input("Type or paste job message...")

if st.button("Send 🚀"):
    if user_input:
        st.session_state.chat.append(("user", user_input))

        result, score, reasons = analyze(user_input)

        # Simulate typing
        bot_reply = f"Analyzing...\n\nResult: {result}\nRisk Score: {score}/6"

        st.session_state.chat.append(("bot", bot_reply))

        for r in reasons:
            st.session_state.chat.append(("bot", r))

# -------------------------
# Display Chat
# -------------------------
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f'<div class="chat-box user">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-box bot">{msg}</div>', unsafe_allow_html=True)

st.write("---")
st.caption("AI Powered Fake Job Detection System 🚀")
