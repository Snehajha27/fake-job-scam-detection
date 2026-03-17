import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="FakeJobShield", layout="wide")

# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= FUNCTIONS =================
FAKE_KEYWORDS = ["work from home","earn money","no experience","urgent hiring",
                 "apply now","investment","fee","whatsapp","telegram"]

def extract_features(text):
    text = text.lower()
    return {
        "keyword_count": sum(1 for kw in FAKE_KEYWORDS if kw in text),
        "exclamation_marks": text.count("!"),
        "caps_ratio": sum(1 for c in text if c.isupper())/max(len(text),1),
        "money_mentions": len(re.findall(r'\$|\₹', text)),
        "has_contact": "whatsapp" in text or "telegram" in text
    }

def rule_score(f):
    score = 0
    score += f["keyword_count"] * 0.1
    score += f["exclamation_marks"] * 0.03
    score += f["caps_ratio"] * 0.1
    score += f["money_mentions"] * 0.1
    score += 0.2 if f["has_contact"] else 0
    return min(score,1)

def analyze_url(text):
    urls = re.findall(r'(https?://\S+|www\.\S+)', text.lower())
    score = 0
    flags = []
    for u in urls:
        if any(x in u for x in ["bit.ly","tinyurl","t.me"]):
            score+=0.3; flags.append("Shortened URL")
        if "login" in u or "verify" in u:
            score+=0.2; flags.append("Phishing URL")
    return score, flags

def analyze_email(text):
    score = 0
    flags = []
    if "@gmail.com" in text.lower():
        score+=0.2; flags.append("Free email domain")
    if "urgent" in text.lower():
        score+=0.2; flags.append("Urgency language")
    if "fee" in text.lower():
        score+=0.3; flags.append("Payment request")
    return score, flags

# ================= MODEL =================
@st.cache_resource
def load_model():
    texts = ["earn money fast","software engineer role","urgent job apply now",
             "python developer job","no experience earn $500"]
    labels = [1,0,1,0,1]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", GradientBoostingClassifier())
    ])
    pipe.fit(texts, labels)
    return pipe

model = load_model()

# ================= SIDEBAR =================
page = st.sidebar.radio("Navigation",[
    "🔍 Detect",
    "📊 Dashboard",
    "📋 Batch",
    "🕒 History",
])

st.sidebar.markdown("### 🤖 AI Assistant")
q = st.sidebar.text_input("Ask something")
if q:
    st.sidebar.write("⚠️ Scam detection uses ML + pattern analysis.")

# ================= PAGE 1 =================
if "Detect" in page:
    st.title("🛡️ Universal Scam Detector")

    text = st.text_area("Paste any text (job/email/message/url)", height=200)

    if st.button("Analyse"):
        if len(text)<20:
            st.warning("Enter more text")
        else:
            ml = model.predict_proba([text])[0][1]
            f = extract_features(text)
            rb = rule_score(f)
            url_s, url_flags = analyze_url(text)
            email_s, email_flags = analyze_email(text)

            final = 0.5*ml + 0.3*rb + 0.1*url_s + 0.1*email_s
            fake = final>0.5

            if fake:
                st.error(f"🚨 FAKE ({final*100:.1f}%)")
            else:
                st.success(f"✅ LEGIT ({final*100:.1f}%)")

            for flag in url_flags+email_flags:
                st.warning(flag)

            st.session_state.history.append({
                "text": text[:50],
                "score": round(final*100,1),
                "result": "Fake" if fake else "Legit"
            })

# ================= DASHBOARD =================
elif "Dashboard" in page:
    st.title("📊 Dashboard")
    st.write("Model running successfully")

# ================= BATCH =================
elif "Batch" in page:
    st.title("📋 Batch Analysis")
    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        probs = model.predict_proba(df["description"])[:,1]
        df["result"] = ["Fake" if p>0.5 else "Legit" for p in probs]
        st.dataframe(df)

# ================= HISTORY =================
elif "History" in page:
    st.title("🕒 History")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history))
    else:
        st.info("No history yet")
