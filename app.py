import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Scam Detector", layout="wide")

# -------------------------
# DARK UI STYLE
# -------------------------
st.markdown("""
<style>
body {background-color: #0e1117;}
h1, h2, h3 {color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("data.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
model = MultinomialNB()
model.fit(X, data["label"])

# -------------------------
# SESSION STORAGE
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

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
        if "gmail" in e or "yahoo" in e:
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
# TABS
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 Detector", 
    "📊 Dashboard", 
    "📁 Upload Dataset", 
    "🧾 History"
])

# -------------------------
# TAB 1: DETECTOR
# -------------------------
with tab1:
    st.title("🤖 AI Scam Detection Chatbot")

    user_input = st.text_area("Paste job / email / URL")

    if st.button("Analyze 🚀"):
        if user_input:
            result, prob, reasons = analyze(user_input)

            # Save history
            st.session_state.history.append({
                "text": user_input,
                "result": result,
                "confidence": round(prob*100,2)
            })

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
        st.subheader("Text Length Trend")
        data["length"] = data["text"].apply(len)
        st.line_chart(data["length"])

    st.dataframe(data)

# -------------------------
# TAB 3: UPLOAD DATA
# -------------------------
with tab3:
    st.title("📁 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        new_data = pd.read_csv(file)
        st.write(new_data.head())

        X_new = vectorizer.fit_transform(new_data["text"])
        model.fit(X_new, new_data["label"])

        st.success("Model retrained successfully!")

# -------------------------
# TAB 4: HISTORY
# -------------------------
with tab4:
    st.title("🧾 Detection History")

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

        st.download_button(
            "Download History",
            hist_df.to_csv(index=False),
            file_name="history.csv"
        )
    else:
        st.info("No history yet")

# -------------------------
# FOOTER
# -------------------------
st.write("---")
st.caption("🚀 Final Year Project | AI Fake Job Detection System (Final Boss Level)")
