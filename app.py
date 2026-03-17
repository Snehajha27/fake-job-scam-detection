import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="JobShield AI", layout="wide")

# ---------- UI STYLE (LIGHT PROFESSIONAL) ----------
st.markdown("""
<style>
.stApp {background-color:#f8fafc;}
.title {text-align:center;font-size:40px;color:#2563eb;font-weight:bold;}
.card {
background:white;
padding:20px;
border-radius:10px;
box-shadow:0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL (SERVER SIDE) ----------
@st.cache_resource
def load_model():
    data = pd.read_csv("dataset.csv")

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["text"])

    model = MultinomialNB()
    model.fit(X, data["label"])

    return vectorizer, model, data

vectorizer, model, data = load_model()

# ---------- HEADER ----------
st.markdown('<p class="title">🛡 JobShield AI - Fake Job Detection</p>', unsafe_allow_html=True)

# ---------- SWITCH BETWEEN CLIENT & SERVER ----------
mode = st.radio("Select View", ["Client Side (Frontend)", "Server Side (Backend)", "Dashboard"])

# ======================================================
# 🧑‍💻 CLIENT SIDE
# ======================================================

if mode == "Client Side (Frontend)":

    st.markdown("## Client Side Interface (User View)")

    option = st.selectbox("Choose Service", [
        "Fake Job Detection",
        "Email Checker",
        "URL Checker"
    ])

    # ---- JOB DETECTOR ----
    if option == "Fake Job Detection":

        msg = st.text_area("Enter Job Message")

        if st.button("Send Request to Server"):

            vec = vectorizer.transform([msg])
            result = model.predict(vec)[0]
            prob = model.predict_proba(vec).max()*100

            if result == "Fake":
                st.error(f"⚠ Fake Job Detected ({prob:.2f}%)")
            else:
                st.success(f"✅ Genuine Job ({prob:.2f}%)")

    # ---- EMAIL CHECK ----
    elif option == "Email Checker":

        email = st.text_input("Enter Email")

        if st.button("Check Email"):

            if "gmail" in email or "yahoo" in email:
                st.warning("Free email → Suspicious")

            elif "hr" in email:
                st.info("Job related email")

            else:
                st.success("Safe email")

    # ---- URL CHECK ----
    elif option == "URL Checker":

        url = st.text_input("Enter URL")

        if st.button("Check URL"):

            if "xyz" in url or "free" in url:
                st.error("Suspicious URL")

            elif "https" in url:
                st.success("Secure URL")

            else:
                st.warning("Not secure")

# ======================================================
# ⚙️ SERVER SIDE
# ======================================================

elif mode == "Server Side (Backend)":

    st.markdown("## Server Side Processing")

    st.write("This section shows how backend works internally.")

    st.markdown("### Dataset Stored on Server")
    st.dataframe(data)

    st.markdown("### ML Pipeline")

    st.code("""
1. Receive request from client
2. Preprocess text
3. Convert to TF-IDF vectors
4. Apply Naive Bayes model
5. Generate prediction
6. Send result to client
""")

    st.success("Server is ready to process requests")

# ======================================================
# 📊 DASHBOARD
# ======================================================

else:

    st.markdown("## System Dashboard")

    fake = len(data[data.label=="Fake"])
    real = len(data[data.label=="Real"])

    col1, col2 = st.columns(2)
    col1.metric("Fake Jobs", fake)
    col2.metric("Real Jobs", real)

    chart = pd.DataFrame({
        "Type": ["Fake", "Real"],
        "Count": [fake, real]
    })

    st.bar_chart(chart.set_index("Type"))

    st.markdown("### System Flow")

    st.write("""
Client → Sends job message  
Server → Processes using ML  
Server → Sends result  
Client → Displays output  
""")
