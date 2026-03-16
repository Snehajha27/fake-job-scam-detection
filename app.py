import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="JobShield AI", page_icon="🛡️", layout="wide")

# ---------------- LIGHT UI ----------------

st.markdown("""
<style>

.stApp{
background-color:#f1f5f9;
}

.title{
text-align:center;
font-size:42px;
font-weight:bold;
color:#2563eb;
}

.subtitle{
text-align:center;
color:#475569;
}

.card{
background:white;
padding:25px;
border-radius:10px;
box-shadow:0 4px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SESSION ----------------

if "login" not in st.session_state:
    st.session_state.login = False


# ---------------- LOGIN PAGE ----------------

def login():

    st.markdown('<p class="title">🛡 JobShield AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fake Job Detection System</p>', unsafe_allow_html=True)

    col1,col2,col3 = st.columns([1,2,1])

    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):

            if username == "admin" and password == "1234":
                st.session_state.login = True
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid Credentials")

        st.markdown('</div>', unsafe_allow_html=True)


# ---------------- SERVER SIDE MODEL ----------------

def load_model():

    data = pd.read_csv("dataset.csv")

    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer()

    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()

    model.fit(X_vec, y)

    return vectorizer, model, data


# ---------------- MAIN APP ----------------

def app():

    vectorizer, model, data = load_model()

    st.sidebar.title("Navigation")

    menu = st.sidebar.radio(
        "Select Page",
        ["Dashboard","Client Side","Server Side"]
    )

# ---------------- DASHBOARD ----------------

    if menu == "Dashboard":

        st.title("System Dashboard")

        st.write("This system follows **Client – Server Architecture**")

        col1,col2 = st.columns(2)

        with col1:
            st.markdown("### Client Side Responsibilities")
            st.write("""
• User Login  
• User Interface  
• Message Input  
• Email Input  
• URL Input  
• Sending request to server
""")

        with col2:
            st.markdown("### Server Side Responsibilities")
            st.write("""
• Dataset storage  
• Text preprocessing  
• Feature extraction (TF-IDF)  
• Machine Learning model  
• Fake job classification  
• Sending result back to client
""")

# ---------------- CLIENT SIDE ----------------

    elif menu == "Client Side":

        st.title("Client Side Interface")

        option = st.selectbox(
            "Choose Service",
            ["Fake Job Detection","Email Checker","URL Checker"]
        )

        if option == "Fake Job Detection":

            message = st.text_area("Enter Job Message")

            if st.button("Send to Server"):

                vec = vectorizer.transform([message])

                prediction = model.predict(vec)[0]

                probability = model.predict_proba(vec).max()*100

                if prediction == "Fake":
                    st.error("⚠ Fake Job Detected")
                    st.write(f"Confidence: {probability:.2f}%")

                else:
                    st.success("✅ Genuine Job Message")
                    st.write(f"Confidence: {probability:.2f}%")

        elif option == "Email Checker":

            email = st.text_input("Enter Email Address")

            if st.button("Send to Server"):

                if "gmail.com" in email or "yahoo.com" in email:
                    st.warning("Free email domain detected")

                elif "hr" in email or "job" in email:
                    st.info("Job related email")

                else:
                    st.success("Email seems safe")

        elif option == "URL Checker":

            url = st.text_input("Enter Website URL")

            if st.button("Send to Server"):

                if "xyz" in url or "free" in url:
                    st.error("Suspicious website")

                elif "https" in url:
                    st.success("Secure website")

                else:
                    st.warning("Website may not be secure")

# ---------------- SERVER SIDE ----------------

    elif menu == "Server Side":

        st.title("Server Side Processing")

        st.write("Server handles data processing and machine learning prediction.")

        st.markdown("### Dataset Stored on Server")
        st.dataframe(data)

        st.markdown("### Machine Learning Pipeline")

        st.write("""
1️⃣ Text Cleaning  
2️⃣ Feature Extraction (TF-IDF)  
3️⃣ Model Training (Naive Bayes)  
4️⃣ Prediction Generation  
5️⃣ Response sent back to Client
""")

        st.success("Server ready to process requests from client")


# ---------------- RUN APP ----------------

if st.session_state.login:
    app()
else:
    login()
