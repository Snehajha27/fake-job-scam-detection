import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("dataset.csv")

X = data['text']
y = data['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

st.title("Fake Job & Internship Scam Detection System")

msg = st.text_area("Enter Job or Internship Message")

if st.button("Check"):
    vect = vectorizer.transform([msg])
    prediction = model.predict(vect)

    if prediction[0] == 1:
        st.error("⚠️ Fake Job Offer Detected!")
    else:
        st.success("✅ Genuine Job Offer")
        
