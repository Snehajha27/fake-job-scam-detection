import streamlit as st

st.set_page_config(page_title="Fake Job Detector")

st.title("🕵️ Fake Job / Internship Detector")

job = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if job == "":
        st.warning("Please enter job description")
    else:
        score = 0

        # Simple rules
        if "pay" in job.lower():
            score += 1
        if "fee" in job.lower():
            score += 2
        if "earn money fast" in job.lower():
            score += 2
        if "no experience" in job.lower():
            score += 1
        if "whatsapp" in job.lower():
            score += 1

        if score >= 3:
            st.error("⚠️ Fake Job Detected")
        else:
            st.success("✅ Looks Real")

st.write("---")
st.caption("Final Year Project - Fake Job Detection System")
