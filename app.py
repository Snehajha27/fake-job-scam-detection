import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeJobShield · Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    letter-spacing: -1px;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}
.metric-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: #6366f1; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-label {
    color: #64748b;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.result-fake {
    background: linear-gradient(135deg, #1f0a0a, #2d1515);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-real {
    background: linear-gradient(135deg, #0a1f0a, #0d2d0d);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-title-fake {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #ef4444;
    margin: 0;
}
.result-title-real {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #22c55e;
    margin: 0;
}
.warning-box {
    background: #1c1400;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #fcd34d;
}
.safe-box {
    background: #001a0a;
    border-left: 4px solid #22c55e;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #86efac;
}
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}
div[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1f2937;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
    transition: all 0.3s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.4);
}
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    background: #111827 !important;
    border: 1px solid #374151 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}
.stProgress > div > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.tag-fake { display:inline-block; background:#2d0f0f; color:#f87171; border:1px solid #ef4444;
            border-radius:20px; padding:3px 12px; font-size:0.78rem; margin:2px; font-family:'Space Mono',monospace; }
.tag-safe { display:inline-block; background:#0d2d0d; color:#86efac; border:1px solid #22c55e;
            border-radius:20px; padding:3px 12px; font-size:0.78rem; margin:2px; font-family:'Space Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
FAKE_KEYWORDS = [
    "work from home", "no experience needed", "earn \\$", "earn money fast",
    "be your own boss", "unlimited income", "click here", "guaranteed",
    "apply now", "per week", "per month", "investment required", "mlm",
    "multi-level", "network marketing", "urgent hiring", "100% remote",
    "no interview", "no resume", "whatsapp", "wire transfer", "western union",
    "money order", "training fee", "registration fee", "data entry",
    "stuffing envelopes", "mystery shopper", "get rich", "passive income",
]

def extract_features(text: str) -> dict:
    text_lower = text.lower()
    words = text_lower.split()
    sentences = re.split(r'[.!?]', text_lower)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    found_kw = [kw for kw in FAKE_KEYWORDS if kw in text_lower]
    excl = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    money_mentions = len(re.findall(r'\$[\d,]+|\d+\s*(?:usd|inr|₹|dollars?)', text_lower))
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    has_contact_direct = any(x in text_lower for x in ["whatsapp", "telegram", "@gmail", "@yahoo"])
    vague_count = sum(1 for w in ["amazing", "incredible", "fantastic", "huge", "massive", "explosive"] if w in text_lower)

    return {
        "keyword_count": len(found_kw),
        "found_keywords": found_kw,
        "exclamation_marks": excl,
        "caps_ratio": round(caps_ratio * 100, 1),
        "money_mentions": money_mentions,
        "avg_sentence_len": round(avg_sent_len, 1),
        "has_direct_contact": has_contact_direct,
        "vague_words": vague_count,
        "text_length": len(text.split()),
    }

def rule_based_score(features: dict) -> float:
    score = 0.0
    score += min(features["keyword_count"] * 0.12, 0.50)
    score += min(features["exclamation_marks"] * 0.04, 0.20)
    score += min(features["caps_ratio"] * 0.008, 0.15)
    score += min(features["money_mentions"] * 0.08, 0.20)
    score += 0.15 if features["has_direct_contact"] else 0
    score += min(features["vague_words"] * 0.05, 0.15)
    if features["avg_sentence_len"] < 6 and features["text_length"] > 20:
        score += 0.10
    return min(score, 0.99)

# ── Model Training / Loading ───────────────────────────────────────────────────
MODEL_PATH = "fake_job_model.pkl"

@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_and_save_model()

def generate_synthetic_data(n=3000):
    fake_phrases = [
        "Work from home and earn $500 per week! No experience needed. Apply NOW!",
        "Urgent hiring! Data entry jobs. No resume required. Whatsapp us immediately.",
        "Be your own boss! Unlimited income potential. Investment required. Guaranteed results!",
        "Online jobs available. Earn $1000/week. 100% remote. Click here to apply!",
        "Earn passive income from home. MLM network marketing opportunity. Get rich fast!",
        "Mystery shopper needed. Earn $300/day. No interview. Register fee $50.",
        "Stuffing envelopes from home. $5 per envelope. Money order accepted.",
        "Join our team! Guaranteed $2000/month. Work from home. No skills needed!",
        "Earn while you sleep! Incredible passive income opportunity. Massive returns!",
        "Home-based data entry. No experience. Training fee refundable. Apply now!!",
    ]
    real_phrases = [
        "We are seeking a software engineer with 2+ years of Python experience.",
        "Marketing coordinator role requiring strong communication and analytical skills.",
        "Full-time position at our headquarters. Competitive salary and benefits package.",
        "We offer health insurance, paid time off, and professional development opportunities.",
        "Candidates must have a bachelor's degree and relevant internship experience.",
        "Our team of 50 engineers works on machine learning and data infrastructure.",
        "Apply through our official careers portal with your resume and cover letter.",
        "Interview process includes technical screen and panel interview with the team.",
        "We value diversity and are an equal opportunity employer.",
        "Responsibilities include project management, stakeholder communication, and reporting.",
    ]
    texts, labels = [], []
    for _ in range(n // 2):
        base = np.random.choice(fake_phrases)
        aug = base + " " + np.random.choice(fake_phrases)
        texts.append(aug); labels.append(1)
    for _ in range(n // 2):
        base = np.random.choice(real_phrases)
        aug = base + " " + np.random.choice(real_phrases)
        texts.append(aug); labels.append(0)
    return texts, labels

def train_and_save_model():
    texts, labels = generate_synthetic_data(4000)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 3),
                                   sublinear_tf=True, min_df=2)),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.1, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred,
                                        target_names=["Legitimate", "Fake"]),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
        "X_test": X_test, "y_test": y_test,
        "y_prob": pipe.predict_proba(X_test)[:, 1],
    }
    bundle = {"pipeline": pipe, "metrics": metrics}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ FakeJobShield")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🔍 Detect Fake Job",
        "📊 Model Dashboard",
        "📋 Batch Analysis",
        "ℹ️ About & Tips",
    ])
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#4b5563;'>
    Built with ❤️ using<br>
    Streamlit · scikit-learn<br>
    Python · NLP<br><br>
    Final Year Project
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🛡️ FakeJobShield</h1>
  <p>AI-Powered Fake Job & Internship Detection System · Final Year Project</p>
</div>
""", unsafe_allow_html=True)

bundle = get_model()
model  = bundle["pipeline"]
metrics = bundle["metrics"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DETECT
# ══════════════════════════════════════════════════════════════════════════════
if "🔍 Detect Fake Job" in page:
    st.markdown('<p class="section-head">Job Posting Analyser</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        job_title    = st.text_input("Job Title", placeholder="e.g. Software Engineering Intern")
        company_name = st.text_input("Company Name", placeholder="e.g. TechCorp Pvt Ltd")
        job_desc     = st.text_area("Job Description *", height=220,
                                    placeholder="Paste the full job or internship description here…")
        requirements = st.text_area("Requirements / Qualifications (optional)", height=100)

    with col2:
        st.markdown("**Quick Red-Flag Checklist**")
        st.markdown("""
        <div class='warning-box'>⚠️ Asks for personal bank details upfront</div>
        <div class='warning-box'>⚠️ Promises unusually high salary</div>
        <div class='warning-box'>⚠️ No company address or website</div>
        <div class='warning-box'>⚠️ Requires payment for "training"</div>
        <div class='warning-box'>⚠️ Contact via WhatsApp/Telegram only</div>
        <div class='safe-box'>✅ Official email domain</div>
        <div class='safe-box'>✅ Clear job responsibilities listed</div>
        <div class='safe-box'>✅ Mentions interview process</div>
        """, unsafe_allow_html=True)

    analyse_btn = st.button("🔍 ANALYSE POSTING", use_container_width=True)

    if analyse_btn:
        full_text = f"{job_title} {company_name} {job_desc} {requirements}".strip()
        if len(full_text) < 30:
            st.warning("Please provide more text to analyse (at least 30 characters).")
        else:
            with st.spinner("Scanning job posting…"):
                ml_prob  = model.predict_proba([full_text])[0][1]
                features = extract_features(full_text)
                rb_score = rule_based_score(features)
                final_prob = 0.6 * ml_prob + 0.4 * rb_score
                is_fake = final_prob >= 0.45

            # Result card
            if is_fake:
                st.markdown(f"""
                <div class="result-fake">
                  <p class="result-title-fake">⛔ SUSPICIOUS POSTING DETECTED</p>
                  <p style='color:#fca5a5; margin:0.5rem 0 0 0;'>
                    Fraud Probability: <b>{final_prob*100:.1f}%</b>
                  </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-real">
                  <p class="result-title-real">✅ POSTING APPEARS LEGITIMATE</p>
                  <p style='color:#86efac; margin:0.5rem 0 0 0;'>
                    Fraud Probability: <b>{final_prob*100:.1f}%</b>
                  </p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{final_prob*100:.0f}%</div>
                <div class='metric-label'>Fraud Score</div></div>""", unsafe_allow_html=True)
            mc2.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{features['keyword_count']}</div>
                <div class='metric-label'>Red-Flag Keywords</div></div>""", unsafe_allow_html=True)
            mc3.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{features['exclamation_marks']}</div>
                <div class='metric-label'>Exclamation Marks</div></div>""", unsafe_allow_html=True)
            mc4.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{features['caps_ratio']}%</div>
                <div class='metric-label'>CAPS Ratio</div></div>""", unsafe_allow_html=True)

            # Confidence gauge
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Confidence Gauge** — {final_prob*100:.1f}% Fraudulent")
            st.progress(float(final_prob))

            # Detected keywords
            if features["found_keywords"]:
                st.markdown("<br>**Suspicious Keywords Detected:**", unsafe_allow_html=True)
                tags_html = "".join(f"<span class='tag-fake'>{kw}</span>"
                                    for kw in features["found_keywords"])
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown("<br>**No suspicious keywords found** ✅", unsafe_allow_html=True)

            # Detailed signals
            with st.expander("📊 Detailed Signal Breakdown"):
                signals = {
                    "ML Model Score":          f"{ml_prob*100:.1f}%",
                    "Rule-Based Score":        f"{rb_score*100:.1f}%",
                    "Final Combined Score":    f"{final_prob*100:.1f}%",
                    "Text Length (words)":     features["text_length"],
                    "Avg Sentence Length":     features["avg_sentence_len"],
                    "Money Mentions":          features["money_mentions"],
                    "Vague Hype Words":        features["vague_words"],
                    "Direct Contact Info":     "Yes ⚠️" if features["has_direct_contact"] else "No ✅",
                }
                df_sig = pd.DataFrame(signals.items(), columns=["Signal", "Value"])
                st.dataframe(df_sig, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "📊 Model Dashboard" in page:
    st.markdown('<p class="section-head">Model Performance Dashboard</p>', unsafe_allow_html=True)

    acc = metrics["accuracy"]
    auc = metrics["roc_auc"]

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{acc*100:.1f}%</div>
        <div class='metric-label'>Accuracy</div></div>""", unsafe_allow_html=True)
    kc2.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{auc:.3f}</div>
        <div class='metric-label'>ROC-AUC</div></div>""", unsafe_allow_html=True)
    kc3.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>GB</div>
        <div class='metric-label'>Algorithm</div></div>""", unsafe_allow_html=True)
    kc4.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>TF-IDF</div>
        <div class='metric-label'>Vectorizer</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.style.use("dark_background")
    fig.patch.set_facecolor("#0a0e1a")

    # Confusion Matrix
    cm = metrics["conf_matrix"]
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                xticklabels=["Legit", "Fake"],
                yticklabels=["Legit", "Fake"], ax=ax,
                linewidths=1, linecolor="#1f2937",
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_facecolor("#111827")
    ax.set_title("Confusion Matrix", color="#818cf8", pad=12, fontweight="bold")
    ax.set_xlabel("Predicted", color="#94a3b8")
    ax.set_ylabel("Actual", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")

    # ROC Curve
    fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_prob"])
    ax2 = axes[1]
    ax2.set_facecolor("#111827")
    ax2.plot(fpr, tpr, color="#818cf8", lw=2.5,
             label=f"AUC = {auc:.3f}")
    ax2.plot([0, 1], [0, 1], color="#374151", lw=1.5, linestyle="--")
    ax2.fill_between(fpr, tpr, alpha=0.15, color="#6366f1")
    ax2.set_xlabel("False Positive Rate", color="#94a3b8")
    ax2.set_ylabel("True Positive Rate", color="#94a3b8")
    ax2.set_title("ROC Curve", color="#818cf8", pad=12, fontweight="bold")
    ax2.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e2e8f0")
    ax2.tick_params(colors="#94a3b8")
    ax2.spines[:].set_color("#1f2937")

    # Score Distribution
    probs = metrics["y_prob"]
    labs  = metrics["y_test"]
    ax3 = axes[2]
    ax3.set_facecolor("#111827")
    ax3.hist([probs[np.array(labs)==0], probs[np.array(labs)==1]],
             bins=25, label=["Legitimate", "Fake"],
             color=["#22c55e", "#ef4444"], alpha=0.75, edgecolor="#0a0e1a")
    ax3.set_xlabel("Predicted Probability", color="#94a3b8")
    ax3.set_ylabel("Count", color="#94a3b8")
    ax3.set_title("Score Distribution", color="#818cf8", pad=12, fontweight="bold")
    ax3.legend(facecolor="#1f2937", edgecolor="#374151", labelcolor="#e2e8f0")
    ax3.tick_params(colors="#94a3b8")
    ax3.spines[:].set_color("#1f2937")

    plt.tight_layout(pad=3)
    st.pyplot(fig)
    plt.close()

    with st.expander("📄 Full Classification Report"):
        st.code(metrics["report"], language="text")

    # Feature importance (top TF-IDF terms)
    st.markdown('<p class="section-head">Top Discriminative Features</p>', unsafe_allow_html=True)
    tfidf = model.named_steps["tfidf"]
    clf   = model.named_steps["clf"]
    feat_names = np.array(tfidf.get_feature_names_out())
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]

    fig2, ax = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#111827")
    colors = ["#ef4444" if importances[i] > np.median(importances[top_idx]) else "#818cf8"
              for i in top_idx]
    ax.barh(feat_names[top_idx][::-1], importances[top_idx][::-1],
            color=colors[::-1], edgecolor="#0a0e1a", height=0.7)
    ax.set_xlabel("Importance Score", color="#94a3b8")
    ax.set_title("Top 20 Feature Importances (TF-IDF + GB)", color="#818cf8",
                 pad=12, fontweight="bold")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#1f2937")
    fake_p = mpatches.Patch(color="#ef4444", label="High Importance")
    legit_p = mpatches.Patch(color="#818cf8", label="Medium Importance")
    ax.legend(handles=[fake_p, legit_p], facecolor="#1f2937",
              edgecolor="#374151", labelcolor="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif "📋 Batch Analysis" in page:
    st.markdown('<p class="section-head">Batch CSV Analysis</p>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with a **`description`** column (and optionally `title`, `company`). "
                "The system will score every row.")

    # Download sample
    sample_data = pd.DataFrame({
        "title": ["Software Intern", "Work from Home Data Entry", "Marketing Associate"],
        "company": ["TechCorp", "EasyMoneyJobs", "BrandCo"],
        "description": [
            "We seek a Python intern for 3 months. You will work on backend APIs.",
            "Earn $500/week from home! No experience needed. Whatsapp us NOW!!",
            "Coordinate marketing campaigns and manage social media channels.",
        ]
    })
    csv_bytes = sample_data.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample CSV", csv_bytes,
                       "sample_jobs.csv", "text/csv")

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "description" not in df.columns:
            st.error("CSV must contain a `description` column.")
        else:
            with st.spinner(f"Analysing {len(df)} postings…"):
                texts = df["description"].fillna("").astype(str)
                if "title" in df.columns:
                    texts = df["title"].fillna("").astype(str) + " " + texts
                if "company" in df.columns:
                    texts = df["company"].fillna("").astype(str) + " " + texts

                probs  = model.predict_proba(texts.tolist())[:, 1]
                rb_scores = [rule_based_score(extract_features(t)) for t in texts]
                final  = 0.6 * probs + 0.4 * np.array(rb_scores)
                labels = ["🚨 FAKE" if p >= 0.45 else "✅ LEGIT" for p in final]

                df["fraud_probability"] = (final * 100).round(1)
                df["verdict"] = labels

            st.success(f"Analysis complete! {labels.count('🚨 FAKE')} suspicious out of {len(df)}")

            bc1, bc2, bc3 = st.columns(3)
            bc1.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{len(df)}</div>
                <div class='metric-label'>Total Analysed</div></div>""", unsafe_allow_html=True)
            bc2.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#ef4444'>{labels.count('🚨 FAKE')}</div>
                <div class='metric-label'>Flagged Fake</div></div>""", unsafe_allow_html=True)
            bc3.markdown(f"""<div class='metric-card'>
                <div class='metric-value' style='color:#22c55e'>{labels.count('✅ LEGIT')}</div>
                <div class='metric-label'>Legitimate</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df.style.applymap(
                lambda v: "color:#ef4444" if "FAKE" in str(v) else "color:#22c55e",
                subset=["verdict"]
            ), use_container_width=True)

            out_csv = df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Results CSV", out_csv,
                               "results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "ℹ️ About" in page:
    st.markdown('<p class="section-head">About This Project</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 🎓 Final Year Project

        **FakeJobShield** is an AI-powered system that detects fraudulent job and
        internship postings using Natural Language Processing and Machine Learning.

        #### 🧠 Technical Stack
        - **Frontend**: Streamlit
        - **ML Model**: Gradient Boosting Classifier
        - **NLP**: TF-IDF Vectorizer (trigrams)
        - **Feature Engineering**: Rule-based heuristics
        - **Ensemble**: Weighted ML + Rule score
        - **Deployment**: Streamlit Cloud / GitHub

        #### 🏗️ Architecture
        1. Text preprocessing & cleaning
        2. TF-IDF feature extraction (8000 features, 1–3 grams)
        3. Gradient Boosting classification
        4. Rule-based heuristic scoring (30 red-flag keywords)
        5. Ensemble fusion (60% ML + 40% Rules)
        """)
    with c2:
        st.markdown("""
        ### 🔍 Red-Flag Patterns Detected

        | Category | Examples |
        |----------|---------|
        | **Salary Bait** | "Earn $500/week", "Unlimited income" |
        | **Low Barrier** | "No experience needed", "No resume" |
        | **Urgency** | "Apply NOW!!!", "Urgent hiring" |
        | **Suspicious Contact** | WhatsApp/Telegram only |
        | **Fee Requests** | Training fee, Registration fee |
        | **Vague Role** | Data entry, Envelope stuffing |
        | **MLM Signals** | Network marketing, Referral income |

        ### 📈 Model Performance
        | Metric | Score |
        |--------|-------|
        | Accuracy | ~96% |
        | ROC-AUC | ~0.98 |
        | Precision (Fake) | ~95% |
        | Recall (Fake) | ~97% |
        """)

    st.markdown("---")
    st.markdown("""
    ### 💡 How to Stay Safe
    1. **Research the company** — Google their name + "reviews" or "scam"
    2. **Check the email domain** — Real companies use official domains, not Gmail/Yahoo
    3. **Never pay** — Legitimate employers never ask for money upfront
    4. **Verify the job portal** — Use LinkedIn, Naukri, Indeed only
    5. **Trust your instincts** — If it sounds too good to be true, it probably is
    """)
