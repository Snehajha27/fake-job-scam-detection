import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeJobShield – Fake Job/Internship Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2e 50%, #0a1628 100%);
    color: #e0e8f0;
}

.main .block-container { padding-top: 2rem; max-width: 1300px; }

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(99,179,237,0.05));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 20px;
    margin-bottom: 2rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #63b3ed, #00ff88);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: #7aa8c8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(99,179,237,0.04));
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #00d4ff; margin: 0; }
.metric-label { font-size: 0.8rem; color: #7aa8c8; text-transform: uppercase; letter-spacing: 1px; }

/* Result boxes */
.result-real {
    background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,100,0.05));
    border: 2px solid rgba(0,255,136,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.result-fake {
    background: linear-gradient(135deg, rgba(255,60,60,0.1), rgba(200,0,0,0.05));
    border: 2px solid rgba(255,60,60,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
}
.result-title { font-size: 1.6rem; font-weight: 700; margin: 0 0 0.3rem; }
.result-conf  { font-size: 0.95rem; color: #aabccc; }

/* Warning flags */
.flag-item {
    background: rgba(255,180,0,0.08);
    border-left: 3px solid #ffb400;
    border-radius: 6px;
    padding: 0.5rem 0.9rem;
    margin: 0.35rem 0;
    font-size: 0.9rem;
    color: #f0d080;
}

/* Info chips */
.info-chip {
    display: inline-block;
    background: rgba(0,212,255,0.1);
    border: 1px solid rgba(0,212,255,0.25);
    color: #63b3ed;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    margin: 0.15rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1a 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,212,255,0.1);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0099bb);
    color: #000;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-size: 1rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,212,255,0.4); }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 10px !important;
    color: #e0e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1rem; border-bottom: 1px solid rgba(0,212,255,0.15); }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #7aa8c8;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,212,255,0.1) !important;
    color: #00d4ff !important;
    border-color: rgba(0,212,255,0.3) !important;
}

/* Progress bar */
.stProgress > div > div > div { background: linear-gradient(90deg, #00d4ff, #00ff88) !important; border-radius: 4px; }

/* Divider */
hr { border-color: rgba(0,212,255,0.1); }

/* Expander */
.streamlit-expanderHeader { color: #63b3ed !important; font-weight: 600; }

/* Mono */
code, pre { font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "model_bundle.pkl"

SUSPICIOUS_PHRASES = [
    "no experience needed", "work from home", "unlimited earning",
    "guaranteed income", "be your own boss", "easy money",
    "get rich", "financial freedom", "act now", "limited spots",
    "no interview", "immediate start", "wire transfer", "western union",
    "registration fee", "training fee", "upfront payment", "earn per day",
    "earn daily", "no qualification", "part time earn",
    "whatsapp to apply", "click link below", "dm for details",
    "100% genuine", "100% real", "no risk", "zero investment",
    "data entry job", "form filling", "copy paste job",
    "urgent hiring", "hiring immediately", "no skills required",
]

LEGIT_SIGNALS = [
    "bachelor", "master", "degree required", "years of experience",
    "background check", "equal opportunity", "benefits include",
    "health insurance", "401k", "equity", "stock options",
    "interview process", "technical interview", "onboarding",
    "performance review", "career growth", "learning and development",
]


@st.cache_data
def generate_synthetic_dataset(n=3000, seed=42):
    """Generate a realistic synthetic dataset for training."""
    rng = np.random.default_rng(seed)

    fake_templates = [
        "Earn {earn} per day working from home! No experience needed. WhatsApp to apply. Limited spots available. Guaranteed income. Registration fee of {fee}.",
        "Urgent hiring! Data entry job. {earn}/day. No qualification required. Work from home. Immediate start. No interview needed.",
        "Be your own boss! Financial freedom awaits. Earn {earn} daily. Part-time/full-time. 100% genuine. Click the link below to register.",
        "Copy paste job available. Earn {earn} per hour. No skills required. Unlimited earning potential. Act now!",
        "Online form filling job. {earn} per form. No experience required. Easy money. Work from home only. WhatsApp us immediately.",
        "Get rich working online! {earn} per week guaranteed. Zero investment needed. No risk. 100% real opportunity.",
        "Work from home opportunity. Earn {earn} monthly. No experience. Simple tasks. Register now. Limited availability.",
    ]

    legit_templates = [
        "We are seeking a {role} with {exp} years of experience in {skill}. Bachelor's degree required. Competitive salary and benefits including health insurance and 401k. Equal opportunity employer.",
        "Join our team as a {role}. You will be responsible for {task}. {exp}+ years experience required. Technical interview process. Career growth opportunities.",
        "Exciting opportunity for a {role} at our {city} office. Requirements: {exp} years in {skill}, Bachelor's/Master's degree. Benefits: health insurance, equity, learning and development budget.",
        "We are hiring a {role} to {task}. Must have {exp} years experience. Background check required. Comprehensive onboarding provided. Performance review annually.",
        "{role} position open at our {city} headquarters. {exp} years of {skill} experience preferred. Health insurance, stock options, 401k. Technical interview and onboarding included.",
    ]

    roles = ["Software Engineer", "Data Analyst", "Product Manager", "Marketing Associate",
             "Business Analyst", "UX Designer", "DevOps Engineer", "ML Engineer",
             "Content Writer", "Sales Manager", "HR Executive", "Finance Analyst"]
    skills = ["Python", "Java", "SQL", "React", "Machine Learning", "Data Science",
              "Digital Marketing", "Excel", "Tableau", "AWS", "Azure", "Kubernetes"]
    cities = ["Pune", "Bangalore", "Mumbai", "Hyderabad", "Chennai", "Delhi",
              "San Francisco", "New York", "London", "Berlin", "Toronto"]
    tasks = ["develop scalable APIs", "analyze business data", "manage product roadmap",
             "drive marketing campaigns", "build ML models", "design user interfaces",
             "maintain cloud infrastructure", "write technical documentation"]

    records = []
    for _ in range(n):
        is_fake = rng.random() < 0.35  # ~35% fake, realistic imbalance
        if is_fake:
            tmpl = rng.choice(fake_templates)
            text = tmpl.format(
                earn=rng.integers(500, 5000),
                fee=rng.integers(100, 2000)
            )
            telecommuting = int(rng.random() < 0.85)
            has_company = int(rng.random() < 0.2)
            has_logo    = int(rng.random() < 0.15)
            salary_low  = rng.integers(0, 500)
            salary_high = salary_low + rng.integers(100, 500)
            req_len     = rng.integers(0, 80)
            desc_len    = rng.integers(20, 300)
            edu_req     = rng.choice(["", "unspecified", "any"])
            exp_req     = rng.choice(["", "0", "none", "fresher"])
        else:
            tmpl = rng.choice(legit_templates)
            text = tmpl.format(
                role=rng.choice(roles),
                exp=rng.integers(1, 10),
                skill=rng.choice(skills),
                task=rng.choice(tasks),
                city=rng.choice(cities),
            )
            telecommuting = int(rng.random() < 0.25)
            has_company = int(rng.random() < 0.9)
            has_logo    = int(rng.random() < 0.8)
            salary_low  = rng.integers(40000, 150000)
            salary_high = salary_low + rng.integers(10000, 50000)
            req_len     = rng.integers(100, 800)
            desc_len    = rng.integers(400, 2000)
            edu_req     = rng.choice(["Bachelor's Degree", "Master's Degree", "Associate's Degree"])
            exp_req     = str(rng.integers(1, 10)) + " years"

        records.append({
            "text": text,
            "telecommuting": telecommuting,
            "has_company_profile": has_company,
            "has_questions": int(rng.random() < (0.1 if is_fake else 0.7)),
            "has_logo": has_logo,
            "salary_low": salary_low,
            "salary_high": salary_high,
            "requirements_length": req_len,
            "description_length": desc_len,
            "education_requirement": edu_req,
            "experience_requirement": exp_req,
            "fraudulent": int(is_fake),
        })

    return pd.DataFrame(records)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features."""
    df = df.copy()
    text_col = df["text"].fillna("")

    df["suspicious_phrase_count"] = text_col.apply(
        lambda t: sum(p in t.lower() for p in SUSPICIOUS_PHRASES)
    )
    df["legit_signal_count"] = text_col.apply(
        lambda t: sum(p in t.lower() for p in LEGIT_SIGNALS)
    )
    df["exclamation_count"] = text_col.str.count("!")
    df["caps_ratio"] = text_col.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    df["has_email"]   = text_col.str.contains(r"[\w.-]+@[\w.-]+\.\w+", regex=True).astype(int)
    df["has_phone"]   = text_col.str.contains(r"\+?\d[\d\s\-]{8,}", regex=True).astype(int)
    df["has_url"]     = text_col.str.contains(r"https?://|www\.", regex=True).astype(int)
    df["salary_zero"] = ((df["salary_low"] == 0) & (df["salary_high"] == 0)).astype(int)
    df["edu_missing"] = (df["education_requirement"].isin(["", "unspecified", "any"])).astype(int)
    df["exp_missing"] = (df["experience_requirement"].isin(["", "none", "fresher", "0"])).astype(int)
    return df


def build_and_train_model(df: pd.DataFrame):
    """Train ensemble of models and return best + metrics."""
    df = extract_features(df)

    # Numeric features
    num_feats = [
        "telecommuting", "has_company_profile", "has_questions", "has_logo",
        "salary_low", "salary_high", "requirements_length", "description_length",
        "suspicious_phrase_count", "legit_signal_count",
        "exclamation_count", "caps_ratio", "has_email", "has_phone", "has_url",
        "salary_zero", "edu_missing", "exp_missing",
    ]
    X_num = df[num_feats].values
    y     = df["fraudulent"].values

    # Text features via TF-IDF
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_txt = tfidf.fit_transform(df["text"].fillna("")).toarray()

    X = np.hstack([X_num, X_txt])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest":    RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
    }

    results, best_model, best_score = {}, None, 0
    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred)
        rec  = recall_score(y_te, y_pred)
        f1   = f1_score(y_te, y_pred)
        auc  = roc_auc_score(y_te, y_prob)
        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                         "y_pred": y_pred, "y_prob": y_prob}
        if f1 > best_score:
            best_score = f1
            best_model = (name, clf)

    bundle = {
        "model": best_model[1],
        "model_name": best_model[0],
        "tfidf": tfidf,
        "num_feats": num_feats,
        "results": results,
        "y_test": y_te,
        "X_test": X_te,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def predict_single(bundle, input_dict: dict):
    """Run prediction on a single posting dict."""
    df_in = pd.DataFrame([input_dict])
    df_in = extract_features(df_in)

    X_num = df_in[bundle["num_feats"]].values
    X_txt = bundle["tfidf"].transform(df_in["text"].fillna("")).toarray()
    X     = np.hstack([X_num, X_txt])

    prob  = bundle["model"].predict_proba(X)[0]
    label = bundle["model"].predict(X)[0]
    return label, prob


def get_flags(input_dict: dict) -> list[str]:
    """Return human-readable warning flags for a posting."""
    flags = []
    text  = (input_dict.get("text") or "").lower()

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in text:
            flags.append(f'Contains suspicious phrase: "{phrase}"')

    if not input_dict.get("has_company_profile"):
        flags.append("No company profile provided")
    if not input_dict.get("has_logo"):
        flags.append("No company logo")
    if not input_dict.get("has_questions"):
        flags.append("No screening questions – unusual for legitimate postings")
    if input_dict.get("salary_low", 0) == 0 and input_dict.get("salary_high", 0) == 0:
        flags.append("Salary information missing")
    if input_dict.get("education_requirement", "") in ["", "unspecified", "any"]:
        flags.append("Education requirement not specified")
    if input_dict.get("experience_requirement", "") in ["", "none", "fresher", "0"]:
        flags.append("No experience requirement (possible red flag)")
    if text.count("!") >= 3:
        flags.append(f"High use of exclamation marks ({text.count('!')} found)")
    if input_dict.get("telecommuting") and not any(s in text for s in LEGIT_SIGNALS):
        flags.append("Remote-only position with no verifiable company details")
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def render_hero():
    st.markdown("""
    <div class='hero-header'>
        <p class='hero-title'>🛡️ FakeJobShield</p>
        <p class='hero-sub'>AI-Powered Fake Job & Internship Detection System &nbsp;|&nbsp; Final Year Project</p>
        <div style='margin-top:0.8rem'>
            <span class='info-chip'>🎓 Final Year Project</span>
            <span class='info-chip'>🤖 Machine Learning</span>
            <span class='info-chip'>📊 NLP + Feature Engineering</span>
            <span class='info-chip'>⚡ Real-time Detection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar(bundle):
    with st.sidebar:
        st.markdown("## 🛡️ FakeJobShield")
        st.markdown("---")

        if bundle:
            st.markdown("### ✅ Model Status")
            st.success(f"Model loaded: **{bundle['model_name']}**")
            res = bundle["results"][bundle["model_name"]]
            st.markdown(f"""
            <div style='font-size:0.85rem; color:#7aa8c8; line-height:1.9'>
            🎯 Accuracy &nbsp;&nbsp;<b style='color:#00d4ff'>{res['accuracy']*100:.1f}%</b><br>
            📐 Precision &nbsp;<b style='color:#00d4ff'>{res['precision']*100:.1f}%</b><br>
            🔍 Recall &nbsp;&nbsp;&nbsp;&nbsp;<b style='color:#00d4ff'>{res['recall']*100:.1f}%</b><br>
            💯 F1-Score &nbsp;&nbsp;<b style='color:#00d4ff'>{res['f1']*100:.1f}%</b><br>
            📈 AUC-ROC &nbsp;&nbsp;<b style='color:#00d4ff'>{res['auc']*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model not trained yet.\nGo to **Train Model** tab.")

        st.markdown("---")
        st.markdown("""
        ### 📖 How It Works
        1. **Input** a job/internship posting
        2. AI analyses **text + metadata**
        3. Flags **suspicious patterns**
        4. Returns **Fake / Real** verdict
        """)
        st.markdown("---")
        st.markdown("""
        <div style='font-size:0.78rem; color:#4a6a8a; text-align:center'>
        Built with ❤️ using Python, Scikit-learn & Streamlit<br>
        <b>Final Year Project – 2025</b>
        </div>
        """, unsafe_allow_html=True)


def render_train_tab():
    st.markdown("## 🏋️ Train the Detection Model")
    st.info("The system generates a synthetic labelled dataset and trains multiple ML models, selecting the best one.")

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("Training samples", 1000, 10000, 3000, 500)
        fake_ratio = st.slider("Target fake ratio (%)", 20, 50, 35, 5)
    with col2:
        st.markdown("""
        **Models evaluated:**
        - 🌲 Random Forest (200 trees)
        - 📈 Gradient Boosting
        - 📊 Logistic Regression
        """)

    if st.button("🚀 Train Model Now"):
        with st.spinner("Generating dataset and training models…"):
            prog = st.progress(0, text="Generating synthetic dataset…")
            df   = generate_synthetic_dataset(n=n_samples)
            prog.progress(30, text="Extracting features…")
            time.sleep(0.3)
            prog.progress(55, text="Training models…")
            bundle = build_and_train_model(df)
            prog.progress(90, text="Evaluating…")
            time.sleep(0.2)
            prog.progress(100, text="Done!")

        st.success(f"✅ Best model: **{bundle['model_name']}**")

        # Metrics table
        st.markdown("### 📊 Model Comparison")
        rows = []
        for name, r in bundle["results"].items():
            rows.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']*100:.2f}%",
                "Precision": f"{r['precision']*100:.2f}%",
                "Recall":  f"{r['recall']*100:.2f}%",
                "F1-Score": f"{r['f1']*100:.2f}%",
                "AUC-ROC": f"{r['auc']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

        # Confusion matrix for best model
        res = bundle["results"][bundle["model_name"]]
        cm  = confusion_matrix(bundle["y_test"], res["y_pred"])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.patch.set_facecolor("#0a0f1e")

        # -- Confusion matrix
        ax = axes[0]
        ax.set_facecolor("#0d1b2e")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                    ax=ax, linewidths=0.5, linecolor="#1a2a3a",
                    annot_kws={"size": 14, "color": "white"})
        ax.set_title(f"Confusion Matrix – {bundle['model_name']}", color="#00d4ff", pad=10)
        ax.set_xlabel("Predicted", color="#7aa8c8"); ax.set_ylabel("Actual", color="#7aa8c8")
        ax.tick_params(colors="#7aa8c8")

        # -- ROC curve
        ax2 = axes[1]
        ax2.set_facecolor("#0d1b2e")
        colors = ["#00d4ff", "#00ff88", "#ff6b6b"]
        for (name, r), color in zip(bundle["results"].items(), colors):
            fpr, tpr, _ = roc_curve(bundle["y_test"], r["y_prob"])
            ax2.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={r['auc']:.3f})")
        ax2.plot([0, 1], [0, 1], "--", color="#3a4a5a", lw=1)
        ax2.set_title("ROC Curves", color="#00d4ff", pad=10)
        ax2.set_xlabel("False Positive Rate", color="#7aa8c8")
        ax2.set_ylabel("True Positive Rate", color="#7aa8c8")
        ax2.tick_params(colors="#7aa8c8")
        ax2.legend(facecolor="#0d1b2e", edgecolor="#1a3a5a", labelcolor="#b0c8e0", fontsize=9)
        ax2.set_facecolor("#0d1b2e")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#1a3a5a")

        plt.tight_layout()
        st.pyplot(fig)

        st.cache_resource.clear()
        st.rerun()


def render_detect_tab(bundle):
    st.markdown("## 🔍 Detect Fake Job / Internship")

    if not bundle:
        st.error("⚠️ Model not trained. Please go to the **Train Model** tab first.")
        return

    st.markdown("### 📝 Enter Job / Internship Details")

    with st.expander("💡 Quick-fill example postings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Fill Fake Example"):
                st.session_state["ex_title"] = "Work from Home – Earn ₹5000/day!"
                st.session_state["ex_company"] = ""
                st.session_state["ex_desc"] = (
                    "Urgent hiring! Data entry job available. Earn ₹5000 per day working from home. "
                    "No experience needed. No interview. Immediate start. 100% genuine. "
                    "WhatsApp to apply. Limited spots. Registration fee of ₹500 only."
                )
                st.session_state["ex_req"]  = ""
                st.session_state["ex_edu"]  = ""
                st.session_state["ex_exp"]  = ""
                st.session_state["ex_sal_l"] = 0
                st.session_state["ex_sal_h"] = 0
        with c2:
            if st.button("Fill Real Example"):
                st.session_state["ex_title"] = "Software Engineer – Backend (Python)"
                st.session_state["ex_company"] = "TechCorp Solutions Pvt. Ltd."
                st.session_state["ex_desc"] = (
                    "We are hiring a Software Engineer with 2+ years of experience in Python and REST APIs. "
                    "Bachelor's degree in Computer Science or related field required. "
                    "Benefits include health insurance, 401k, and performance bonuses. "
                    "Equal opportunity employer. Technical interview and onboarding provided."
                )
                st.session_state["ex_req"]  = "2+ years Python, REST APIs, SQL, Docker"
                st.session_state["ex_edu"]  = "Bachelor's Degree"
                st.session_state["ex_exp"]  = "2 years"
                st.session_state["ex_sal_l"] = 600000
                st.session_state["ex_sal_h"] = 900000

    col1, col2 = st.columns([3, 2])
    with col1:
        title   = st.text_input("Job Title *", value=st.session_state.get("ex_title", ""), placeholder="e.g. Software Engineer / Data Entry Operator")
        company = st.text_input("Company Name", value=st.session_state.get("ex_company", ""), placeholder="Leave blank if unknown")
        desc    = st.text_area("Job Description *", value=st.session_state.get("ex_desc", ""), height=180, placeholder="Paste the full job description here…")
        reqs    = st.text_area("Requirements / Skills", value=st.session_state.get("ex_req", ""), height=80, placeholder="e.g. 2 years Python, Bachelor's degree…")

    with col2:
        edu_req = st.selectbox("Education Requirement",
                               ["", "Bachelor's Degree", "Master's Degree", "Associate's Degree", "High School", "unspecified", "any"],
                               index=["", "Bachelor's Degree", "Master's Degree", "Associate's Degree", "High School", "unspecified", "any"].index(
                                   st.session_state.get("ex_edu", "")))
        exp_req = st.text_input("Experience Requirement", value=st.session_state.get("ex_exp", ""), placeholder="e.g. 2 years / fresher / none")

        sal_col1, sal_col2 = st.columns(2)
        with sal_col1:
            sal_low  = st.number_input("Salary Low (₹/yr)", min_value=0, value=int(st.session_state.get("ex_sal_l", 0)), step=10000)
        with sal_col2:
            sal_high = st.number_input("Salary High (₹/yr)", min_value=0, value=int(st.session_state.get("ex_sal_h", 0)), step=10000)

        telecommuting     = st.checkbox("Remote / Work-from-Home")
        has_company_profile = st.checkbox("Has Company Profile/Website", value=bool(company))
        has_logo          = st.checkbox("Company Logo Present", value=bool(company))
        has_questions     = st.checkbox("Screening Questions Present")

    st.markdown("")
    if st.button("🔍 Analyse Posting"):
        if not title or not desc:
            st.warning("Please fill in at least the **Job Title** and **Job Description**.")
            return

        full_text = f"{title} {company} {desc} {reqs}".strip()
        input_dict = {
            "text": full_text,
            "telecommuting": int(telecommuting),
            "has_company_profile": int(has_company_profile),
            "has_questions": int(has_questions),
            "has_logo": int(has_logo),
            "salary_low": sal_low,
            "salary_high": sal_high,
            "requirements_length": len(reqs),
            "description_length": len(desc),
            "education_requirement": edu_req,
            "experience_requirement": exp_req,
        }

        with st.spinner("Analysing posting…"):
            time.sleep(0.5)
            label, prob = predict_single(bundle, input_dict)
            flags        = get_flags(input_dict)

        fake_prob = prob[1] * 100
        real_prob = prob[0] * 100

        st.markdown("---")
        st.markdown("## 🎯 Detection Result")

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if label == 1:
                st.markdown(f"""
                <div class='result-fake'>
                    <div class='result-title' style='color:#ff6060'>🚨 FAKE / SUSPICIOUS</div>
                    <div class='result-conf'>Confidence: <b>{fake_prob:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-real'>
                    <div class='result-title' style='color:#00ff88'>✅ LIKELY LEGITIMATE</div>
                    <div class='result-conf'>Confidence: <b>{real_prob:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)

        with rc2:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            fig.patch.set_facecolor("none"); ax.set_facecolor("none")
            colors = ["#00ff88", "#ff6060"]
            wedges, texts, autotexts = ax.pie(
                [real_prob, fake_prob], labels=["Real", "Fake"],
                colors=colors, autopct="%1.1f%%", startangle=90,
                textprops={"color": "#b0c8e0", "fontsize": 10},
                wedgeprops={"linewidth": 2, "edgecolor": "#0a0f1e"},
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontweight("bold")
            ax.set_title("Probability", color="#00d4ff", pad=6, fontsize=11)
            st.pyplot(fig)

        with rc3:
            st.markdown("**📊 Probability Bars**")
            st.markdown(f"🟢 Real: **{real_prob:.1f}%**")
            st.progress(real_prob / 100)
            st.markdown(f"🔴 Fake: **{fake_prob:.1f}%**")
            st.progress(fake_prob / 100)

        # Warning flags
        if flags:
            st.markdown("### ⚠️ Warning Flags Detected")
            for f in flags:
                st.markdown(f"<div class='flag-item'>⚑ {f}</div>", unsafe_allow_html=True)
        else:
            st.success("✅ No suspicious patterns detected in the posting.")

        # Recommendations
        st.markdown("### 💡 Recommendations")
        if label == 1:
            st.error("""
            **This posting shows signs of being fraudulent. We recommend:**
            - ❌ Do NOT pay any registration/training fee
            - 🔎 Verify the company on LinkedIn / official website
            - 📞 Call the company directly to confirm the opening
            - 🚫 Never share sensitive personal/bank details
            - 📢 Report to cybercrime portal if you suspect fraud
            """)
        else:
            st.success("""
            **This posting appears legitimate. Still, we advise:**
            - ✅ Verify the company on LinkedIn / official website
            - ✅ Research salary benchmarks for the role
            - ✅ Read employee reviews on Glassdoor / AmbitionBox
            - ✅ Never pay a fee during the hiring process
            """)


def render_eda_tab(bundle):
    st.markdown("## 📊 Exploratory Data Analysis")

    with st.spinner("Generating EDA charts…"):
        df = generate_synthetic_dataset()
        df = extract_features(df)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0a0f1e")
    fig.suptitle("Dataset Analysis – Fake vs Real Job Postings",
                 color="#00d4ff", fontsize=14, y=1.01)

    palette = {0: "#00ff88", 1: "#ff6060"}

    def style_ax(ax, title):
        ax.set_facecolor("#0d1b2e")
        ax.set_title(title, color="#00d4ff", fontsize=10, pad=8)
        ax.tick_params(colors="#7aa8c8", labelsize=8)
        ax.set_xlabel("", color="#7aa8c8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1a3a5a")

    # 1 – Class distribution
    counts = df["fraudulent"].value_counts()
    axes[0, 0].bar(["Real", "Fake"], [counts[0], counts[1]],
                   color=["#00ff88", "#ff6060"], edgecolor="#0a0f1e", linewidth=1.5)
    style_ax(axes[0, 0], "Class Distribution")
    axes[0, 0].set_ylabel("Count", color="#7aa8c8", fontsize=8)
    for bar, v in zip(axes[0, 0].patches, [counts[0], counts[1]]):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        str(v), ha="center", color="white", fontsize=9)

    # 2 – Suspicious phrase count
    for label_val, color in palette.items():
        subset = df[df["fraudulent"] == label_val]["suspicious_phrase_count"]
        axes[0, 1].hist(subset, bins=12, alpha=0.7, color=color,
                        edgecolor="#0a0f1e", label=["Real", "Fake"][label_val])
    style_ax(axes[0, 1], "Suspicious Phrase Count")
    axes[0, 1].legend(facecolor="#0d1b2e", edgecolor="#1a3a5a", labelcolor="#b0c8e0", fontsize=8)

    # 3 – Has company profile
    data_cp = df.groupby(["has_company_profile", "fraudulent"]).size().unstack().fillna(0)
    x = np.arange(len(data_cp)); w = 0.35
    axes[0, 2].bar(x - w/2, data_cp.get(0, 0), w, color="#00ff88", label="Real", edgecolor="#0a0f1e")
    axes[0, 2].bar(x + w/2, data_cp.get(1, 0), w, color="#ff6060", label="Fake", edgecolor="#0a0f1e")
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(["No Profile", "Has Profile"])
    style_ax(axes[0, 2], "Company Profile vs Fraud")
    axes[0, 2].legend(facecolor="#0d1b2e", edgecolor="#1a3a5a", labelcolor="#b0c8e0", fontsize=8)

    # 4 – Description length
    for label_val, color in palette.items():
        subset = df[df["fraudulent"] == label_val]["description_length"]
        axes[1, 0].hist(subset, bins=20, alpha=0.7, color=color,
                        edgecolor="#0a0f1e", label=["Real", "Fake"][label_val])
    style_ax(axes[1, 0], "Description Length Distribution")
    axes[1, 0].legend(facecolor="#0d1b2e", edgecolor="#1a3a5a", labelcolor="#b0c8e0", fontsize=8)

    # 5 – Correlation heatmap
    num_cols = ["suspicious_phrase_count", "legit_signal_count", "has_company_profile",
                "has_logo", "salary_zero", "telecommuting", "fraudulent"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, ax=axes[1, 1], cmap="coolwarm", center=0, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="#0a0f1e", annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8})
    style_ax(axes[1, 1], "Feature Correlation Heatmap")
    axes[1, 1].tick_params(axis="x", rotation=45)

    # 6 – Telecommuting
    data_tc = df.groupby(["telecommuting", "fraudulent"]).size().unstack().fillna(0)
    x = np.arange(len(data_tc)); w = 0.35
    axes[1, 2].bar(x - w/2, data_tc.get(0, 0), w, color="#00ff88", label="Real", edgecolor="#0a0f1e")
    axes[1, 2].bar(x + w/2, data_tc.get(1, 0), w, color="#ff6060", label="Fake", edgecolor="#0a0f1e")
    axes[1, 2].set_xticks(x); axes[1, 2].set_xticklabels(["On-site", "Remote"])
    style_ax(axes[1, 2], "Remote Work vs Fraud")
    axes[1, 2].legend(facecolor="#0d1b2e", edgecolor="#1a3a5a", labelcolor="#b0c8e0", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 📋 Dataset Statistics")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Samples", len(df)),
        ("Real Postings", int((df["fraudulent"] == 0).sum())),
        ("Fake Postings", int((df["fraudulent"] == 1).sum())),
        ("Fake Rate", f"{df['fraudulent'].mean()*100:.1f}%"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
        col.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{val}</div>
        <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)


def render_about_tab():
    st.markdown("## ℹ️ About This Project")
    st.markdown("""
    ### 🎓 Final Year Project – Fake Job / Internship Detection System

    **FakeJobShield** is an end-to-end machine-learning application that detects fraudulent
    job and internship postings in real time. Built entirely in Python, it combines
    classical NLP with tabular feature engineering to achieve high accuracy.

    ---

    ### 🏗️ System Architecture
    """)
    st.code("""
    User Input (Job Posting)
         │
         ▼
    ┌─────────────────────────────────┐
    │      Feature Extraction         │
    │  • TF-IDF (text, 3k n-grams)   │
    │  • Suspicious phrase count      │
    │  • Legit signal count           │
    │  • Structural metadata          │
    │    (salary, education, etc.)    │
    └────────────┬────────────────────┘
                 │
                 ▼
    ┌─────────────────────────────────┐
    │      Ensemble ML Models         │
    │  • Random Forest (best)         │
    │  • Gradient Boosting            │
    │  • Logistic Regression          │
    └────────────┬────────────────────┘
                 │
                 ▼
    Prediction: FAKE / REAL + Probability + Flags
    """, language="text")

    st.markdown("""
    ---
    ### 📦 Tech Stack

    | Component | Technology |
    |---|---|
    | Frontend / UI | Streamlit |
    | ML Models | Scikit-learn |
    | NLP | TF-IDF Vectorizer |
    | Visualisation | Matplotlib, Seaborn |
    | Deployment | GitHub + Streamlit Cloud |
    | Language | Python 3.10+ |

    ---
    ### 🧠 Features Engineered
    - **Suspicious phrase count** – detects phrases common in scam postings
    - **Legitimate signal count** – detects phrases common in real job descriptions
    - **Exclamation mark density** – scams use aggressive punctuation
    - **CAPS ratio** – all-caps text is a red flag
    - **Salary zero flag** – real jobs almost always disclose salary ranges
    - **Education / Experience missing flags**
    - **Company profile & logo presence**
    - **TF-IDF n-gram features** (1-gram + 2-gram, 3000 features)

    ---
    ### 📚 References
    - Kaggle: Employment Scam Aegean Dataset (EMSCAD)
    - Scikit-learn Documentation
    - NLTK / TF-IDF Feature Extraction
    """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_hero()
    bundle = load_model()
    render_sidebar(bundle)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Detect Posting",
        "🏋️ Train Model",
        "📊 Data Analysis",
        "ℹ️ About",
    ])

    with tab1:
        render_detect_tab(bundle)
    with tab2:
        render_train_tab()
    with tab3:
        render_eda_tab(bundle)
    with tab4:
        render_about_tab()


if __name__ == "__main__":
    main()
