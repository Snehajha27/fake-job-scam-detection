import streamlit as st
import re
import urllib.parse
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from detector import FakeJobDetector

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JobShield – Fake Job Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a0e1a 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1527 0%, #111827 100%);
    border-right: 1px solid #1e3a5f;
}

/* Cards */
.risk-card {
    border-radius: 16px;
    padding: 28px;
    margin: 16px 0;
    border: 1px solid;
    backdrop-filter: blur(10px);
}
.risk-high {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border-color: rgba(239,68,68,0.4);
    box-shadow: 0 0 30px rgba(239,68,68,0.15);
}
.risk-medium {
    background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.08));
    border-color: rgba(245,158,11,0.4);
    box-shadow: 0 0 30px rgba(245,158,11,0.15);
}
.risk-low {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border-color: rgba(16,185,129,0.4);
    box-shadow: 0 0 30px rgba(16,185,129,0.15);
}

/* Flag badges */
.flag-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.flag-critical { background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.5); color: #fca5a5; }
.flag-warning  { background: rgba(245,158,11,0.2); border: 1px solid rgba(245,158,11,0.5); color: #fcd34d; }
.flag-info     { background: rgba(59,130,246,0.2); border: 1px solid rgba(59,130,246,0.5); color: #93c5fd; }

/* Score ring label */
.score-label {
    font-size: 3.2rem;
    font-weight: 700;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.2) !important;
    color: #60a5fa !important;
}

/* Metric cards */
.metric-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: all 0.2s;
    font-family: 'Space Grotesk', sans-serif;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.35);
}

/* Input fields */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Header hero */
.hero-header {
    text-align: center;
    padding: 40px 20px 20px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 8px;
}
.hero-sub {
    color: #64748b;
    font-size: 1.05rem;
    margin-bottom: 0;
}

hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">🛡️ JobShield</div>
  <p class="hero-sub">AI-powered fake job & internship message detector — paste a message, email, or URL</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    sensitivity = st.select_slider(
        "Sensitivity",
        options=["Low", "Medium", "High"],
        value="Medium",
        help="Higher sensitivity catches more scams but may flag legitimate offers"
    )
    check_url = st.toggle("Analyze embedded URLs", value=True)
    check_email = st.toggle("Analyze email patterns", value=True)
    check_urgency = st.toggle("Detect urgency tactics", value=True)
    check_grammar = st.toggle("Grammar & tone analysis", value=True)

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
    **JobShield** analyzes recruitment messages for:
    - 🔗 Suspicious links & domains
    - 💸 Unrealistic salary promises
    - ⚡ Urgency / pressure tactics
    - 📧 Shady email patterns
    - 🏢 Missing company info
    - 🚩 Known scam phrases
    """)
    st.markdown("---")
    st.caption("Final Year Project · Built with Streamlit")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Message", "🔗 Analyze URL / Link", "📖 How It Works"])

detector = FakeJobDetector(
    sensitivity=sensitivity,
    check_url=check_url,
    check_email=check_email,
    check_urgency=check_urgency,
    check_grammar=check_grammar,
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Message Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Paste the recruitment message / email / WhatsApp text below")

    example_msgs = {
        "— Select an example —": "",
        "🚨 Scam: Work-from-home offer": (
            "URGENT!! Congratulations! You have been SELECTED for a Work From Home Job paying $5000/week!! "
            "No experience needed. Just WhatsApp us immediately at +1-555-0199. Limited slots available. "
            "Send your Aadhaar, PAN, and bank details to hr.jobs2024@gmail.com to confirm. Act NOW before offer expires!!"
        ),
        "🚨 Scam: Internship with fee": (
            "Dear Candidate, You are shortlisted for internship at Amazon (remote). Stipend: Rs.50,000/month. "
            "To register please pay Rs.999 registration fee via UPI to jobs_hr@paytm and share screenshot. "
            "Reply within 2 hours or seat will be given to next candidate. Contact: recruiter.amazon2024@gmail.com"
        ),
        "✅ Legit: Campus placement": (
            "Hi, I'm Sarah from Infosys Talent Acquisition. We visited your campus last week and reviewed your profile on LinkedIn. "
            "We'd like to invite you for a virtual interview for our Systems Engineer role. Please apply at careers.infosys.com/apply. "
            "No fee involved. Interview scheduled for next Monday. Let me know if you have questions."
        ),
    }

    chosen = st.selectbox("Try an example:", list(example_msgs.keys()))
    prefill = example_msgs[chosen]

    message_input = st.text_area(
        "Message text",
        value=prefill,
        height=200,
        placeholder="Paste any recruitment message, email body, or SMS here...",
        label_visibility="collapsed"
    )

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        analyze_msg = st.button("🔍 Analyze Message", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            message_input = ""

    if analyze_msg and message_input.strip():
        with st.spinner("Scanning message for red flags..."):
            result = detector.analyze_message(message_input)
        _render_result(result)
    elif analyze_msg:
        st.warning("Please paste a message to analyze.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — URL Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Paste a job application link or company URL")

    url_input = st.text_input(
        "URL",
        placeholder="https://example.com/apply-now or any link from a recruitment message",
        label_visibility="collapsed"
    )

    if st.button("🔗 Analyze URL", use_container_width=True):
        if url_input.strip():
            with st.spinner("Inspecting URL..."):
                result = detector.analyze_url(url_input.strip())
            _render_result(result)
        else:
            st.warning("Please enter a URL.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### How JobShield Detects Fake Jobs")

    features = [
        ("🔗 URL & Domain Analysis", "Checks for URL shorteners, free hosting, typosquatting on real company domains, suspicious TLDs, and mismatched domains."),
        ("💸 Unrealistic Offers", "Flags impossibly high salaries, 'no experience needed' claims, and guaranteed income promises."),
        ("⚡ Urgency Tactics", "Detects pressure phrases like 'act now', 'limited seats', '2-hour deadline' used to rush victims."),
        ("📧 Email Pattern Check", "Legit companies don't recruit from Gmail/Yahoo. Flags personal email domains for corporate claims."),
        ("💰 Fee Requests", "Any request for registration, processing, or training fees is a major scam signal."),
        ("🪪 Data Harvesting", "Flags requests for Aadhaar, PAN, bank account, or passport details upfront."),
        ("🏢 Missing Legitimacy", "Checks for missing company name, office address, or verifiable contact info."),
        ("📝 Grammar & Tone", "Excessive caps, multiple exclamation marks, and poor grammar are common in scam messages."),
    ]

    cols = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px;padding:16px 18px;margin-bottom:12px;">
              <strong style="color:#60a5fa;">{title}</strong>
              <p style="color:#94a3b8;font-size:0.88rem;margin-top:6px;margin-bottom:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Shared render function — defined BEFORE tabs so both can use it
# ════════════════════════════════════════════════════════════════════════════
def _render_result(result: dict):
    score = result["risk_score"]
    level = result["risk_level"]
    flags = result["flags"]

    color_map = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
    card_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}
    emoji_map = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
    verdict = {
        "HIGH":   "Very likely a SCAM — do not respond or share personal info.",
        "MEDIUM": "Suspicious — verify independently before proceeding.",
        "LOW":    "Appears legitimate — standard caution advised.",
    }

    st.markdown("---")
    st.markdown("### 📋 Detection Report")

    # ── Score gauge ──
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 40, "color": color_map[level]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color_map[level], "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40],  "color": "rgba(16,185,129,0.15)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                {"range": [70, 100],"color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {
                "line": {"color": color_map[level], "width": 4},
                "thickness": 0.75,
                "value": score
            }
        },
        title={"text": f"Risk Score  {emoji_map[level]} {level}", "font": {"size": 18, "color": "#e2e8f0"}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=280,
        margin=dict(t=40, b=10, l=30, r=30),
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"""
        <div class="risk-card {card_class[level]}" style="margin-top:20px;">
          <h3 style="margin-top:0;color:{color_map[level]};">{emoji_map[level]} {level} RISK</h3>
          <p style="font-size:1rem;color:#e2e8f0;">{verdict[level]}</p>
          <hr style="border-color:rgba(255,255,255,0.1);"/>
          <p style="font-size:0.85rem;color:#94a3b8;margin-bottom:0;">
            <strong style="color:#e2e8f0;">{len(flags)}</strong> red flag(s) detected
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Flags ──
    if flags:
        st.markdown("#### 🚩 Detected Red Flags")
        flag_html = ""
        for f in flags:
            cls = "flag-critical" if f["severity"] == "critical" else ("flag-warning" if f["severity"] == "warning" else "flag-info")
            icon = "🔴" if f["severity"] == "critical" else ("🟡" if f["severity"] == "warning" else "🔵")
            flag_html += f'<span class="flag-badge {cls}">{icon} {f["label"]}</span>'
        st.markdown(flag_html, unsafe_allow_html=True)

        st.markdown("#### 📝 Details")
        for f in flags:
            color = "#fca5a5" if f["severity"] == "critical" else ("#fcd34d" if f["severity"] == "warning" else "#93c5fd")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border-left:3px solid {color};
                        padding:10px 16px;border-radius:0 8px 8px 0;margin:6px 0;">
              <strong style="color:{color};">{f['label']}</strong>
              <p style="color:#94a3b8;margin:4px 0 0;font-size:0.88rem;">{f['detail']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Category breakdown bar chart ──
    if result.get("category_scores"):
        st.markdown("#### 📊 Category Breakdown")
        cats = result["category_scores"]
        fig2 = go.Figure(go.Bar(
            x=list(cats.values()),
            y=list(cats.keys()),
            orientation="h",
            marker=dict(
                color=list(cats.values()),
                colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                cmin=0, cmax=100,
                line=dict(width=0)
            ),
            text=[f"{v}%" for v in cats.values()],
            textposition="outside",
            textfont=dict(color="#e2e8f0"),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            xaxis=dict(range=[0, 110], showgrid=False, zeroline=False, color="#475569"),
            yaxis=dict(showgrid=False, color="#e2e8f0"),
            height=280,
            margin=dict(t=10, b=10, l=10, r=60),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Safety tips ──
    if level == "HIGH":
        st.error("🛑 **Safety Advice:** Do NOT share personal documents, bank details, or pay any fee. Report this to cybercrime.gov.in or your institution's placement cell.")
    elif level == "MEDIUM":
        st.warning("⚠️ **Safety Advice:** Verify the company on LinkedIn and their official website. Call the HR directly from the number on the official website — not from the message.")
    else:
        st.success("✅ **Looks legitimate.** Always good practice to verify the company website independently before sharing documents.")


# Patch the tabs — Python functions defined after tabs work fine; just make sure
# _render_result is accessible at call time (it is, since calls happen on button click).
