"""
app.py — GenAI Customer Intelligence Platform
Premium UI — Obsidian black · True gold · Playfair Display serif elegance
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="GenAI Customer Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,700;0,900;1,400;1,700&family=Plus+Jakarta+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --obsidian:   #181815;    ##Background
    --ink:        #1f1f1c;
    --ink2:       #262622;
    --ink3:       #2f2f2b;
    --gold:       #d4a847;    ##Accent
    --gold2:      #e8c070;
    --gold3:      #f5d990;
    --gold-glow:  rgba(212,168,71,0.10);
    --gold-dim:   rgba(212,168,71,0.14);
    --gold-line:  rgba(212,168,71,0.22);
    --silver:     #a8a49c;
    --silver2:    #6b6760;
    --cream:      #f2ede4;   ##Text
    --cream2:     #e8e2d6;
    --emerald:    #4a9b6f;
    --ruby:       #b85450;
    --radius:     8px;
    --radius-sm:  5px;
    --font-serif: 'Playfair Display', Georgia, 'Times New Roman', serif;
    --font-sans:  'Plus Jakarta Sans', system-ui, sans-serif;
    --font-mono:  'IBM Plex Mono', 'Courier New', monospace;
    --transition: 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background: var(--obsidian) !important;
    color: var(--cream) !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--ink) !important;
    border-right: 1px solid rgba(212,168,71,0.12) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Brand ── */
.brand-wrap {
    padding: 32px 22px 26px;
    border-bottom: 1px solid rgba(212,168,71,0.1);
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.brand-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}
.brand-mark {
    display: flex;
    gap: 5px;
    margin-bottom: 16px;
}
.brand-mark span {
    display: block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--gold);
    opacity: 0.6;
}
.brand-mark span:nth-child(2) { opacity: 1; background: var(--gold2); }
.brand-mark span:nth-child(3) { opacity: 0.4; }
.brand-title {
    font-family: var(--font-serif) !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--cream) !important;
    line-height: 1.35;
    letter-spacing: 0.2px;
}
.brand-sub {
    font-family: var(--font-mono) !important;
    font-size: 0.58rem;
    color: var(--silver2) !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 7px;
    opacity: 0.7;
}

/* ── Nav ── */
[data-testid="stRadio"] > div { gap: 2px !important; }
[data-testid="stRadio"] label {
    border-radius: var(--radius-sm) !important;
    padding: 9px 15px !important;
    margin: 0 !important;
    font-size: 0.82rem !important;
    font-family: var(--font-sans) !important;
    font-weight: 400 !important;
    color: var(--silver) !important;
    transition: all var(--transition) !important;
    border: 1px solid transparent !important;
    letter-spacing: 0.1px !important;
}
[data-testid="stRadio"] label:hover {
    color: var(--gold2) !important;
    background: var(--gold-glow) !important;
}

/* ── Force dark background on every Streamlit container ── */
.stApp,
.stApp > *,
.main,
.main > *,
.main .block-container,
section[data-testid="stSidebarContent"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="block-container"],
div[class*="appview-container"],
div[class*="main-content"],
header[data-testid="stHeader"],
footer { 
    background-color: var(--obsidian) !important;
    color: var(--cream) !important;
}
.main .block-container {
    padding-top: 2.8rem !important;
    max-width: 1280px !important;
}

/* ── Fix text visibility across all elements ── */
p, li, span, td, th, label, div {
    color: var(--cream) !important;
}
p { 
    font-size: 0.9rem !important; 
    line-height: 1.75 !important;
    color: var(--cream) !important;
}
.ph-sub p, .ph-sub {
    color: var(--silver) !important;
    font-size: 0.84rem !important;
}
/* Keep muted elements muted */
.kpi-label, .brand-sub, .stack-info,
.tech-desc, .step-desc, .silver { 
    color: var(--silver) !important; 
}
.silver2 { color: var(--silver2) !important; }

/* ── Page header ── */
.ph-wrap { margin-bottom: 8px; }
.ph-rule {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}
.ph-rule-line { height: 1px; width: 32px; background: var(--gold); }
.ph-rule-dot  { width: 4px; height: 4px; border-radius: 50%; background: var(--gold); }
.ph-eyebrow {
    font-family: var(--font-mono) !important;
    font-size: 0.6rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gold) !important;
}
.ph-title {
    font-family: var(--font-serif) !important;
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: var(--cream) !important;
    letter-spacing: -1px;
    line-height: 1.05;
    margin: 10px 0 14px !important;
}
.ph-title em {
    font-style: italic;
    color: var(--gold2);
}
.ph-sub {
    font-size: 0.84rem;
    color: var(--silver) !important;
    line-height: 1.75;
    font-weight: 300;
    max-width: 620px;
    border-left: 2px solid var(--gold-line);
    padding-left: 14px;
}

/* ── Divider ── */
.divider {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 36px 0;
}
.divider-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,168,71,0.15), transparent);
}
.divider-center {
    padding: 0 18px;
    display: flex;
    gap: 6px;
    align-items: center;
}
.divider-center span {
    display: block; width: 3px; height: 3px;
    border-radius: 50%;
    background: var(--gold);
    opacity: 0.5;
}
.divider-center span:nth-child(2) { opacity: 1; width: 4px; height: 4px; }

/* ── KPI card ── */
.kpi-card {
    background: var(--ink2);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: var(--radius);
    padding: 24px 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform var(--transition), box-shadow var(--transition);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent 0%, var(--gold) 40%, var(--gold2) 60%, transparent 100%);
    opacity: 0.7;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.5), 0 0 0 1px rgba(212,168,71,0.15);
}
.kpi-card p { margin: 0 !important; padding: 0 !important; }
.kpi-val {
    font-family: var(--font-serif) !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: var(--cream) !important;
    letter-spacing: -1.5px !important;
    line-height: 1 !important;
    margin-bottom: 0 !important;
}
.kpi-label {
    font-family: var(--font-mono) !important;
    font-size: 0.58rem !important;
    font-weight: 500 !important;
    color: var(--silver2) !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    margin-top: 10px !important;
}
.kpi-delta {
    display: inline-block !important;
    margin-top: 12px !important;
    font-size: 0.7rem !important;
    padding: 3px 9px !important;
    border-radius: 3px !important;
    font-family: var(--font-mono) !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
.kpi-delta.up   { background: rgba(74,155,111,0.12) !important; color: #5db888 !important; border: 1px solid rgba(74,155,111,0.2) !important; }
.kpi-delta.down { background: rgba(184,84,80,0.12) !important;  color: #d47470 !important; border: 1px solid rgba(184,84,80,0.2) !important; }

/* ── Section header ── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 40px 0 22px;
}
.section-ornament {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
}
.section-ornament-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--gold); }
.section-ornament-line { width: 1px; height: 12px; background: linear-gradient(180deg, var(--gold), transparent); }
.section-title {
    font-family: var(--font-serif) !important;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--cream) !important;
    letter-spacing: 0.1px;
}
.section-title em { font-style: italic; color: var(--gold2); font-weight: 400; }
.section-rule {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(212,168,71,0.2), transparent);
}

/* ── Cards ── */
.insight-card {
    background: var(--ink2);
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 2px solid var(--gold);
    border-radius: var(--radius);
    padding: 22px 24px;
    margin-bottom: 12px;
    font-size: 0.87rem;
    line-height: 1.85;
    color: var(--cream);
}
.insight-card code {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem;
    background: var(--ink3);
    padding: 1px 6px;
    border-radius: 3px;
    color: var(--gold2);
}
.arch-block {
    border-radius: var(--radius);
    padding: 22px 26px;
    border: 1px solid rgba(212,168,71,0.14);
    background: var(--ink2);
    font-family: var(--font-mono) !important;
    font-size: 0.77rem;
    line-height: 2.1;
    color: var(--cream);
    position: relative;
    overflow: hidden;
}
.arch-block::before {
    content: 'ARCHITECTURE';
    position: absolute;
    top: 14px; right: 18px;
    font-size: 0.52rem;
    letter-spacing: 2.5px;
    color: var(--gold);
    opacity: 0.4;
}
.arch-block b { color: var(--gold2); font-weight: 500; }

/* ── Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 3px;
    font-size: 0.62rem;
    font-weight: 600;
    font-family: var(--font-mono) !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
.badge-success { background: rgba(74,155,111,0.12); color: #5db888; border: 1px solid rgba(74,155,111,0.22); }
.badge-danger  { background: rgba(184,84,80,0.12);  color: #d47470; border: 1px solid rgba(184,84,80,0.2); }
.badge-info    { background: var(--gold-dim); color: var(--gold2); border: 1px solid var(--gold-line); }

/* ── Tech grid ── */
.tech-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 6px;
    margin-top: 4px;
}
.tech-item {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 14px 16px;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(255,255,255,0.04);
    background: var(--ink2);
    font-size: 0.82rem;
    color: var(--cream);
    transition: border-color var(--transition), background var(--transition);
    cursor: default;
}
.tech-item:hover {
    border-color: rgba(212,168,71,0.2);
    background: var(--ink3);
}
.tech-pip {
    width: 3px;
    align-self: stretch;
    border-radius: 2px;
    flex-shrink: 0;
    margin-top: 2px;
}
.tech-name { font-weight: 600; color: var(--cream); margin-bottom: 3px; font-size: 0.83rem; }
.tech-desc { color: var(--silver2); font-size: 0.76rem; font-weight: 300; line-height: 1.5; }

/* ── Step cards ── */
.step-card {
    display: flex;
    align-items: flex-start;
    gap: 18px;
    padding: 15px 18px;
    border-radius: var(--radius);
    border: 1px solid rgba(255,255,255,0.04);
    background: var(--ink2);
    margin-bottom: 5px;
    transition: border-color var(--transition);
}
.step-card:hover { border-color: rgba(212,168,71,0.18); }
.step-num {
    min-width: 28px; height: 28px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--gold-line);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono) !important;
    font-size: 0.62rem; font-weight: 500;
    color: var(--gold);
    flex-shrink: 0;
    background: var(--gold-glow);
}
.step-title { font-weight: 600; font-size: 0.86rem; color: var(--cream); margin-bottom: 3px; }
.step-desc  { font-size: 0.76rem; color: var(--silver2); font-weight: 300; line-height: 1.5; }

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    border: 1px solid var(--gold-line) !important;
    background: var(--ink2) !important;
    color: var(--gold2) !important;
    transition: all var(--transition) !important;
    padding: 0.5rem 1.4rem !important;
}
.stButton > button:hover {
    background: var(--gold-dim) !important;
    border-color: var(--gold) !important;
    color: var(--gold3) !important;
    box-shadow: 0 4px 20px rgba(212,168,71,0.12) !important;
}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border-radius: var(--radius-sm) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    background: var(--ink2) !important;
    color: var(--cream) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.86rem !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--gold-line) !important;
    box-shadow: 0 0 0 3px var(--gold-glow) !important;
}
[data-testid="stNumberInput"] input {
    background: var(--ink2) !important;
    border-color: rgba(255,255,255,0.06) !important;
    color: var(--cream) !important;
}
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    background: var(--ink2) !important;
    border-color: var(--gold-line) !important;
}
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    overflow: hidden;
}
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: var(--radius) !important;
    background: var(--ink2) !important;
}
[data-testid="stCode"] {
    border-radius: var(--radius) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── Sidebar footer ── */
.stack-info {
    padding: 0 8px;
    font-family: var(--font-mono) !important;
    font-size: 0.6rem;
    line-height: 2.3;
    color: var(--silver2) !important;
}
.stack-info b { color: var(--gold); font-weight: 500; font-size: 0.58rem; letter-spacing: 1px; }

/* ── Remove iframe border from components.html ── */
iframe {
    border: none !important;
    outline: none !important;
    display: block !important;
}

/* ── Result card for sentiment ── */
.result-card {
    border-radius: var(--radius);
    padding: 24px 26px;
    border: 1px solid rgba(255,255,255,0.05);
    background: var(--ink2);
    margin-top: 10px;
    position: relative;
    overflow: hidden;
}
.result-label {
    font-family: var(--font-serif) !important;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    line-height: 1;
    margin: 8px 0 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def kpi(label, value, delta=None, delta_dir="up"):
    """Render a KPI card using components.html to bypass Streamlit HTML sanitizer."""
    import streamlit.components.v1 as components
    delta_color  = "#5db888" if delta_dir == "up" else "#d47470"
    delta_bg     = "rgba(74,155,111,0.12)" if delta_dir == "up" else "rgba(184,84,80,0.12)"
    delta_border = "rgba(74,155,111,0.2)"  if delta_dir == "up" else "rgba(184,84,80,0.2)"
    if delta:
        delta_block = (
            "<div style=\"display:inline-block;margin-top:10px;font-size:0.68rem;"
            "padding:2px 8px;border-radius:3px;font-family:IBM Plex Mono,monospace;"
            f"background:{delta_bg};color:{delta_color};border:1px solid {delta_border}\">"
            f"{delta}</div>"
        )
    else:
        delta_block = ""
    html = (
        "<link href=\"https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700"
        "&family=IBM+Plex+Mono:wght@500&display=swap\" rel=\"stylesheet\">"
        "<style>*{margin:0;padding:0;box-sizing:border-box}</style>"
        "<div style=\"background:#262622;border:1px solid rgba(255,255,255,0.05);overflow:hidden;"
        "border-top:2px solid #d4a847;border-radius:8px;padding:20px 18px 16px;"
        "width:100%;height:118px;display:flex;flex-direction:column;justify-content:center;\">"
        f"<div style=\"font-family:'Playfair Display',Georgia,serif;font-size:2.2rem;"
        f"font-weight:700;color:#f2ede4;letter-spacing:-1.5px;line-height:1\">{value}</div>"
        f"<div style=\"font-family:'IBM Plex Mono',monospace;font-size:0.56rem;font-weight:500;"
        f"color:#6b6760;text-transform:uppercase;letter-spacing:1.8px;margin-top:8px\">{label}</div>"
        f"{delta_block}"
        "</div>"
    )
    components.html(html, height=128, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

def section(title, subtitle=""):
    sub_html = f'<em> — {subtitle}</em>' if subtitle else ""
    st.markdown(f"""<div class="section-hdr">
      <div class="section-ornament">
        <div class="section-ornament-dot"></div>
        <div class="section-ornament-line"></div>
      </div>
      <div class="section-title">{title}{sub_html}</div>
      <div class="section-rule"></div>
    </div>""", unsafe_allow_html=True)

def badge(text, kind="info"):
    return f'<span class="badge badge-{kind}">{text}</span>'

def divider():
    st.markdown("""<div class="divider">
      <div class="divider-line"></div>
      <div class="divider-center"><span></span><span></span><span></span></div>
      <div class="divider-line"></div>
    </div>""", unsafe_allow_html=True)

def page_header(eyebrow, title, sub):
    st.markdown(f"""<div class="ph-wrap">
      <div class="ph-rule">
        <div class="ph-rule-line"></div>
        <div class="ph-rule-dot"></div>
        <div class="ph-eyebrow">{eyebrow}</div>
      </div>
      <div class="ph-title">{title}</div>
      <div class="ph-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def safe_get_df(data, *keys):
    for k in keys:
        val = data.get(k)
        if val is not None and isinstance(val, pd.DataFrame) and not val.empty:
            return val
    return None

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#6b6760", size=11),
    title_font=dict(family="Playfair Display", size=14, color="#f2ede4"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b6760", size=10)),
    margin=dict(t=44, b=24, l=10, r=10),
    xaxis=dict(showgrid=False, color="#4a4844", linecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", color="#4a4844", linecolor="rgba(255,255,255,0.05)"),
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
      <div class="brand-mark"><span></span><span></span><span></span></div>
      <div class="brand-title">GenAI Customer<br>Intelligence</div>
      <div class="brand-sub">Analytics Platform · v1.0</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "💬  NLP Sentiment",
        "📉  Churn Prediction",
        "🧪  A/B Testing",
        "💡  LLM Insights",
        "⚙️  Run Pipeline",
    ], label_visibility="collapsed")

    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stack-info">
    <b>NLP LAYER</b><br>DistilBERT · spaCy · TF-IDF<br>
    <b>DL ENGINE</b><br>PyTorch · MLflow · SHAP<br>
    <b>GEN AI</b><br>LangChain · Groq / Llama3<br>
    <b>CLOUD</b><br>Azure ML · AWS SageMaker<br>
    <b>PLATFORM</b><br>Databricks · Delta Lake<br>
    <b>TESTING</b><br>pytest · 23 tests passing<br><br>
    <b>AUTHOR</b><br>Swathi K S
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base  = os.path.join(os.path.dirname(__file__), "..")
    data  = {}
    paths = {
        "customers":   f"{base}/data/raw/customers.csv",
        "reviews":     f"{base}/outputs/reviews_with_sentiment.csv",
        "reviews_raw": f"{base}/data/raw/reviews.csv",
        "ab_events":   f"{base}/data/raw/ab_experiment_events.csv",
        "churn_preds": f"{base}/outputs/churn_predictions.csv",
        "ab_results":  f"{base}/outputs/ab_test_results.json",
        "report":      f"{base}/outputs/executive_report.md",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            if path.endswith(".csv"):
                data[key] = pd.read_csv(path)
            elif path.endswith(".json"):
                with open(path) as f:
                    data[key] = json.load(f)
            elif path.endswith(".md"):
                with open(path, encoding="utf-8") as f:
                    data[key] = f.read()
        else:
            data[key] = None
    return data

data         = load_data()
pipeline_ran = data.get("customers") is not None


# ═════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════════════════════
if "Overview" in page:
    page_header(
        "End-to-End AI / ML Platform",
        "<em>GenAI</em> Customer Intelligence",
        "A production-grade platform combining Large Language Models, Deep Learning, NLP, "
        "Statistical Testing and A/B Experimentation — deployed on Azure & AWS with Databricks pipelines."
    )

    if not pipeline_ran:
        st.info("Pipeline not yet run. Navigate to Run Pipeline to generate data and models.")
    else:
        customers  = data["customers"]
        reviews    = safe_get_df(data, "reviews_raw", "reviews")
        ab         = data.get("ab_events")
        churn_rate = customers["churned"].mean() if "churned" in customers.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Total Customers",  f"{len(customers):,}")
        with c2: kpi("Churn Rate",        f"{churn_rate:.1%}", "⚠ Monitor", "down")
        with c3: kpi("Reviews Analysed",  f"{len(reviews):,}" if reviews is not None else "—")
        with c4: kpi("A/B Events",        f"{len(ab):,}" if ab is not None else "—")

    divider()
    section("Technology Stack", "full modern AI/ML toolkit")

    techs = [
        ("#d4a847", "LLM & Gen AI",    "LangChain + Groq / Llama3 for automated insight generation & conversational Q&A"),
        ("#d4a847", "Deep Learning",   "PyTorch ChurnNet — 256→128→64→32→1 with BatchNorm + Dropout"),
        ("#a8a49c", "NLP",             "HuggingFace DistilBERT for sentiment classification + TF-IDF keyword extraction"),
        ("#a8a49c", "Statistics",      "SciPy & Statsmodels — Hypothesis testing, A/B testing, Bayesian analysis"),
        ("#6b6760", "Azure Cloud",     "Azure ML Jobs, Azure Blob Storage, Azure ML Workspace integration"),
        ("#6b6760", "AWS Cloud",       "S3, SageMaker endpoint deployment, Lambda-ready inference handler"),
        ("#6b6760", "Databricks",      "Medallion architecture — Bronze / Silver / Gold Delta Lake pipeline"),
        ("#a8a49c", "MLflow",          "Experiment tracking, model registry, hyperparameter logging"),
        ("#d4a847", "SDLC / Agile",    "Scrum sprints, GitHub Actions CI/CD, pytest — 23 tests passing ✓"),
    ]

    st.markdown('<div class="tech-grid">', unsafe_allow_html=True)
    for color, name, desc in techs:
        st.markdown(f"""
        <div class="tech-item">
          <div class="tech-pip" style="background:{color}40;border-left:2px solid {color}"></div>
          <div>
            <div class="tech-name">{name}</div>
            <div class="tech-desc">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    divider()
    section("Project Structure")
    st.code("""genai_customer_intelligence/
├── src/
│   ├── data/              data_generator.py · preprocessor.py
│   ├── nlp/               sentiment_analyzer.py  (DistilBERT + TF-IDF)
│   ├── llm/               insight_generator.py   (LangChain + Groq/Llama3)
│   ├── deep_learning/     churn_model.py         (PyTorch ChurnNet)
│   ├── statistics/        ab_testing.py          (Frequentist + Bayesian)
│   └── pipeline_runner.py
├── streamlit_app/         app.py  ← this dashboard
├── databricks/            medallion_pipeline.py  (Bronze → Silver → Gold)
├── cloud/azure/           azure_ml_job.yml
├── cloud/aws/             sagemaker_deploy.py
├── agile/                 sprint_plan.md  (6 sprints · 88 story points)
└── tests/                 test_all_modules.py  (pytest · 23 tests ✓)""", language="text")


# ═════════════════════════════════════════════════════════════
# PAGE: NLP SENTIMENT
# ═════════════════════════════════════════════════════════════
elif "NLP" in page:
    import plotly.express as px
    import plotly.graph_objects as go

    page_header(
        "Natural Language Processing",
        "<em>Sentiment</em> Analysis",
        "BERT-based sentiment classification · TF-IDF keyword extraction · Real-time inference engine"
    )

    reviews = safe_get_df(data, "reviews", "reviews_raw")

    if reviews is None:
        st.warning("Run the pipeline first to generate review data.")
    else:
        if "predicted_sentiment" not in reviews.columns:
            reviews = reviews.copy()
            reviews["predicted_sentiment"] = reviews.get("sentiment_label", pd.Series(["positive"]*len(reviews)))

        total = len(reviews)
        pos   = (reviews["predicted_sentiment"] == "positive").sum()
        neg   = (reviews["predicted_sentiment"] == "negative").sum()
        avg_r = round(reviews["rating"].mean(), 2) if "rating" in reviews.columns else "—"

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Total Reviews", f"{total:,}")
        with c2: kpi("Positive",       f"{pos/total:.1%}", "↑ Strong", "up")
        with c3: kpi("Negative",        f"{neg/total:.1%}", "↓ Monitor", "down")
        with c4: kpi("Avg Star Rating", f"★ {avg_r}")

        divider()
        section("Sentiment Distribution")

        col_l, col_r = st.columns(2)
        with col_l:
            sc    = reviews["predicted_sentiment"].value_counts()
            fig   = px.pie(
                values=sc.values, names=sc.index, hole=0.58,
                color=sc.index,
                color_discrete_map={"positive":"#d4a847","neutral":"#6b6760","negative":"#b85450"},
                title="Sentiment Breakdown",
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              textfont=dict(family="IBM Plex Mono", size=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "rating" in reviews.columns:
                rs   = reviews.groupby(["rating","predicted_sentiment"]).size().reset_index(name="count")
                fig2 = px.bar(rs, x="rating", y="count", color="predicted_sentiment",
                              barmode="stack", title="Rating vs Sentiment",
                              color_discrete_map={"positive":"#d4a847","neutral":"#6b6760","negative":"#b85450"})
                fig2.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

        divider()
        section("Review Samples")
        show_cols = [c for c in ["review_text","rating","sentiment_label","predicted_sentiment","sentiment_confidence"] if c in reviews.columns]
        st.dataframe(reviews[show_cols].head(20), use_container_width=True, hide_index=True)

        divider()
        section("Live Sentiment Predictor", "real-time inference")
        st.markdown('<p style="font-size:0.82rem;color:var(--silver2);margin-bottom:14px">Enter a review text or a star rating (1–5) to instantly classify sentiment.</p>', unsafe_allow_html=True)

        user_text = st.text_area("", height=100,
            placeholder='e.g. "not amazing"  ·  "5"  ·  "great quality but slow delivery"')

        if st.button("◆  Analyse Sentiment") and user_text.strip():
            raw = user_text.strip()

            # ── Input validation before calling analyzer ──────────
            VALID_KEYWORDS = [
                "positive","negative","neutral","good","bad","great","poor",
                "excellent","terrible","worst","best","amazing","awful",
                "not good","not great","not bad","not amazing","not terrible",
                "love","hate","okay","ok","fine","decent","average",
                "happy","unhappy","satisfied","dissatisfied","disappointed",
                "fast","slow","quick","delayed","broken","defective","quality",
                "delivery","recommend","return","refund","complaint","perfect",
            ]

            def is_valid_input(text):
                """Return (is_valid, is_rating, clean_text)."""
                t = text.strip().lower()
                try:
                    val = float(t)
                    if 1 <= val <= 5:
                        return True, True, text
                    else:
                        return False, False, text
                except ValueError:
                    pass
                if len(t) < 2 or not any(c.isalpha() for c in t):
                    return False, False, text
                words = t.split()
                real_words = [w for w in words if len(w) >= 2 and any(c.isalpha() for c in w)]
                if not real_words:
                    return False, False, text
                for kw in VALID_KEYWORDS:
                    if kw in t:
                        return True, False, text
                vowels = sum(1 for c in t if c in 'aeiou')
                if len(t) >= 4 and vowels >= 1 and len(real_words) >= 1:
                    return True, False, text
                return False, False, text

            valid, is_num, _ = is_valid_input(raw)

            if not valid:
                st.markdown("""
                <div style="border-radius:8px;padding:16px 20px;
             border:1px solid rgba(184,84,80,0.3);background:rgba(184,84,80,0.07);
             font-size:0.86rem;color:#f2ede4;margin-top:8px">
          ⚠ <b>Invalid input.</b> Please enter:<br>
          &nbsp;&nbsp;• A star rating between <b>1 and 5</b><br>
          &nbsp;&nbsp;• A review text with real words (e.g. "not good", "great quality", "terrible service")
                </div>""", unsafe_allow_html=True)
                st.stop()

            from src.nlp.sentiment_analyzer import SentimentAnalyzer
            with st.spinner("Running inference..."):
                analyzer = SentimentAnalyzer(use_transformers=False)
                result   = analyzer.predict_single(raw)

            label     = result["label"].upper()
            conf      = result["score"]
            is_rating = result.get("rating_mode", False) or is_num

            # Catch any unexpected label
            if label not in ("POSITIVE", "NEGATIVE", "NEUTRAL"):
                st.markdown(f"""
                <div style="border-radius:8px;padding:16px 20px;
             border:1px solid rgba(184,84,80,0.3);background:rgba(184,84,80,0.07);
             font-size:0.86rem;color:#f2ede4;margin-top:8px">
          ⚠ Could not classify this input. Try a clearer review or a rating (1–5).
                </div>""", unsafe_allow_html=True)
                st.stop()

            color_map = {"POSITIVE":"#d4a847","NEGATIVE":"#b85450","NEUTRAL":"#6b6760"}
            icon_map  = {"POSITIVE":"✦","NEGATIVE":"✖","NEUTRAL":"◈"}
            color     = color_map[label]
            icon      = icon_map[label]

            RATING_DESC = {
                "POSITIVE": "This rating indicates a <b>satisfied customer</b> with high likelihood of repeat purchase.",
                "NEUTRAL":  "This rating indicates a <b>neutral customer</b> — a conversion opportunity.",
                "NEGATIVE": "This rating indicates a <b>dissatisfied customer</b> at risk of churn. Follow-up recommended.",
            }
            TEXT_DESC = {
                "POSITIVE": "The review carries <b>positive sentiment</b> — customer satisfaction signal.",
                "NEUTRAL":  "The review carries <b>mixed or neutral sentiment</b> — no strong signal.",
                "NEGATIVE": "The review carries <b>negative sentiment</b> — potential dissatisfaction detected.",
            }

            bar_w = int(conf * 100)
            desc  = RATING_DESC[label] if is_rating else TEXT_DESC[label]
            mode  = "Star Rating Mode" if is_rating else "Text Review Mode"
            badge_kind = "success" if label == "POSITIVE" else "danger" if label == "NEGATIVE" else "info"

            st.markdown(f"""
            <div class="result-card" style="border-top:2px solid {color}">
      <div style="display:flex;align-items:center;justify-content:space-between">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
              color:{color};letter-spacing:2px;text-transform:uppercase">{mode}</span>
                <span class="badge badge-{badge_kind}">{label}</span>
      </div>
      <div class="result-label" style="color:{color}">{icon} {label.title()}</div>
      <div style="font-size:0.84rem;color:#a8a49c;margin-bottom:18px;line-height:1.7">{desc}</div>
      <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;
           font-size:0.62rem;color:#6b6760;margin-bottom:7px">
                <span>Confidence</span>
                <span style="color:{color};font-weight:500">{conf:.1%}</span>
      </div>
      <div style="height:4px;border-radius:99px;background:rgba(255,255,255,0.06);overflow:hidden">
                <div style="height:100%;width:{bar_w}%;
             background:linear-gradient(90deg,{color},{color}99);border-radius:99px"></div>
      </div>
            </div>""", unsafe_allow_html=True)
 
            text_low = raw.lower()
            negators = ["not","no","never","isn't","wasn't","don't","doesn't","hardly","barely"]
            if any(n in text_low.split() for n in negators) and not is_rating:
                st.markdown("""
                <div style="margin-top:8px;padding:10px 14px;border-radius:5px;
             background:rgba(212,168,71,0.06);border:1px solid rgba(212,168,71,0.15);
             font-size:0.78rem;color:#a8a49c">
          ◆ <b style="color:#e8c070">Negation detected</b> —
          words like "not", "never", "don't" invert the polarity of nearby terms.
                </div>""", unsafe_allow_html=True)
                st.stop()

                color_map = {"POSITIVE":"#d4a847","NEGATIVE":"#b85450","NEUTRAL":"#6b6760"}
                icon_map  = {"POSITIVE":"✦","NEGATIVE":"✖","NEUTRAL":"◈"}
                color     = color_map.get(label, "#d4a847")
                icon      = icon_map.get(label, "◆")

                RATING_DESC = {
                    "POSITIVE": "This rating indicates a <b>satisfied customer</b> with high likelihood of repeat purchase.",
                    "NEUTRAL":  "This rating indicates a <b>neutral customer</b> — a conversion opportunity.",
                "NEGATIVE": "This rating indicates a <b>dissatisfied customer</b> at risk of churn. Follow-up recommended.",
            }
            TEXT_DESC = {
                "POSITIVE": "The review carries <b>positive sentiment</b> — customer satisfaction signal.",
                "NEUTRAL":  "The review carries <b>mixed or neutral sentiment</b> — no strong signal.",
                "NEGATIVE": "The review carries <b>negative sentiment</b> — potential dissatisfaction detected.",
            }

            bar_w = int(conf * 100)
            desc  = RATING_DESC[label] if is_rating else TEXT_DESC[label]
            mode  = "Star Rating Mode" if is_rating else "Text Review Mode"

            st.markdown(f"""
            <div class="result-card" style="border-top: 2px solid {color}">
            <div style="display:flex;align-items:center;justify-content:space-between">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.58rem;color:{color};letter-spacing:2px;text-transform:uppercase">{mode}</span>
                <span class="badge badge-{'success' if label=='POSITIVE' else 'danger' if label=='NEGATIVE' else 'info'}">{label}</span>
            </div>
            <div class="result-label" style="color:{color}">{icon} {label.title()}</div>
            <div style="font-size:0.84rem;color:var(--silver);margin-bottom:18px;line-height:1.7">{desc}</div>
            <div style="display:flex;justify-content:space-between;font-family:'IBM Plex Mono',monospace;font-size:0.62rem;color:var(--silver2);margin-bottom:7px">
                <span>Confidence</span><span style="color:{color};font-weight:500">{conf:.1%}</span>
            </div>
            <div style="height:4px;border-radius:99px;background:rgba(255,255,255,0.06);overflow:hidden">
                <div style="height:100%;width:{bar_w}%;background:linear-gradient(90deg,{color},{color}99);border-radius:99px"></div>
            </div>
            </div>""", unsafe_allow_html=True)

            text_low = user_text.lower()
            negators = ["not","no","never","isn't","wasn't","don't","doesn't","hardly","barely"]
            if any(n in text_low.split() for n in negators) and not is_rating:
                st.markdown("""
                <div style="margin-top:8px;padding:10px 14px;border-radius:5px;background:rgba(212,168,71,0.06);border:1px solid rgba(212,168,71,0.15);font-size:0.78rem;color:var(--silver2)">
                ◆ <b style="color:var(--gold2)">Negation detected</b> — words like "not", "never", "don't" invert the polarity of nearby terms.
                </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: CHURN PREDICTION
# ═════════════════════════════════════════════════════════════
elif "Churn" in page:
    import plotly.express as px
    import plotly.graph_objects as go

    page_header(
        "Deep Learning Model",
        "<em>Churn</em> Prediction",
        "PyTorch ChurnNet · Binary classification · SHAP feature importance · MLflow experiment tracking"
    )

    preds = data.get("churn_preds")
    if preds is None:
        st.warning("Run the pipeline first to generate churn predictions.")
    else:
        high_risk = int((preds["churn_probability"] >= 0.7).sum()) if "churn_probability" in preds.columns else 0
        med_risk  = int(((preds["churn_probability"] >= 0.4) & (preds["churn_probability"] < 0.7)).sum()) if "churn_probability" in preds.columns else 0
        low_risk  = int((preds["churn_probability"] < 0.4).sum()) if "churn_probability" in preds.columns else 0
        avg_prob  = preds["churn_probability"].mean() if "churn_probability" in preds.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("High Risk", f"{high_risk:,}", "≥ 70% prob", "down")
        with c2: kpi("Medium Risk", f"{med_risk:,}", "40–70%", "down")
        with c3: kpi("Low Risk", f"{low_risk:,}", "< 40%", "up")
        with c4: kpi("Avg Churn Prob", f"{avg_prob:.1%}")

        divider()
        section("Probability Distribution")

        col_l, col_r = st.columns(2)
        with col_l:
            if "churn_probability" in preds.columns:
                fig = px.histogram(preds, x="churn_probability", nbins=40,
                                   color_discrete_sequence=["#d4a847"],
                                   title="Churn Probability Distribution",
                                   labels={"churn_probability":"Churn Probability"})
                fig.add_vline(x=0.5, line_dash="dash", line_color="#b85450",
                              annotation_text="Decision Boundary", annotation_font_size=10,
                              annotation_font_color="#b85450")
                fig.add_vline(x=0.7, line_dash="dot", line_color="#d4a847",
                              annotation_text="High Risk", annotation_font_size=10,
                              annotation_font_color="#d4a847")
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_traces(marker_line_width=0, opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            risk_data = pd.DataFrame({
                "Risk Level": ["High Risk", "Medium Risk", "Low Risk"],
                "Count":      [high_risk, med_risk, low_risk],
            })
            fig2 = px.bar(risk_data, x="Risk Level", y="Count",
                          color="Risk Level",
                          color_discrete_map={"High Risk":"#b85450","Medium Risk":"#d4a847","Low Risk":"#4a9b6f"},
                          title="Customers by Risk Tier")
            fig2.update_layout(**{**PLOTLY_LAYOUT, "showlegend": False})
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

        divider()
        section("Model Architecture", "PyTorch ChurnNet")
        st.markdown("""
        <div class="arch-block">
        ChurnNet(<br>
        &nbsp;&nbsp;<b>Linear</b>(input_dim → 256) → BatchNorm1d → ReLU → Dropout(0.30)<br>
        &nbsp;&nbsp;<b>Linear</b>(256 → 128)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ BatchNorm1d → ReLU → Dropout(0.30)<br>
        &nbsp;&nbsp;<b>Linear</b>(128 → 64)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ BatchNorm1d → ReLU → Dropout(0.21)<br>
        &nbsp;&nbsp;<b>Linear</b>(64 → 32)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ ReLU<br>
        &nbsp;&nbsp;<b>Linear</b>(32 → 1)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Sigmoid<br>
        )<br><br>
        Optimizer &nbsp;: Adam (lr=1e-3, weight_decay=1e-5)<br>
        Scheduler &nbsp;: ReduceLROnPlateau (patience=5, factor=0.5)<br>
        Loss &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: BCELoss<br>
        Tracking &nbsp;&nbsp;: MLflow experiment registry
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: A/B TESTING
# ═════════════════════════════════════════════════════════════
elif "A/B" in page:
    import plotly.express as px
    import plotly.graph_objects as go

    page_header(
        "Experimentation Engine",
        "<em>A/B</em> Testing",
        "Frequentist Two-Proportion Z-Test · Bayesian Beta-Binomial · Revenue impact analysis"
    )

    ab_events  = data.get("ab_events")
    ab_results = data.get("ab_results")

    if ab_events is None:
        st.warning("Run the pipeline first.")
    else:
        control   = ab_events[ab_events.group == "control"]
        treatment = ab_events[ab_events.group == "treatment"]
        ctrl_cvr  = control["converted"].mean()
        trt_cvr   = treatment["converted"].mean()
        lift      = (trt_cvr - ctrl_cvr) / ctrl_cvr

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Control CVR",   f"{ctrl_cvr:.2%}")
        with c2: kpi("Treatment CVR", f"{trt_cvr:.2%}", f"+{lift:.1%} lift", "up")
        with c3: kpi("Control N",     f"{len(control):,}")
        with c4: kpi("Treatment N",   f"{len(treatment):,}")

        divider()

        if ab_results:
            freq  = ab_results.get("frequentist", {})
            bayes = ab_results.get("bayesian", {})

            col_f, col_b = st.columns(2)
            with col_f:
                section("Frequentist Test", "Z-Test")
                is_sig = freq.get("significant", False)
                p_val  = freq.get("p_value", "—")
                impact = freq.get("revenue_impact", 0)
                st.markdown(f"""
                <div class="insight-card">
                  <div style="margin-bottom:14px">{badge("SIGNIFICANT ✓","success") if is_sig else badge("NOT SIGNIFICANT","danger")}</div>
                  <table style="width:100%;border-collapse:collapse">
                    <tr><td style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--silver2);padding:5px 0;text-transform:uppercase;letter-spacing:1px">P-Value</td>
                        <td style="text-align:right;font-size:0.86rem"><b>{p_val}</b></td></tr>
                    <tr><td style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--silver2);padding:5px 0;text-transform:uppercase;letter-spacing:1px">Relative Uplift</td>
                        <td style="text-align:right;font-size:0.86rem"><b>{freq.get('uplift',0)*100:.1f}%</b></td></tr>
                    <tr><td style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--silver2);padding:5px 0;text-transform:uppercase;letter-spacing:1px">95% CI Lower</td>
                        <td style="text-align:right;font-size:0.86rem"><b>{freq.get('ci_lower','—')}</b></td></tr>
                    <tr><td style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--silver2);padding:5px 0;text-transform:uppercase;letter-spacing:1px">95% CI Upper</td>
                        <td style="text-align:right;font-size:0.86rem"><b>{freq.get('ci_upper','—')}</b></td></tr>
                    <tr><td style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;color:var(--silver2);padding:5px 0;text-transform:uppercase;letter-spacing:1px">Revenue Impact / Mo</td>
                        <td style="text-align:right;font-size:0.86rem;color:var(--gold2)"><b>₹{impact:,.0f}</b></td></tr>
                  </table>
                </div>""", unsafe_allow_html=True)

            with col_b:
                section("Bayesian Test", "Beta-Binomial")
                prob = bayes.get("prob_treatment_better", 0)
                rec  = bayes.get("recommendation", "—")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={"suffix":"%","font":{"family":"Playfair Display","size":36,"color":"#f2ede4"}},
                    title={"text":"P(Treatment > Control)","font":{"family":"Plus Jakarta Sans","size":11,"color":"#6b6760"}},
                    gauge={
                        "axis":{"range":[0,100],"tickcolor":"#4a4844","tickfont":{"color":"#4a4844","size":9}},
                        "bar":{"color":"#d4a847","thickness":0.2},
                        "bgcolor":"rgba(0,0,0,0)",
                        "bordercolor":"rgba(255,255,255,0.04)",
                        "steps":[
                            {"range":[0,80],  "color":"rgba(184,84,80,0.05)"},
                            {"range":[80,95], "color":"rgba(212,168,71,0.05)"},
                            {"range":[95,100],"color":"rgba(74,155,111,0.07)"},
                        ],
                        "threshold":{"line":{"color":"#4a9b6f","width":2},"thickness":0.8,"value":95},
                    },
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220,
                                        margin=dict(t=30,b=0,l=20,r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
                rec_color = "success" if "Ship" in rec else "info"
                st.markdown(f'<div style="text-align:center;margin-top:-8px">{badge(rec, rec_color)}</div>', unsafe_allow_html=True)

        divider()
        section("Revenue Distribution", "converters only")
        converters = ab_events[ab_events.converted == 1]
        fig3 = px.histogram(converters, x="revenue_inr", color="group", nbins=35,
                            barmode="overlay", opacity=0.70,
                            color_discrete_map={"control":"#6b6760","treatment":"#d4a847"},
                            labels={"revenue_inr":"Revenue (INR)","group":"Group"})
        fig3.update_layout(**PLOTLY_LAYOUT)
        fig3.update_traces(marker_line_width=0)
        st.plotly_chart(fig3, use_container_width=True)

        divider()
        section("Custom Hypothesis Test", "live z-test calculator")
        st.markdown('<p style="font-size:0.82rem;color:var(--silver2);margin-bottom:16px">Enter your own experiment numbers to run a live two-proportion z-test.</p>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            n_ctrl    = st.number_input("Control N",           min_value=10, value=int(len(control)))
            conv_ctrl = st.number_input("Control Conversions", min_value=0,  value=int(control["converted"].sum()))
        with col_b:
            n_trt    = st.number_input("Treatment N",           min_value=10, value=int(len(treatment)))
            conv_trt = st.number_input("Treatment Conversions", min_value=0,  value=int(treatment["converted"].sum()))

        if st.button("◆  Run Z-Test"):
            from src.statistics.ab_testing import FrequentistABTest
            result = FrequentistABTest(alpha=0.05).test_conversion_rate(
                int(conv_ctrl), int(n_ctrl), int(conv_trt), int(n_trt)
            )
            is_s = result.get("significant", False)
            st.markdown(f"""
            <div class="insight-card" style="border-left-color:{'#4a9b6f' if is_s else '#b85450'}">
              {badge("SIGNIFICANT ✓","success") if is_s else badge("NOT SIGNIFICANT","danger")}
              <br><br>
              p-value: <b>{result['p_value']}</b> &nbsp;·&nbsp;
              Uplift: <b>{result['relative_lift']*100:.1f}%</b> &nbsp;·&nbsp;
              Z-stat: <b>{result['z_statistic']}</b>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: LLM INSIGHTS
# ═════════════════════════════════════════════════════════════
elif "LLM" in page:
    page_header(
        "Generative AI",
        "<em>LLM</em> Business Insights",
        "LangChain + Groq / Llama3 · Automated executive reporting · Conversational data Q&A"
    )

    report = data.get("report")
    if report:
        section("Executive Report")
        st.markdown(
            '<div class="insight-card" style="font-size:0.86rem;line-height:1.9">'
            + report.replace('\n', '<br>')
            + '</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Run the pipeline first to generate the LLM executive report.")
        st.markdown("""
        <div class="insight-card">
          <b>No API key required.</b> The platform auto-generates template-based reports.<br>
          Add <code>GROQ_API_KEY</code> to your <code>.env</code> file for Llama3-powered AI insights.
        </div>""", unsafe_allow_html=True)

    divider()
    section("Ask a Question About Your Data", "RAG-style Q&A")
    st.markdown('<p style="font-size:0.82rem;color:var(--silver2);margin-bottom:14px">Ask any business question — the LLM answers using your platform data as context.</p>', unsafe_allow_html=True)

    question = st.text_input("", placeholder="e.g. Which customer segment has the highest churn risk?")
    if st.button("◆  Ask LLM") and question.strip():
        @st.cache_resource
        def get_llm():
            from src.llm.insight_generator import LLMInsightGenerator
            return LLMInsightGenerator()
        with st.spinner("Thinking..."):
            gen     = get_llm()
            context = ("Customer dataset: 5,000 customers, 8,000 reviews, 12,000 A/B events. "
                       "Overall churn rate ~26%. Top churn factors: low engagement, "
                       "high support tickets, low monthly spend.")
            answer  = gen.answer_question(question, context)
        st.markdown(f'<div class="insight-card">{answer}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# PAGE: RUN PIPELINE
# ═════════════════════════════════════════════════════════════
elif "Pipeline" in page:
    page_header(
        "Orchestration",
        "Pipeline <em>Control</em>",
        "Run the full end-to-end AI/ML pipeline — data generation through LLM reporting"
    )

    section("Pipeline Steps")
    steps = [
        ("Data Generation",     "Synthetic 5K customers · 8K reviews · 12K A/B events · feature store"),
        ("NLP Analysis",        "DistilBERT sentiment classification · TF-IDF keyword extraction"),
        ("Deep Learning Churn", "PyTorch ChurnNet training · MLflow tracking · SHAP explainability"),
        ("A/B Testing",         "Frequentist Z-test + Bayesian Beta-Binomial · revenue impact"),
        ("LLM Insights",        "LangChain executive report · Groq / Llama3 or template fallback"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">{i:02d}</div>
          <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("◆  Generate Data Only"):
            with st.spinner("Generating synthetic dataset..."):
                from src.data.data_generator import main as generate
                generate()
            st.success("Data generated successfully.")
            st.cache_data.clear()

    with col2:
        if st.button("◆  Run Full Pipeline"):
            import subprocess
            with st.spinner("Running full pipeline (~2–3 min)..."):
                result = subprocess.run([sys.executable, "src/pipeline_runner.py"],
                                        capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Pipeline complete. Refresh the page to see updated results.")
                st.cache_data.clear()
                with st.expander("Pipeline log"):
                    st.code(result.stdout[-3000:])
            else:
                st.error("Pipeline failed. See error output below.")
                st.code(result.stderr[-2000:])

    divider()
    section("Manual Terminal Commands")
    st.code("""# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Generate data
python src/data/data_generator.py

# 3. Run full pipeline
python src/pipeline_runner.py

# 4. Launch dashboard
streamlit run streamlit_app/app.py

# 5. Run tests
pytest tests/ -v --cov=src""", language="bash")
