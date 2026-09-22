from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from resume_screening.model import MODEL_PATH, ensure_model
from resume_screening.pipeline import screen_resume
from resume_screening.ranking import rank_candidates
from resume_screening.resume_parser import extract_text_from_upload
from resume_screening.skill_extractor import extract_skills, load_skill_dictionary

st.set_page_config(page_title="Talent Lens | Resume Intelligence", page_icon="TL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
    --ink: #eff5f3;
    --muted: #9badab;
    --line: rgba(255,255,255,.12);
    --teal: #43d6b2;
    --teal-glow: rgba(67, 214, 178, 0.25);
    --surface: #0c1a1c;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background: radial-gradient(circle at 76% 0%, rgba(67,214,178,.12), transparent 30rem), linear-gradient(135deg,#071311 0%,#0b1719 48%,#111329 100%);
    color: var(--ink);
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.04em; color: var(--ink); }
.hero { padding:1.2rem 0 .5rem; }
.eyebrow { color:var(--teal); text-transform:uppercase; letter-spacing:.16em; font-size:.75rem; font-weight:700; }
.hero h1 { font-size:clamp(2.2rem,5vw,4.2rem); line-height:.98; margin:.35rem 0 .7rem; max-width:760px; }
.hero p { color:var(--muted); max-width:680px; font-size:1.05rem; line-height:1.6; }

/* Container cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(11, 23, 25, 0.72) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(12px) !important;
    padding: 1.2rem !important;
}

/* Form labels - high contrast readability */
label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
    color: #e2edea !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

/* Inputs and textareas - crisp dark surface with light text */
.stTextArea textarea, .stTextInput input {
    background-color: var(--surface) !important;
    color: #f1f7f5 !important;
    border: 1px solid rgba(67, 214, 178, 0.3) !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #43d6b2 !important;
    box-shadow: 0 0 0 2px var(--teal-glow) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: rgba(160, 185, 180, 0.75) !important;
}

/* Selectbox dropdown */
div[data-baseweb="select"] > div {
    background-color: var(--surface) !important;
    color: #f1f7f5 !important;
    border: 1px solid rgba(67, 214, 178, 0.3) !important;
    border-radius: 12px !important;
}
div[data-baseweb="select"] * {
    color: #f1f7f5 !important;
}
div[data-baseweb="popover"], ul[role="listbox"] {
    background-color: var(--surface) !important;
    border: 1px solid rgba(67, 214, 178, 0.35) !important;
}
li[role="option"] {
    color: #f1f7f5 !important;
    background-color: transparent !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: rgba(67, 214, 178, 0.18) !important;
    color: #43d6b2 !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(67, 214, 178, 0.45) !important;
    border-radius: 16px !important;
    background: rgba(67, 214, 178, 0.04) !important;
    padding: 0.8rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--surface) !important;
    border: 1px solid rgba(67, 214, 178, 0.25) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #cad8d4 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(67, 214, 178, 0.15) !important;
    border: 1px solid rgba(67, 214, 178, 0.4) !important;
    color: #43d6b2 !important;
    font-weight: 600 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #9badab !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #43d6b2 !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: #43d6b2 !important;
}
div[data-baseweb="tab-border"] {
    background-color: rgba(255, 255, 255, 0.12) !important;
}

/* Metrics and badges */
.metric { border-left:2px solid var(--teal); padding-left:.9rem; }
.metric .value { font:700 1.65rem 'Space Grotesk'; color: #eff5f3; }
.metric .label { color:var(--muted); font-size:.78rem; }
.badge { display:inline-block; border-radius:999px; padding:.26rem .65rem; font-size:.75rem; font-weight:700; background:rgba(67,214,178,.15); color:var(--teal); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #061113 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}
section[data-testid="stSidebar"] * {
    color: #eff5f3;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] .stCaption {
    color: #9badab !important;
}

/* Primary buttons */
.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(67, 214, 178, 0.45) !important;
    background: linear-gradient(135deg, #35c9a6, #208d88) !important;
    color: #021210 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 14px rgba(67, 214, 178, 0.25) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #43d6b2, #27a39d) !important;
    box-shadow: 0 6px 20px rgba(67, 214, 178, 0.4) !important;
}
.stButton > button:disabled {
    background: rgba(67, 214, 178, 0.12) !important;
    border-color: rgba(67, 214, 178, 0.15) !important;
    color: rgba(239, 245, 243, 0.35) !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None

with st.sidebar:
    st.markdown("## Talent Lens")
    st.caption("Competency-first screening workspace")
    st.markdown("---")
    st.markdown("**Workflow**")
    st.markdown("1. Define the role\n2. Upload resumes\n3. Review ranked evidence")
    st.markdown("---")
    st.caption("Decision support only. Keep a human in the loop and avoid using sensitive characteristics.")

st.markdown('<div class="hero"><div class="eyebrow">Resume intelligence / 01</div><h1>Find the signal in every application.</h1><p>Talent Lens turns job requirements and PDF resumes into transparent, skill-based rankings powered by NLP, similarity features, and a supervised model.</p></div>', unsafe_allow_html=True)

setup, results_tab = st.tabs(["Screen candidates", "Model & methodology"])
with setup:
    col1, col2 = st.columns([1.05, .95], gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("### 01 / Define the role")
            job_title = st.selectbox(
                "Job position",
                [
                    "Custom role",
                    "Software Developer",
                    "Data Analyst",
                    "Machine Learning Engineer",
                    "React Developer",
                    "SQL / Database Developer",
                    "Network Security Engineer",
                    "Operations Manager",
                    "Mechanical Engineer",
                    "Civil Engineer",
                    "Electrical Engineer",
                    "Blockchain Engineer",
                    "Accounting / Finance",
                    "Digital Media & Design",
                    "Automobile / Automotive",
                    "Aviation & Aerospace",
                    "Plumber / Trades",
                    "Project Manager",
                ],
            )
            if job_title == "Custom role":
                job_title = st.text_input("Role name", placeholder="e.g. Plumber, Product Operations Lead, Nurse...")
            job_text = st.text_area("Job description & requirements", height=210, placeholder="Paste responsibilities, must-have skills, qualifications, and experience requirements...")
            job_upload = st.file_uploader("Or upload a job description", type=["txt", "md"], key="job")
            if job_upload and not job_text:
                job_text = job_upload.getvalue().decode("utf-8", errors="ignore")
            if job_text:
                skills = sorted(extract_skills(job_text, load_skill_dictionary()))
                st.markdown(f"**{len(skills)} competencies detected**")
                st.write(" ".join(f"`{skill}`" for skill in skills) if skills else "Add more specific requirements to detect competencies.")
    with col2:
        with st.container(border=True):
            st.markdown("### 02 / Add candidate resumes")
            uploads = st.file_uploader("Drop multiple PDF resumes here", type=["pdf"], accept_multiple_files=True)
            st.markdown(f'<span class="badge">{len(uploads)} resume(s) ready</span>', unsafe_allow_html=True)
            st.markdown("\n")
            if uploads:
                for upload in uploads:
                    st.caption(f"{upload.name} · {round(upload.size / 1024)} KB")
            analyze = st.button("Run screening", use_container_width=True, disabled=not (job_text and uploads))


    if analyze:
        with st.status("Processing candidate evidence...", expanded=True) as status:
            bundle = ensure_model(MODEL_PATH)
            rows = []
            for upload in uploads:
                try:
                    text = extract_text_from_upload(upload)
                    rows.append(screen_resume(upload.name, text, job_text, bundle))
                    st.write(f"Processed {upload.name}")
                except Exception as exc:
                    st.warning(f"Could not process {upload.name}: {exc}")
            st.session_state.results = rank_candidates(rows) if rows else None
            status.update(label="Screening complete", state="complete")

    if st.session_state.results is not None:
        frame = st.session_state.results
        st.markdown("## Candidate ranking")
        m1, m2, m3, m4 = st.columns(4)
        for container, value, label in [(m1, len(frame), "candidates reviewed"), (m2, f"{frame.score.mean():.0%}", "average score"), (m3, len(frame[frame.prediction == "Strong Match"]), "strong matches"), (m4, f"{frame.matched_skills.map(len).mean():.1f}", "avg. matched skills")]:
            with container:
                st.markdown(f'<div class="metric"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)
        chart = px.bar(frame, x="score", y="candidate", orientation="h", color="score", color_continuous_scale=["#263a4a", "#43d6b2"], range_x=[0, 1], text=frame.score.map(lambda x: f"{x:.0%}"))
        chart.update_layout(height=max(260, 62 * len(frame)), margin=dict(l=0,r=0,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce8e5", coloraxis_showscale=False)
        st.plotly_chart(chart, use_container_width=True)
        display = frame[["rank", "candidate", "score", "prediction", "matched_skills", "missing_skills"]].copy()
        display["score"] = display["score"].map(lambda x: f"{x:.0%}")
        display["matched_skills"] = display["matched_skills"].map(lambda x: ", ".join(x) or "—")
        display["missing_skills"] = display["missing_skills"].map(lambda x: ", ".join(x) or "—")
        st.dataframe(display, use_container_width=True, hide_index=True, column_config={"rank":"Rank", "candidate":"Candidate", "score":"Score", "prediction":"Assessment", "matched_skills":"Matched skills", "missing_skills":"Missing requirements"})
        selected = st.selectbox("Inspect candidate evidence", frame.candidate.tolist())
        candidate = frame[frame.candidate == selected].iloc[0]
        a, b = st.columns(2)
        with a:
            st.markdown(f"### {candidate.candidate}")
            st.metric("Suitability", f"{candidate.score:.0%}", candidate.prediction)
            st.markdown("**Matched skills**")
            st.write(" · ".join(candidate.matched_skills) or "No direct matches found")
        with b:
            st.markdown("**Missing requirements**")
            st.write(" · ".join(candidate.missing_skills) or "None detected")
            st.markdown(f"**Experience signal:** {candidate.years_experience:g} years  \n**Degree signal:** {'Detected' if candidate.degree_match else 'Not detected'}")
        csv = frame.to_csv(index=False).encode("utf-8")
        st.download_button("Export screening report (CSV)", csv, "talent-lens-screening.csv", "text/csv")

with results_tab:
    st.markdown("### How the decision is made")
    st.info("The ranking is decision support, not an automated hiring decision. The model is intentionally restricted to job-relevant competency and qualification signals.")
    st.markdown("**Pipeline**")
    st.code("PDF → text extraction → normalization → lemmatization → skill extraction → feature engineering → supervised model → ranking", language="text")
    try:
        bundle = ensure_model(MODEL_PATH)
        st.markdown(f"**Active model:** `{bundle.get('model_name', 'trained classifier')}`")
        if bundle.get("reports"):
            report = pd.DataFrame(bundle["reports"]).T
            st.dataframe(report.style.format("{:.2f}"), use_container_width=True)
    except Exception as exc:
        st.warning(f"Model evaluation is unavailable until dependencies are installed: {exc}")
    st.markdown("**Interpretation**")
    st.write("Matched and missing competencies, Jaccard and cosine similarity, experience signals, education signals, certification evidence, and text coverage are combined into the model feature vector. Sensitive characteristics are not used as ranking factors.")
