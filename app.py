"""
app.py — Streamlit entrypoint for Clinical Document Intelligence

A single-page application that ingests clinical documents, extracts structured
data via NVIDIA Nemotron LLM (OpenAI-compatible API), assigns risk flags, and
renders decision-ready patient summary cards with confidence indicators and
evidence traceability.
"""

import os
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from document_loader import load_document
from extraction import extract_clinical_data, ClinicalExtraction
from reasoning import assess_risk, compare_documents, RiskAssessment, MultiDocComparison

# ─────────────────────────────────────────────────────────────
# Page Config & Logging
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Clinical Document Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Log file setup ──────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "clinical_app.log"

# Root logger → file + console
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# File handler — all logs with timestamps
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

# Console handler — brief
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "[%(levelname)s] %(name)s: %(message)s"
))

# Only add handlers if not already added (Streamlit reruns)
if not root_logger.handlers:
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)
logger.info(f"Log file: {LOG_FILE.resolve()}")


# ─────────────────────────────────────────────────────────────
# Custom CSS — Premium card design
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────────────────── */
.stApp {
    background: radial-gradient(circle at top left, #111827 0%, #050505 50%, #09090b 100%);
    color: #e2e8f0;
}

/* ── Typography Override (Safe) ─────────────────────── */
/* We only apply this to our custom elements to prevent breaking Streamlit icons */
.header-bar, .patient-card, .comparison-card, .info-box, .error-box {
    font-family: 'Outfit', sans-serif;
}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #09090b;
    border-right: 1px solid rgba(0, 229, 255, 0.1);
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-family: 'Outfit', sans-serif;
}

/* ── Header Bar ─────────────────────────────────────── */
.header-bar {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.1) 0%, rgba(179, 136, 255, 0.1) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}

.header-bar::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,229,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: pulse-glow 8s ease-in-out infinite alternate;
}

@keyframes pulse-glow {
    0% { transform: scale(1); opacity: 0.5; }
    100% { transform: scale(1.2); opacity: 1; }
}

.header-bar h1 {
    color: #ffffff;
    font-weight: 800;
    font-size: 2.5rem;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.03em;
    background: linear-gradient(to right, #ffffff, #00E5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-bar p {
    color: #94a3b8;
    font-weight: 400;
    font-size: 1.1rem;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Patient Summary Card ───────────────────────────── */
.patient-card {
    background: rgba(17, 24, 39, 0.6);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 0;
    margin-bottom: 2rem;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.patient-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 229, 255, 0.3);
    box-shadow: 0 15px 50px rgba(0, 229, 255, 0.15);
}

.card-header {
    padding: 1.75rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    background: rgba(0, 0, 0, 0.2);
}

.card-header-left h2 {
    color: #f8fafc;
    font-weight: 700;
    font-size: 1.5rem;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-header-meta {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    align-items: center;
}

.meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 0.3rem 0.8rem;
    border-radius: 24px;
    font-size: 0.8rem;
    color: #cbd5e1;
    font-weight: 500;
    transition: all 0.2s ease;
}

.patient-card:hover .meta-tag {
    background: rgba(255, 255, 255, 0.06);
}

/* ── Risk Badges ────────────────────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    box-shadow: 0 0 20px inset rgba(0,0,0,0.2);
}

.risk-routine {
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid #00E5FF;
    color: #00E5FF;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
}

.risk-needs-review {
    background: rgba(179, 136, 255, 0.1);
    border: 1px solid #B388FF;
    color: #B388FF;
    box-shadow: 0 0 15px rgba(179, 136, 255, 0.2);
}

.risk-urgent {
    background: rgba(255, 42, 109, 0.1);
    border: 1px solid #FF2A6D;
    color: #FF2A6D;
    box-shadow: 0 0 20px rgba(255, 42, 109, 0.4);
    animation: neon-pulse 2s ease-in-out infinite alternate;
}

@keyframes neon-pulse {
    from { box-shadow: 0 0 10px rgba(255, 42, 109, 0.2); }
    to { box-shadow: 0 0 25px rgba(255, 42, 109, 0.6); }
}

/* ── Card Body ──────────────────────────────────────── */
.card-body {
    padding: 2rem 2.5rem;
}

.section-title {
    color: #00E5FF;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin: 1.5rem 0 0.75rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title::before {
    content: '';
    display: block;
    width: 8px;
    height: 8px;
    background: #00E5FF;
    border-radius: 50%;
    box-shadow: 0 0 8px #00E5FF;
}

.section-title:first-child {
    margin-top: 0;
}

.chief-complaint {
    color: #f1f5f9;
    font-size: 1.1rem;
    font-weight: 400;
    line-height: 1.6;
    margin: 0;
}

/* ── Data Items ─────────────────────────────────────── */
.data-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    gap: 1rem;
    transition: background 0.2s ease;
}

.data-item:hover {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 8px;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
}

.data-item:last-child {
    border-bottom: none;
}

.data-label {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
    flex: 1;
}

.data-value-normal {
    color: #f8fafc;
    font-size: 0.95rem;
    font-weight: 500;
}

.data-value-abnormal {
    color: #FF2A6D;
    font-size: 0.95rem;
    font-weight: 600;
    background: rgba(255, 42, 109, 0.1);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
}

/* ── Confidence Tag ─────────────────────────────────── */
.conf-tag {
    display: inline-flex;
    align-items: center;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    margin-left: 0.5rem;
    vertical-align: middle;
    letter-spacing: 0.05em;
}

.conf-high {
    background: rgba(0, 229, 255, 0.15);
    color: #00E5FF;
}

.conf-med {
    background: rgba(179, 136, 255, 0.15);
    color: #B388FF;
}

.conf-low {
    background: rgba(255, 42, 109, 0.15);
    color: #FF2A6D;
}

/* ── Evidence Snippet ───────────────────────────────── */
.evidence {
    color: #64748b;
    font-size: 0.8rem;
    font-style: italic;
    margin-top: 0.25rem;
    line-height: 1.5;
    border-left: 2px solid rgba(255,255,255,0.1);
    padding-left: 0.75rem;
}

/* ── Card Footer ────────────────────────────────────── */
.card-footer {
    padding: 1.5rem 2.5rem;
    background: rgba(0, 0, 0, 0.2);
    border-top: 1px solid rgba(255, 255, 255, 0.05);
}

.recommendation-box {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.05), rgba(179, 136, 255, 0.05));
    border: 1px solid rgba(0, 229, 255, 0.15);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
}

.recommendation-text {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
    line-height: 1.5;
}

.justification-text {
    color: #94a3b8;
    font-size: 0.85rem;
    margin: 0.75rem 0 0 0;
    font-style: italic;
}

.driving-factors {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.factor-chip {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 16px;
    font-weight: 500;
    transition: all 0.2s ease;
}

.factor-chip:hover {
    background: rgba(0, 229, 255, 0.1);
    border-color: rgba(0, 229, 255, 0.3);
    color: #00E5FF;
}

/* ── Medications Table ──────────────────────────────── */
.med-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin: 0.5rem 0;
}

.med-table th {
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.med-table td {
    color: #f1f5f9;
    font-size: 0.9rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.med-table tr:hover td {
    background: rgba(255, 255, 255, 0.02);
}

.med-table tr:last-child td {
    border-bottom: none;
}

/* ── Multi-doc Comparison Card ──────────────────────── */
.comparison-card {
    background: linear-gradient(135deg, rgba(179, 136, 255, 0.08), rgba(0, 229, 255, 0.08));
    backdrop-filter: blur(24px);
    border: 1px solid rgba(179, 136, 255, 0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin: 2.5rem 0;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

.comparison-card h3 {
    color: #fff;
    font-weight: 700;
    font-size: 1.35rem;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.trend-item, .conflict-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.5rem 0;
    color: #e2e8f0;
    font-size: 0.95rem;
    line-height: 1.5;
}

.trend-icon { color: #00E5FF; font-size: 1.1rem; }
.conflict-icon { color: #FF2A6D; font-size: 1.1rem; }

/* ── Loading animation ──────────────────────────────── */
.loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem;
}

.loading-spinner {
    width: 60px;
    height: 60px;
    border: 4px solid rgba(0, 229, 255, 0.1);
    border-top: 4px solid #00E5FF;
    border-radius: 50%;
    animation: spin-neon 1s cubic-bezier(0.5, 0, 0.5, 1) infinite;
    margin-bottom: 1.5rem;
    box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
}

@keyframes spin-neon {
    to { transform: rotate(360deg); }
}

.loading-text {
    color: #00E5FF;
    font-size: 1rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* ── Info boxes ─────────────────────────────────────── */
.info-box {
    background: rgba(0, 229, 255, 0.05);
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-left: 4px solid #00E5FF;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    color: #e2e8f0;
    font-size: 0.95rem;
    margin: 1rem 0;
}

.error-box {
    background: rgba(231, 76, 60, 0.08);
    border: 1px solid rgba(231, 76, 60, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #f0a0a0;
    font-size: 0.9rem;
    margin: 1rem 0;
}

/* ── Streamlit overrides ────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stExpander"] summary {
    color: #a0a0c0 !important;
    font-weight: 500;
}

h1, h2, h3, h4 {
    color: #fff !important;
}

p, li {
    color: #c8c8e0;
}

/* ── Sample data buttons ────────────────────────────── */
.sample-btn-container {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin: 0.5rem 0;
}

/* ── Divider ────────────────────────────────────────── */
.subtle-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# NVIDIA API Configuration
# ─────────────────────────────────────────────────────────────

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def get_nvidia_client(api_key: str):
    """Initialize OpenAI client pointed at NVIDIA API."""
    from openai import OpenAI
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


def get_confidence_tag(confidence: float) -> str:
    """Return an HTML confidence tag with color-coding."""
    if confidence >= 0.85:
        cls = "conf-high"
        label = f"{confidence:.0%}"
    elif confidence >= 0.60:
        cls = "conf-med"
        label = f"{confidence:.0%}"
    else:
        cls = "conf-low"
        label = f"{confidence:.0%}"
    return f'<span class="conf-tag {cls}">{label}</span>'


def get_risk_badge(risk_level: str) -> str:
    """Return an HTML risk badge."""
    risk_map = {
        "Routine": ("risk-routine", "✓ Routine"),
        "Needs Review": ("risk-needs-review", "⚠ Needs Review"),
        "Urgent": ("risk-urgent", "🚨 Urgent"),
    }
    cls, label = risk_map.get(risk_level, ("risk-needs-review", risk_level))
    return f'<span class="risk-badge {cls}">{label}</span>'


def render_patient_card(
    extraction: ClinicalExtraction,
    risk: RiskAssessment | None,
):
    """Render a full patient summary card."""
    # ── Header
    header_html = f"""
    <div class="patient-card">
        <div class="card-header">
            <div class="card-header-left">
                <h2>🧑‍⚕️ {extraction.patient_name.value}
                    {get_confidence_tag(extraction.patient_name.confidence)}</h2>
                <div class="card-header-meta">
                    <span class="meta-tag">📋 {extraction.document_type.value}</span>
                    <span class="meta-tag">📅 {extraction.document_date.value}</span>
                    <span class="meta-tag">👤 {extraction.patient_age.value} · {extraction.patient_sex.value}</span>
                </div>
            </div>
            <div>
                {get_risk_badge(risk.risk_level) if risk else '<span class="meta-tag">⏳ Risk not assessed</span>'}
            </div>
        </div>
    """

    # ── Body
    body_html = '<div class="card-body">'

    # Chief Complaint
    body_html += f"""
        <div class="section-title">Chief Complaint</div>
        <p class="chief-complaint">{extraction.chief_complaint.value}
            {get_confidence_tag(extraction.chief_complaint.confidence)}</p>
        <div class="evidence">📌 "{extraction.chief_complaint.evidence_snippet}"</div>
    """

    # Diagnoses
    if extraction.diagnoses:
        body_html += '<div class="section-title">Diagnoses</div>'
        for dx in extraction.diagnoses:
            body_html += f"""
            <div class="data-item">
                <span class="data-label">• {dx.description.value}
                    {get_confidence_tag(dx.description.confidence)}</span>
            </div>
            """

    # Medications
    if extraction.medications:
        body_html += '<div class="section-title">Medications</div>'
        body_html += """
        <table class="med-table">
            <thead><tr>
                <th>Medication</th><th>Dosage</th><th>Frequency</th>
            </tr></thead>
            <tbody>
        """
        for med in extraction.medications:
            body_html += f"""
            <tr>
                <td>{med.name.value} {get_confidence_tag(med.name.confidence)}</td>
                <td>{med.dosage.value}</td>
                <td>{med.frequency.value}</td>
            </tr>
            """
        body_html += "</tbody></table>"

    # Vital Signs / Lab Values — highlight abnormal
    if extraction.vital_signs:
        body_html += '<div class="section-title">Vital Signs & Lab Values</div>'
        for vs in extraction.vital_signs:
            status = vs.status.value.lower()
            value_cls = "data-value-abnormal" if status == "abnormal" else "data-value-normal"
            flag = " ⚠" if status == "abnormal" else ""
            body_html += f"""
            <div class="data-item">
                <span class="data-label">{vs.name.value}</span>
                <span class="{value_cls}">{vs.value.value} {vs.unit.value}{flag}
                    {get_confidence_tag(vs.status.confidence)}</span>
            </div>
            """

    # Allergies
    if extraction.allergies:
        body_html += '<div class="section-title">Allergies</div>'
        for allergy in extraction.allergies:
            body_html += f"""
            <div class="data-item">
                <span class="data-label">⚠️ {allergy.value}
                    {get_confidence_tag(allergy.confidence)}</span>
            </div>
            """

    # Follow-up Instructions
    if extraction.follow_up_instructions:
        body_html += '<div class="section-title">Follow-Up Instructions</div>'
        for instr in extraction.follow_up_instructions:
            body_html += f"""
            <div class="data-item">
                <span class="data-label">→ {instr.value}
                    {get_confidence_tag(instr.confidence)}</span>
            </div>
            """

    body_html += '</div>'

    # ── Footer (risk assessment)
    footer_html = '<div class="card-footer">'
    if risk:
        footer_html += f"""
        <div class="section-title">Recommended Next Step</div>
        <div class="recommendation-box">
            <p class="recommendation-text">💡 {risk.recommended_next_step}</p>
            <p class="justification-text">Reason: {risk.justification}</p>
            <div class="driving-factors">
        """
        for factor in risk.driving_factors:
            footer_html += f'<span class="factor-chip">{factor}</span>'
        footer_html += '</div></div>'
    else:
        footer_html += '<div class="info-box">Risk assessment not available.</div>'

    footer_html += '</div>'

    # Combine
    full_html = header_html + body_html + footer_html + '</div>'
    st.html(full_html)

    # Raw JSON expander
    with st.expander(f"Raw JSON — {extraction.patient_name.value}", expanded=False):
        combined = {
            "extraction": json.loads(extraction.model_dump_json()),
        }
        if risk:
            combined["risk_assessment"] = json.loads(risk.model_dump_json())
        st.json(combined)


def render_comparison_card(comparison: MultiDocComparison):
    """Render the multi-document comparison card."""
    html = f"""
    <div class="comparison-card">
        <h3>📊 Multi-Document Comparison {get_risk_badge(comparison.overall_risk_level)}</h3>
        <p style="color: #c8c8e0; font-size: 0.95rem; margin-bottom: 1rem;">
            {comparison.summary}
        </p>
    """

    if comparison.trending_observations:
        html += '<div class="section-title">Trending Observations</div>'
        for trend in comparison.trending_observations:
            html += f"""
            <div class="trend-item">
                <span class="trend-icon">📈</span>
                <span>{trend}</span>
            </div>
            """

    if comparison.conflicts:
        html += '<div class="section-title">Conflicts / Discrepancies</div>'
        for conflict in comparison.conflicts:
            html += f"""
            <div class="conflict-item">
                <span class="conflict-icon">⚡</span>
                <span>{conflict}</span>
            </div>
            """

    html += '</div>'
    st.html(html)

    with st.expander("Raw Comparison JSON", expanded=False):
        st.json(json.loads(comparison.model_dump_json()))


# ─────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────

def main():
    # ── Header
    st.markdown("""
    <div class="header-bar">
        <h1>🏥 Clinical Document Intelligence</h1>
        <p>AI-powered extraction and risk assessment for clinical documents.
           Upload discharge summaries, lab reports, intake forms, or physician notes.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        # API Key
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        api_key_input = st.text_input(
            "NVIDIA API Key",
            value=api_key,
            type="password",
            help="Your NVIDIA API key. Set NVIDIA_API_KEY env var or paste here.",
        )
        if api_key_input:
            api_key = api_key_input

        st.markdown("---")

        # File uploader
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose clinical documents",
            type=["txt", "pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Supported: .txt, .pdf, .png, .jpg",
        )

        st.markdown("---")

        # Sample data quick-load
        st.markdown("### 📁 Sample Documents")
        st.markdown(
            '<p style="color: #8888b0; font-size: 0.82rem;">'
            "Load pre-built synthetic documents for demo purposes.</p>",
            unsafe_allow_html=True,
        )

        sample_dir = Path(__file__).parent / "sample_data"
        sample_files = []
        if sample_dir.exists():
            for ext in ("*.txt", "*.pdf", "*.png", "*.jpg", "*.jpeg"):
                sample_files.extend(sample_dir.glob(ext))
            sample_files = sorted(set(sample_files), key=lambda p: p.name)

        sample_labels = {
            # .txt samples
            "discharge_summary_chf.txt": "🟡 Discharge — CHF Exacerbation",
            "lab_report_moderate.txt": "🟡 Lab Report — Moderate (Lipids/A1c)",
            # .pdf samples
            "discharge_summary_routine.pdf": "📕 [PDF] Discharge — Routine",
            "lab_report_urgent.pdf": "📕 [PDF] Lab Report — Urgent",
            # .png samples
            "intake_form_moderate.png": "🖼️ [PNG] Intake — Moderate",
            "physician_note_routine.png": "🖼️ [PNG] Physician Note — Routine",
        }

        selected_samples = []
        for sf in sample_files:
            label = sample_labels.get(sf.name, sf.name)
            if st.checkbox(label, key=f"sample_{sf.name}"):
                selected_samples.append(sf)

        if selected_samples:
            st.markdown(
                f'<div class="info-box">📋 {len(selected_samples)} sample document(s) selected</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Analyze button
        analyze_btn = st.button("🚀 Analyze Documents", use_container_width=True)

        st.markdown("---")
        st.markdown(
            '<p style="color: #666690; font-size: 0.75rem; text-align: center;">'
            "⚠️ This prototype uses <strong>synthetic data only</strong>. "
            "No real patient data is processed.<br><br>"
            "Built with NVIDIA Nemotron · Streamlit · Pydantic</p>",
            unsafe_allow_html=True,
        )

    # ── Main content area
    if not api_key:
        st.markdown(
            '<div class="info-box">'
            "🔑 Please provide your NVIDIA API key in the sidebar to get started."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Gather all documents to process
    documents = []  # list of (filename, bytes)

    if analyze_btn or st.session_state.get("results"):
        # Collect uploaded files
        if uploaded_files:
            for uf in uploaded_files:
                documents.append((uf.name, uf.getvalue()))

        # Collect selected samples
        for sf in selected_samples:
            documents.append((sf.name, sf.read_bytes()))

        if not documents and not st.session_state.get("results"):
            st.markdown(
                '<div class="error-box">'
                "📭 No documents to analyze. Please upload files or select sample documents."
                "</div>",
                unsafe_allow_html=True,
            )
            return

    # Process documents if analyze was clicked
    if analyze_btn and documents:
        try:
            client = get_nvidia_client(api_key)
        except Exception as e:
            st.markdown(
                f'<div class="error-box">❌ Failed to initialize API client: {e}</div>',
                unsafe_allow_html=True,
            )
            return

        results = []
        timing_log = []  # Collect timing info for display
        progress_bar = st.progress(0, text="Initializing...")
        pipeline_start = time.perf_counter()

        for i, (filename, file_bytes) in enumerate(documents):
            step = i + 1
            total = len(documents)
            doc_start = time.perf_counter()

            # Step 1: Load document
            progress_bar.progress(
                (step - 0.7) / (total + 0.5),
                text=f"📄 Loading {filename} ({step}/{total})..."
            )
            t0 = time.perf_counter()
            text = load_document(file_bytes, filename)
            load_elapsed = time.perf_counter() - t0
            logger.info(f"[TIMING][APP] Document loading '{filename}': {load_elapsed:.3f}s")

            if text.startswith("[ERROR]"):
                timing_log.append({
                    "file": filename,
                    "load_s": round(load_elapsed, 2),
                    "extract_s": None,
                    "risk_s": None,
                    "total_s": round(load_elapsed, 2),
                    "status": "❌ Load Error",
                })
                results.append({
                    "filename": filename,
                    "error": text,
                    "extraction": None,
                    "risk": None,
                })
                continue

            # Step 2: Extract
            progress_bar.progress(
                (step - 0.3) / (total + 0.5),
                text=f"🧠 Extracting data from {filename} ({step}/{total})..."
            )
            t0 = time.perf_counter()
            extraction = extract_clinical_data(text, client)
            extract_elapsed = time.perf_counter() - t0
            logger.info(f"[TIMING][APP] Data extraction '{filename}': {extract_elapsed:.3f}s")

            if extraction is None:
                timing_log.append({
                    "file": filename,
                    "load_s": round(load_elapsed, 2),
                    "extract_s": round(extract_elapsed, 2),
                    "risk_s": None,
                    "total_s": round(time.perf_counter() - doc_start, 2),
                    "status": "❌ Extraction Failed",
                })
                results.append({
                    "filename": filename,
                    "error": "Extraction failed after retries.",
                    "extraction": None,
                    "risk": None,
                })
                continue

            # Step 3: Risk assessment
            progress_bar.progress(
                (step) / (total + 0.5),
                text=f"⚖️ Assessing risk for {filename} ({step}/{total})..."
            )
            t0 = time.perf_counter()
            risk = assess_risk(extraction, client)
            risk_elapsed = time.perf_counter() - t0
            logger.info(f"[TIMING][APP] Risk assessment '{filename}': {risk_elapsed:.3f}s")

            doc_elapsed = time.perf_counter() - doc_start
            timing_log.append({
                "file": filename,
                "load_s": round(load_elapsed, 2),
                "extract_s": round(extract_elapsed, 2),
                "risk_s": round(risk_elapsed, 2),
                "total_s": round(doc_elapsed, 2),
                "status": "✅ Success",
            })
            logger.info(f"[TIMING][APP] Total for '{filename}': {doc_elapsed:.3f}s")

            results.append({
                "filename": filename,
                "error": None,
                "extraction": extraction,
                "risk": risk,
            })

        # Step 4: Multi-doc comparison
        extractions_for_compare = [
            r["extraction"] for r in results if r["extraction"] is not None
        ]
        comparison = None
        compare_elapsed = 0.0
        if len(extractions_for_compare) >= 2:
            progress_bar.progress(0.95, text="📊 Comparing documents...")
            t0 = time.perf_counter()
            comparison = compare_documents(extractions_for_compare, client)
            compare_elapsed = time.perf_counter() - t0
            logger.info(f"[TIMING][APP] Multi-doc comparison: {compare_elapsed:.3f}s")

        pipeline_elapsed = time.perf_counter() - pipeline_start
        logger.info(f"[TIMING][APP] ═══ TOTAL PIPELINE: {pipeline_elapsed:.3f}s ═══")
        progress_bar.progress(1.0, text=f"✅ Analysis complete in {pipeline_elapsed:.1f}s!")

        # Store results in session state
        st.session_state["results"] = results
        st.session_state["comparison"] = comparison
        st.session_state["timing_log"] = timing_log
        st.session_state["pipeline_elapsed"] = pipeline_elapsed
        st.session_state["compare_elapsed"] = compare_elapsed

    # ── Render results from session state
    if st.session_state.get("results"):
        results = st.session_state["results"]
        comparison = st.session_state.get("comparison")
        timing_log = st.session_state.get("timing_log", [])
        pipeline_elapsed = st.session_state.get("pipeline_elapsed", 0)
        compare_elapsed = st.session_state.get("compare_elapsed", 0)

        st.markdown(f"### 📋 Analysis Results — {len(results)} Document(s)")

        # ── Timing Summary (collapsible)
        if timing_log:
            with st.expander(f"⏱️ Performance Timing — Total: {pipeline_elapsed:.1f}s", expanded=False):
                # Build a timing summary table
                timing_html = """
                <table class="med-table">
                    <thead><tr>
                        <th>Document</th><th>Load</th><th>Extract</th><th>Risk</th><th>Total</th><th>Status</th>
                    </tr></thead>
                    <tbody>
                """
                for t in timing_log:
                    timing_html += f"""
                    <tr>
                        <td>{t['file']}</td>
                        <td>{t['load_s']:.2f}s</td>
                        <td>{f"{t['extract_s']:.2f}s" if t['extract_s'] is not None else '—'}</td>
                        <td>{f"{t['risk_s']:.2f}s" if t['risk_s'] is not None else '—'}</td>
                        <td><strong>{t['total_s']:.2f}s</strong></td>
                        <td>{t['status']}</td>
                    </tr>
                    """
                if compare_elapsed > 0:
                    timing_html += f"""
                    <tr>
                        <td><em>Multi-doc Comparison</em></td>
                        <td>—</td><td>—</td><td>—</td>
                        <td><strong>{compare_elapsed:.2f}s</strong></td>
                        <td>{'✅' if comparison else '❌'}</td>
                    </tr>
                    """
                timing_html += f"""
                    <tr style="border-top: 2px solid rgba(255,255,255,0.15);">
                        <td><strong>TOTAL PIPELINE</strong></td>
                        <td>—</td><td>—</td><td>—</td>
                        <td><strong>{pipeline_elapsed:.2f}s</strong></td>
                        <td>—</td>
                    </tr>
                </tbody></table>
                """
                st.markdown(timing_html, unsafe_allow_html=True)

        # Multi-doc comparison (if available) — show at top
        if comparison:
            render_comparison_card(comparison)

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

        # Render individual cards
        for i, result in enumerate(results):
            if result["error"]:
                st.markdown(
                    f'<div class="error-box">❌ <strong>{result["filename"]}</strong>: '
                    f'{result["error"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                render_patient_card(result["extraction"], result["risk"])

    elif not analyze_btn:
        # Show welcome state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <p style="font-size: 3rem; margin-bottom: 1rem;">📄</p>
            <h3 style="color: #a0a0c0; font-weight: 500;">Upload or select documents to begin</h3>
            <p style="color: #7070a0; font-size: 0.9rem; max-width: 500px; margin: 0 auto;">
                Upload clinical documents (PDF, images, or text files) using the sidebar,
                or select from the pre-built sample documents to see the system in action.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
