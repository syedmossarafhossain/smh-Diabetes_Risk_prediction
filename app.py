import streamlit as st
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes AI | Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(59,130,246,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(139,92,246,0.10), transparent 25%),
        #090d16;
    color: #f8fafc;
}


/* Remove default top spacing */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0d1320 0%,
        #0a0f19 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff;
}


/* Header */

.hero {
    position: relative;
    overflow: hidden;

    margin-top: 50px;
    margin-bottom: 45px;

    padding: 42px 48px;

    min-height: 260px;

    border-radius: 28px;

    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(124, 58, 237, 0.28),
            transparent 32%
        ),
        radial-gradient(
            circle at 15% 80%,
            rgba(37, 99, 235, 0.20),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.98),
            rgba(17, 24, 39, 0.94)
        );

    border: 1px solid rgba(148, 163, 184, 0.14);

    box-shadow:
        0 25px 80px rgba(0, 0, 0, 0.35),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);

    backdrop-filter: blur(20px);
}


/* Decorative glow */

.hero::before {
    content: "";

    position: absolute;

    width: 280px;
    height: 280px;

    right: -90px;
    top: -120px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.30),
            transparent 70%
        );

    filter: blur(10px);

    pointer-events: none;
}


/* Decorative grid */

.hero::after {
    content: "";

    position: absolute;

    inset: 0;

    background-image:
        linear-gradient(
            rgba(255,255,255,0.025) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(255,255,255,0.025) 1px,
            transparent 1px
        );

    background-size: 32px 32px;

    mask-image: linear-gradient(
        to bottom right,
        black,
        transparent 70%
    );

    pointer-events: none;
}


/* Hero content */

.hero-content {
    position: relative;
    z-index: 2;
}


/* Badge */

.badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 8px 14px;

    border-radius: 999px;

    background:
        rgba(59, 130, 246, 0.10);

    border: 1px solid
        rgba(96, 165, 250, 0.25);

    color: #93c5fd;

    font-size: 12px;
    font-weight: 700;

    letter-spacing: 0.4px;

    margin-bottom: 18px;

    box-shadow:
        0 8px 25px rgba(37, 99, 235, 0.12);
}


/* Badge dot */

.badge-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #60a5fa;

    box-shadow:
        0 0 12px rgba(96, 165, 250, 0.8);
}


/* Hero title */

.hero-title {
    margin: 0;

    max-width: 760px;

    font-size: clamp(34px, 4vw, 54px);

    line-height: 1.05;

    font-weight: 800;

    letter-spacing: -2px;

    color: #f8fafc;
}


/* Gradient title */

.hero-title span {
    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #818cf8,
            #c084fc
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    background-clip: text;
}


/* Subtitle */

.hero-subtitle {
    margin-top: 18px;

    max-width: 680px;

    font-size: 16px;

    line-height: 1.7;

    color: #94a3b8;
}


/* Hero bottom information */

.hero-meta {
    display: flex;

    align-items: center;

    gap: 12px;

    margin-top: 26px;

    flex-wrap: wrap;
}


/* Meta cards */

.hero-meta-item {
    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 9px 13px;

    border-radius: 10px;

    background: rgba(255,255,255,0.035);

    border: 1px solid
        rgba(255,255,255,0.07);

    color: #cbd5e1;

    font-size: 12px;

    font-weight: 500;
}


/* Right visual */

.hero-visual {
    position: absolute;

    z-index: 2;

    right: 48px;
    top: 50%;

    transform: translateY(-50%);

    width: 210px;
    height: 210px;

    display: flex;

    align-items: center;
    justify-content: center;
}


/* AI pulse circle */

.hero-orb {
    width: 150px;
    height: 150px;

    border-radius: 50%;

    display: flex;

    align-items: center;
    justify-content: center;

    background:
        radial-gradient(
            circle at 35% 30%,
            rgba(96,165,250,0.35),
            rgba(124,58,237,0.12) 45%,
            rgba(15,23,42,0.9) 70%
        );

    border: 1px solid
        rgba(129,140,248,0.35);

    box-shadow:
        0 0 50px rgba(99,102,241,0.22),
        inset 0 0 35px rgba(96,165,250,0.08);

    animation: heroPulse 4s ease-in-out infinite;
}


/* Orb icon */

.hero-orb-icon {
    font-size: 58px;

    filter:
        drop-shadow(
            0 0 18px rgba(96,165,250,0.45)
        );
}


/* Orb animation */

@keyframes heroPulse {

    0%, 100% {
        transform: scale(1);
        box-shadow:
            0 0 50px rgba(99,102,241,0.22),
            inset 0 0 35px rgba(96,165,250,0.08);
    }

    50% {
        transform: scale(1.05);
        box-shadow:
            0 0 75px rgba(99,102,241,0.35),
            inset 0 0 45px rgba(96,165,250,0.12);
    }
}


/* Responsive */

@media (max-width: 900px) {

    .hero {
        padding: 34px 30px;
    }

    .hero-visual {
        opacity: 0.25;
        right: 15px;
    }

    .hero-title {
        max-width: 650px;
    }
}


@media (max-width: 600px) {

    .hero {
        margin-top: 20px;
        padding: 28px 22px;
        min-height: 300px;
    }

    .hero-title {
        font-size: 34px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .hero-visual {
        display: none;
    }

    .hero-meta {
        gap: 8px;
    }
}

/* Section title */

.section-header {
    width: 100%;
    min-height: 90px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-sizing: border-box;
    margin-top: 35px;
    margin-bottom: 24px;
    padding: 18px 24px;
    border-radius: 18px;
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.95),
            rgba(17, 24, 39, 0.82)
        );

    border: 1px solid rgba(148, 163, 184, 0.14);
    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);
    position: relative;
    overflow: hidden;
}


/* Subtle glow inside the box */

.section-header::before {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -80px;
    top: -100px;
    border-radius: 50%;
    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.12),
            transparent 70%
        );
    pointer-events: none;
}

/* Left side */
.section-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    z-index: 2;
}


/* Icon box */
.section-icon {
    width: 52px;
    height: 52px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.20),
            rgba(124, 58, 237, 0.20)
        );

    border: 1px solid rgba(96, 165, 250, 0.22);
    font-size: 23px;
    box-shadow:
        0 8px 25px rgba(37, 99, 235, 0.12);
}


/* Title */

.section-title {
    margin: 0;
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.5px;
}


/* Description */

.section-description {
    margin-top: 6px;
    color: #64748b;
    font-size: 13px;
    font-weight: 500;
    line-height: 1.5;
}


/* Status badge */

.section-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
    padding: 9px 14px;
    border-radius: 999px;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.18);
    color: #86efac;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
    position: relative;
    z-index: 2;
}


/* Green status dot */

.section-status-dot {
    width: 7px;
    height: 7px;
    flex-shrink: 0;
    display: inline-block;
    border-radius: 50%;
    background: #4ade80;
    box-shadow:
        0 0 12px rgba(74, 222, 128, 0.85);
    animation: livePulse 1.8s infinite;
}


/*CLINICAL INPUT PANEL*/
.clinical-panel {

    padding: 24px;

    border-radius: 24px;

    background:
        linear-gradient(
            145deg,
            rgba(15,23,42,0.65),
            rgba(20,26,39,0.45)
        );

    border: 1px solid
        rgba(255,255,255,0.06);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.16);

    margin-bottom: 25px;
}



/* ADVANCED INPUT AREA */

div[data-testid="stNumberInput"],
div[data-testid="stTextInput"],
div[data-testid="stDateInput"] {

    padding: 16px 18px 14px 18px;

    margin-bottom: 14px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(20,26,39,0.92),
            rgba(15,23,42,0.88)
        );

    border: 1px solid
        rgba(255,255,255,0.07);

    box-shadow:
        0 8px 30px rgba(0,0,0,0.14);

    transition:
        transform 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}


/* Hover */

div[data-testid="stNumberInput"]:hover,
div[data-testid="stTextInput"]:hover,
div[data-testid="stDateInput"]:hover {

    transform: translateY(-2px);

    border-color:
        rgba(96,165,250,0.25);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.22);
}


/* Labels */

div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stDateInput"] label {

    color: #cbd5e1 !important;

    font-size: 13px !important;

    font-weight: 600 !important;

    margin-bottom: 8px !important;
}


/* Input */

div[data-baseweb="input"] {

    background:
        rgba(15,23,42,0.95) !important;

    border-radius: 12px !important;

    border: 1px solid
        rgba(255,255,255,0.08) !important;

    min-height: 42px;
}


/* Input hover */

div[data-baseweb="input"]:hover {

    border-color:
        rgba(96,165,250,0.35) !important;
}


/* Input focus */

div[data-baseweb="input"]:focus-within {

    border-color:
        #3b82f6 !important;

    box-shadow:
        0 0 0 3px
        rgba(59,130,246,0.12) !important;
}


/* Input text */

div[data-baseweb="input"] input {

    color: #f8fafc !important;

    font-size: 14px !important;

    font-weight: 500 !important;
}


/* Placeholder */

div[data-baseweb="input"] input::placeholder {

    color: #64748b !important;
}

/* NUMBER INPUT CONTROL*/
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {

    color: #94a3b8 !important;

    background:
        rgba(255,255,255,0.035) !important;

    border: none !important;

    transition: all 0.2s ease;
}

button[data-testid="stNumberInputStepDown"]:hover,
button[data-testid="stNumberInputStepUp"]:hover {

    color: #ffffff !important;

    background:
        rgba(59,130,246,0.15) !important;
}

/* CLINICAL INTELLIGENCE DASHBOARD */
.clinical-dashboard {
    margin-top: 35px;
    margin-bottom: 30px;
}

.clinical-dashboard-header {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;

    margin-top: 35px;
    margin-bottom: 22px;

    padding: 18px 22px;

    box-sizing: border-box;

    border-radius: 20px;

    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.92),
            rgba(17, 24, 39, 0.78)
        );

    border: 1px solid rgba(255, 255, 255, 0.07);

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.16);
}

.clinical-dashboard-title {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}

.clinical-dashboard-icon {
    width: 48px;
    height: 48px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.20),
            rgba(124, 58, 237, 0.20)
        );
    border: 1px solid rgba(96, 165, 250, 0.20);
    font-size: 22px;
    box-shadow:
        0 10px 30px rgba(37, 99, 235, 0.12);
}

.clinical-dashboard-heading {
    margin: 0;
    color: #f8fafc;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.clinical-dashboard-subtitle {
    margin-top: 5px;
    color: #64748b;
    font-size: 12px;
    font-weight: 500;
    line-height: 1.5;
}

.live-analysis {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    flex-shrink: 0;
    padding: 8px 13px;
    border-radius: 999px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.18);
    color: #86efac;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
    white-space: nowrap;
}

.live-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
    display: inline-block;
    background: #4ade80;
    box-shadow:
        0 0 12px rgba(74,222,128,0.85);
    animation: livePulse 1.8s infinite;
}

@keyframes livePulse {

    0% {
        opacity: 1;
        transform: scale(1);
    }

    50% {
        opacity: 0.45;
        transform: scale(0.75);
    }

    100% {
        opacity: 1;
        transform: scale(1);
    }
}


/* =========================================================
   PROFILE INDEX
   ========================================================= */
.profile-index-card {
    width: 100%;
    height: 390px;
    min-height: 390px;
    max-height: 390px;
    padding: 26px;
    margin: 0;
    box-sizing: border-box;
    border-radius: 24px;
    background:
        radial-gradient(
            circle at 50% 30%,
            rgba(59,130,246,0.12),
            transparent 45%
        ),
        linear-gradient(
            145deg,
            rgba(15,23,42,0.96),
            rgba(17,24,39,0.82)
        );
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 20px 60px rgba(0,0,0,0.20);
    text-align: center;
    overflow: hidden;
}

.profile-index-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #94a3b8;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.8px;
}

.profile-index-circle {
    width: 205px;
    height: 205px;
    margin: 25px auto 18px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background:
        radial-gradient(
            circle,
            #0f172a 58%,
            transparent 59%
        );
    border: 8px solid rgba(59,130,246,0.18);
    box-shadow:
        0 0 0 2px rgba(96,165,250,0.08),
        0 0 50px rgba(59,130,246,0.18),
        inset 0 0 40px rgba(59,130,246,0.08);

    box-sizing: border-box;
}

.profile-index-value {
    font-size: 58px;
    font-weight: 800;
    line-height: 1;
    background:
        linear-gradient(
            135deg,
            #60a5fa,
            #818cf8,
            #c084fc
        );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.profile-index-unit {
    margin-top: 6px;
    color: #64748b;
    font-size: 11px;
}

.profile-index-title {
    color: #cbd5e1;
    font-size: 15px;
    font-weight: 700;
}

.profile-index-description {
    max-width: 300px;
    margin: 9px auto 0;
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}


/* =========================================================
   CLINICAL PARAMETER PROFILE
   EXACTLY 390px HEIGHT
   ========================================================= */

.st-key-radar_card {
    width: 100% !important;

    height: 390px !important;
    min-height: 390px !important;
    max-height: 390px !important;

    box-sizing: border-box !important;

    padding: 0 26px !important;
    margin: 0 !important;

    border-radius: 24px !important;

    background:
        radial-gradient(
            circle at 50% 30%,
            rgba(59,130,246,0.12),
            transparent 45%
        ),
        linear-gradient(
            145deg,
            rgba(15,23,42,0.96),
            rgba(17,24,39,0.82)
        ) !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    box-shadow:
        0 20px 60px rgba(0,0,0,0.20) !important;

    overflow: hidden !important;

    flex: 0 0 390px !important;
}


/* Force Streamlit's internal wrapper to stay inside 390px */

.st-key-radar_card > div {
    width: 100% !important;

    min-height: 0 !important;

    box-sizing: border-box !important;

    overflow: hidden !important;
}



/* =========================================================
   RADAR HEADER
   ========================================================= */

.st-key-radar_card .radar-card-header {
    width: 100% !important;

    height: 58px !important;
    min-height: 58px !important;
    max-height: 58px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;

    box-sizing: border-box !important;

    padding: 0 0 !important;
    margin: 0 !important;

    flex-shrink: 0 !important;
}

/* Clinical Parameter Profile */

.st-key-radar_card .radar-card-title {
    color: #94a3b8 !important;

    font-size: 17px !important;
    font-weight: 700 !important;

    letter-spacing: 0.8px !important;

    line-height: 1.2 !important;
}


/* 8 PARAMETERS */

.st-key-radar_card .radar-card-badge {
    color: #94a3b8 !important;

    font-size: 17px !important;
    font-weight: 700 !important;

    letter-spacing: 0.8px !important;

    line-height: 1.2 !important;
}

    /* ========================================================= 
    RADAR GRAPH AREA 
    ========================================================= */


/* =========================================================
   PLOTLY CHART
   ========================================================= */

.st-key-radar_card [data-testid="stPlotlyChart"] {
    width: 100% !important;

    height: 300px !important;
    min-height: 300px !important;
    max-height: 300px !important;

    margin: 0 auto !important;
    padding: 0 !important;

    overflow: hidden !important;

    flex-shrink: 0 !important;
}


/* =========================================================
   PLOTLY INTERNAL WRAPPER
   ========================================================= */

.st-key-radar_card [data-testid="stPlotlyChart"] > div {
    width: 100% !important;

    height: 300px !important;
    min-height: 300px !important;
    max-height: 300px !important;
}


/* =========================================================
   PLOTLY IFRAME
   ========================================================= */

.st-key-radar_card [data-testid="stPlotlyChart"] iframe {
    width: 100% !important;

    height: 300px !important;
    min-height: 300px !important;
    max-height: 300px !important;

    display: block !important;
}
/* =========================================================
   PARAMETER HEADER
   ========================================================= */
.parameter-analysis-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 28px 0 16px;
}

.parameter-analysis-title {
    font-size: 17px;
    font-weight: 800;
    color: #f8fafc;
}

.parameter-analysis-count {
    padding: 6px 10px;
    border-radius: 999px;
    background:
        rgba(96,165,250,0.08);
    border:
        1px solid rgba(96,165,250,0.15);
    color: #93c5fd;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
}



/* =========================================================
   PARAMETER GRID
   ========================================================= */

[data-testid="stHorizontalBlock"] {
    width: 100%;
    column-gap: 24px !important;
    row-gap: 0 !important;
}


/* Streamlit columns */

[data-testid="column"] {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: flex-start;

    min-width: 0;
}

/* =========================================================
   PARAMETER CARDS
   ========================================================= */
   
.advanced-parameter-card {
    width: 100%;
    min-height: 200px;
    box-sizing: border-box;

    padding: 24px;
    margin:  0 0 24px 0;

    background: linear-gradient(
        145deg,
        #121a2b,
        #0e1525
    );

    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 24px;

    display: flex;
    flex-direction: column;

    overflow: hidden;
    transition:
        transform 0.2s ease,
        border-color 0.2s ease,
        box-shadow 0.2s ease;
}

.advanced-parameter-card:hover {
    transform: translateY(-4px);
    border-color:
        rgba(96,165,250,0.25);
    box-shadow:
        0 18px 40px rgba(0,0,0,0.25);
}

.parameter-card-top {
    width: 100%;

    display: flex;
    align-items: center;
    justify-content: space-between;

    margin-bottom: 24px;
}

.parameter-card-icon {
    width: 48px;
    height: 48px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: #16233b;
    border: 1px solid rgba(70, 130, 220, 0.25);
    border-radius: 14px;

    font-size: 24px;
}


.parameter-status {
    padding: 7px 13px;

    border-radius: 20px;

    font-size: 11px;
    font-weight: 700;

    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap
}

.status-low {
    color: #93c5fd;
    background:
        rgba(59,130,246,0.08);
    border:
        1px solid rgba(59,130,246,0.15);
}

.status-normal {
    color: #86efac;
    background:
        rgba(34,197,94,0.08);
    border:
        1px solid rgba(34,197,94,0.15);
}

.status-moderate {
    color: #fde68a;
    background:
        rgba(245,158,11,0.08);
    border:
        1px solid rgba(245,158,11,0.15);
}

.status-elevated {
    color: #fca5a5;
    background:
        rgba(239,68,68,0.08);
    border:
        1px solid rgba(239,68,68,0.15);
}

.parameter-card-name {
    margin-bottom: 10px;

    color: #959eae;

    font-size: 13px;
    font-weight: 600;
}


.parameter-card-value {
    color: #f5f7fb;

    font-size: 30px;
    font-weight: 800;

    line-height: 1.2;

    margin-bottom: 25px;
}

.parameter-card-unit {
    margin-left: 5px;

    color: #748194;

    font-size: 12px;
    font-weight: 500;
}


.parameter-progress {
    width: 100%;
    height: 8px;

    margin-top: auto;
    margin-bottom: 12px;

    background: #202a3d;

    border-radius: 10px;

    overflow: hidden;
}

.parameter-progress-fill {
    height: 100%;

    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );

    border-radius: 10px;

    transition: width 0.4s ease;
}
.parameter-card-footer {
    width: 100%;

    display: flex;
    align-items: center;
    justify-content: space-between;

    color: #7c7f84;

    font-size: 12px;
}

/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 1100px) {

    [data-testid="stHorizontalBlock"] {
        column-gap: 18px !important;
    }

}


@media (max-width: 800px) {

    [data-testid="stHorizontalBlock"] {
        column-gap: 14px !important;
    }

}


@media (max-width: 600px) {

    .advanced-parameter-card {
        min-height: 280px;
        padding: 20px;
        margin-bottom: 18px;
    }

}

/* =========================================================
   DASHBOARD STATUS FOOTER
   ========================================================= */

.clinical-dashboard-footer {
    width: 100%;
    min-height: 72px;

    display: flex;
    align-items: center;
    justify-content: space-between;

    box-sizing: border-box;

    margin-top: 18px;
    margin-bottom: 28px;

    padding: 16px 22px;

    border-radius: 18px;

    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.95),
            rgba(17, 24, 39, 0.82)
        );

    border: 1px solid rgba(148, 163, 184, 0.14);

    box-shadow:
        0 15px 45px rgba(0, 0, 0, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);

    position: relative;
    overflow: hidden;
}


/* Subtle background glow */

.clinical-dashboard-footer::before {
    content: "";

    position: absolute;

    width: 180px;
    height: 180px;

    right: -80px;
    top: -100px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(99, 102, 241, 0.10),
            transparent 70%
        );

    pointer-events: none;
}


/* Footer items */

.clinical-dashboard-footer .footer-item {
    display: flex;
    align-items: center;

    gap: 8px;

    position: relative;
    z-index: 2;

    color: #b2b4b6;

    font-size: 13px;
    font-weight: 500;

    white-space: nowrap;
}


/* Footer icon */

.clinical-dashboard-footer .footer-icon {
    width: 30px;
    height: 30px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 9px;

    background:
        linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.12),
            rgba(124, 58, 237, 0.12)
        );

    border: 1px solid rgba(96, 165, 250, 0.15);

    font-size: 14px;
}


/* Bold number */

.clinical-dashboard-footer b {
    color: #cbd5e1;
    font-weight: 700;
}


/* Mobile */

@media (max-width: 700px) {

    .clinical-dashboard-footer {
        flex-direction: column;
        align-items: flex-start;

        gap: 12px;

        padding: 16px 18px;
    }

    .clinical-dashboard-footer .footer-item {
        white-space: normal;
    }

}
@media (max-width: 700px) {

    .clinical-dashboard-header {
        align-items: flex-start;
         padding: 16px;
    }
    .clinical-dashboard-heading {
        font-size: 18px;
    }

    .clinical-dashboard-subtitle {
        font-size: 11px;
    }

    .live-analysis {
        display: none;
    }

    .clinical-dashboard-footer {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }
}

@media (max-width: 520px) {

    .clinical-dashboard-header {
        flex-direction: column;
        align-items: flex-start;
    }

    .live-analysis {
        align-self: flex-start;
    }

}



/* Streamlit input */

div[data-baseweb="input"] {
    background: #171d2a !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
}


/* Buttons */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 10px 30px rgba(37,99,235,0.25);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 35px rgba(37,99,235,0.35);
}


/* Prediction card */

.result-card {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        rgba(15,23,42,0.95),
        rgba(30,41,59,0.85)
    );
    border: 1px solid rgba(255,255,255,0.10);
    text-align: center;
    margin-top: 50px;
}

.risk-number {
    font-size: 52px;
    font-weight: 800;
    margin: 10px 0;
}

.risk-label {
    color: #94a3b8;
    font-size: 14px;
}


/* Metric cards */

.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(20,26,39,0.85);
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-title {
    color: #94a3b8;
    font-size: 13px;
}

.metric-value {
    font-size: 25px;
    font-weight: 700;
    margin-top: 5px;
}


/* Footer */

.footer {
    text-align: center;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #64748b;
    font-size: 12px;
}


/* Disclaimer */

.disclaimer {
    padding: 15px 18px;
    border-radius: 14px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.20);
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "Model",
    "diabetes_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "Model",
    "scaler.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.markdown(
        """
        <div style="
            padding:15px;
            border-radius:15px;
            background:rgba(59,130,246,0.08);
            border:1px solid rgba(59,130,246,0.15);
        ">
        <b>AI Risk Assessment</b><br>
        <span style="color:#94a3b8;font-size:13px;">
        Machine Learning powered diabetes risk estimation.
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("**Algorithm**")
    st.caption("Random Forest Classifier")

    st.write("**Model Accuracy**")
    st.caption("82%")

    st.write("**Input Features**")
    st.caption("8 clinical parameters")

    st.markdown("---")

    st.markdown("### 📌 Features")

    st.markdown("""
    - Pregnancy history
    - Glucose level
    - Blood pressure
    - Skin thickness
    - Insulin level
    - BMI
    - Diabetes pedigree
    - Age
    """)

    st.markdown("---")

    st.caption("AI Diabetes Risk Prediction")
    st.caption("Educational / Research Project")


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="hero">'
    
    '<div class="hero-content">'
    
    '<div class="badge">'
    '<span class="badge-dot"></span>'
    'AI-POWERED HEALTH ANALYTICS'
    '</div>'
    
    '<div class="hero-title">'
    'Diabetes Risk '
    '<span>Prediction</span>'
    '</div>'
    
    '<div class="hero-subtitle">'
    'Analyze clinical health parameters with a '
    'machine learning model and receive an '
    'instant diabetes risk assessment.'
    '</div>'
    
    '<div class="hero-meta">'
    
    '<div class="hero-meta-item">'
    '🤖 Random Forest'
    '</div>'
    
    '<div class="hero-meta-item">'
    '⚡ Instant Prediction'
    '</div>'
    
    '<div class="hero-meta-item">'
    '🔒 Secure Analysis'
    '</div>'
    
    '</div>'
    
    '</div>'
    
    
    '<div class="hero-visual">'
    
    '<div class="hero-orb">'
    
    '<div class="hero-orb-icon">'
    '🧬'
    '</div>'
    
    '</div>'
    
    '</div>'
    
    '</div>',
    
    unsafe_allow_html=True
)

# =========================================================
# INPUT SECTION
# =========================================================

# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-header">'
    '<div class="section-header-left">'
    '<div class="section-icon">'
    '👤'
    '</div>'
    '<div>'
    '<div class="section-title">'
    'Patient Information'
    '</div>'
    '<div class="section-description">'
    'Enter basic patient details for the assessment'
    '</div>'
    '</div>'
    '</div>'
    '<div class="section-status">'
    '<span class="section-status-dot"></span>'
    'Ready'
    '</div>'
    '</div>',
    
    unsafe_allow_html=True
)

# Patient details
patient_col1, patient_col2 = st.columns(
    2,
    gap="large"
)

with patient_col1:

    patient_name = st.text_input(
        "👤 Patient Name",
        placeholder="Enter patient name"
    )

with patient_col2:

    prediction_date = st.date_input(
        "📅 Assessment Date"
    )


# =========================================================
# CLINICAL PARAMETERS
# =========================================================

st.markdown(
'<div class="section-header">'
'<div class="section-header-left">'
'<div class="section-icon">'
'🧪'
'</div>'
'<div>'
'<div class="section-title">'
'Clinical Parameters'
'</div>'
'<div class="section-description">'
'Provide the clinical measurements required for risk analysis'
'</div>'
'</div>'
'</div>'
'<div class="section-status">'
'<span class="section-status-dot"></span>'
'8 Parameters'
'</div>'
'</div>',
unsafe_allow_html=True
)


# CREATE TWO COLUMNS
col1, col2 = st.columns(
2,
gap="large"
)

# =========================================================
# LEFT INPUTS
# =========================================================
with col1:

    pregnancies = st.number_input(
        "🤰 Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "🧪 Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    bp = st.number_input(
        "❤️ Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=80.0,
        step=1.0
    )

    skin = st.number_input(
        "📏 Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

# RIGHT INPUTS
with col2:

    insulin = st.number_input(
        "💉 Insulin Level",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# ADVANCED CLINICAL INTELLIGENCE DASHBOARD
st.markdown(
    '<div class="clinical-dashboard-header">'
    '<div class="clinical-dashboard-title">'
    '<div class="clinical-dashboard-icon">'
    '🧬'
    '</div>'
    '<div>'
    '<div class="clinical-dashboard-heading">'
    'Clinical Intelligence Dashboard'
    '</div>'
    '<div class="clinical-dashboard-subtitle">'
    'Real-time visualization of entered clinical parameters'
    '</div>'
    '</div>'
    '</div>'
    '<div class="live-analysis">'
    '<span class="live-dot"></span>'
    'LIVE ANALYSIS'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CLINICAL DATA
# =========================================================

clinical_data = {

    "Pregnancies": pregnancies,

    "Glucose": glucose,

    "Blood Pressure": bp,

    "Skin Thickness": skin,

    "Insulin": insulin,

    "BMI": bmi,

    "Diabetes Pedigree": dpf,

    "Age": age

}


# =========================================================
# NORMALIZATION FOR VISUALIZATION
# =========================================================

visual_values = {

    "Pregnancies":
        min((pregnancies / 20) * 100, 100),

    "Glucose":
        min((glucose / 300) * 100, 100),

    "Blood Pressure":
        min((bp / 200) * 100, 100),

    "Skin Thickness":
        min((skin / 100) * 100, 100),

    "Insulin":
        min((insulin / 900) * 100, 100),

    "BMI":
        min((bmi / 70) * 100, 100),

    "Diabetes Pedigree":
        min((dpf / 3) * 100, 100),

    "Age":
        min((age / 120) * 100, 100)

}


# =========================================================
# PROFILE INDEX
# =========================================================

clinical_profile_index = (
    np.mean(
        list(visual_values.values())
    )
)

clinical_profile_index = max(
    0,
    min(
        100,
        clinical_profile_index
    )
)


# =========================================================
# TOP DASHBOARD
# =========================================================

profile_col, radar_col = st.columns(
    [1, 1.65],
    gap="medium"
)
# =========================================================
# PROFILE INDEX CARD
# =========================================================

with profile_col:

    st.markdown(
        '<div class="profile-index-card">'
        '<div class="profile-index-header">'
        '<span>'
        'CLINICAL PROFILE INDEX'
        '</span>'
        '<span>'
        '📊'
        '</span>'
        '</div>'
        '<div class="profile-index-circle">'
        '<div class="profile-index-value">'
        f'{clinical_profile_index:.0f}'
        '</div>'
        '<div class="profile-index-unit">'
        '/ 100'
        '</div>'
        '</div>'
        '<div class="profile-index-title">'
        'Parameter Visualization Index'
        '</div>'
        '<div class="profile-index-description">'
        'A normalized visualization score based '
        'on the entered clinical parameters.'
        '<br><br>'
        '<b>Not a medical diagnosis.</b>'    
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RADAR CHART
# =========================================================

# =========================================================
# RADAR CHART
# =========================================================

with radar_col:

    with st.container(
        height=390,
        key="radar_card"
    ):

        st.markdown(
            '<div class="radar-card-header">'
            '<div class="radar-card-title">'
            'Clinical Parameter Profile'
            '</div>'
            '<div class="radar-card-badge">'
            '8 PARAMETERS'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        radar_labels = [
            "Pregnancy",
            "Glucose",
            "Blood Pressure",
            "Skin",
            "Insulin",
            "BMI",
            "Pedigree",
            "Age"
        ]

        radar_values = [
            visual_values["Pregnancies"],
            visual_values["Glucose"],
            visual_values["Blood Pressure"],
            visual_values["Skin Thickness"],
            visual_values["Insulin"],
            visual_values["BMI"],
            visual_values["Diabetes Pedigree"],
            visual_values["Age"]
        ]

        fig_radar = go.Figure()

        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_labels + [radar_labels[0]],

                fill="toself",

                line=dict(
                    width=2
                ),

                opacity=0.85,

                name="Clinical Profile"
            )
        )

        fig_radar.update_layout(

            height=300,

            margin=dict(
                l=35,
                r=35,
                t=25,
                b=25
            ),

            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",

            showlegend=False,

            font=dict(
                color="#cbd5e1",
                size=10
            ),

            polar=dict(

                bgcolor="rgba(15,23,42,0.15)",

                radialaxis=dict(
                    visible=True,
                    range=[0, 100],

                    gridcolor="rgba(148,163,184,0.14)",
                    linecolor="rgba(148,163,184,0.10)",

                    tickfont=dict(
                        color="#64748b",
                        size=8
                    )
                ),

                angularaxis=dict(
                    gridcolor="rgba(148,163,184,0.12)",
                    linecolor="rgba(148,163,184,0.10)",

                    tickfont=dict(
                        color="#cbd5e1",
                        size=13
                    )
                )
            )
        )

        st.plotly_chart(
            fig_radar,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True
            }
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# =========================================================
# PARAMETER ANALYSIS HEADER
# =========================================================

st.markdown(
    '<div class="section-header">'
    '<div class="section-header-left">'
    
    '<div class="section-icon">'
    '📊'
    '</div>'
    '<div>'
    '<div class="section-title">'
    'Parameter Analysis'
    '</div>'
    '<div class="section-description">'
    'Detailed visualization of clinical parameter values'
    '</div>'
    
    '</div>'
    
    '</div>'
    '<div class="section-status">'
    '<span class="section-status-dot"></span>'
    '8 Parameters'
    '</div>'
    
    '</div>',
    
    unsafe_allow_html=True
)


# =========================================================
# PARAMETER CARD DATA
# =========================================================

parameter_cards = [

    (
        "🤰",
        "Pregnancies",
        pregnancies,
        "count",
        20
    ),

    (
        "🧪",
        "Glucose",
        glucose,
        "mg/dL",
        300
    ),

    (
        "❤️",
        "Blood Pressure",
        bp,
        "mmHg",
        200
    ),

    (
        "📏",
        "Skin Thickness",
        skin,
        "mm",
        100
    ),

    (
        "💉",
        "Insulin",
        insulin,
        "μU/mL",
        900
    ),

    (
        "⚖️",
        "BMI",
        bmi,
        "kg/m²",
        70
    ),

    (
        "🧬",
        "Diabetes Pedigree",
        dpf,
        "index",
        3
    ),

    (
        "🎂",
        "Age",
        age,
        "years",
        120
    )

]


# =========================================================
# PARAMETER CARDS
# =========================================================

parameter_columns = st.columns(
    4,
    gap="medium"
)


for i, (
    icon,
    name,
    value,
    unit,
    max_value
) in enumerate(parameter_cards):

    # -----------------------------------------
    # CALCULATE PERCENTAGE
    # -----------------------------------------

    percentage = min(
        max(
            (float(value) / max_value) * 100,
            0
        ),
        100
    )

    # -----------------------------------------
    # UI STATUS
    # -----------------------------------------

    if percentage < 30:

        status = "LOW"
        status_class = "status-low"

    elif percentage < 50:

        status = "NORMAL"
        status_class = "status-normal"

    elif percentage < 70:

        status = "MODERATE"
        status_class = "status-moderate"

    else:

        status = "ELEVATED"
        status_class = "status-elevated"

    # -----------------------------------------
    # FORMAT VALUE
    # -----------------------------------------

    if name == "BMI":

        display_value = f"{float(value):.1f}"

    elif name == "Diabetes Pedigree":

        display_value = f"{float(value):.2f}"

    else:

        display_value = f"{float(value):.0f}"

    # -----------------------------------------
    # CARD
    # -----------------------------------------

    with parameter_columns[i % 4]:

        st.markdown(
            '<div class="advanced-parameter-card">'

            '<div class="parameter-card-top">'

            '<div class="parameter-card-icon">'
            f'{icon}'
            '</div>'

            f'<div class="parameter-status {status_class}">'
            f'{status}'
            '</div>'

            '</div>'

            '<div class="parameter-card-name">'
            f'{name}'
            '</div>'

            '<div class="parameter-card-value">'
            f'{display_value}'
            '<span class="parameter-card-unit">'
            f'{unit}'
            '</span>'
            '</div>'

            '<div class="parameter-progress">'

            '<div class="parameter-progress-fill" '
            f'style="width: {percentage:.1f}%;">'
            '</div>'

            '</div>'

            '<div class="parameter-card-footer">'
            '<span>Visualization</span>'
            f'<span>{percentage:.0f}%</span>'
            '</div>'

            '</div>',

            unsafe_allow_html=True
        )

# =========================================================
# DASHBOARD FOOTER
# =========================================================

st.markdown(
    '<div class="clinical-dashboard-footer">'

    '<div class="footer-item">'
    '<span class="footer-icon">🧠</span>'
    '<span><b>8</b> clinical parameters visualized</span>'
    '</div>'

    '<div class="footer-item">'
    '<span class="footer-icon">⚡</span>'
    '<span>Ready for machine learning prediction</span>'
    '</div>'

    '<div class="footer-item">'
    '<span class="footer-icon">🔒</span>'
    '<span>Application-level data processing</span>'
    '</div>'

    '</div>',

    unsafe_allow_html=True
)



# =========================================================
# FEATURE OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 Clinical Data Overview</div>',
    unsafe_allow_html=True
)

features = [
    pregnancies,
    glucose,
    bp,
    skin,
    insulin,
    bmi,
    dpf,
    age
]

labels = [
    "Pregnancy",
    "Glucose",
    "Blood Pressure",
    "Skin",
    "Insulin",
    "BMI",
    "DPF",
    "Age"
]

fig = plt.figure(figsize=(12, 4))

plt.bar(
    labels,
    features
)

plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.15)

plt.tight_layout()

st.pyplot(fig, use_container_width=True)

plt.close(fig)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔍 Analyze Diabetes Risk"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = np.array([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    scaled_data = scaler.transform(input_data)

    # -----------------------------------------------------
    # MAKE PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    risk_prob = probability[0][1] * 100

    # -----------------------------------------------------
    # PREDICTION TEXT
    # -----------------------------------------------------

    prediction_text = (
        "HIGH RISK"
        if int(prediction[0]) == 1
        else "LOW RISK"
    )

    prediction_color = (
        "#f87171"
        if int(prediction[0]) == 1
        else "#34d399"
    )

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # METRIC COLUMNS
    # =====================================================

    metric1, metric2, metric3 = st.columns(
        3,
        gap="medium"
    )

    # =====================================================
    # RISK PROBABILITY
    # =====================================================

    with metric1:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">'
            'RISK PROBABILITY'
            '</div>'
            f'<div class="metric-value">'
            f'{risk_prob:.2f}%'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # =====================================================
    # PREDICTION
    # =====================================================

    with metric2:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">'
            'PREDICTION'
            '</div>'
            f'<div class="metric-value" '
            f'style="color:{prediction_color};">'
            f'{prediction_text}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # =====================================================
    # MODEL
    # =====================================================

    with metric3:

        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-title">'
            'MODEL'
            '</div>'
            '<div class="metric-value">'
            'Random Forest'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # =====================================================
    # PROBABILITY BAR
    # =====================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.progress(
        min(max(risk_prob / 100, 0.0), 1.0),
        text=f"Estimated Risk: {risk_prob:.2f}%"
    )

    # =====================================================
    # PDF REPORT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📄 Prediction Report'
        '</div>',
        unsafe_allow_html=True
    )

    # Create PDF buffer
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer
    )

    styles = getSampleStyleSheet()

    elements = []

    # =====================================================
    # PDF TITLE
    # =====================================================

    elements.append(
        Paragraph(
            "<b>Diabetes Risk Prediction Report</b>",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(
            1,
            0.4 * inch
        )
    )

    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

    elements.append(
        Paragraph(
            f"<b>Patient Name:</b> "
            f"{patient_name if patient_name else 'Not Provided'}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Assessment Date:</b> {prediction_date}",
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(
            1,
            0.25 * inch
        )
    )

    # =====================================================
    # CLINICAL INFORMATION
    # =====================================================

    elements.append(
        Paragraph(
            f"Pregnancies: {pregnancies}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Glucose Level: {glucose}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Blood Pressure: {bp}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Skin Thickness: {skin}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Insulin Level: {insulin}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"BMI: {bmi}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Diabetes Pedigree Function: {dpf}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Age: {age}",
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(
            1,
            0.3 * inch
        )
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    elements.append(
        Paragraph(
            f"<b>Prediction:</b> "
            f"{prediction_text}",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Risk Probability:</b> "
            f"{risk_prob:.2f}%",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            "<b>Model:</b> Random Forest",
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(
            1,
            0.3 * inch
        )
    )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    elements.append(
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by an "
            "educational Machine Learning application and should "
            "not be considered a medical diagnosis.",
            styles["Normal"]
        )
    )

    # =====================================================
    # BUILD PDF
    # =====================================================

    doc.build(
        elements
    )

    pdf_buffer.seek(0)

    # =====================================================
    # DOWNLOAD BUTTON
    # =====================================================

    st.download_button(
        label="📥 Download Prediction Report",
        data=pdf_buffer,
        file_name="Diabetes_Prediction_Report.pdf",
        mime="application/pdf"
    )
# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🩺 Diabetes Risk Prediction &nbsp; • &nbsp;
    Machine Learning Healthcare Project

    </div>
    """,
    unsafe_allow_html=True
)
