"""
app.py
------
Main Streamlit entrypoint for the Natural Language → SQL Engine.

Run with:
    streamlit run app.py

Prerequisites:
    1. pip install -r requirements.txt
    2. python seed.py          (creates ecommerce.db)
    3. Copy .env.example → .env and fill in HF_TOKEN
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db import SCHEMA_TEXT, is_safe_query, run_query
from nl_to_sql import DEFAULT_MODEL, generate_sql

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NL → SQL Engine",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ════════════════════════════════════════════════════════
       BASE RESET & DARK THEME
    ════════════════════════════════════════════════════════ */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #0b0f19 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.12) 0px, transparent 50%);
        background-attachment: fixed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background-color: transparent !important;
    }
    [data-testid="block-container"] {
        padding: 2.5rem 3rem 4rem !important;
        max-width: 1100px;
    }

    /* ════════════════════════════════════════════════════════
       SIDEBAR
    ════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: rgba(13, 18, 36, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] > div {
        background: transparent !important;
        padding: 2rem 1.25rem !important;
    }
    [data-testid="stSidebar"] * {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Sidebar expander */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"]:hover {
        background: rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }

    /* ════════════════════════════════════════════════════════
       TYPOGRAPHY
    ════════════════════════════════════════════════════════ */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em !important;
    }

    /* ════════════════════════════════════════════════════════
       HERO HEADER BANNER
    ════════════════════════════════════════════════════════ */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.7) 0%, rgba(49, 46, 129, 0.4) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 3rem 3.5rem;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2), inset 0 0 0 1px rgba(255,255,255,0.05);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 0.75rem 0;
        letter-spacing: -0.04em;
        line-height: 1.1;
        background: linear-gradient(to right, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
        line-height: 1.6;
        max-width: 600px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 30px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #c7d2fe !important;
        margin-bottom: 1.25rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
    }

    /* ════════════════════════════════════════════════════════
       INPUT SECTION CARD
    ════════════════════════════════════════════════════════ */
    .section-card {
        background: rgba(17, 24, 39, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.25rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #818cf8 !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ════════════════════════════════════════════════════════
       STREAMLIT WIDGET OVERRIDES
    ════════════════════════════════════════════════════════ */
    /* Text areas */
    .stTextArea textarea, .stTextInput input {
        background: rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
        caret-color: #818cf8;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #818cf8 !important;
        background: rgba(0, 0, 0, 0.4) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }
    .stTextArea textarea::placeholder {
        color: #475569 !important;
    }
    /* Labels */
    .stTextArea label, .stTextInput label {
        color: #cbd5e1 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* ════════════════════════════════════════════════════════
       BUTTONS
    ════════════════════════════════════════════════════════ */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em;
    }
    /* Primary run button — gradient */
    div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #fff !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 15px rgba(79,70,229,0.3), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    }
    div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(79,70,229,0.5), inset 0 1px 0 rgba(255,255,255,0.3) !important;
        filter: brightness(1.1);
    }
    div[data-testid="stHorizontalBlock"] > div:first-child .stButton > button:active {
        transform: translateY(0) !important;
    }
    /* Secondary clear button */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        padding: 0.75rem 1.2rem !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
    }
    /* Run SQL / example buttons */
    .stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        color: #a5b4fc !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }

    /* ════════════════════════════════════════════════════════
       DIVIDER
    ════════════════════════════════════════════════════════ */
    hr { border-color: #1e293b !important; }
    [data-testid="stHorizontalRule"] hr { border-color: #1e293b !important; }

    /* ════════════════════════════════════════════════════════
       SQL SECTION
    ════════════════════════════════════════════════════════ */
    .sql-section {
        background: rgba(17, 24, 39, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 2rem 0 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .sql-section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1.25rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #818cf8 !important;
    }
    .sql-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #818cf8;
        box-shadow: 0 0 10px #818cf8;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(129, 140, 248, 0.4); }
        70% { box-shadow: 0 0 0 6px rgba(129, 140, 248, 0); }
        100% { box-shadow: 0 0 0 0 rgba(129, 140, 248, 0); }
    }
    /* SQL textarea gets monospace */
    .sql-section .stTextArea textarea {
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
        background: rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #60a5fa !important;
        padding: 1.25rem !important;
    }

    /* ════════════════════════════════════════════════════════
       RESULTS SECTION
    ════════════════════════════════════════════════════════ */
    .results-section {
        background: rgba(17, 24, 39, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .results-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
    }
    .results-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #2dd4bf !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .cyan-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #2dd4bf;
        box-shadow: 0 0 10px #2dd4bf;
        display: inline-block;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stDataFrame"] iframe {
        border-radius: 12px !important;
    }

    /* ════════════════════════════════════════════════════════
       ROW COUNT PILL
    ════════════════════════════════════════════════════════ */
    .row-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(34,211,238,0.1);
        color: #22d3ee !important;
        border: 1px solid rgba(34,211,238,0.25);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* ════════════════════════════════════════════════════════
       ALERTS / MESSAGES
    ════════════════════════════════════════════════════════ */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        font-size: 0.88rem !important;
    }
    /* error */
    [data-testid="stAlert"][data-baseweb="notification"][kind="error"],
    .stError { border-left: 4px solid #f87171 !important; }
    /* warning */
    [data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
    .stWarning { border-left: 4px solid #fbbf24 !important; }
    /* info */
    [data-testid="stAlert"][data-baseweb="notification"][kind="info"],
    .stInfo { border-left: 4px solid #22d3ee !important; }

    /* ════════════════════════════════════════════════════════
       EXPANDERS (main area)
    ════════════════════════════════════════════════════════ */
    [data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.75rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover { color: #c7d2fe !important; }
    [data-testid="stExpander"] summary svg { fill: #6366f1 !important; }

    /* ════════════════════════════════════════════════════════
       CODE BLOCKS
    ════════════════════════════════════════════════════════ */
    .stCode, [data-testid="stCode"] {
        border-radius: 10px !important;
        border: 1px solid #21262d !important;
    }
    .stCode code {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    /* ════════════════════════════════════════════════════════
       SCROLLBAR
    ════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #6366f1; }

    /* ════════════════════════════════════════════════════════
       SPINNER
    ════════════════════════════════════════════════════════ */
    [data-testid="stSpinner"] { color: #6366f1 !important; }

    /* ════════════════════════════════════════════════════════
       DOWNLOAD BUTTON
    ════════════════════════════════════════════════════════ */
    [data-testid="stDownloadButton"] > button {
        background: rgba(34,211,238,0.1) !important;
        color: #22d3ee !important;
        border: 1px solid rgba(34,211,238,0.3) !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        padding: 0.35rem 1rem !important;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(34,211,238,0.2) !important;
        border-color: #22d3ee !important;
    }

    /* ════════════════════════════════════════════════════════
       SIDEBAR HISTORY ITEMS
    ════════════════════════════════════════════════════════ */
    .hist-ts {
        font-size: 0.68rem;
        color: #475569;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }
    .hist-nl {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 6px;
        line-height: 1.4;
    }

    /* ════════════════════════════════════════════════════════
       EXAMPLE QUERY PILLS (inside expander)
    ════════════════════════════════════════════════════════ */
    .ex-pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
    .ex-pill {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: #a5b4fc !important;
        cursor: pointer;
        transition: all 0.15s;
    }
    .ex-pill:hover {
        background: rgba(99,102,241,0.25);
        border-color: rgba(99,102,241,0.5);
    }

    /* ════════════════════════════════════════════════════════
       MISC FIXES
    ════════════════════════════════════════════════════════ */
    /* Remove ugly red "Required" asterisk colour */
    span[data-testid="InputInstructions"] { display: none !important; }
    /* Captions */
    [data-testid="stCaptionContainer"] p { color: #475569 !important; font-size: 0.75rem !important; }
    /* Spinner text */
    [data-testid="stSpinner"] p { color: #94a3b8 !important; }
    /* Divider in sidebar */
    [data-testid="stSidebar"] hr { border-color: rgba(99,102,241,0.15) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "history":       [],
        "current_nl":    "",
        "current_sql":   "",
        "current_df":    None,
        "current_error": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ── Helper: check DB exists ───────────────────────────────────────────────────

_DB_PATH = Path(__file__).parent / "ecommerce.db"


def _db_exists() -> bool:
    return _DB_PATH.exists() and _DB_PATH.stat().st_size > 0


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="margin-bottom:1.2rem;">
              <div style="font-size:1.35rem;font-weight:800;color:#a5b4fc;letter-spacing:-0.02em;">
                🗄️ NL → SQL
              </div>
              <div style="font-size:0.72rem;color:#475569;margin-top:2px;font-family:'JetBrains Mono',monospace;">
                E-Commerce Intelligence Engine
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── DB status ──────────────────────────────────────────
        if _db_exists():
            st.success("ecommerce.db connected", icon="✅")
        else:
            st.error("DB not found — run `python seed.py`", icon="🚫")

        st.markdown(
            f'<div style="font-size:0.72rem;color:#475569;margin:6px 0 16px;'
            f'font-family:\'JetBrains Mono\',monospace;">model: {DEFAULT_MODEL.split("/")[-1]}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Schema explorer ────────────────────────────────────
        with st.expander("📋  Schema", expanded=False):
            st.code(SCHEMA_TEXT, language="text")

        st.divider()

        # ── Query history ──────────────────────────────────────
        st.markdown(
            '<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;'
            'text-transform:uppercase;color:#6366f1;margin-bottom:0.75rem;">🕑 History</div>',
            unsafe_allow_html=True,
        )

        history: list[dict] = st.session_state["history"]

        if not history:
            st.markdown(
                '<div style="font-size:0.78rem;color:#334155;text-align:center;'
                'padding:1rem 0;">No queries yet</div>',
                unsafe_allow_html=True,
            )
        else:
            for i, entry in enumerate(reversed(history)):
                idx = len(history) - 1 - i
                ts_str = entry["timestamp"].strftime("%H:%M:%S")
                nl_short = entry["nl"][:42] + ("…" if len(entry["nl"]) > 42 else "")
                row_count = (
                    len(entry["df"])
                    if isinstance(entry["df"], pd.DataFrame) and not entry["df"].empty
                    else 0
                )

                label = f"#{len(history)-i} · {nl_short}"
                with st.expander(label, expanded=False):
                    st.markdown(
                        f'<div class="hist-ts">🕐 {ts_str} · {row_count} rows</div>'
                        f'<div class="hist-nl">{entry["nl"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.code(entry["sql"], language="sql")
                    if st.button("↩ Reload", key=f"reload_{idx}", use_container_width=True):
                        st.session_state["current_nl"]    = entry["nl"]
                        st.session_state["current_sql"]   = entry["sql"]
                        st.session_state["current_df"]    = entry["df"]
                        st.session_state["current_error"] = None
                        st.rerun()


# ── Results section ───────────────────────────────────────────────────────────

def _render_results(sql: str, df: "pd.DataFrame | None", error: str | None) -> None:

    # ── Generated SQL card ──────────────────────────────────
    st.markdown('<div class="sql-section">', unsafe_allow_html=True)
    st.markdown(
        '<div class="sql-section-header">'
        '<span class="sql-dot"></span> Generated SQL &nbsp;'
        '<span style="color:#334155;font-weight:400;text-transform:none;letter-spacing:0;">'
        '— edit and re-run below</span></div>',
        unsafe_allow_html=True,
    )

    edited_sql = st.text_area(
        label="SQL Editor",
        value=sql,
        height=130,
        key="sql_editor",
        label_visibility="collapsed",
    )
    st.session_state["current_sql"] = edited_sql

    c1, c2 = st.columns([1, 5])
    with c1:
        run_edited = st.button("▶  Run SQL", key="run_edited_sql", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if run_edited:
        ok, reason = is_safe_query(edited_sql)
        if not ok:
            st.session_state["current_df"]    = None
            st.session_state["current_error"] = f"Only read queries are permitted. {reason}"
        else:
            with st.spinner("🔍  Executing query…"):
                new_df, err = run_query(edited_sql)
            st.session_state["current_df"]    = new_df
            st.session_state["current_error"] = err
            
            if not err:
                st.session_state["history"].append(
                    {
                        "nl":        st.session_state.get("current_nl", ""),
                        "sql":       edited_sql,
                        "df":        new_df,
                        "timestamp": datetime.now(),
                    }
                )
        st.session_state["current_sql"] = edited_sql
        st.rerun()

    # ── Error ───────────────────────────────────────────────
    if st.session_state.get("current_error"):
        st.error(f"⚠️  {st.session_state['current_error']}")
        return

    if st.session_state.get("current_df") is None:
        return

    # ── Results card ────────────────────────────────────────
    st.markdown('<div class="results-section">', unsafe_allow_html=True)

    col_title, col_export = st.columns([3, 1])
    with col_title:
        st.markdown(
            f'<div class="results-title">'
            f'<span class="cyan-dot"></span> Results'
            f'&nbsp;&nbsp;<span class="row-pill">{len(df)} row{"s" if len(df)!=1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col_export:
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Export CSV",
                data=csv_bytes,
                file_name="query_results.csv",
                mime="text/csv",
                key="csv_download",
            )

    if df.empty:
        st.info("Query ran successfully, but returned no results.", icon="ℹ️")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _render_sidebar()

    # ── Hero banner ─────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
          <div class="hero-badge">⚡ AI-Powered · SQLite · Qwen 2.5</div>
          <div class="hero-title">Natural Language → SQL Engine</div>
          <div class="hero-subtitle">
            Ask questions about the e-commerce database in plain English.<br>
            The AI translates your intent into SQL — you see every query before it runs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _db_exists():
        st.error(
            "**Database not found.** Run `python seed.py` to create and seed `ecommerce.db`.",
            icon="🚫",
        )
        st.stop()

    # ── Input card ──────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-label">💬 Ask your question</div>',
        unsafe_allow_html=True,
    )

    # Example queries
    example_queries = [
        "Top 10 customers by total spending",
        "Electronics products priced above $100",
        "Orders placed each month this year",
        "Products that have never been ordered",
        "Customers who signed up in the last 6 months",
        "Average order value by product category",
    ]

    with st.expander("💡 Try an example query", expanded=False):
        cols = st.columns(2)
        for i, ex in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"→ {ex}", key=f"ex_{i}", use_container_width=True):
                    st.session_state["current_nl"]    = ex
                    st.session_state["current_sql"]   = ""
                    st.session_state["current_df"]    = None
                    st.session_state["current_error"] = None
                    st.rerun()

    nl_input: str = st.text_area(
        label="Describe the data you need:",
        value=st.session_state.get("current_nl", ""),
        height=90,
        placeholder="e.g.  Show me the top 5 customers by total revenue, broken down by city",
        key="nl_input_area",
        label_visibility="collapsed",
    )

    col_run, col_clear, col_pad = st.columns([2, 2, 5])
    with col_run:
        run_clicked = st.button("🤖  Generate SQL", key="run_query_btn", use_container_width=True)
    with col_clear:
        if st.button("✕  Clear", key="clear_btn", use_container_width=True):
            st.session_state["current_nl"]    = ""
            st.session_state["current_sql"]   = ""
            st.session_state["current_df"]    = None
            st.session_state["current_error"] = None
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Validation + generation ─────────────────────────────
    if run_clicked:
        if not nl_input or len(nl_input.strip()) < 4:
            st.warning("Please describe the data you need (at least 4 characters).", icon="⚠️")
        else:
            st.session_state["current_nl"]    = nl_input.strip()
            st.session_state["current_sql"]   = ""
            st.session_state["current_df"]    = None
            st.session_state["current_error"] = None

            with st.spinner("🤖  Generating SQL with AI…"):
                try:
                    sql = generate_sql(nl_input.strip())
                    st.session_state["current_sql"] = sql
                except ValueError as exc:
                    st.session_state["current_error"] = str(exc)
                    sql = None

            if sql:
                st.rerun()

    # ── Render results ──────────────────────────────────────
    current_sql   = st.session_state.get("current_sql", "")
    current_df    = st.session_state.get("current_df")
    current_error = st.session_state.get("current_error")

    if current_sql or current_error:
        _render_results(current_sql, current_df, current_error)


if __name__ == "__main__":
    main()
