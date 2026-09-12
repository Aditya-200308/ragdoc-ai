# ============================================================
# FILE: src/app.py
# PURPOSE: The web interface — a beautiful Streamlit app
#          that lets users upload documents and ask questions.
#
# HOW TO RUN:
#   streamlit run src/app.py
# ============================================================

import os
import sys
import types
import tempfile
import json
import time

# Ensure root directory and src directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.abspath(os.path.dirname(__file__))

for d in [root_dir, src_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Virtual module mapping for Streamlit Cloud
if "src" not in sys.modules:
    m = types.ModuleType("src")
    m.__path__ = [src_dir]
    sys.modules["src"] = m

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from src.rag_chain import RAGChain
from src.evaluator import Evaluator

# Load environment variables
load_dotenv()


def play_chime():
    """Plays a pleasant synthetic audio chime notification when AI finishes generating an answer."""
    sound_js = """
    <script>
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const osc1 = audioCtx.createOscillator();
        const osc2 = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc1.type = 'sine';
        osc2.type = 'sine';

        osc1.frequency.setValueAtTime(523.25, audioCtx.currentTime);
        osc1.frequency.exponentialRampToValueAtTime(659.25, audioCtx.currentTime + 0.1);

        osc2.frequency.setValueAtTime(659.25, audioCtx.currentTime + 0.1);
        osc2.frequency.exponentialRampToValueAtTime(783.99, audioCtx.currentTime + 0.25);

        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.4);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(audioCtx.destination);

        osc1.start();
        osc2.start(audioCtx.currentTime + 0.1);
        osc1.stop(audioCtx.currentTime + 0.2);
        osc2.stop(audioCtx.currentTime + 0.4);
    } catch(e) {}
    </script>
    """
    components.html(sound_js, height=0, width=0)

# ================================================================
# PAGE CONFIGURATION
# ================================================================
st.set_page_config(
    page_title="RAGDoc AI | Smart Document Q&A & Benchmarking",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CUSTOM CSS — Premium Glassmorphism & Modern Dark Theme
# ================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Fira+Code:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* ─── STREAMLIT HEADER ─── */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* ─── MAIN APP CANVAS: Deep Indigo / Cosmic Violet Gradient ─── */
    .stApp, [data-testid="stAppViewContainer"] {
        background: 
            radial-gradient(ellipse 85% 55% at 50% -12%, rgba(147, 51, 234, 0.32) 0%, transparent 65%),
            radial-gradient(circle at 8% 22%, rgba(99, 102, 241, 0.24) 0%, transparent 42%),
            radial-gradient(circle at 92% 78%, rgba(236, 72, 153, 0.22) 0%, transparent 45%),
            radial-gradient(circle at 50% 92%, rgba(14, 165, 233, 0.18) 0%, transparent 50%),
            linear-gradient(165deg, #070314 0%, #0d0622 25%, #130a33 55%, #060210 100%) !important;
        color: #f8fafc !important;
    }

    /* Main container padding */
    .main .block-container {
        padding: 1.8rem 3rem 3.5rem 3rem;
        max-width: 1280px;
    }

    /* ─── TOP UTILITY RIBBON ─── */
    .top-utility-bar {
        background: linear-gradient(90deg, rgba(26, 14, 62, 0.88) 0%, rgba(42, 20, 95, 0.88) 50%, rgba(18, 10, 48, 0.88) 100%) !important;
        border: 1px solid rgba(168, 85, 247, 0.4) !important;
        border-radius: 14px;
        padding: 0.65rem 1.4rem;
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(14px);
    }

    .utility-badge {
        display: flex;
        align-items: center;
        gap: 0.48rem;
        font-size: 0.88rem;
        color: #e2e8f0;
        font-weight: 600;
    }

    /* ─── SIDEBAR: Rich Royal Violet with Glowing Border ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c051e 0%, #150838 40%, #090317 100%) !important;
        border-right: 1.5px solid rgba(168, 85, 247, 0.35) !important;
        box-shadow: 6px 0 32px rgba(0, 0, 0, 0.65) !important;
    }

    [data-testid="stSidebar"] * {
        color: #ede9fe !important;
    }

    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #c4b5fd !important;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(168, 85, 247, 0.25) !important;
    }

    /* ─── HERO HEADER BANNER: Vibrant Glassmorphic Card ─── */
    .hero-header {
        border-radius: 24px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, rgba(42, 22, 98, 0.78) 0%, rgba(68, 28, 130, 0.62) 50%, rgba(24, 12, 60, 0.85) 100%) !important;
        border: 1.5px solid rgba(192, 132, 252, 0.45) !important;
        box-shadow: 0 20px 50px -10px rgba(147, 51, 234, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.25) !important;
        backdrop-filter: blur(18px);
    }

    .hero-title {
        background: linear-gradient(135deg, #f0abfc 0%, #c084fc 25%, #818cf8 50%, #38bdf8 75%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        color: #cbd5e1 !important;
        font-size: 1.15rem;
        line-height: 1.75;
    }

    .hero-subtitle strong {
        color: #f8fafc !important;
        font-weight: 700;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.28) 0%, rgba(99, 102, 241, 0.28) 100%);
        border: 1.2px solid rgba(192, 132, 252, 0.55);
        color: #f5d0fe;
        padding: 0.35rem 0.95rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.25);
    }

    /* ─── CARD STRUCTURES: Polished Glassmorphism ─── */
    .rag-card, .metric-card {
        background: linear-gradient(145deg, rgba(32, 18, 75, 0.78) 0%, rgba(20, 12, 50, 0.88) 100%) !important;
        border: 1.5px solid rgba(168, 85, 247, 0.35) !important;
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(14px);
        transition: all 0.25s ease;
    }

    .rag-card:hover, .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(192, 132, 252, 0.6) !important;
        box-shadow: 0 14px 40px rgba(147, 51, 234, 0.35) !important;
    }

    /* ─── ANSWER CARD: Vivid Glowing Purple/Indigo Shell ─── */
    .answer-card {
        background: linear-gradient(145deg, rgba(36, 20, 88, 0.88) 0%, rgba(22, 13, 60, 0.95) 100%) !important;
        border: 1.5px solid rgba(192, 132, 252, 0.48) !important;
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin: 1.2rem 0;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45), 0 0 28px rgba(168, 85, 247, 0.22) !important;
        backdrop-filter: blur(14px);
        font-size: 1.15rem;
        line-height: 1.85;
        letter-spacing: 0.01em;
        color: #f8fafc !important;
    }

    .answer-card p, .answer-card div, .answer-card span, .answer-card li {
        color: #f8fafc !important;
        font-size: 1.15rem !important;
        line-height: 1.85 !important;
    }

    /* ─── SOURCE CARD: Neon Cyan Glow & Crisp Citations ─── */
    .source-card {
        background: linear-gradient(145deg, rgba(15, 28, 70, 0.82) 0%, rgba(10, 18, 50, 0.92) 100%) !important;
        border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
        border-left: 5px solid #38bdf8 !important;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
        font-size: 0.98rem;
        line-height: 1.65;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(10px);
        color: #f1f5f9 !important;
    }

    .source-card div, .source-card p, .source-card span {
        color: #cbd5e1 !important;
        font-size: 0.98rem !important;
    }

    /* ─── ARCHITECTURE FLOW CARDS (Tab 3) ─── */
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }

    @media (max-width: 1100px) {
        .flow-grid {
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        }
    }

    .flow-step {
        background: linear-gradient(145deg, rgba(32, 18, 75, 0.78) 0%, rgba(20, 12, 50, 0.88) 100%) !important;
        border: 1.5px solid rgba(147, 51, 234, 0.35) !important;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .flow-step:hover {
        transform: translateY(-4px);
        border-color: rgba(192, 132, 252, 0.65) !important;
        box-shadow: 0 14px 34px rgba(147, 51, 234, 0.35) !important;
    }

    .flow-num {
        background: linear-gradient(135deg, #c084fc 0%, #6366f1 100%);
        color: white !important;
        width: 34px;
        height: 34px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.92rem;
        margin: 0 auto 0.7rem auto;
        box-shadow: 0 4px 14px rgba(168, 85, 247, 0.5);
    }

    .flow-title {
        font-weight: 800;
        font-size: 1.02rem;
        color: #f8fafc !important;
        margin-bottom: 0.4rem;
    }

    .flow-desc {
        font-size: 0.86rem;
        line-height: 1.55;
        color: #cbd5e1 !important;
    }

    /* ─── STATUS BADGES: Radiant Neon Pills ─── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        border-radius: 14px;
        padding: 0.5rem 1.1rem;
        font-size: 0.92rem;
        font-weight: 700;
    }

    .doc-pill {
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(167, 139, 250, 0.5);
        border-radius: 8px;
        padding: 0.2rem 0.65rem;
        font-size: 0.82rem;
        font-weight: 700;
        font-family: 'Fira Code', monospace;
    }

    /* ─── STREAMLIT BUTTON STYLING ─── */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #06b6d4 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.8rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 6px 24px rgba(168, 85, 247, 0.45) !important;
        text-transform: none !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px rgba(168, 85, 247, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
    }

    /* ─── TABS STYLING ─── */
    [data-testid="stTabs"] {
        border-bottom: 2px solid rgba(168, 85, 247, 0.25) !important;
        margin-bottom: 1.6rem !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.2rem !important;
        background: transparent !important;
    }

    button[data-baseweb="tab"] {
        color: #a5b4fc !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.75rem 1.4rem !important;
        border: none !important;
        border-bottom: 3.5px solid transparent !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #f8fafc !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 3.5px solid #c084fc !important;
        font-weight: 800 !important;
    }

    /* ─── FILE UPLOADER & CHAT INPUT ─── */
    [data-testid="stFileUploader"] {
        background: rgba(26, 15, 60, 0.65) !important;
        border: 1.5px dashed rgba(168, 85, 247, 0.5) !important;
        border-radius: 16px !important;
        padding: 1.2rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(192, 132, 252, 0.85) !important;
        box-shadow: 0 0 24px rgba(168, 85, 247, 0.3) !important;
    }

    [data-testid="stChatInput"] {
        border-radius: 16px !important;
        border: 1.5px solid rgba(168, 85, 247, 0.45) !important;
        background: rgba(22, 12, 54, 0.85) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #c084fc !important;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.5) !important;
    }

    /* ─── DEDICATED SCROLLABLE CHAT CONTAINER (ChatGPT/Gemini Style) ─── */
    [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatMessage"]) {
        border-radius: 20px !important;
        border: 1.5px solid rgba(168, 85, 247, 0.4) !important;
        background: linear-gradient(180deg, rgba(16, 9, 40, 0.82) 0%, rgba(22, 12, 54, 0.9) 100%) !important;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(16px) !important;
        padding: 0.8rem 1rem !important;
    }

    /* Modern slim glowing scrollbar for the scrollable container */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar,
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar {
        width: 7px !important;
        height: 7px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-track,
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar-track {
        background: rgba(14, 8, 32, 0.5) !important;
        border-radius: 10px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb,
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #a855f7 0%, #6366f1 100%) !important;
        border-radius: 10px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover,
    div[data-testid="stVerticalBlock"]::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #c084fc 0%, #818cf8 100%) !important;
    }

    /* ChatGPT/Gemini Chat Message Spacing & Clean Container */
    [data-testid="stChatMessage"] {
        padding: 0.6rem 0.8rem !important;
        margin-bottom: 0.5rem !important;
        border-radius: 16px !important;
        background: transparent !important;
    }

    [data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid rgba(168, 85, 247, 0.28) !important;
        background: rgba(18, 10, 44, 0.65) !important;
        margin-top: 0.6rem !important;
    }

    /* ─── HEADINGS & TEXT ─── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.015em;
    }

    p, span, label, div, .stMarkdown p, .stMarkdown li {
        color: #e2e8f0;
    }

    table, th, td {
        color: #f8fafc !important;
        background-color: rgba(20, 14, 48, 0.6) !important;
        border-color: rgba(139, 92, 246, 0.25) !important;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def initialize_rag(strategy, use_reranker):
    with st.spinner("🔧 Initializing RAG pipeline (loading models)..."):
        rag = RAGChain(
            chunking_strategy=strategy,
            use_reranker=use_reranker
        )
    return rag


def display_sources(sources):
    """Displays retrieved source chunks in styled cards with document citations."""
    if not sources:
        return

    st.markdown("**📄 Retrieved Sources:**")
    for i, source in enumerate(sources):
        score_pct = int(source.get("score", 0) * 100)
        src_name = source.get("source") or source.get("metadata", {}).get("source", "Document")
        st.markdown(f"""
        <div class="source-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <div>
                    <strong style="color: #f1f5f9;">Source {i+1}</strong>
                    <span class="doc-pill" style="margin-left: 0.5rem;">📁 {src_name}</span>
                </div>
                <div style="color: #60a5fa; font-weight: 600; font-size: 0.8rem;">
                    Re-rank Match: {score_pct}%
                </div>
            </div>
            <div style="line-height: 1.5; color: #cbd5e1; white-space: pre-wrap;">
                {source.get('text', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# TOP UTILITY RIBBON & HERO HEADER
# ================================================================
st.markdown("""
<div class="top-utility-bar">
    <div class="utility-badge"><span style="font-size: 1.15rem;">🦜</span> <strong>Pipeline:</strong> <span style="color: #c084fc;">LangChain LCEL</span></div>
    <div class="utility-badge"><span style="font-size: 1.15rem;">⚡</span> <strong>LLM Engine:</strong> <span style="color: #34d399;">Google Gemini 3.8 Flash</span></div>
    <div class="utility-badge"><span style="font-size: 1.15rem;">🎯</span> <strong>Retrieval:</strong> <span style="color: #38bdf8;">2-Stage Neural Re-Ranking</span></div>
    <div class="utility-badge"><span style="font-size: 1.15rem;">🗄️</span> <strong>Vector Store:</strong> <span style="color: #f472b6;">ChromaDB</span></div>
</div>

<div class="hero-header">
    <div style="display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.7rem; flex-wrap: wrap;">
        <span style="font-size: 3.2rem; line-height: 1; filter: drop-shadow(0 0 16px rgba(168, 85, 247, 0.6));">🧠</span>
        <span class="hero-title" style="font-size: 3.2rem; line-height: 1.1;">RAGDoc AI</span>
        <span class="hero-pill">
            🦜 LANGCHAIN LCEL • ⚡ GEMINI 3.8 FLASH
        </span>
    </div>
    <div class="hero-subtitle">
        <strong>LangChain LCEL RAG & Benchmarking Engine</strong> — Upload multi-format document(s) (PDF, DOCX, TXT), explore high-precision <strong>2-stage neural retrieval</strong> (Bi-Encoder Dense Indexing + Cross-Encoder Re-Ranking), benchmark chunking strategies (LangChain Recursive, Semantic, Sentence, Fixed), and synthesize source-grounded answers powered by <strong>Google Gemini 3.8 Flash</strong>.
    </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# SIDEBAR CONTROLS
# ================================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # Ensure .env is explicitly loaded from project root
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)
    gemini_key_env = os.environ.get("GEMINI_API_KEY", "")
    user_api_key = gemini_key_env

    st.markdown("#### ⚡ AI Engine")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 182, 212, 0.18) 100%); border: 1.5px solid rgba(52, 211, 153, 0.5); border-radius: 14px; padding: 0.85rem 1rem; box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2); margin-bottom: 0.8rem;">
        <div style="display: flex; align-items: center; gap: 0.65rem;">
            <span style="font-size: 1.35rem;">⚡</span>
            <div>
                <div style="color: #ffffff; font-weight: 800; font-size: 0.95rem; letter-spacing: -0.01em;">Google Gemini 3.8 Flash</div>
                <div style="color: #6ee7b7; font-size: 0.78rem; font-weight: 600;">Cloud Turbo • Active Engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # RAG Pipeline Engine (LangChain LCEL only)
    st.markdown("#### 🦜 RAG Architecture")
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(147, 51, 234, 0.25) 0%, rgba(99, 102, 241, 0.22) 100%); border: 1.5px solid rgba(168, 85, 247, 0.55); border-radius: 14px; padding: 0.85rem 1rem; box-shadow: 0 4px 20px rgba(147, 51, 234, 0.25); margin-bottom: 0.8rem;">
        <div style="display: flex; align-items: center; gap: 0.65rem;">
            <span style="font-size: 1.35rem;">🦜</span>
            <div>
                <div style="color: #ffffff; font-weight: 800; font-size: 0.95rem; letter-spacing: -0.01em;">LangChain LCEL Pipeline</div>
                <div style="color: #c084fc; font-size: 0.78rem; font-weight: 600;">Active Pipeline • Neural 2-Stage</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state["rag_engine_mode"] = "langchain"

    st.markdown("---")

    # Strategy Selector (Default: LangChain)
    st.markdown("#### ✂️ Chunking Strategy")
    strategy = st.selectbox(
        "Choose a strategy",
        options=["langchain", "semantic", "sentence", "fixed"],
        index=0,
        help="LangChain: RecursiveCharacterTextSplitter | Semantic: splits by topic change | Sentence: splits by grammatical sentence | Fixed: splits by character count"
    )

    # Strategy descriptions
    descriptions = {
        "langchain": "🦜 **LangChain Recursive (Recommended):** Hierarchical paragraph & sentence splitting via RecursiveCharacterTextSplitter for optimal semantic cohesion.",
        "semantic": "🧠 **Semantic:** Dynamically groups text by contextual embedding similarity thresholds.",
        "sentence": "📜 **Sentence-Based:** Preserves natural grammatical sentence boundaries.",
        "fixed": "⚡ **Fixed-Size (500 chars):** Strict character window baseline with 50-char overlap."
    }
    st.info(descriptions[strategy])

    st.markdown("---")

    # Audio Notifications Toggle
    st.markdown("#### 🔊 Audio Notifications")
    enable_sound = st.toggle(
        "Enable Response Chime",
        value=True,
        help="Plays a pleasant audio chime when AI finishes generating an answer."
    )

    # Defaults (Re-ranking is always ON by default in background)
    use_reranker = True


def clean_trailing_syntax(text):
    """Ensures clean markdown formatting without dangling artifacts or unclosed tags."""
    if not text:
        return ""
    import re
    # Strip any trailing horizontal rules (--- or ***)
    text = re.sub(r'\s*(?:---+|\*\*\*+)\s*$', '', text).strip()
    # Strip any dangling partial emoji/headers at the very end
    text = re.sub(r'\s*💡\s*(?:\*\*)?\s*$', '', text).strip()
    # Ensure any unclosed bold formatting (odd number of **) is balanced
    if text.count("**") % 2 != 0:
        text += "**"
    return text


def extract_followups(answer_text):
    """Separates main answer text from suggested follow-up questions using robust pattern matching."""
    if not answer_text:
        return "", []

    import re
    # Match any variation of follow-up questions heading (bold, headers, emoji, dividers)
    pattern = r'(?:\r?\n)*(?:---+|\*\*\*+)?\s*(?:###?\s*)?(?:\*\*)?💡\s*(?:\*\*)?\s*(?:Suggested Follow-up Questions|Suggested Questions|Follow-up Questions):?(?:\*\*)?'

    match = re.search(pattern, answer_text, re.IGNORECASE)
    if match:
        main_ans = answer_text[:match.start()].strip()
        followup_section = answer_text[match.end():].strip()
        raw_qs = followup_section.split("\n")
        qs = []
        for line in raw_qs:
            cleaned = line.strip(" -*123456789.💡`").strip()
            if cleaned and len(cleaned) > 5:
                qs.append(cleaned)
        return clean_trailing_syntax(main_ans), qs[:3]

    # Clean any dangling/incomplete follow-up markers (e.g. "💡 **", "💡", "💡 Suggested")
    dangling_pattern = r'(?:\r?\n)*(?:---+|\*\*\*+)?\s*(?:###?\s*)?(?:\*\*)?💡.*$'
    cleaned_ans = re.sub(dangling_pattern, '', answer_text, flags=re.DOTALL).strip()
    return clean_trailing_syntax(cleaned_ans), []


# ================================================================
# MAIN NAVIGATION TABS
# ================================================================
tab_qa, tab_eval, tab_about = st.tabs([
    "💬 Q&A",
    "📊 Evaluate",
    "📖 About & Architecture"
])


# ================================================================
# TAB 1: CONVERSATIONAL Q&A
# ================================================================
with tab_qa:
    # ── TOP SECTION: Document Upload & Indexing ──
    st.markdown("### 📂 Document Ingestion & Vector Indexing")

    upload_col, action_col = st.columns([3.2, 1.2], gap="medium")
    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload up to 5 document(s) (.txt, .pdf, or .docx)",
            type=["txt", "pdf", "docx"],
            accept_multiple_files=True,
            help="Upload up to 5 document(s) (up to 500MB each) to index into ChromaDB vector database"
        )
    with action_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        index_clicked = st.button("🚀 Index Document(s)", key="btn_index_docs", use_container_width=True)

    if uploaded_files and len(uploaded_files) > 5:
        st.warning("⚠️ Maximum 5 files allowed. Only the first 5 will be processed.")
        uploaded_files = uploaded_files[:5]

    # Index Button Handler
    if index_clicked:
        if not uploaded_files:
            st.warning("⚠️ Please choose at least one document file to upload.")
        else:
            try:
                session_key = f"rag_{strategy}"

                if session_key not in st.session_state or not hasattr(st.session_state[session_key], "load_documents"):
                    st.session_state[session_key] = initialize_rag(strategy, use_reranker)

                rag = st.session_state[session_key]

                # Process uploaded files with original filenames preserved
                doc_inputs = []
                for uf in uploaded_files:
                    suffix = os.path.splitext(uf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.getvalue())
                        doc_inputs.append((tmp.name, uf.name))

                # Index the documents
                with st.spinner("✂️ Chunking and indexing document(s)..."):
                    rag.retriever.clear()
                    if hasattr(rag, "load_documents"):
                        chunks = rag.load_documents(doc_inputs)
                    else:
                        chunks = rag.load_document(doc_inputs)

                # Store in session state
                st.session_state["indexed"] = True
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["current_rag_key"] = session_key
                st.session_state["uploaded_doc_inputs"] = doc_inputs
                st.session_state["uploaded_file_paths"] = [di[0] for di in doc_inputs]
                st.session_state["uploaded_file_names"] = [di[1] for di in doc_inputs]

                st.success(f"✅ Indexed {len(chunks)} chunks from {len(doc_inputs)} document(s) using {strategy.capitalize()} strategy!")

            except Exception as e:
                st.error(f"❌ Error during indexing: {str(e)}")

    # Indexing Summary Banner (Horizontal & Space-Saving)
    if st.session_state.get("indexed"):
        chunk_count = st.session_state.get("chunk_count", 0)
        file_names = st.session_state.get("uploaded_file_names", [])
        st.markdown(f"""
        <div class="rag-card" style="padding: 0.95rem 1.4rem; margin: 0.8rem 0 1.4rem 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.4rem;">📁</span>
                <div>
                    <div style="font-weight: 700; color: #f8fafc; font-size: 0.94rem;">Indexed Documents: <span style="color: #c084fc;">{', '.join(file_names)}</span></div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">ChromaDB Vector Store Ready • Dense + Cross-Encoder Active</div>
                </div>
            </div>
            <div style="display: flex; gap: 1.8rem; align-items: center;">
                <div style="text-align: center;">
                    <span style="color: #34d399; font-weight: 800; font-size: 1.2rem;">{chunk_count}</span>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Chunks</div>
                </div>
                <div style="text-align: center;">
                    <span style="color: #60a5fa; font-weight: 800; font-size: 1.2rem;">{strategy.capitalize()}</span>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Chunker</div>
                </div>
                <div style="text-align: center;">
                    <span style="color: #a78bfa; font-weight: 800; font-size: 1.2rem;">2-Stage ON</span>
                    <div style="color: #94a3b8; font-size: 0.76rem;">Re-Ranker</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SECTION 2: CONVERSATIONAL Q&A STUDIO (ChatGPT / Gemini Style) ──
    header_col, clear_col = st.columns([3.5, 1], gap="medium")
    with header_col:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 0.65rem;">
            <span style="font-size: 1.5rem;">💬</span>
            <span style="font-size: 1.35rem; font-weight: 800; color: #f8fafc;">Document Q&A Conversation</span>
        </div>
        """, unsafe_allow_html=True)
    with clear_col:
        if st.button("🗑️ Clear Conversation", key="btn_clear_chat", use_container_width=True):
            st.session_state["chat_messages"] = []
            st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Dedicated scrollable conversation box with independent scroll up/down
    chat_container = st.container(height=520, border=True)

    with chat_container:
        if not st.session_state["chat_messages"]:
            st.markdown("""
            <div style="text-align: center; padding: 3.5rem 1.5rem; color: #94a3b8;">
                <div style="font-size: 3.2rem; margin-bottom: 0.8rem; filter: drop-shadow(0 0 16px rgba(168, 85, 247, 0.45));">🧠</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.45rem;">What would you like to explore from your documents?</div>
                <div style="font-size: 0.95rem; color: #cbd5e1; max-width: 540px; margin: 0 auto 1.4rem auto; line-height: 1.6;">
                    Ask questions, compare sections across multiple files, or request in-depth summaries. Every response is synthesized by Google Gemini 3.8 Flash and grounded directly in your uploaded documents.
                </div>
                <div style="display: inline-flex; gap: 0.5rem; font-size: 0.84rem; color: #c4b5fd; background: rgba(147, 51, 234, 0.18); padding: 0.45rem 1rem; border-radius: 20px; border: 1px solid rgba(168, 85, 247, 0.35);">
                    ⚡ Upload & Index your documents above, then ask below!
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display full multi-turn conversation history (ChatGPT / Gemini style)
            for msg_idx, msg in enumerate(st.session_state["chat_messages"]):
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.28) 0%, rgba(139, 92, 246, 0.22) 100%); border: 1px solid rgba(168, 85, 247, 0.45); border-radius: 14px; padding: 0.85rem 1.2rem; color: #f8fafc; font-size: 1.05rem; font-weight: 500; display: inline-block; line-height: 1.6;">
                            {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with st.chat_message("assistant", avatar="🧠"):
                        main_ans, followups = extract_followups(msg["content"])
                        st.markdown(f"""
                        <div class="answer-card" style="margin: 0.2rem 0 0.6rem 0;">
                            <div style="line-height: 1.85; font-size: 1.15rem; font-weight: 450; color: #f8fafc;">
                                {main_ans}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Clean expandable sources container (ChatGPT / Gemini style citations)
                        if "sources" in msg and msg["sources"]:
                            with st.expander(f"📄 View Cited Sources ({len(msg['sources'])} chunks)", expanded=False):
                                display_sources(msg["sources"])

                        # Interactive 1-click follow-up question pills
                        if followups:
                            st.markdown("<div style='margin-top: 0.7rem; margin-bottom: 0.35rem; font-weight: 700; color: #c084fc; font-size: 0.88rem;'>💡 Suggested Follow-up Questions:</div>", unsafe_allow_html=True)
                            for q_idx, q in enumerate(followups):
                                if st.button(f"👉 {q}", key=f"fq_{msg_idx}_{q_idx}", use_container_width=True):
                                    st.session_state["clicked_followup"] = q
                                    st.rerun()

    # ── CHAT INPUT (Anchored below the scrollable response box) ──
    pending_query = None
    if "clicked_followup" in st.session_state and st.session_state["clicked_followup"]:
        pending_query = st.session_state.pop("clicked_followup")

    chat_placeholder = "Ask any follow-up question..." if st.session_state.get("chat_messages") else "Ask a question about your indexed document(s)..."
    user_typed = st.chat_input(chat_placeholder)

    user_query = pending_query or user_typed

    if user_query:
        if not st.session_state.get("indexed"):
            st.warning("⚠️ Please upload and index document(s) first using the panel above.")
        else:
            # Append user message to conversation history (preserves old responses)
            st.session_state["chat_messages"].append({"role": "user", "content": user_query})

            # Get active RAG instance
            session_key = st.session_state.get("current_rag_key", f"rag_{strategy}")
            rag = st.session_state.get(session_key)

            if not rag:
                rag = initialize_rag(strategy, use_reranker)
                st.session_state[session_key] = rag

            # Generate Answer with full conversation history
            with st.spinner("🧠 Searching vector database & generating answer..."):
                try:
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["chat_messages"][:-1]
                    ]

                    result = rag.format_answer_with_sources(
                        question=user_query,
                        top_k=5,
                        chat_history=history,
                        api_key=user_api_key.strip() if user_api_key else None,
                        provider="gemini"
                    )

                    # Append Assistant response to history
                    st.session_state["chat_messages"].append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

                    # Play audio chime if enabled
                    if enable_sound:
                        play_chime()

                    # Rerun to cleanly update the scrollable chat container
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")


# ================================================================
# TAB 2: EVALUATION
# ================================================================
with tab_eval:
    st.markdown("### 📊 Strategy Comparison Evaluation")

    has_uploaded_docs = "uploaded_file_paths" in st.session_state and st.session_state["uploaded_file_paths"]

    if has_uploaded_docs:
        file_names = st.session_state.get("uploaded_file_names", [os.path.basename(p) for p in st.session_state["uploaded_file_paths"]])
        st.markdown(
            f"Evaluate your **uploaded document(s)** ({', '.join(file_names)}) "
            "across chunking strategies (`LangChain Recursive`, `Semantic`, `Sentence`, `Fixed`)."
        )

        if st.button("▶️ Run Evaluation", use_container_width=True):
            with st.spinner("🔄 Running evaluation on your document(s)..."):
                try:
                    doc_inputs = st.session_state.get("uploaded_doc_inputs", st.session_state["uploaded_file_paths"])
                    results_df = Evaluator.evaluate_uploaded_docs(
                        doc_inputs=doc_inputs
                    )
                    st.session_state["eval_results"] = results_df
                except Exception as e:
                    st.error(f"❌ Evaluation error: {str(e)}")
                    st.exception(e)
    else:
        st.warning(
            "⚠️ Please upload and index your document(s) in the **💬 Q&A** tab first "
            "before running strategy evaluation."
        )

    # Show results table & chart if evaluated
    if "eval_results" in st.session_state:
        df = st.session_state["eval_results"]
        st.success("✅ Evaluation complete!")

        # Results Dataframe
        st.dataframe(df, use_container_width=True)

        # Show saved chart
        chart_path = "results/strategy_comparison.png"
        if os.path.exists(chart_path):
            st.image(chart_path, caption="RAG Chunking Strategy Performance Comparison")

        # Resume Bullet Generator
        st.markdown("### 📝 Portfolio Resume Bullet")
        if len(df) >= 3:
            best_row = df.loc[df["Hit Rate (%)"].idxmax()]
            worst_row = df.loc[df["Hit Rate (%)"].idxmin()]

            best_strategy = best_row["Strategy"]
            best_rate = best_row["Hit Rate (%)"]
            worst_rate = worst_row["Hit Rate (%)"]
            best_faith = best_row["Avg Faithfulness (0-5)"]

            resume_bullet = (
                f"Built enterprise RAG pipeline with LangChain LCEL and Gemini 3.8 Flash, "
                f"benchmarking chunking strategies (LangChain Recursive, Semantic, Sentence, Fixed); "
                f"{best_strategy} chunking achieved retrieval hit rate of {best_rate}% on evaluation harness "
                f"with faithfulness score of {best_faith}/5.0. Implemented 2-stage neural retrieval "
                f"with Cross-Encoder re-ranking (ms-marco-MiniLM)."
            )

            st.markdown(f"""
            <div class="answer-card">
                <div style="color: #c4b5fd; font-size: 0.8rem; margin-bottom: 0.4rem; font-weight: 700;">
                    📋 Copy this bullet to your resume / portfolio
                </div>
                <div style="font-size: 0.95rem; line-height: 1.6; color: #f8fafc; font-style: italic;">
                    "{resume_bullet}"
                </div>
            </div>
            """, unsafe_allow_html=True)


# ================================================================
# TAB 3: ABOUT & ARCHITECTURE
# ================================================================
with tab_about:
    st.markdown("""
    <div style="font-size: 2.2rem; font-weight: 800; margin-bottom: 0.6rem; letter-spacing: -0.01em;">
        📖 How This RAG System Works
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Section 1: Architecture Pipeline
    st.markdown("""
    <div style="font-size: 1.4rem; font-weight: 700; margin-top: 0.8rem; margin-bottom: 0.6rem;">
        🏗️ 5-Step System Architecture Flow
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="flow-grid">
        <div class="flow-step">
            <div class="flow-num">1</div>
            <div class="flow-title">Multi-Doc Ingestion</div>
            <div class="flow-desc">Parses PDF, DOCX, and TXT files, extracting raw text and metadata.</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">2</div>
            <div class="flow-title">LangChain Chunking</div>
            <div class="flow-desc">Splits documents via LangChain Recursive, Semantic, Sentence, or Fixed rules.</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">3</div>
            <div class="flow-title">Vector Indexing</div>
            <div class="flow-desc">Embeds chunks with <code>all-MiniLM-L6-v2</code> into ChromaDB vector store.</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">4</div>
            <div class="flow-title">2-Stage Re-Ranking</div>
            <div class="flow-desc">Cross-Encoder (<code>ms-marco-MiniLM</code>) re-scores top-15 candidates.</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">5</div>
            <div class="flow-title">Grounded Generation</div>
            <div class="flow-desc">Google Gemini 3.8 Flash synthesizes accurate answers strictly from top-5 context.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Deep Dive on Re-ranking
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown("#### 🎯 Why 2-Stage Retrieval (Re-Ranking)?")
        st.markdown("""
        Standard RAG systems use **Bi-Encoder Vector Search** to find chunks by cosine similarity.
        While fast, vector search can retrieve chunks that are *topically similar* but fail to answer the exact question.

        **Our Solution:** We implement a **2-Stage Retrieval Pipeline**:
        1. **Stage 1 (Fast Retrieval):** ChromaDB retrieves top **15 candidate chunks** using embeddings.
        2. **Stage 2 (Precision Re-Ranking):** A **Cross-Encoder model** (`ms-marco-MiniLM-L-6-v2`) performs joint attention on `(Question + Passage)` to re-score candidates, returning the **top 5 most accurate chunks**.
        """)

    with col_b:
        st.markdown("#### ✂️ The 4 Chunking Strategies Compared")
        st.markdown("""
        | Strategy | How it works | Best for |
        | :--- | :--- | :--- |
        | **LangChain Recursive** | Hierarchical splitting by paragraph, newline, and space budgets. | Complex documents & high-accuracy QA (Recommended). |
        | **Semantic** | Measures embedding distance between consecutive sentences to group text by topic changes. | Complex multi-topic documents. |
        | **Sentence-Based** | Cuts strictly at grammatical sentence boundaries. | Structured narrative text. |
        | **Fixed-Size** | Cuts text strictly every 500 characters. | Quick baseline testing. |
        """)

    st.markdown("---")

    # Section 3: Tech Stack & Credits
    st.markdown("#### 🛠️ Tech Stack & Components")

    col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)

    with col_t1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🦜</div>
            <strong style="color: #f8fafc;">Framework</strong>
            <div style="color: #c084fc; font-size: 0.85rem; margin-top: 0.3rem; font-weight: 700;">LangChain LCEL</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">⚡</div>
            <strong style="color: #f8fafc;">Cloud LLM</strong>
            <div style="color: #6ee7b7; font-size: 0.85rem; margin-top: 0.3rem; font-weight: 700;">Gemini 3.8 Flash</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t3:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">📐</div>
            <strong style="color: #f8fafc;">Embeddings</strong>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">all-MiniLM-L6-v2</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t4:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🗄️</div>
            <strong style="color: #f8fafc;">Vector DB</strong>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">ChromaDB (Local)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_t5:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🎯</div>
            <strong style="color: #f8fafc;">Re-Ranker</strong>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">ms-marco-MiniLM</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Author Card
    st.markdown("""
    <div style="text-align: center; padding: 1.2rem; color: #a5b4fc; font-size: 0.95rem;">
        🚀 <strong>AI Engineering Portfolio • Project #01</strong> | Powered by <strong>LangChain LCEL</strong> & <strong>Google Gemini 3.8 Flash</strong> | Developed by <strong>Aditya</strong>
    </div>
    """, unsafe_allow_html=True)
