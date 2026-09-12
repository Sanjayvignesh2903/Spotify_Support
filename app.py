# app.py - Spotify AI Support Agent Studio Dashboard
import os
import re
import json
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ==========================================
# 1. PAGE SETUP & ENHANCED VISUAL ENGINE
# ==========================================
st.set_page_config(
    page_title="Spotify AI Support Ops Hub",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Manage session state for interactive preset buttons
if 'tweet_input' not in st.session_state:
    st.session_state.tweet_input = ""
if 'auto_run' not in st.session_state:
    st.session_state.auto_run = False

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: linear-gradient(-45deg, #090b0e, #0e1e17, #13141f, #071c14, #121316) !important;
        background-size: 400% 400% !important;
        animation: gradientShift 18s ease infinite !important;
        color: #f1f5f9;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.75) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(22px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 15px rgba(29, 185, 84, 0.3); }
        50% { box-shadow: 0 0 30px rgba(29, 185, 84, 0.7); }
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        animation: slideInUp 0.6s ease-out forwards;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(29, 185, 84, 0.4);
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.55), 0 0 20px rgba(29, 185, 84, 0.15);
    }

    .gradient-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 20%, #1DB954 65%, #1ed760 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }

    .badge-auto {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #1DB954 0%, #179443 100%);
        color: #031408;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
        padding: 6px 14px;
        border-radius: 999px;
        box-shadow: 0 0 16px rgba(29, 185, 84, 0.45);
        animation: pulseGlow 3s infinite;
    }

    .badge-escalate {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
        padding: 6px 14px;
        border-radius: 999px;
        box-shadow: 0 0 18px rgba(255, 65, 108, 0.5);
    }

    .tweet-preview-box {
        background: linear-gradient(145deg, rgba(29, 161, 242, 0.08), rgba(20, 23, 26, 0.7));
        border: 1px solid rgba(29, 161, 242, 0.25);
        border-radius: 14px;
        padding: 18px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc;
        line-height: 1.6;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        position: relative;
    }

    .tweet-author {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 700;
        color: #1DA1F2;
        font-size: 0.95rem;
        margin-bottom: 10px;
    }

    div.stButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #FFF !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: rgba(29, 185, 84, 0.2) !important;
        border-color: #1DB954 !important;
    }
    
    /* Primary Process Button Override */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1DB954 0%, #169c45 100%) !important;
        color: #041409 !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(29, 185, 84, 0.35) !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 10px 28px rgba(29, 185, 84, 0.55) !important;
    }

    .stat-chip {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. GROQ & RETRIEVER CORE SETUP
# ==========================================
@st.cache_resource
def get_client():
    return Groq(api_key="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")

client = get_client()

def clean_tweet(text: str) -> str:
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return ' '.join(text.lower().split())

class SupportRetriever:
    def __init__(self, kb_csv_path: str):
        self.df = pd.read_csv(kb_csv_path).dropna(subset=['Customer_Message', 'Brand_Reply'])
        self.df['Clean_Message'] = self.df['Customer_Message'].apply(clean_tweet)
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=15000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Clean_Message'])

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        clean_query = clean_tweet(query)
        query_vec = self.vectorizer.transform([clean_query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "past_query": self.df.iloc[idx]['Customer_Message'],
                "past_reply": self.df.iloc[idx]['Brand_Reply'],
                "similarity": float(scores[idx])
            })
        return results

@st.cache_resource
def load_retriever(path: str):
    try:
        return SupportRetriever(path)
    except:
        return None

# ==========================================
# 3. CLASSIFICATION, RAG & JUDGE LOGIC
# ==========================================
def classify_ticket(customer_message: str, model: str = "openai/gpt-oss-20b") -> dict:
    prompt = f"""You are an automated support classifier for Spotify.
Analyze this tweet and classify the intent and action.

Taxonomy intents:
- account_access
- billing_subscription
- playback_streaming_issue
- playlist_library_management
- content_availability
- general_feedback_ui

Actions:
- Escalate: ONLY for sensitive account security (passwords, login, numbers), billing disputes, or account lookup.
- Auto: for troubleshooting, links, feedback, or general availability.

Tweet: "{customer_message}"

Output JSON strictly:
{{"intent": "<intent>", "action": "<Auto|Escalate>", "reason": "<brief justification>"}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

def draft_reply(customer_message: str, retrieved_context: list[dict], action: str, model: str = "openai/gpt-oss-20b") -> str:
    if action == "Escalate":
        return 'I cannot securely process account changes or billing requests through this AI interface. To get assistance with this, please <a href="https://support.spotify.com/in-en/smarthelp/" target="_blank" style="color:#1DB954; font-weight:bold;">contact Spotify support</a>.'

    context_str = "\n".join([
        f"- Past: {item['past_query']}\n  Reply: {item['past_reply']}"
        for item in retrieved_context
    ])

    prompt = f"""You are @SpotifyCares on Twitter. Reply to this customer directly and concisely (< 280 characters).
Customer: "{customer_message}"

Similar historical resolutions:
{context_str}

Guidelines: Friendly, proactive, and concise. Do not promise refunds directly. Provide clear troubleshooting steps.
Drafted Tweet:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=120
    )
    reply_text = str(response.choices[0].message.content).strip(' "\'\n')
    return reply_text

def run_llm_judge(customer_message: str, drafted_reply: str, model: str = "openai/gpt-oss-20b") -> dict:
    prompt = f"""Grade this Spotify support reply from 1 to 5:
- 1: Toxic, completely unhelpful, or promises refunds.
- 2: Robotic or misses the problem.
- 3: Acceptable but dry tone.
- 4: Good, helpful, matches Spotify brand tone.
- 5: Excellent, empathetic, concise, and actionable.

Customer: "{customer_message}"
Reply: "{drafted_reply}"

Output JSON strictly:
{{"score": 5, "reasoning": "brief critique"}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

# ==========================================
# 4. SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="38" />
            <h2 style="margin: 0; font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">Control Deck</h2>
        </div>
    """, unsafe_allow_html=True)
    st.caption("Proof-Focused Operations & Benchmark Harness")

    st.markdown("---")
    st.markdown("### 📈 Evaluation Benchmarks")

    st.markdown("""
        <div class="stat-chip">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">LLM Intent Accuracy</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #1DB954;">64.0% <span style="font-size: 0.8rem; color: #38bdf8;">(+23% vs Baseline)</span></div>
        </div>
        <div class="stat-chip">
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Action Routing Accuracy</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #38bdf8;">79.0% <span style="font-size: 0.8rem; color: #94a3b8;">(0.87 F1 Auto)</span></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Baseline Matrix (Golden Set n=200):**
    - **Trivial (Majority):** `41% Acc` | `0.10 F1`
    - **Simple (Keyword):** `56% Acc` | `0.42 F1`
    - **Zero-Shot LLM Agent:** `64% Acc` | `0.66 Macro F1`
    """)

    st.markdown("---")
    kb_path = st.text_input(
        "Knowledge Base CSV Path",
        value=r"C:\Users\sanja\OneDrive\Documents\Hiver\spotify_knowledge_base.csv"
    )

# ==========================================
# 5. MAIN DASHBOARD CONTENT
# ==========================================
st.markdown('<div class="gradient-header">🎧 Spotify Support Operations Hub</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; margin-top: -8px; font-size: 1.05rem;'>Real-time zero-shot intent triage, bi-gram TF-IDF RAG context retrieval, and automated QA grading.</p>", unsafe_allow_html=True)
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Main Text Input
tweet_input = st.text_area(
    "Incoming Customer Tweet:",
    value=st.session_state.tweet_input,
    height=120,
    placeholder="Type or paste the customer message here..."
)

# Quick Preset Buttons
st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;'>Quick Ticket Scenarios (Click to test):</p>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

def set_preset(text):
    st.session_state.tweet_input = text
    st.session_state.auto_run = True

with col_btn1:
    if st.button("🚨 Change Phone Number", use_container_width=True):
        set_preset("how to change my number")
        st.rerun()
with col_btn2:
    if st.button("💳 Billing Double Charge", use_container_width=True):
        set_preset("@SpotifyCares I got charged $11.99 twice for premium this month on my card. Need a refund ASAP")
        st.rerun()
with col_btn3:
    if st.button("📱 Offline Playback Bug", use_container_width=True):
        set_preset("Every time I switch to offline mode on my iPhone 15, my downloaded playlists disappear and the app crashes!")
        st.rerun()
with col_btn4:
    if st.button("🌍 Content Licensing", use_container_width=True):
        set_preset("Why is the entire new track list greyed out in my region @SpotifyCares? Is it locked?")
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
process_clicked = st.button("⚡ Process & Triage Tweet", type="primary", use_container_width=True)

if process_clicked or st.session_state.auto_run:
    st.session_state.auto_run = False
    current_input = tweet_input if tweet_input.strip() else st.session_state.tweet_input
    
    if current_input.strip():
        with st.spinner("Processing pipeline: Intent Triage -> TF-IDF Bi-gram Search -> Response Generation..."):
            retriever = load_retriever(kb_path)
            if not retriever:
                st.error(f"Failed to load knowledge base from `{kb_path}`")
                st.stop()

            classification = classify_ticket(current_input)
            intent = classification.get("intent", "general_feedback_ui")
            action = classification.get("action", "Auto")
            reason = classification.get("reason", "No justification provided.")

            matches = retriever.retrieve(current_input, top_k=2)
            reply = draft_reply(current_input, matches, action)

        # 3-Column Results Layout
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        res_col1, res_col2, res_col3 = st.columns([1, 1.3, 1.4])

        with res_col1:
            badge_html = '<span class="badge-escalate">🚨 ESCALATE</span>' if action == "Escalate" else '<span class="badge-auto">⚡ AUTO-RESOLVE</span>'
            st.markdown(f"""
<div class="glass-card">
<h4 style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 8px;">1. Triage & Routing</h4>
<div style="margin-bottom: 12px;">{badge_html}</div>
<div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">CLASSIFIED INTENT</div>
<div style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #38bdf8; font-size: 1rem; margin-bottom: 10px;">{intent}</div>
<div style="font-size: 0.8rem; color: #94a3b8;">SYSTEM REASONING</div>
<div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">{reason}</div>
</div>
""", unsafe_allow_html=True)

        with res_col2:
            match_cards_html = ""
            for i, match in enumerate(matches, 1):
                match_cards_html += f"""
<div style="background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.2); border-radius: 12px; padding: 18px; margin-bottom: 16px; box-shadow: 0 6px 16px rgba(0,0,0,0.15);">
<div style="display: flex; justify-content: space-between; font-size: 1.0rem; color: #1DB954; font-weight: 800; margin-bottom: 12px;">
<span>Match #{i}</span>
<span>Similarity: {match['similarity']:.2f}</span>
</div>
<div style="font-size: 1.1rem; color: #ffffff; margin-bottom: 8px; line-height: 1.5;"><b>Query:</b> {match['past_query']}</div>
<div style="font-size: 1.05rem; color: #e2e8f0; line-height: 1.5;"><b>Resolution:</b> {match['past_reply']}</div>
</div>
"""
            st.markdown(f"""
<div class="glass-card">
<h4 style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 10px;">2. Historical Context (RAG)</h4>
{match_cards_html}
</div>
""", unsafe_allow_html=True)

        with res_col3:
            char_len = len(re.sub('<[^<]+>', '', reply))
            status_color = "#1DB954" if char_len <= 280 else "#ff4b2b"
            st.markdown(f"""
<div class="glass-card">
<h4 style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 12px;">3. Response Generation</h4>
<div class="tweet-preview-box">
<div class="tweet-author">
<img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="18" />
@SpotifyCares
</div>
<div>{reply}</div>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 0.8rem; color: #94a3b8;">
<span>Character Budget (Raw Text)</span>
<span style="color: {status_color}; font-weight: 700;">{char_len} / 280 chars</span>
</div>
</div>
""", unsafe_allow_html=True)

            with st.expander("🧪 Run LLM-as-a-Judge Evaluation", expanded=False):
                if st.button("Evaluate Draft with QA Judge", use_container_width=True):
                    with st.spinner("Grading response against QA Rubric..."):
                        judge_eval = run_llm_judge(current_input, reply)
                        score = judge_eval.get("score", 4)
                        reasoning = judge_eval.get("reasoning", "")
                        
                        st.markdown(f"""
<div style="background: rgba(29, 185, 84, 0.08); border: 1px solid rgba(29, 185, 84, 0.25); border-radius: 10px; padding: 14px; margin-top: 10px;">
<div style="font-size: 1.2rem; font-weight: 800; color: #1DB954;">Score: {score} / 5</div>
<div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">{reasoning}</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.warning("Please type a customer message or select a preset scenario above.")