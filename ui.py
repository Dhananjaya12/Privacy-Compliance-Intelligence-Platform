import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="PDF RAG Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #f5f0eb;
}

[data-testid="stSidebar"] {
    background-color: #eee8e0;
    border-right: 1px solid #d9d0c4;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #3d3530 !important;
}

#MainMenu, footer, header { visibility: hidden; }

.user-bubble {
    background: #ffffff;
    border: 1px solid #ddd5c8;
    border-radius: 16px 16px 4px 16px;
    padding: 13px 18px;
    margin: 6px 0 6px 80px;
    font-size: 15px;
    line-height: 1.65;
    color: #2c2420;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.assistant-bubble {
    background: #ffffff;
    border: 1px solid #ddd5c8;
    border-left: 3px solid #c17f4a;
    border-radius: 4px 16px 16px 16px;
    padding: 13px 18px;
    margin: 6px 80px 6px 0;
    font-size: 15px;
    line-height: 1.75;
    color: #2c2420;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.msg-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.you-label  { color: #8a7a6e; text-align: right; }
.bot-label  { color: #c17f4a; }

.meta-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    padding: 2px 9px;
    border-radius: 20px;
    margin-top: 10px;
    margin-right: 5px;
}
.badge-web   { background: #edf7ee; color: #2e7d32; border: 1px solid #a5d6a7; }
.badge-rag   { background: #ede8f5; color: #5c35a0; border: 1px solid #c5b3e6; }
.badge-score { background: #fef3e2; color: #a05c10; border: 1px solid #f5c98a; }

.stChatInput textarea {
    background: #ffffff !important;
    border: 1px solid #d0c8be !important;
    color: #2c2420 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}
.stChatInput textarea:focus {
    border-color: #c17f4a !important;
    box-shadow: 0 0 0 3px rgba(193,127,74,0.12) !important;
}

.stButton > button {
    background: #ffffff !important;
    color: #3d3530 !important;
    border: 1px solid #d0c8be !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: #c17f4a !important;
    color: #c17f4a !important;
    background: #fdf7f0 !important;
}

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1.5px dashed #d0c8be;
    border-radius: 10px;
    padding: 6px;
}

hr { border-color: #d9d0c4 !important; margin: 14px 0 !important; }

.sdot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: middle;
}
.sdot-green  { background: #43a047; }
.sdot-yellow { background: #f9a825; }
.sdot-red    { background: #e53935; }

.sidebar-title {
    font-size: 20px;
    font-weight: 600;
    color: #2c2420;
    letter-spacing: -0.01em;
}
.sidebar-sub {
    font-size: 12px;
    color: #8a7a6e;
    margin-top: 2px;
}

.mode-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 6px;
    background: #fdf7f0;
    color: #c17f4a;
    border: 1px solid #f5d8b8;
}

.empty-state {
    text-align: center;
    padding: 90px 40px;
}
.empty-icon { font-size: 48px; margin-bottom: 16px; }
.empty-title {
    font-size: 22px;
    font-weight: 600;
    color: #3d3530;
    margin-bottom: 10px;
    letter-spacing: -0.02em;
}
.empty-hint {
    font-size: 14px;
    color: #8a7a6e;
    line-height: 1.7;
    max-width: 340px;
    margin: 0 auto 24px auto;
}
.example-chip {
    display: inline-block;
    background: #ffffff;
    border: 1px solid #ddd5c8;
    border-radius: 20px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 13px;
    color: #5c4f44;
}

/* Sidebar toggle arrow — make it a visible button */
[data-testid="collapsedControl"] {
    background-color: #ffffff !important;
    border: 1px solid #d0c8be !important;
    border-radius: 8px !important;
    width: 2rem !important;
    height: 2rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    top: 14px !important;
    left: 14px !important;
}
[data-testid="collapsedControl"]:hover {
    border-color: #c17f4a !important;
    background-color: #fdf7f0 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #3d3530 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "query"


# ── Helpers ────────────────────────────────────────────────────────────────
def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json().get("status") == "ready"
    except:
        return None


def send_query(question: str) -> dict:
    try:
        r = requests.post(f"{API_BASE}/api/query", json={"query": question}, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Is the server running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The model may still be loading."}
    except Exception as e:
        return {"error": str(e)}


def upload_pdfs(files) -> dict:
    try:
        file_tuples = [("files", (f.name, f.read(), "application/pdf")) for f in files]
        r = requests.post(f"{API_BASE}/api/ingest", files=file_tuples, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to the API. Is the server running?"}
    except Exception as e:
        return {"error": str(e)}


def render_message(role: str, content: str, meta: dict = None):
    if role == "user":
        st.markdown(f"""
            <div class="msg-label you-label">You</div>
            <div class="user-bubble">{content}</div>
        """, unsafe_allow_html=True)
    else:
        badges = ""
        if meta:
            if meta.get("used_web_search"):
                badges += '<span class="meta-badge badge-web">⚡ web search</span>'
            else:
                badges += '<span class="meta-badge badge-rag">📄 from PDFs</span>'
            if meta.get("retrieval_score") is not None:
                score = round(meta["retrieval_score"], 3)
                badges += f'<span class="meta-badge badge-score">relevance {score}</span>'
        st.markdown(f"""
            <div class="msg-label bot-label">Assistant</div>
            <div class="assistant-bubble">
                {content}
                <div>{badges}</div>
            </div>
        """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">PDF RAG Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Document Intelligence</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    health = check_health()
    if health is True:
        st.markdown('<span class="sdot sdot-green"></span> API Ready', unsafe_allow_html=True)
    elif health is False:
        st.markdown('<span class="sdot sdot-yellow"></span> API Loading…', unsafe_allow_html=True)
    else:
        st.markdown('<span class="sdot sdot-red"></span> API Offline', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:11px;font-weight:600;color:#8a7a6e;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px'>Mode</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Query", key="btn_query"):
            st.session_state.mode = "query"
            st.rerun()
    with col2:
        if st.button("📤 Upload", key="btn_upload"):
            st.session_state.mode = "upload"
            st.rerun()

    current_mode = st.session_state.mode
    st.markdown(f'<div class="mode-pill">● {current_mode.upper()}</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if current_mode == "upload":
        st.markdown("<div style='font-size:13px;font-weight:600;color:#3d3530;margin-bottom:8px'>Upload PDFs</div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "label",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.markdown(f"<div style='font-size:12px;color:#8a7a6e;margin:6px 0'>{len(uploaded_files)} file(s) selected</div>", unsafe_allow_html=True)
            if st.button("Ingest Files", key="ingest_btn"):
                with st.spinner("Processing…"):
                    result = upload_pdfs(uploaded_files)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✓ {result['files_processed']} file(s) · {result['chunks_created']} chunks created")
        st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("🗑 Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        n = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"<div style='font-size:11px;color:#8a7a6e;margin-top:8px'>{n} question(s) this session</div>", unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">What would you like to know?</div>
        <div class="empty-hint">
            Your documents are ready to explore.<br>Ask anything below to get started.
        </div>
        <div>
            <span class="example-chip">Summarise the key findings</span>
            <span class="example-chip">What methodology was used?</span>
            <span class="example-chip">Compare results across sections</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("meta"))


# ── Input ──────────────────────────────────────────────────────────────────
if current_mode == "query":
    if prompt := st.chat_input("Ask something about your documents…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)

        with st.spinner("Thinking…"):
            result = send_query(prompt)

        if "error" in result:
            msg = f"⚠ {result['error']}"
            st.session_state.messages.append({"role": "assistant", "content": msg})
            render_message("assistant", msg)
        else:
            answer = result.get("answer", "No answer returned.")
            meta = {
                "used_web_search": result.get("used_web_search", False),
                "retrieval_score": result.get("retrieval_score"),
            }
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})
            render_message("assistant", answer, meta)

        st.rerun()

elif current_mode == "upload":
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;color:#8a7a6e;font-size:14px;'>
        Use the sidebar to upload and ingest your PDFs,<br>
        then switch to <strong style='color:#3d3530'>Query</strong> mode to start asking questions.
    </div>
    """, unsafe_allow_html=True)