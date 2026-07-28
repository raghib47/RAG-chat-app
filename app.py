"""
DocMind AI — Futuristic RAG PDF Assistant
------------------------------------------
Premium dark-mode Streamlit UI over a Mistral + Chroma RAG pipeline.
Upload PDFs, build a vector store, and chat with your documents.
"""

import os
import time
import shutil
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CUSTOM CSS  (all styling lives here)
# ======================================================================
CSS = """
<style>
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap)');
@import url('[fonts.googleapis.com](https://fonts.googleapis.com/icon?family=Material+Icons+Round)');

:root {
    --bg:        #0B0F19;
    --bg-2:      #111827;
    --blue:      #3B82F6;
    --cyan:      #22D3EE;
    --purple:    #8B5CF6;
    --glass:     rgba(255,255,255,0.04);
    --glass-brd: rgba(255,255,255,0.09);
    --txt:       #E5E7EB;
    --txt-dim:   #94A3B8;
}

/* ---------- Base ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    color: var(--txt);
}
.stApp {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(59,130,246,0.10), transparent),
        radial-gradient(1000px 500px at 90% 0%, rgba(139,92,246,0.10), transparent),
        var(--bg);
}
#MainMenu, header[data-testid="stHeader"], footer {visibility: hidden;}
.block-container {padding-top: 1rem; max-width: 1250px;}

/* ---------- Floating orbs ---------- */
.orb {
    position: fixed; border-radius: 50%; filter: blur(60px);
    opacity: .35; z-index: 0; pointer-events: none;
    animation: float 14s ease-in-out infinite;
}
.orb1 {width:280px;height:280px;background:var(--blue);   top:12%;left:6%;}
.orb2 {width:240px;height:240px;background:var(--purple); top:55%;right:8%;animation-delay:-4s;}
.orb3 {width:180px;height:180px;background:var(--cyan);   bottom:8%;left:35%;animation-delay:-8s;}
@keyframes float {
    0%,100% {transform: translateY(0) translateX(0);}
    50%     {transform: translateY(-40px) translateX(20px);}
}

/* ---------- Navbar ---------- */
.navbar {
    position: sticky; top: 0; z-index: 50;
    display: flex; align-items: center; justify-content: space-between;
    padding: .85rem 1.5rem; margin-bottom: 1.4rem;
    border-radius: 18px;
    background: rgba(17,24,39,0.65);
    backdrop-filter: blur(18px);
    border: 1px solid var(--glass-brd);
    box-shadow: 0 8px 40px rgba(0,0,0,.45), inset 0 0 0 1px rgba(59,130,246,.06);
}
.nav-left {display:flex; align-items:center; gap:.75rem;}
.nav-logo {
    width:40px;height:40px;border-radius:12px;
    display:grid;place-items:center;font-size:22px;
    background: linear-gradient(135deg,var(--blue),var(--purple));
    box-shadow: 0 0 22px rgba(59,130,246,.55);
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {0%,100%{box-shadow:0 0 18px rgba(59,130,246,.45);}50%{box-shadow:0 0 34px rgba(139,92,246,.7);}}
.nav-title {font-size:1.25rem;font-weight:700;letter-spacing:-.5px;}
.nav-badge {
    font-size:.62rem;font-weight:700;letter-spacing:1px;
    padding:.15rem .5rem;border-radius:999px;margin-left:.4rem;
    background:linear-gradient(90deg,var(--blue),var(--cyan));color:#031122;
}
.nav-right {display:flex; align-items:center; gap:.6rem;}
.nav-btn {
    display:flex;align-items:center;gap:.4rem;
    padding:.45rem .85rem;border-radius:11px;font-size:.82rem;font-weight:500;
    color:var(--txt);text-decoration:none;
    background:var(--glass);border:1px solid var(--glass-brd);
    transition:all .25s ease;
}
.nav-btn:hover {border-color:var(--blue);box-shadow:0 0 18px rgba(59,130,246,.35);transform:translateY(-1px);}

/* ---------- Hero ---------- */
.hero {text-align:center; padding:2.2rem 1rem 1.6rem;}
.hero h1 {
    font-size:3.1rem;font-weight:800;line-height:1.05;margin:0;letter-spacing:-1.5px;
    background:linear-gradient(90deg,#fff,var(--cyan),var(--purple),#fff);
    background-size:300% auto;-webkit-background-clip:text;background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shine 6s linear infinite;
}
@keyframes shine {to{background-position:300% center;}}
.hero p {color:var(--txt-dim);font-size:1.05rem;max-width:620px;margin:1rem auto 0;line-height:1.6;}

/* ---------- Generic glass card ---------- */
.glass {
    background:var(--glass);border:1px solid var(--glass-brd);
    border-radius:20px;padding:1.1rem 1.25rem;backdrop-filter:blur(14px);
    box-shadow:0 8px 30px rgba(0,0,0,.35);
    transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.glass:hover {transform:translateY(-3px);border-color:rgba(59,130,246,.5);box-shadow:0 12px 40px rgba(59,130,246,.18);}

/* ---------- Stat cards ---------- */
.stat-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.4rem 0;}
.stat {
    position:relative;overflow:hidden;border-radius:18px;padding:1.15rem 1.25rem;
    background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.01));
    border:1px solid var(--glass-brd);backdrop-filter:blur(12px);
    transition:all .3s ease;
}
.stat::before {
    content:"";position:absolute;inset:0;border-radius:18px;padding:1px;
    background:linear-gradient(120deg,var(--blue),var(--cyan),var(--purple));
    -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
    -webkit-mask-composite:xor;mask-composite:exclude;opacity:.5;transition:opacity .3s;
}
.stat:hover {transform:translateY(-4px);}
.stat:hover::before {opacity:1;}
.stat .ico {font-size:1.5rem;}
.stat .num {font-size:1.9rem;font-weight:800;margin-top:.3rem;
    background:linear-gradient(90deg,#fff,var(--cyan));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;}
.stat .lbl {color:var(--txt-dim);font-size:.78rem;font-weight:500;text-transform:uppercase;letter-spacing:.6px;}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background:rgba(17,24,39,0.75);backdrop-filter:blur(20px);
    border-right:1px solid var(--glass-brd);
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {color:var(--txt-dim)!important;font-weight:500;}
.side-card {
    background:var(--glass);border:1px solid var(--glass-brd);
    border-radius:16px;padding:.9rem 1rem;margin-bottom:.9rem;
}
.profile {display:flex;align-items:center;gap:.7rem;}
.avatar {
    width:44px;height:44px;border-radius:50%;display:grid;place-items:center;font-size:20px;
    background:linear-gradient(135deg,var(--purple),var(--blue));
    box-shadow:0 0 16px rgba(139,92,246,.55);
}
.profile .name {font-weight:600;font-size:.95rem;}
.profile .role {color:var(--txt-dim);font-size:.75rem;}
.side-title {font-size:.72rem;font-weight:700;letter-spacing:1.2px;color:var(--txt-dim);
    text-transform:uppercase;margin:.4rem 0 .6rem;}

/* ---------- Chat bubbles ---------- */
.chat-row {display:flex;margin:.6rem 0;animation:slideUp .35s ease;}
@keyframes slideUp {from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.chat-row.user {justify-content:flex-end;}
.bubble {
    max-width:74%;padding:.85rem 1.1rem;border-radius:18px;line-height:1.55;font-size:.95rem;
    box-shadow:0 6px 20px rgba(0,0,0,.3);
}
.bubble.user {
    background:linear-gradient(135deg,var(--blue),#2563EB);color:#fff;
    border-bottom-right-radius:5px;box-shadow:0 6px 22px rgba(59,130,246,.4);
}
.bubble.bot {
    background:var(--glass);border:1px solid var(--glass-brd);backdrop-filter:blur(12px);
    border-bottom-left-radius:5px;
}
.msg-meta {font-size:.68rem;color:var(--txt-dim);margin-top:.35rem;display:flex;gap:.5rem;align-items:center;}
.msg-avatar {
    width:32px;height:32px;border-radius:10px;display:grid;place-items:center;font-size:16px;flex-shrink:0;margin:0 .55rem;
}
.av-bot  {background:linear-gradient(135deg,var(--cyan),var(--blue));}
.av-user {background:linear-gradient(135deg,var(--purple),var(--blue));}

/* ---------- Thinking animation ---------- */
.thinking {display:flex;align-items:center;gap:.55rem;padding:.5rem .2rem;}
.dot {width:9px;height:9px;border-radius:50%;background:var(--cyan);animation:blink 1.4s infinite both;}
.dot:nth-child(2){animation-delay:.2s;}.dot:nth-child(3){animation-delay:.4s;}
@keyframes blink {0%,80%,100%{opacity:.25;transform:scale(.8);}40%{opacity:1;transform:scale(1.15);}}

/* ---------- Suggestion chips (streamlit buttons) ---------- */
div[data-testid="stHorizontalBlock"] .stButton>button {
    background:var(--glass);border:1px solid var(--glass-brd);border-radius:999px;
    color:var(--txt);font-size:.8rem;font-weight:500;padding:.35rem .9rem;
    transition:all .25s ease;
}
div[data-testid="stHorizontalBlock"] .stButton>button:hover {
    border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 16px rgba(34,211,238,.35);transform:translateY(-2px);
}

/* ---------- Primary buttons ---------- */
.stButton>button {border-radius:12px;font-weight:600;transition:all .25s ease;}
button[kind="primary"], .stButton>button[kind="primary"] {
    background:linear-gradient(135deg,var(--blue),var(--purple))!important;
    border:none!important;color:#fff!important;
    box-shadow:0 0 22px rgba(59,130,246,.45)!important;
}
button[kind="primary"]:hover {box-shadow:0 0 34px rgba(139,92,246,.6)!important;transform:translateY(-1px);}

/* ---------- File uploader ---------- */
div[data-testid="stFileUploaderDropzone"] {
    background:var(--glass)!important;border:2px dashed var(--glass-brd)!important;
    border-radius:18px!important;transition:all .3s ease;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    border-color:var(--cyan)!important;box-shadow:0 0 24px rgba(34,211,238,.25);
}

/* ---------- Doc cards ---------- */
.doc-card {
    display:flex;align-items:center;gap:.9rem;padding:.8rem 1rem;margin-bottom:.6rem;
    background:var(--glass);border:1px solid var(--glass-brd);border-radius:14px;
    transition:all .25s ease;
}
.doc-card:hover {border-color:rgba(59,130,246,.5);transform:translateX(3px);}
.doc-ico {width:38px;height:38px;border-radius:10px;display:grid;place-items:center;font-size:18px;
    background:linear-gradient(135deg,#EF4444,#B91C1C);color:#fff;flex-shrink:0;}
.doc-info {flex:1;min-width:0;}
.doc-name {font-weight:600;font-size:.88rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.doc-sub {color:var(--txt-dim);font-size:.73rem;}
.doc-status {font-size:.68rem;font-weight:700;padding:.2rem .55rem;border-radius:999px;
    background:rgba(34,197,94,.15);color:#4ADE80;border:1px solid rgba(34,197,94,.35);}

/* ---------- Chat input ---------- */
div[data-testid="stChatInput"] {
    background:rgba(17,24,39,.85)!important;backdrop-filter:blur(16px);
    border:1px solid var(--glass-brd)!important;border-radius:16px!important;
    box-shadow:0 0 26px rgba(59,130,246,.18);
}

/* ---------- Source box ---------- */
.source-box {
    background:rgba(255,255,255,.03);border:1px solid var(--glass-brd);border-radius:10px;
    padding:.6rem .8rem;margin-bottom:.45rem;font-size:.8rem;color:var(--txt-dim);
}
.source-box b {color:var(--cyan);}

/* ---------- Footer ---------- */
.footer {text-align:center;color:var(--txt-dim);font-size:.82rem;
    padding:2rem 0 1rem;margin-top:2rem;border-top:1px solid var(--glass-brd);}
.footer a {color:var(--cyan);text-decoration:none;}
.footer a:hover {text-shadow:0 0 10px rgba(34,211,238,.6);}

/* ---------- Status pill ---------- */
.pill {display:inline-block;padding:.3rem .8rem;border-radius:999px;font-size:.75rem;font-weight:600;}
.pill.on  {background:rgba(34,197,94,.15);color:#4ADE80;border:1px solid rgba(34,197,94,.35);}
.pill.off {background:rgba(148,163,184,.12);color:var(--txt-dim);border:1px solid rgba(148,163,184,.3);}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ======================================================================
# SESSION STATE
# ======================================================================
def init_state():
    defaults = {
        "vectorstore": None,
        "chat_history": [],          # list of dicts: {role, content, sources, ts}
        "processed": [],             # list of dicts: {name, pages, size}
        "workdir": None,
        "num_chunks": 0,
        "prefill": "",
        "avg_time": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.workdir is None:
        st.session_state.workdir = tempfile.mkdtemp(prefix="docmind_")

init_state()

# ======================================================================
# RAG CORE (unchanged pipeline logic)
# ======================================================================
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant.\n\n"
     "Use ONLY the provided context to answer the question.\n\n"
     "If the answer is not present in the context, "
     'say: "I could not find the answer in the document."'),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}\n"),
])


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    return ChatMistralAI(model=model_name, temperature=temperature)


def build_vectorstore(files, persist_dir, chunk_size, chunk_overlap):
    """Save uploads, load, split, embed, and index. Returns (vs, chunks, meta)."""
    embedding = get_embedding_model()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    all_chunks, meta = [], []
    tmp = tempfile.mkdtemp(prefix="docmind_uploads_")
    try:
        for f in files:
            path = os.path.join(tmp, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata["source"] = f.name
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            meta.append({
                "name": f.name,
                "pages": len(docs),
                "size": f"{len(f.getbuffer()) / 1024:.0f} KB",
            })
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not all_chunks:
        return None, 0, []

    vs = Chroma.from_documents(
        documents=all_chunks, embedding=embedding, persist_directory=persist_dir
    )
    return vs, len(all_chunks), meta


def answer_question(query, vectorstore, model_name, temperature):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    prompt = PROMPT.invoke({"context": context, "question": query})
    llm = get_llm(model_name, temperature)
    resp = llm.invoke(prompt)
    return resp.content, docs


# ======================================================================
# UI HELPERS
# ======================================================================
def navbar():
    st.markdown("""
    <div class="navbar">
      <div class="nav-left">
        <div class="nav-logo">🧠</div>
        <div>
          <span class="nav-title">DocMind AI</span>
          <span class="nav-badge">AI</span>
        </div>
      </div>
      <div class="nav-right">
        <a class="nav-btn" href="[github.com](https://github.com)" target="_blank">
          <span class="material-icons-round" style="font-size:17px;">code</span> GitHub
        </a>
        <a class="nav-btn" href="#">
          <span class="material-icons-round" style="font-size:17px;">settings</span>
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)


def hero():
    st.markdown("""
    <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
    <div class="hero">
      <h1>Your AI Knowledge Assistant</h1>
      <p>Upload PDFs and chat with them using advanced Retrieval-Augmented
         Generation powered by Large Language Models.</p>
    </div>
    """, unsafe_allow_html=True)


def stat_cards():
    n_docs = len(st.session_state.processed)
    n_pages = sum(d["pages"] for d in st.session_state.processed)
    n_chunks = st.session_state.num_chunks
    n_emb = n_chunks
    db = "Chroma" if st.session_state.vectorstore else "—"
    rt = f"{st.session_state.avg_time:.1f}s" if st.session_state.avg_time else "—"

    cards = [
        ("description", n_docs, "Documents"),
        ("menu_book", n_pages, "Pages"),
        ("grid_view", n_chunks, "Chunks"),
        ("hub", n_emb, "Embeddings"),
        ("storage", db, "Vector DB"),
        ("bolt", rt, "Response Time"),
    ]
    html = '<div class="stat-grid">'
    for ico, num, lbl in cards:
        html += f"""
        <div class="stat">
          <div class="ico"><span class="material-icons-round"
               style="color:var(--cyan);">{ico}</span></div>
          <div class="num">{num}</div>
          <div class="lbl">{lbl}</div>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_message(role, content, sources, ts):
    if role == "user":
        st.markdown(f"""
        <div class="chat-row user">
          <div class="bubble user">{content}
            <div class="msg-meta" style="justify-content:flex-end;">🕐 {ts}</div>
          </div>
          <div class="msg-avatar av-user">🧑</div>
        </div>""", unsafe_allow_html=True)
    else:
        # bubble shell via HTML, markdown body via st.markdown for code/format support
        st.markdown(f"""
        <div class="chat-row bot">
          <div class="msg-avatar av-bot">🤖</div>
          <div class="bubble bot">""", unsafe_allow_html=True)
        st.markdown(content)
        st.markdown(f'<div class="msg-meta">🕐 {ts}</div></div></div>',
                    unsafe_allow_html=True)
        if sources:
            with st.expander("📎 Sources"):
                for i, d in enumerate(sources, 1):
                    page = d.metadata.get("page", "?")
                    src = d.metadata.get("source", "document")
                    snip = d.page_content[:280].strip().replace("\n", " ")
                    st.markdown(
                        f'<div class="source-box"><b>{i}. {src} · page {page}</b>'
                        f'<br>{snip}…</div>', unsafe_allow_html=True)


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div class="side-card profile">
      <div class="avatar">👤</div>
      <div><div class="name">Guest User</div>
      <div class="role">Free Workspace</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="side-title">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Choose LLM",
        ["mistral-small-2603"],
        label_visibility="collapsed",
    )
    embed_model = st.selectbox(
        "Embedding model", ["mistral-embed"], label_visibility="collapsed"
    )



    st.markdown('<div class="side-title">Session</div>', unsafe_allow_html=True)
    if st.button("🧹 Reset conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    status = "on" if st.session_state.vectorstore else "off"
    label = "● KB Ready" if st.session_state.vectorstore else "○ No KB"
    st.markdown(f'<span class="pill {status}">{label}</span>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="side-card" style="margin-top:1rem;">
      <div class="side-title" style="margin:0 0 .4rem;">About</div>
      <div style="font-size:.8rem;color:var(--txt-dim);line-height:1.5;">
        DocMind AI is a private, document-grounded assistant built on
        LangChain, Mistral & Chroma.
      </div>
      <div style="font-size:.72rem;color:var(--txt-dim);margin-top:.6rem;">
        🌙 Dark Mode · v1.0.0
      </div>
    </div>""", unsafe_allow_html=True)

# ======================================================================
# MAIN
# ======================================================================
navbar()
hero()
stat_cards()

# ---------------- Upload section ----------------
st.markdown("### 📤 Document Upload")
up_col, act_col = st.columns([3, 1])
with up_col:
    uploaded = st.file_uploader(
        "Drag & drop PDFs here",
        type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
with act_col:
    process = st.button("⚡ Build KB", type="primary", use_container_width=True)

if process:
    if not uploaded:
        st.warning("Upload at least one PDF first.")
    else:
        prog = st.progress(0, text="Reading documents…")
        try:
            prog.progress(30, text="Chunking & embedding…")
            vs, n_chunks, meta = build_vectorstore(
                uploaded, st.session_state.workdir, chunk_size, chunk_overlap
            )
            prog.progress(90, text="Indexing vectors…")
            if vs is None:
                st.error("Couldn't extract text from the PDF(s).")
            else:
                st.session_state.vectorstore = vs
                st.session_state.num_chunks = n_chunks
                st.session_state.processed = meta
                st.session_state.chat_history = []
                prog.progress(100, text="Done!")
                time.sleep(0.4)
                st.rerun()
        except Exception as e:
            st.error(f"Failed to build knowledge base: {e}")

# ---------------- Uploaded doc cards ----------------
if st.session_state.processed:
    st.markdown("#### 📁 Indexed Documents")
    for d in st.session_state.processed:
        st.markdown(f"""
        <div class="doc-card">
          <div class="doc-ico">📄</div>
          <div class="doc-info">
            <div class="doc-name">{d['name']}</div>
            <div class="doc-sub">{d['pages']} pages · {d['size']}</div>
          </div>
          <div class="doc-status">✓ Indexed</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Chat area ----------------
st.markdown("### 💬 Chat")

if st.session_state.vectorstore is None:
    st.info("👆 Upload PDFs and click **Build KB** to start chatting.")
else:
    # history
    for m in st.session_state.chat_history:
        render_message(m["role"], m["content"], m.get("sources"), m["ts"])

    # suggestion chips
    st.markdown("###### 💡 Suggested")
    suggestions = [
        "Summarize this PDF", "Explain chapter 1", "Key points", "Create quiz",
        "Generate notes", "Important formulas", "Translate", "Extract tables",
    ]
    cols = st.columns(4)
    for i, s in enumerate(suggestions):
        if cols[i % 4].button(s, key=f"chip_{i}", use_container_width=True):
            st.session_state.prefill = s

    # input (chat_input can't be pre-filled, so we honor prefill as a submitted query)
    typed = st.chat_input("Ask anything about your documents…")
    query = typed or st.session_state.prefill
    st.session_state.prefill = ""

    if query:
        now = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append(
            {"role": "user", "content": query, "sources": None, "ts": now})
        render_message("user", query, None, now)

        # thinking animation
        holder = st.empty()
        holder.markdown("""
        <div class="chat-row bot">
          <div class="msg-avatar av-bot">🤖</div>
          <div class="bubble bot"><div class="thinking">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            <span style="color:var(--txt-dim);font-size:.85rem;">Thinking…</span>
          </div></div>
        </div>""", unsafe_allow_html=True)

        t0 = time.time()
        try:
            answer, sources = answer_question(
                query, st.session_state.vectorstore, model_name, temperature)
        except Exception as e:
            answer, sources = f"⚠️ Something went wrong: {e}", []
        elapsed = time.time() - t0
        st.session_state.avg_time = elapsed

        holder.empty()
        now = datetime.now().strftime("%H:%M")
        render_message("assistant", answer, sources, now)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources, "ts": now})
        st.rerun()

# ---------------- Footer ----------------
st.markdown("""
<div class="footer">
  Made with ❤️ using <b>Streamlit</b> · <b>LangChain</b> · <b>Mistral AI</b><br>
  <a href="[github.com](https://github.com)" target="_blank">GitHub</a> &nbsp;·&nbsp; v1.0.0
</div>""", unsafe_allow_html=True)
