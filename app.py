"""
RAG PDF Chatbot - Streamlit UI
--------------------------------
Upload PDFs, build a vector store with Mistral embeddings, and chat with
your documents using ChatMistralAI. Combines create_database.py + main.py
into a single interactive app.
"""

import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="RAG PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: #0e1117;
    }

    /* Header */
    .rag-header {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #2d3748;
        margin-bottom: 1.5rem;
    }
    .rag-header h1 {
        margin: 0;
        font-size: 1.9rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .rag-header p {
        margin: 0.3rem 0 0 0;
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #2d3748;
    }

    /* Status pill */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .status-ready {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .status-empty {
        background: rgba(148, 163, 184, 0.12);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Source chunk box */
    .source-box {
        background: #1a2130;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    .source-box b { color: #93c5fd; }

    div[data-testid="stChatMessage"] {
        background: #161b26;
        border: 1px solid #232b3b;
        border-radius: 14px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="rag-header">
        <h1>📄 RAG PDF Chat</h1>
        <p>Upload your PDFs, build a knowledge base, and ask questions grounded in your documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, sources)
if "processed_names" not in st.session_state:
    st.session_state.processed_names = []
if "workdir" not in st.session_state:
    st.session_state.workdir = tempfile.mkdtemp(prefix="rag_chroma_")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2603")


def build_vectorstore(uploaded_files, persist_dir):
    """Save uploads to disk, load, split, embed, and index them."""
    embedding_model = get_embedding_model()
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    tmp_pdf_dir = tempfile.mkdtemp(prefix="rag_uploads_")
    try:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(tmp_pdf_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uploaded_file.name

            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
    finally:
        shutil.rmtree(tmp_pdf_dir, ignore_errors=True)

    if not all_chunks:
        return None

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    return vectorstore


def answer_question(query, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = PROMPT.invoke({"context": context, "question": query})
    llm = get_llm()
    response = llm.invoke(final_prompt)
    return response.content, docs


# ----------------------------------------------------------------------
# Sidebar - upload & build knowledge base
# ----------------------------------------------------------------------
with st.sidebar:
    st.subheader("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        process_clicked = st.button("⚡ Process", use_container_width=True)
    with col2:
        reset_clicked = st.button("🗑️ Reset", use_container_width=True)

    if reset_clicked:
        shutil.rmtree(st.session_state.workdir, ignore_errors=True)
        st.session_state.workdir = tempfile.mkdtemp(prefix="rag_chroma_")
        st.session_state.vectorstore = None
        st.session_state.chat_history = []
        st.session_state.processed_names = []
        st.rerun()

    if process_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading, chunking and embedding your documents..."):
                try:
                    vs = build_vectorstore(uploaded_files, st.session_state.workdir)
                    if vs is None:
                        st.error("Couldn't extract any text from the uploaded PDF(s).")
                    else:
                        st.session_state.vectorstore = vs
                        st.session_state.processed_names = [f.name for f in uploaded_files]
                        st.session_state.chat_history = []
                        st.success(f"Indexed {len(uploaded_files)} document(s).")
                except Exception as e:
                    st.error(f"Failed to build knowledge base: {e}")

    if st.session_state.vectorstore is not None:
        st.markdown('<span class="status-pill status-ready">● Ready to chat</span>', unsafe_allow_html=True)
        st.caption("Indexed files:")
        for name in st.session_state.processed_names:
            st.markdown(f"- 📄 {name}")
    else:
        st.markdown('<span class="status-pill status-empty">○ No documents indexed yet</span>', unsafe_allow_html=True)

    st.divider()
    st.caption("Powered by LangChain · Mistral AI · Chroma")

# ----------------------------------------------------------------------
# Main - chat interface
# ----------------------------------------------------------------------
if st.session_state.vectorstore is None:
    st.info("👈 Upload PDF(s) in the sidebar and click **Process** to start chatting.")
else:
    for role, content, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and sources:
                with st.expander("📎 View sources used"):
                    for i, doc in enumerate(sources, 1):
                        page = doc.metadata.get("page", "?")
                        src = doc.metadata.get("source", "document")
                        snippet = doc.page_content[:300].strip().replace("\n", " ")
                        st.markdown(
                            f'<div class="source-box"><b>{i}. {src} — page {page}</b><br>{snippet}...</div>',
                            unsafe_allow_html=True,
                        )

    query = st.chat_input("Ask something about your document(s)...")
    if query:
        st.session_state.chat_history.append(("user", query, None))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = answer_question(query, st.session_state.vectorstore)
                except Exception as e:
                    answer, sources = f"Something went wrong: {e}", []
            st.markdown(answer)
            if sources:
                with st.expander("📎 View sources used"):
                    for i, doc in enumerate(sources, 1):
                        page = doc.metadata.get("page", "?")
                        src = doc.metadata.get("source", "document")
                        snippet = doc.page_content[:300].strip().replace("\n", " ")
                        st.markdown(
                            f'<div class="source-box"><b>{i}. {src} — page {page}</b><br>{snippet}...</div>',
                            unsafe_allow_html=True,
                        )

        st.session_state.chat_history.append(("assistant", answer, sources))