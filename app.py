import streamlit as st  #type: ignore
import os
import shutil
from modules.ingestion import ingest_documents
from modules.embeddings import get_embedding_model
from modules.vectorstore import (
    build_vectorstore, save_vectorstore,
    load_vectorstore, vectorstore_exists
)
from modules.retriever import get_relevant_context
from modules.llm_chain import get_prompt_template, get_llm, build_rag_chain, get_answer
from config import DATA_DIR, VECTORSTORE_DIR

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Doc Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .status-box {
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache expensive resources ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load embedding model and LLM once, cache for session."""
    embedding_model = get_embedding_model()
    llm             = get_llm()
    prompt          = get_prompt_template()
    chain           = build_rag_chain(llm, prompt)
    return embedding_model, chain


@st.cache_resource(show_spinner=False)
def load_vs(embedding_model):
    """Load vectorstore if it exists."""
    if vectorstore_exists():
        return load_vectorstore(embedding_model)
    return None


# ── Initialize session state ──────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False


# ── Header ────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🤖 Enterprise Document Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">RAG-powered Q&A — Ask anything about your documents</div>', unsafe_allow_html=True)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Manager")
    st.markdown("Upload PDFs to build your knowledge base.")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents"
    )

    # Process button
    if uploaded_files:
        if st.button("⚙️ Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents..."):

                # Save uploaded files to data/
                os.makedirs(DATA_DIR, exist_ok=True)
                for file in uploaded_files:
                    save_path = os.path.join(DATA_DIR, file.name)
                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                # Run ingestion pipeline
                st.info("📄 Extracting text from PDFs...")
                chunks = ingest_documents(DATA_DIR)

                # Build vectorstore
                st.info("🔢 Building vector index...")
                embedding_model, chain = load_models()
                vectorstore = build_vectorstore(chunks, embedding_model)
                save_vectorstore(vectorstore)

                # Store in session
                st.session_state.vectorstore    = vectorstore
                st.session_state.docs_processed = True

                # Clear cache so it reloads fresh
                load_vs.clear()

            st.success(f"✅ Processed {len(uploaded_files)} file(s) — {len(chunks)} chunks indexed!")

    st.divider()

    # Load existing index
    st.subheader("📦 Existing Index")
    if vectorstore_exists():
        if st.button("📂 Load Saved Index", use_container_width=True):
            with st.spinner("Loading saved index..."):
                embedding_model, chain = load_models()
                st.session_state.vectorstore    = load_vectorstore(embedding_model)
                st.session_state.docs_processed = True
            st.success("✅ Index loaded successfully!")
    else:
        st.info("No saved index found.\nUpload PDFs to get started.")

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=6, value=3)


# ── Main chat area ────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat")

    # Status indicator
    if st.session_state.docs_processed and st.session_state.vectorstore:
        st.success("✅ Knowledge base ready — Ask your questions below!")
    else:
        st.warning("⚠️ No documents loaded. Upload PDFs in the sidebar first.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Sources"):
                    for src in message["sources"]:
                        st.markdown(f"""
<div class="source-box">
📄 <b>{src['source']}</b> — Page {src['page']}<br>
<small>{src['preview']}</small>
</div>
""", unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):

        if not st.session_state.docs_processed or not st.session_state.vectorstore:
            st.error("Please upload and process documents first!")
        else:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents and generating answer..."):

                    embedding_model, chain = load_models()

                    # Retrieve context
                    context, chunks = get_relevant_context(
                        prompt,
                        st.session_state.vectorstore,
                        k=top_k
                    )

                    # Get answer
                    answer = get_answer(prompt, context, chain)

                    # Display answer
                    st.markdown(answer)

                    # Build source list
                    sources = [
                        {
                            "source": c.metadata.get("source", "Unknown"),
                            "page":   c.metadata.get("page", "?"),
                            "preview": c.page_content[:150] + "..."
                        }
                        for c in chunks
                    ]

                    # Show sources
                    with st.expander("📄 View Sources"):
                        for src in sources:
                            st.markdown(f"""
<div class="source-box">
📄 <b>{src['source']}</b> — Page {src['page']}<br>
<small>{src['preview']}</small>
</div>
""", unsafe_allow_html=True)

                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

# ── Right column — stats ──────────────────────────────────────────────
with col2:
    st.subheader("📊 Stats")

    if st.session_state.docs_processed:
        # Count PDFs in data/
        pdf_count = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]) if os.path.exists(DATA_DIR) else 0
        st.metric("📄 PDFs Loaded", pdf_count)
        st.metric("💬 Questions Asked", len([m for m in st.session_state.chat_history if m["role"] == "user"]))
        st.metric("🔍 Chunks per Query", top_k)

        st.divider()
        st.subheader("📁 Loaded Files")
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                if f.endswith(".pdf"):
                    st.markdown(f"📄 `{f}`")
    else:
        st.info("Load documents to see stats.")