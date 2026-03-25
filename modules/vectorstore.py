import os
from langchain_community.vectorstores import FAISS #type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  #type: ignore
from langchain_core.documents import Document #type: ignore
from config import VECTORSTORE_DIR, FAISS_INDEX_NAME


# ── Build FAISS index from chunks ─────────────────────────────────────
def build_vectorstore(chunks: list[Document], embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    Takes document chunks + embedding model.
    Converts every chunk into a vector and stores in FAISS index.
    """
    print(f"⚙️  Building FAISS index from {len(chunks)} chunks...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    print(f"✅ FAISS index built successfully\n")
    return vectorstore


# ── Save FAISS index to disk ──────────────────────────────────────────
def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Persists the FAISS index to vectorstore/ folder.
    So you don't rebuild every time you restart the app.
    """
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    vectorstore.save_local(
        folder_path=VECTORSTORE_DIR,
        index_name=FAISS_INDEX_NAME
    )

    print(f"💾 FAISS index saved to: {VECTORSTORE_DIR}/{FAISS_INDEX_NAME}")
    print(f"   Files created: {FAISS_INDEX_NAME}.faiss + {FAISS_INDEX_NAME}.pkl\n")


# ── Load FAISS index from disk ────────────────────────────────────────
def load_vectorstore(embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    Loads a previously saved FAISS index from disk.
    Much faster than rebuilding from scratch every run.
    """
    index_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)

    if not os.path.exists(f"{index_path}.faiss"):
        raise FileNotFoundError(
            f"No FAISS index found at '{index_path}'.\n"
            "Run build_vectorstore() first to create it."
        )

    print(f"📂 Loading FAISS index from: {VECTORSTORE_DIR}")

    vectorstore = FAISS.load_local(
        folder_path=VECTORSTORE_DIR,
        embeddings=embedding_model,
        index_name=FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True
        # Safe here — we created this file ourselves
    )

    print(f"✅ FAISS index loaded successfully\n")
    return vectorstore


# ── Check if saved index exists ───────────────────────────────────────
def vectorstore_exists() -> bool:
    """
    Returns True if a saved FAISS index already exists on disk.
    Used in app.py to skip rebuilding if index is fresh.
    """
    index_path = os.path.join(VECTORSTORE_DIR, FAISS_INDEX_NAME)
    return os.path.exists(f"{index_path}.faiss")