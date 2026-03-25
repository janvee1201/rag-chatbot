import os
from langchain_huggingface import HuggingFaceEmbeddings   #type: ignore
from config import EMBEDDING_MODEL


# ── Load embedding model (downloads once, cached locally) ─────────────
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads the sentence-transformer embedding model locally.
    Model: all-MiniLM-L6-v2
    - Size  : ~80MB (downloads once, cached after)
    - Speed : Very fast, runs on CPU
    - Output: 384-dimensional vectors per chunk
    No API key needed — runs 100% locally.
    """
    print(f"🔄 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (First run downloads ~80MB — cached after)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True}
        # normalize=True improves cosine similarity search accuracy
    )

    print("✅ Embedding model loaded successfully\n")
    return embeddings


# ── Quick test: embed a single sentence ───────────────────────────────
def test_embedding(model: HuggingFaceEmbeddings, text: str = "Hello world") -> list:
    """
    Embeds a single string and returns its vector.
    Use this to verify the model is working.
    """
    vector = model.embed_query(text)
    return vector