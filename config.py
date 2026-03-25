import os
from dotenv import load_dotenv  #type: ignore

load_dotenv()

# ── Paths ────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# ── Chunking ─────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 50))

# ── Embeddings (local, free) ──────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#  Lightweight (80MB), fast, high quality for semantic search

# ── LLM (free HuggingFace) ───────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL         = "arcee-ai/trinity-large-preview:free"
#  Best free LLM on HuggingFace for Q&A tasks

# ── Retrieval ─────────────────────────────────────────
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS", 3))

# ── FAISS index name ──────────────────────────────────
FAISS_INDEX_NAME = "doc_index"