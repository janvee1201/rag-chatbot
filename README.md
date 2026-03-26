# 🤖 Enterprise Document Chatbot
### RAG-Powered Intelligent Document Q&A System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-FF6F00?style=for-the-badge&logo=meta&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-Free_LLM-6366F1?style=for-the-badge&logoColor=white)

**Ask questions. Get answers. Grounded in your documents.**

[Features](#-features) • [Architecture](#-architecture) • [Setup](#-quick-start) • [Usage](#-usage) • [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

A production-grade **Retrieval-Augmented Generation (RAG)** system that lets you chat with your PDF documents using natural language. Upload any PDF, and instantly query it with semantic search — powered by local embeddings, FAISS vector store, and a free LLM via OpenRouter.

> **Zero paid APIs required.** Everything runs free — local embeddings + OpenRouter free tier.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Ingestion** | Upload and process multiple PDFs with smart chunking |
| 🔢 **Semantic Embeddings** | Local `sentence-transformers` model — no API needed |
| ⚡ **FAISS Vector Search** | Lightning-fast similarity search across 100+ document chunks |
| 🤖 **Free LLM Answers** | Powered by OpenRouter's free LLaMA/Mistral models |
| 💬 **Chat Interface** | Clean Streamlit UI with full chat history |
| 📎 **Source Citations** | Every answer shows exactly which page it came from |
| 💾 **Persistent Index** | FAISS index saved to disk — no re-processing on restart |
| ⚙️ **Configurable** | Tune chunk size, overlap, top-k retrieval via `.env` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                                                             │
│   PDF Files → pdfplumber → Text Chunks → Embeddings → FAISS │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                           │
│                                                             │
│   User Query → Embed Query → FAISS Search → Top-K Chunks    │
│                                    │                        │
│              LLM Answer ← Prompt Template ← Context         │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Document Parsing** | `pdfplumber` | Accurate text + layout extraction |
| **Chunking** | `LangChain RecursiveCharacterTextSplitter` | Smart boundary-aware splitting |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 80MB, runs locally, 384-dim vectors |
| **Vector Store** | `FAISS` | Sub-millisecond similarity search |
| **LLM** | `OpenRouter` (LLaMA / Mistral free models) | Free, fast, no GPU needed |
| **Orchestration** | `LangChain` | RAG pipeline management |
| **UI** | `Streamlit` | Rapid, clean chat interface |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Free accounts at [HuggingFace](https://huggingface.co) and [OpenRouter](https://openrouter.ai)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/enterprise-doc-chatbot.git
cd enterprise-doc-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
# HuggingFace (free token - huggingface.co/settings/tokens)
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter (free key - openrouter.ai/keys)
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx

# Chunking config (optional - defaults work great)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=3
```

### 4. Launch the app

```bash
# Windows
python -m streamlit run app.py

# Mac/Linux
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📖 Usage

### Via Streamlit UI (Recommended)

1. **Upload PDFs** — Drag and drop files in the sidebar
2. **Process** — Click "⚙️ Process Documents" to build the index
3. **Ask** — Type any question in the chat box
4. **Explore** — Click "View Sources" under any answer to see citations

### Via Python (Programmatic)

```python
from modules.ingestion import ingest_documents
from modules.embeddings import get_embedding_model
from modules.vectorstore import build_vectorstore, save_vectorstore
from modules.retriever import get_relevant_context
from modules.llm_chain import get_llm, get_prompt_template, build_rag_chain, get_answer

# Build pipeline
chunks      = ingest_documents()
model       = get_embedding_model()
vectorstore = build_vectorstore(chunks, model)
save_vectorstore(vectorstore)

# Query
llm     = get_llm()
prompt  = get_prompt_template()
chain   = build_rag_chain(llm, prompt)

context, chunks = get_relevant_context("What are the required skills?", vectorstore)
answer          = get_answer("What are the required skills?", context, chain)
print(answer)
```

---

## 📁 Project Structure

```
enterprise-doc-chatbot/
│
├── 📂 data/                         # Drop your PDFs here
│   └── your_document.pdf
│
├── 📂 vectorstore/                  # Auto-generated FAISS index
│   ├── doc_index.faiss
│   └── doc_index.pkl
│
├── 📂 modules/                      # Core pipeline modules
│   ├── __init__.py
│   ├── ingestion.py                 # PDF loading & chunking
│   ├── embeddings.py                # HuggingFace embedding model
│   ├── vectorstore.py               # FAISS build/save/load
│   ├── retriever.py                 # Semantic search & context formatting
│   └── llm_chain.py                 # LLM integration & RAG chain
│
├── app.py                           # Streamlit chat UI
├── config.py                        # Central configuration
├── requirements.txt                 # Dependencies
├── .env                             # API keys (never commit this!)
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

All settings in `config.py` / `.env`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `3` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `LLM_MODEL` | `arcee-ai/trinity-large-preview:free` | OpenRouter model ID |

---

## 🧪 Testing Modules Individually

```bash
# Test ingestion
python test_ingestion.py

# Test embeddings
python test_embeddings.py

# Test vector store
python test_vectorstore.py

# Test retriever
python test_retriever.py

# Test full RAG pipeline
python test_llm_chain.py
```

---

## 🔧 Troubleshooting

<details>
<summary><b>ModuleNotFoundError on langchain imports</b></summary>

```bash
pip install langchain-core langchain-community langchain-text-splitters langchain-huggingface
```
</details>

<details>
<summary><b>Rate limit error from OpenRouter (429)</b></summary>

Switch to a different free model in `config.py`:
```python
LLM_MODEL = "google/gemma-3-1b-it:free"
# or
LLM_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"
```
Browse all free models at [openrouter.ai/models](https://openrouter.ai/models?order=newest&supported_parameters=free)
</details>

<details>
<summary><b>streamlit: command not found (Windows)</b></summary>

```bash
python -m streamlit run app.py
```
</details>

<details>
<summary><b>No PDFs found error</b></summary>

Make sure your PDF is inside the `data/` folder:
```bash
ls data/   # should show your PDF file
```
</details>

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Embedding model size | ~80MB (cached locally) |
| Avg. ingestion time (10-page PDF) | ~2 seconds |
| Avg. query response time | < 3 seconds |
| Chunk retrieval (FAISS) | < 50ms |
| Supported document size | 100+ pages |

---

## 🗺️ Roadmap

- [ ] Multi-document comparison queries
- [ ] Chat history export (PDF / CSV)
- [ ] Support for `.docx` and `.txt` files
- [ ] Re-ranking retrieved chunks for better accuracy
- [ ] Docker deployment support
- [ ] Conversation memory across sessions

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — RAG orchestration framework
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI Similarity Search
- [Sentence Transformers](https://www.sbert.net) — Local embedding models
- [OpenRouter](https://openrouter.ai) — Free LLM inference API
- [Streamlit](https://streamlit.io) — Rapid UI framework

---

<div align="center">

Built with ❤️ | RAG • LangChain • FAISS • Streamlit

⭐ **Star this repo if it helped you!** ⭐

</div>