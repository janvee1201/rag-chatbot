import os
import pdfplumber  #type: ignore
from tqdm import tqdm
from langchain_core.documents import Document  #type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  #type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter   #type: ignore
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATA_DIR

# ── Step 1: Extract raw text from a single PDF ─────────────────────────
def load_pdf(file_path: str) -> list[Document]:
    """
    Reads a PDF page by page using pdfplumber.
    Returns a list of LangChain Document objects (one per page).
    """
    documents = []

    print(f"📄 Loading: {os.path.basename(file_path)}")

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="  Reading pages")):
            text = page.extract_text()

            # Skip empty/unreadable pages
            if not text or text.strip() == "":
                continue

            # Each page becomes a Document with metadata
            doc = Document(
                page_content=text.strip(),
                metadata={
                    "source": os.path.basename(file_path),
                    "page": page_num + 1
                }
            )
            documents.append(doc)

    print(f"  ✅ Extracted {len(documents)} pages\n")
    return documents


# ── Step 2: Load all PDFs from the data/ folder ────────────────────────
def load_all_pdfs(data_dir: str = DATA_DIR) -> list[Document]:
    """
    Scans the data/ folder and loads every PDF found.
    Returns combined list of Documents from all files.
    """
    all_docs = []
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{data_dir}'. Drop some PDFs in the data/ folder first.")

    print(f"🗂️  Found {len(pdf_files)} PDF(s): {pdf_files}\n")

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        docs = load_pdf(path)
        all_docs.extend(docs)

    print(f"📦 Total pages loaded: {len(all_docs)}\n")
    return all_docs


# ── Step 3: Split pages into smaller overlapping chunks ────────────────
def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits large page-level documents into smaller overlapping chunks.

    Why overlap? So context at chunk boundaries isn't lost.
    Example: chunk_size=500, overlap=50 means each chunk shares
    50 chars with the next → no hard cutoffs mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        # Tries to split at paragraphs first, then lines, then sentences
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️  Chunking complete:")
    print(f"   Pages input  : {len(documents)}")
    print(f"   Chunks output: {len(chunks)}")
    print(f"   Chunk size   : {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars\n")

    return chunks


# ── Step 4: Master pipeline function ──────────────────────────────────
def ingest_documents(data_dir: str = DATA_DIR) -> list[Document]:
    """
    Full ingestion pipeline:
    data/ folder → load PDFs → extract text → chunk → return

    Call this from other modules.
    """
    print("=" * 50)
    print("🚀 Starting Document Ingestion Pipeline")
    print("=" * 50 + "\n")

    documents = load_all_pdfs(data_dir)
    chunks    = chunk_documents(documents)

    print("=" * 50)
    print(f"✅ Ingestion complete — {len(chunks)} chunks ready")
    print("=" * 50 + "\n")

    return chunks