from langchain_community.vectorstores import FAISS #type: ignore
from langchain_core.documents import Document #type: ignore
from config import TOP_K_RESULTS


# ── Core retrieval function ───────────────────────────────────────────
def retrieve_chunks(
    query: str,
    vectorstore: FAISS,
    k: int = TOP_K_RESULTS
) -> list[Document]:
    """
    Takes a user query, embeds it, searches FAISS index,
    returns top-k most semantically similar chunks.
    """
    print(f"🔍 Retrieving top {k} chunks for query:")
    print(f"   '{query}'\n")

    results = vectorstore.similarity_search(query, k=k)

    print(f"   Found {len(results)} relevant chunks\n")
    return results


# ── Retrieval with similarity scores ─────────────────────────────────
def retrieve_with_scores(
    query: str,
    vectorstore: FAISS,
    k: int = TOP_K_RESULTS
) -> list[tuple[Document, float]]:
    """
    Same as retrieve_chunks but also returns similarity scores.
    Lower score = more similar (FAISS uses L2 distance by default).
    Useful for debugging retrieval quality.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


# ── Format retrieved chunks into a single context string ─────────────
def format_context(chunks: list[Document]) -> str:
    """
    Merges retrieved chunks into one clean context block.
    This gets passed directly to the LLM as context.

    Format:
    [Source: file.pdf | Page: 2]
    chunk text here...
    ---
    """
    context_parts = []

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "Unknown")
        page   = chunk.metadata.get("page", "?")

        part = (
            f"[Source: {source} | Page: {page}]\n"
            f"{chunk.page_content.strip()}"
        )
        context_parts.append(part)

    context = "\n---\n".join(context_parts)
    return context


# ── Master retriever pipeline ─────────────────────────────────────────
def get_relevant_context(
    query: str,
    vectorstore: FAISS,
    k: int = TOP_K_RESULTS
) -> tuple[str, list[Document]]:
    """
    Full retrieval pipeline:
    query → search FAISS → format context → return

    Returns:
        context  : formatted string ready for LLM prompt
        chunks   : raw Document list for citations in UI
    """
    chunks  = retrieve_chunks(query, vectorstore, k)
    context = format_context(chunks)
    return context, chunks