from modules.embeddings import get_embedding_model
from modules.vectorstore import load_vectorstore
from modules.retriever import retrieve_with_scores, get_relevant_context

# Load model and vectorstore (already built in Step 4)
model       = get_embedding_model()
vectorstore = load_vectorstore(model)

# ── Test 1: Retrieve with scores ──────────────────────────────────────
print("=" * 50)
print("TEST 1 — Retrieval with Similarity Scores")
print("=" * 50)

query   = "What skills are required for this job?"
results = retrieve_with_scores(query, vectorstore, k=3)

for i, (doc, score) in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Score  : {round(score, 4)}  (lower = more similar)")
    print(f"Source : {doc.metadata['source']}")
    print(f"Page   : {doc.metadata['page']}")
    print(f"Preview: {doc.page_content[:200]}")

# ── Test 2: Formatted context for LLM ────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2 — Formatted Context Block for LLM")
print("=" * 50)

query2           = "What is the job location?"
context, chunks  = get_relevant_context(query2, vectorstore, k=2)

print(f"\nQuery  : '{query2}'")
print(f"Chunks : {len(chunks)}\n")
print("--- Formatted Context ---")
print(context)