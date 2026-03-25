from modules.ingestion import ingest_documents
from modules.embeddings import get_embedding_model
from modules.vectorstore import build_vectorstore, save_vectorstore, load_vectorstore, vectorstore_exists

# Step 1 — Ingest
chunks = ingest_documents()

# Step 2 — Load embedding model
model = get_embedding_model()

# Step 3 — Build FAISS index
vectorstore = build_vectorstore(chunks, model)

# Step 4 — Save to disk
save_vectorstore(vectorstore)

# Step 5 — Reload from disk and verify
print("🔄 Testing reload from disk...")
reloaded = load_vectorstore(model)

# Step 6 — Quick similarity search test
query = "What are the required skills?"
results = reloaded.similarity_search(query, k=2)

print(f"\n🔍 Test Search Query: '{query}'")
print(f"   Results found: {len(results)}\n")

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Source : {doc.metadata['source']}")
    print(f"Page   : {doc.metadata['page']}")
    print(f"Preview: {doc.page_content[:200]}")
    print()

print(f"✅ vectorstore_exists() → {vectorstore_exists()}")