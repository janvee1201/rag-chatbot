from modules.ingestion import ingest_documents

chunks = ingest_documents()

for i, chunk in enumerate(chunks[:2]):
    print(f"--- Chunk {i+1} ---")
    print(f"Source : {chunk.metadata['source']}")
    print(f"Page   : {chunk.metadata['page']}")
    print(f"Length : {len(chunk.page_content)} chars")
    print(f"Preview: {chunk.page_content[:200]}")
    print()