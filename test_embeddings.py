from modules.embeddings import get_embedding_model, test_embedding

# Load model
model = get_embedding_model()

# Test single embedding
vector = test_embedding(model, "What are the job responsibilities?")

print(f"✅ Embedding test passed!")
print(f"   Input : 'What are the job responsibilities?'")
print(f"   Vector dimensions : {len(vector)}")
print(f"   First 5 values    : {[round(v, 4) for v in vector[:5]]}")