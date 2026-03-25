from modules.embeddings import get_embedding_model
from modules.vectorstore import load_vectorstore
from modules.retriever import get_relevant_context
from modules.llm_chain import get_prompt_template, get_llm, build_rag_chain, get_answer

# ── Load all components ───────────────────────────────────────────────
model       = get_embedding_model()
vectorstore = load_vectorstore(model)
prompt      = get_prompt_template()
llm         = get_llm()
chain       = build_rag_chain(llm, prompt)

# ── Test questions ────────────────────────────────────────────────────
questions = [
    "What is the job location?",
    "What programming skills are required?",
    "What does the company offer to employees?"
]

print("=" * 55)
print("🧪 RAG PIPELINE — END TO END TEST")
print("=" * 55)

for question in questions:
    print(f"\n❓ Question: {question}")
    print("-" * 40)

    # Retrieve relevant context
    context, chunks = get_relevant_context(question, vectorstore, k=3)

    # Generate answer
    answer = get_answer(question, context, chain)

    print(f"🤖 Answer: {answer}")
    print(f"📄 Based on {len(chunks)} chunks from document")
    print()