import requests
from langchain_core.prompts import PromptTemplate   #type: ignore
from config import OPENROUTER_API_KEY, LLM_MODEL


# ── Build the prompt template ─────────────────────────────────────────
def get_prompt_template() -> PromptTemplate:
    template = """You are a helpful assistant that answers questions based strictly on the provided document context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer only using the context provided above
- If the answer is not in the context, say "I could not find this information in the provided documents"
- Be concise and factual
- Do not make up information

Answer:"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


# ── Load LLM client ───────────────────────────────────────────────────
def get_llm() -> dict:
    """
    Uses OpenRouter free API.
    - Free tier available, no credit card needed
    - Supports LLaMA-3, Mistral and more
    - Simple REST API, no library issues
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OpenRouter API key not found!\n"
            "Add OPENROUTER_API_KEY=your_key to your .env file\n"
            "Get free key at: https://openrouter.ai/keys"
        )

    print(f"🤖 Loading LLM: {LLM_MODEL}")
    print("   (Using OpenRouter free API)\n")

    # Test key works
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Enterprise Doc Chatbot"
    }

    print("✅ LLM ready\n")
    return {"headers": headers, "model": LLM_MODEL}


# ── Build chain ───────────────────────────────────────────────────────
def build_rag_chain(llm: dict, prompt: PromptTemplate) -> dict:
    return {"llm": llm, "prompt": prompt}


# ── Master function: query the RAG system ────────────────────────────
def get_answer(question: str, context: str, chain: dict) -> str:
    """
    Sends question + context to OpenRouter free LLM.
    Returns clean answer string.
    """
    print(f"💬 Generating answer for: '{question}'\n")

    llm     = chain["llm"]
    prompt  = chain["prompt"]
    headers = llm["headers"]
    model   = llm["model"]

    # Format full prompt
    full_prompt = prompt.format(context=context, question=question)

    # Call OpenRouter API
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions strictly based on provided document context."
            },
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return f"❌ API Error {response.status_code}: {response.text}"

    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()

    # Clean up if model echoes "Answer:"
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    return answer