# 🚀 RAG Chatbot System

## 📌 Overview
This project is a Retrieval-Augmented Generation (RAG) based chatbot that retrieves relevant information from documents and generates accurate responses using a Large Language Model (LLM).

---

## ⚙️ Features
- 📄 Document ingestion pipeline
- 🔍 Semantic search using embeddings
- 🧠 LLM-based response generation
- 📦 Modular architecture (embedding, retriever, LLM chain)
- ⚡ Fast and scalable pipeline

---

## 🧠 Tech Stack
- Python
- LangChain / LangGraph
- FastAPI
- Vector Database (FAISS / Chroma)
- OpenRouter / LLM APIs

---

## 📂 Project Structure
chatbot/
│── modules/
│   ├── embedding.py
│   ├── ingestion.py
│   ├── retriever.py
│   ├── llm_chain.py
│   └── vectorstore.py
│
│── data/              # Input documents
│── vectorstore/       # Stored embeddings
│── app.py             # Main application
│── config.py
│── requirements.txt

## 🚀 Setup Instructions

### 1️⃣ Clone the repository
git clone https://github.com/janvee1201/rag-chatbot.git
cd rag-chatbot

### 2️⃣ Install dependencies
pip install -r requirements.txt

### 3️⃣ Create `.env` file

GMAIL_SENDER=your_email
GMAIL_APP_PASSWORD=your_password
OPENROUTER_API_KEY=your_api_key

### 4️⃣ Run the project

python app.py

## 🔄 Workflow
1. Documents are ingested and converted into embeddings  
2. Embeddings are stored in a vector database  
3. User query is converted into embedding  
4. Relevant documents are retrieved  
5. LLM generates final response  

---

## 📌 Future Improvements
- 🌐 Web UI (React / Streamlit)
- 🤖 Agentic workflows (LangGraph)
- 📡 Real-time streaming responses
- ☁️ Deployment (AWS / Render)

---

