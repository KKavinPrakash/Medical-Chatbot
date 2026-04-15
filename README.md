# Medical Chatbot - Dual Architecture RAG

This repository houses a comprehensive Medical Question Answering system built on Retrieval-Augmented Generation (RAG). It uniquely features **two separate technological implementations** running side-by-side:

1. **FAISS Dense Vector Database** implementation (Original)
2. **BM25 Vectorless Page Index** implementation (Free-Tier Native)

Both iterations leverage the **Groq API** (`llama-3.1-8b-instant`) for ultra-low latency response generation.

## 🏗️ Architectures Available

### 1. The Vector Setup (FAISS)
* **Ingestion**: `create_memory_for_llm.py`
* **Execution**: `medibot.py`
Uses HuggingFace BioBERT embeddings to convert PDF text chunks into dense semantic vectors and performs cosine similarity matching inside a local FAISS database.

### 2. The Vectorless Setup (BM25 Page Index)
* **Ingestion**: `create_memory_vectorless.py`
* **Execution**: `medibot_vectorless.py`
* **Memory Support**: Integrated LangChain `History-Aware Retriever`
A completely Free-Tier setup that requires ZERO vector embeddings. It indexes exact PDF pages using structural statistical keyword analysis (BM25) and utilizes pure semantic node navigation. It excels at exact medical term mapping and strictly preserves page context, complete with conversational memory.

## 🚀 How to Run Manually

### Prerequisites
1. Access your internal virtual workspace (`venv/`).
2. Ensure you have installed all base dependencies including `rank_bm25`:
   ```bash
   venv/bin/pip install -r requirements.txt
   venv/bin/pip install rank_bm25
   ```
3. Set your Groq API Key in a `.env` file at the root:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
4. Place your PDF documentation inside the `data/` directory.

### Running the Vectorless Setup (Recommended)
1. **Build the Index** (Only needs to be run once when data changes):
   ```bash
   venv/bin/python create_memory_vectorless.py
   ```
2. **Launch the Chatbot App**:
   ```bash
   venv/bin/python -m streamlit run medibot_vectorless.py
   ```

### Running the Vector Setup (FAISS)
1. **Build the Vector Store** (Only needs to be run once when data changes):
   ```bash
   venv/bin/python create_memory_for_llm.py
   ```
2. **Launch the Chatbot App**:
   ```bash
   venv/bin/python -m streamlit run medibot.py
   ```

## 📚 Technical Documentation
* [Vectorless Reverse Engineering & Architecture](VECTORLESS_REVERSE_ENGINEERING_DOCUMENTATION.md)
* [Legacy FAISS Reverse Engineering Analysis](MEDICAL_CHATBOT_REVERSE_ENGINEERING_DOCUMENTATION.md)