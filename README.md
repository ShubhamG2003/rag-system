# rag-system
Local RAG System using Ollama + FAISS

Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using a local LLM via Ollama and the Gemma 2B model.
It enables querying custom text data with context-aware answers—fully offline.

# Architecture
User Query → Embedding → FAISS → Top-K Chunks → LLM → Answer

# Tech Stack
Python
FAISS (vector search)
Sentence Transformers (embeddings)
Ollama (LLM inference)

# How to Run
git clone <your-repo>
cd rag-system

python3 -m venv rag_env
source rag_env/bin/activate

pip install -r requirements.txt
python app/main.py

# Features
Fully local LLM (no API dependency)
Semantic retrieval using FAISS
Context-grounded answers (reduces hallucination)
Debug mode showing retrieved chunks

# Future Improvements
PDF ingestion
Streamlit UI
Multi-document support
Hybrid search (keyword + vector)
