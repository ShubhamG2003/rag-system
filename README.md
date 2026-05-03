# Local RAG System (Offline Q&A over Documents)

Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using a local LLM via Ollama and the Gemma 2B model.
It enables users to query custom data and receive context-grounded answers without external APIs.

Architecture
Query → Embeddings → FAISS → Top-K Chunks → LLM → Answer

Tech Stack
Python
FAISS (vector search)
Sentence Transformers (embeddings)
Ollama (local LLM)

How to Run
git clone <rag-system>
cd rag-system

python3 -m venv rag_env
source rag_env/bin/activate

pip install -r requirements.txt
python app/main.py

Features
Fully offline (no API cost)
Semantic retrieval using FAISS
Context-aware responses (reduces hallucination)
Debug mode showing retrieved chunks

Future Improvements
PDF ingestion
Streamlit UI
Multi-document support
Hybrid search
