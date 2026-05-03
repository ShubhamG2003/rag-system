# Local RAG System (Offline Q&A over Documents)<br>

Overview<br>

This project implements a Retrieval-Augmented Generation (RAG) pipeline using a local LLM via Ollama and the Gemma 2B model.<br>
It enables users to query custom data and receive context-grounded answers without external APIs.<br>

Architecture<br>
Query → Embeddings → FAISS → Top-K Chunks → LLM → Answer<br>

Tech Stack<br>
Python<br>
FAISS (vector search)<br>
Sentence Transformers (embeddings)<br>
Ollama (local LLM)<br>

How to Run<br>
git clone <rag-system><br>
cd rag-system<br>

python3 -m venv rag_env<br>
source rag_env/bin/activate<br>

pip install -r requirements.txt<br>
python app/main.py<br>

Features<br>
Fully offline (no API cost)<br>
Semantic retrieval using FAISS<br>
Context-aware responses (reduces hallucination)<br>
Debug mode showing retrieved chunks<br>

Future Improvements<br>
PDF ingestion<br>
Streamlit UI<br>
Multi-document support<br>
Hybrid search<br>
