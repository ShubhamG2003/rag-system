# RAG System using FAISS + Ollama

A lightweight Retrieval-Augmented Generation (RAG) pipeline built using:

- FAISS for vector similarity search
- Sentence Transformers for embeddings
- Ollama for local LLM inference
- Gemma 2B as the generation model

## Architecture

Document
→ Chunking
→ Embedding Generation
→ FAISS Indexing
→ Vector Retrieval
→ Prompt Assembly
→ Ollama Response Generation

## Features

- Semantic search using embeddings
- Local inference using Ollama
- Modular architecture
- Configurable chunking
- Cosine similarity retrieval
- Defensive error handling

## Installation

```bash
pip install -r requirements.txt
```

## Run Ollama

```bash
ollama run gemma:2b
```

## Usage

Create a `sample.txt` file.

Run:

```bash
python app.py
```

## Example Query

```txt
Ask a question:
What is the leave policy?
```

## Future Improvements

- Metadata filtering
- Hybrid retrieval (BM25 + vector search)
- Persistent FAISS index
- Reranking
- Streamlit interface
- PDF ingestion
