"""
RAG package.

Provides:
- chunking
- embeddings
- retrieval
- local LLM generation
"""

from rag.pipeline import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "RAGPipeline"
]
