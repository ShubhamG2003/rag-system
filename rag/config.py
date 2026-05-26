from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gemma:2b"

    chunk_size: int = 500
    chunk_overlap: int = 100

    top_k: int = 3

    ollama_url: str = "http://localhost:11434/api/generate"
