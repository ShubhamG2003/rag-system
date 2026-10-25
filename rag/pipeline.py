from pathlib import Path

from rag.chunking import chunk_pages, chunk_text
from rag.config import Config
from rag.document_loader import load_pdf
from rag.embeddings import EmbeddingService
from rag.llm import OllamaClient
from rag.retrieval import VectorStore


class RAGPipeline:

    def __init__(self):
        self.config = Config()

        self.embedding_service = EmbeddingService(
            self.config.embedding_model
        )

        self.llm_client = OllamaClient(
            model_name=self.config.llm_model,
            url=self.config.ollama_url
        )

        self.chunks: list[dict] = []
        self.vector_store = None

    def _index_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            raise ValueError("No text chunks available to index.")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.embedding_service.encode(texts)

        dimension = embeddings.shape[1]
        vector_store = VectorStore(dimension)
        vector_store.add(embeddings)

        # Update pipeline state only after indexing succeeds.
        self.chunks = chunks
        self.vector_store = vector_store

    def ingest(self, text: str) -> None:
        """Ingest plain text, preserving compatibility with existing usage."""

        text_chunks = chunk_text(
            text=text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )

        chunks = [
            {
                "text": chunk,
                "source": "provided_text",
                "page": None
            }
            for chunk in text_chunks
        ]

        self._index_chunks(chunks)

    def ingest_pdf(self, file_path: str | Path) -> None:
        """Ingest a PDF and preserve its source and page metadata."""

        pages = load_pdf(file_path)

        chunks = chunk_pages(
            pages=pages,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )

        self._index_chunks(chunks)

    def query(self, question: str) -> str:
        if self.vector_store is None:
            raise RuntimeError("No document has been ingested.")

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding = self.embedding_service.encode(
            [question]
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            chunks=self.chunks,
            k=self.config.top_k
        )

        context_parts = []

        for result in results:
            chunk = result.chunk

            source = chunk["source"]
            page = chunk["page"]

            if page is not None:
                label = f"Source: {source}, page: {page}"
            else:
                label = f"Source: {source}"

            context_parts.append(
                f"[{label}]\n{chunk['text']}"
            )

        context = "\n\n".join(
        f"[Source: {r.chunk['source']}, Page: {r.chunk['page']}]\n{r.chunk['text']}"
        for r in results
        )

        return self.llm_client.generate(
            query=question,
            context=context
        )
