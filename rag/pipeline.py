from rag.chunking import chunk_text
from rag.config import Config
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

        self.chunks: list[str] = []

        self.vector_store = None

    def ingest(self, text: str) -> None:

        self.chunks = chunk_text(
            text=text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )

        embeddings = self.embedding_service.encode(
            self.chunks
        )

        dimension = embeddings.shape[1]

        self.vector_store = VectorStore(dimension)

        self.vector_store.add(embeddings)

    def query(self, question: str) -> str:

        if self.vector_store is None:
            raise RuntimeError(
                "No document has been ingested."
            )

        query_embedding = self.embedding_service.encode(
            [question]
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            chunks=self.chunks,
            k=self.config.top_k
        )

        context = "\n\n".join(
            result.chunk
            for result in results
        )

        return self.llm_client.generate(
            query=question,
            context=context
        )
