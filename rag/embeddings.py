import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class EmbeddingService:

    def __init__(self, model_name: str):

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:

        embeddings = self.model.encode(texts)

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        faiss.normalize_L2(embeddings)

        return embeddings
