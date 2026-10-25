from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class RetrievalResult:
    chunk: dict
    score: float


class VectorStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        self.index.add(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        chunks: list[dict],
        k: int
    ) -> list[RetrievalResult]:
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunks[idx],
                    score=float(score)
                )
            )

        return results
