from typing import List


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:

    if not text.strip():
        raise ValueError("Input text is empty.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks

def chunk_pages(
    pages: list[dict],
    chunk_size: int,
    overlap: int
) -> list[dict]:
    """Chunk PDF pages while preserving source and page metadata."""

    chunks = []

    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks