import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------
# 1. Chunk
# -----------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks


# -----------------------
# 2. Load embedding model
# -----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(texts):
    return model.encode(texts)


# -----------------------
# 3. Build index
# -----------------------
def build_index(chunks):
    embeddings = get_embeddings(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings


# -----------------------
# 4. Retrieve
# -----------------------
def retrieve(query, chunks, index, k=3):
    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]


# -----------------------
# 5. Generate (Ollama)
# -----------------------
def generate_answer(query, context):
    prompt = f"""
Answer ONLY from the given context.
If the answer is not present, say "Not found".

Context:
{context}

Question:
{query}
"""

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
    )

    return res.json()["response"]

def run_rag(text):
    chunks = chunk_text(text)
    index, _ = build_index(chunks)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        retrieved_chunks = retrieve(query, chunks, index)
        context = "\n".join(retrieved_chunks)

        print("\n[DEBUG] Retrieved Chunks:\n", context)

        answer = generate_answer(query, context)
        print("\nAnswer:\n", answer)