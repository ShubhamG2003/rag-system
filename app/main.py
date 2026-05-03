from rag_pipeline import run_rag

if __name__ == "__main__":
    text = """
    Machine learning is a subset of artificial intelligence.
    It allows systems to learn from data and improve over time.
    Supervised learning uses labeled data.
    Unsupervised learning finds hidden patterns.
    """

    run_rag(text)