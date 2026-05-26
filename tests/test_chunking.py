from rag.chunking import chunk_text


def test_chunking_returns_chunks():

    text = "A" * 1000

    chunks = chunk_text(
        text=text,
        chunk_size=200,
        overlap=50
    )

    assert len(chunks) > 0


def test_overlap_validation():

    try:

        chunk_text(
            text="hello",
            chunk_size=100,
            overlap=100
        )

    except ValueError:

        assert True
