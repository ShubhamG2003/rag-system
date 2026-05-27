from rag.pipeline import RAGPipeline


def main() -> None:
    with open("sample.txt", "r", encoding="utf-8") as file:
        document = file.read()

    pipeline = RAGPipeline()
    pipeline.ingest(document)

    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        answer = pipeline.query(query)

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()
