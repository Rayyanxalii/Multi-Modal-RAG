"""Main entrypoint for the RAG package."""

from rag.answer_generation import generate_answer


def main():
    result = generate_answer()
    print("\n--- Question ---\n")
    print(result.get("question", ""))
    print("\n--- Answer ---\n")
    print(result.get("answer", ""))
    print("\n--- Sources ---\n")
    for source in result.get("sources", []):
        print(
            f"Source {source['source']} | "
            f"Chunk: {source['chunk_id']} | "
            f"Pages: {source['pages'] or 'Unknown'}"
        )


if __name__ == "__main__":
    main()
