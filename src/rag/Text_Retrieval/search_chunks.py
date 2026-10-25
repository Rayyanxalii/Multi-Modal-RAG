
import os
from pathlib import Path

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(PROJECT_ROOT / ".env")

COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "nomic-embed-text"


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "Set QDRANT_URL and QDRANT_API_KEY in .env"
        )

    return QdrantClient(url=url, api_key=api_key)





def search_chunks(question: str, top_k: int = 5) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    client = get_client()

    # 1. Convert the question into a vector.
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )
    query_embedding = response["embeddings"][0]

    # 2. Retrieve the closest vectors from Qdrant.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    # 3. Return readable results.
    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append({
            "dense_score": point.score,
            "chunk_id": payload.get("chunk_id"),
            "document_id": payload.get("document_id"),
            "page_numbers": payload.get("page_numbers", []),
            "headings": payload.get("headings", []),
            "text": payload.get("text", ""),
            "contextualized_text": payload.get(
                "contextualized_text", ""
            ),
        })

    return results






def main():
    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist. "
            "Run the indexing script first."
        )

    question = input("Ask a question about the paper: ")
    results = search_chunks(question, top_k=10)

    for rank, result in enumerate(results, start=1):
        print("\n" + "=" * 75)
        print(f"Rank: {rank}")
        print(f"Score: {result['dense_score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Pages: {result['page_numbers']}")
        print(f"Headings: {result['headings']}")
        print(f"Text:\n{result['text']}")


if __name__ == "__main__":
    main()
