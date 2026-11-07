from pathlib import Path
import ollama
from qdrant_client import QdrantClient

from rag.config import (
    get_qdrant_credentials,
    TEXT_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def get_client() -> QdrantClient:
    url, api_key = get_qdrant_credentials()
    return QdrantClient(url=url, api_key=api_key)


def search_chunks(question: str, top_k: int = 5, doc_id: str | None = None) -> list[dict]:
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

    # Optional filter by document_id
    query_filter = None
    if doc_id:
        from qdrant_client import models
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(doc_id).lower()),
                )
            ]
        )

    # 2. Retrieve the closest vectors from Qdrant.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
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
