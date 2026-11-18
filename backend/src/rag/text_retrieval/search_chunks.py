"""Dense vector search over indexed text chunks in Qdrant."""

from pathlib import Path
import ollama
from qdrant_client import QdrantClient

from rag.config import (
    get_qdrant_credentials,
    get_ollama_client,
    TEXT_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def get_client() -> QdrantClient:
    """Create a Qdrant client from configured credentials."""
    url, api_key = get_qdrant_credentials()
    return QdrantClient(url=url, api_key=api_key)


def search_chunks(
    question: str,
    top_k: int = 5,
    doc_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """Search Qdrant for chunks nearest to the question embedding."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    client = get_client()

    from rag.services.embedding_service import embed_query
    query_embedding = embed_query(question)

    query_filter = None
    must_conditions = []
    if doc_id:
        clean_doc = str(doc_id).strip().lower()
        from qdrant_client import models
        must_conditions.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=clean_doc),
            )
        )
    else:
        print("[Warning] search_chunks called without doc_id! This allows cross-document results.")

    clean_user = str(user_id or "").strip()
    if clean_user and clean_user != "default":
        from qdrant_client import models
        must_conditions.append(
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=clean_user),
            )
        )

    if must_conditions:
        from qdrant_client import models
        query_filter = models.Filter(must=must_conditions)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

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
    """Run interactive dense search and print ranked results."""
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
