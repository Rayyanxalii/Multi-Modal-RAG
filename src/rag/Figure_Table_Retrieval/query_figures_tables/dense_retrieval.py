
import os
from pathlib import Path

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# ==================================================
# Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ENV_PATH = PROJECT_ROOT / ".env"

COLLECTION_NAME = "research_figures_tables"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768

# The existing collection uses an empty-string vector name.
VECTOR_NAME = ""

load_dotenv(ENV_PATH, override=True)


# ==================================================
# Qdrant client
# ==================================================

def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            f"QDRANT_URL or QDRANT_API_KEY is missing in {ENV_PATH}"
        )

    return QdrantClient(url=url, api_key=api_key)


# ==================================================
# Query embedding
# ==================================================

def embed_query(query: str) -> list[float]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query,
    )

    embeddings = response["embeddings"]

    if not embeddings:
        raise RuntimeError("Ollama returned no query embedding.")

    embedding = embeddings[0]

    if len(embedding) != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE}-dimensional embedding, "
            f"received {len(embedding)}."
        )

    return embedding


# ==================================================
# Dense retrieval
# ==================================================

def dense_retrieve(
    query: str,
    top_k: int = 5,
    record_type: str | None = None,
) -> list[dict]:
    """
    Retrieve figures and tables using dense semantic similarity.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return.
        record_type: Optional filter: "figure" or "table".

    Returns:
        Ranked records containing their identifiers and similarity scores.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if record_type not in (None, "figure", "table"):
        raise ValueError(
            "record_type must be None, 'figure', or 'table'."
        )

    client = get_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection '{COLLECTION_NAME}' does not exist. "
            "Run the indexing script first."
        )

    query_vector = embed_query(query.strip())

    query_filter = None

    if record_type is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="record_type",
                    match=models.MatchValue(value=record_type),
                )
            ]
        )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using=VECTOR_NAME,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}

        results.append({
            "rank": rank,
            "record_type": payload.get("record_type"),
            "record_id": payload.get("record_id"),
            "page": payload.get("page_number"),
            "score": float(point.score),
            "searchable_text": payload.get("searchable_text"),
        })

    return results


# ==================================================
# Quick test
# ==================================================

if __name__ == "__main__":
    query = input("Enter your search query: ").strip()

    results = dense_retrieve(query, top_k=5)

    if not results:
        print("No matching figures or tables found.")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Type: {result['record_type']}")
        print(f"ID: {result['record_id']}")
        print(f"Page: {result['page']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Content:\n{result['searchable_text']}")
