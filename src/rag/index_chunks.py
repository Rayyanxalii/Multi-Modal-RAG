import json
import os
import uuid
from pathlib import Path

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = (
    PROJECT_ROOT / "src" / "pdf" / "chunks"
    / "attention_chunks.jsonl"
)

# Configuration
COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768
BATCH_SIZE = 16

load_dotenv(PROJECT_ROOT / ".env")




def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("The chunks file is empty.")

    return chunks




def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError(
            "Set QDRANT_URL and QDRANT_API_KEY in .env"
        )

    return QdrantClient(url=url, api_key=api_key)




def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        info = client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")

        # Prevent accidentally indexing into a collection
        # configured for a different embedding dimension.
        vector_config = info.config.params.vectors
        if (
            not isinstance(vector_config, dict)
            and vector_config.size != VECTOR_SIZE
        ):
            raise ValueError(
                f"Collection vector size does not match "
                f"{VECTOR_SIZE}. Inspect the collection config."
            )




def index_chunks() -> None:
    chunks = load_chunks()
    client = get_client()

    ensure_collection(client)
    
    indexed_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count
    
    print(f"Chunks in JSONL: {len(chunks)}")
    print(f"Points currently in Qdrant: {indexed_count}")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            chunk["contextualized_text"]
            for chunk in batch
        ]

        # Generate all embeddings in this batch with Ollama.
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )
        embeddings = response["embeddings"]

        if len(embeddings) != len(batch):
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        points = []

        for chunk, embedding in zip(batch, embeddings):
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Unexpected vector size for "
                    f"{chunk['chunk_id']}: {len(embedding)}"
                )

            # Qdrant accepts UUIDs or unsigned integer IDs.
            # UUID5 provides a stable ID for each document chunk.
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk['document_id']}:{chunk['chunk_id']}",
                )
            )

            # Preserve metadata as the Qdrant payload.
            payload = {
                key: value
                for key, value in chunk.items()
                if key not in {"contextualized_text"}
            }

            payload["contextualized_text"] = (
                chunk["contextualized_text"]
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"Indexed {min(start + len(batch), len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nIndexing complete.")
    print(f"Points in collection: {info.points_count}")




if __name__ == "__main__":
    index_chunks()
