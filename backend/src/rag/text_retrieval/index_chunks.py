import json
import uuid
from pathlib import Path

import ollama
from qdrant_client import QdrantClient, models

from rag.config import (
    TEXT_CHUNKS_PATH as CHUNKS_PATH,
    TEXT_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_SIZE,
    get_qdrant_credentials,
)

BATCH_SIZE = 16


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
    url, api_key = get_qdrant_credentials()
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

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk['document_id']}:{chunk['chunk_id']}",
                )
            )

            payload = {
                key: value
                for key, value in chunk.items()
                if key not in {"contextualized_text"}
            }
            payload["contextualized_text"] = chunk["contextualized_text"]

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
