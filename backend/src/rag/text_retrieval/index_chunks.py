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
    get_ollama_client,
)

BATCH_SIZE = 16


def load_chunks(chunks_path: Path = CHUNKS_PATH) -> list[dict]:
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    with chunks_path.open("r", encoding="utf-8") as file:
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


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created collection: {collection_name}")
    else:
        info = client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")

        vector_config = info.config.params.vectors
        if (
            not isinstance(vector_config, dict)
            and vector_config.size != VECTOR_SIZE
        ):
            raise ValueError(
                f"Collection vector size does not match "
                f"{VECTOR_SIZE}. Inspect the collection config."
            )

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def index_text_chunks(
    chunks_file_path: Path = CHUNKS_PATH,
    collection_name: str = COLLECTION_NAME,
    doc_id: str | None = None,
    user_id: str | None = None,
) -> int:
    """Index text chunks into Qdrant vector database with optional document_id and user_id filtering."""
    chunks = load_chunks(chunks_file_path)
    client = get_client()

    ensure_collection(client, collection_name=collection_name)

    indexed_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    print(f"Chunks in JSONL: {len(chunks)}")
    print(f"Points currently in Qdrant: {indexed_count}")

    clean_user = str(user_id or "default").strip()

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            chunk["contextualized_text"]
            for chunk in batch
        ]

        from rag.services.embedding_service import embed_documents
        embeddings = embed_documents(texts)

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

            chunk_doc_id = str(doc_id or chunk.get("document_id") or "default").lower()
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{clean_user}:{chunk_doc_id}:{chunk['chunk_id']}",
                )
            )

            payload = {
                key: value
                for key, value in chunk.items()
                if key not in {"contextualized_text"}
            }
            payload["document_id"] = chunk_doc_id
            payload["user_id"] = clean_user
            payload["contextualized_text"] = chunk["contextualized_text"]

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        print(
            f"Indexed {min(start + len(batch), len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    info = client.get_collection(collection_name)
    print(f"\nIndexing complete.")
    print(f"Points in collection: {info.points_count}")
    return len(chunks)


index_chunks = index_text_chunks


if __name__ == "__main__":
    index_chunks()
