import json
import uuid
from pathlib import Path

import ollama
from qdrant_client import QdrantClient, models

from rag.config import (
    get_qdrant_credentials,
    FIGURES_CLEANED_JSONL as FIGURES_PATH,
    TABLES_JSONL as TABLES_PATH,
    VISUAL_COLLECTION_NAME as COLLECTION_NAME,
    EMBEDDING_MODEL,
    VECTOR_SIZE,
)

BATCH_SIZE = 16
VECTOR_NAME = ""


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path.name}, line {line_number}"
                )

            records.append(record)

    return records


def build_searchable_text(
    record: dict,
    record_type: str,
) -> str | None:
    parts = []

    if record_type == "figure":
        for key in (
            "parent_figure",
            "panel",
            "panel_caption",
            "shared_caption",
            "caption",
        ):
            value = record.get(key)
            if value is not None and str(value).strip():
                parts.append(f"{key}: {str(value).strip()}")

    elif record_type == "table":
        caption = record.get("caption")
        table_markdown = record.get("table_markdown")

        if caption and str(caption).strip():
            parts.append(f"caption: {str(caption).strip()}")

        if table_markdown and str(table_markdown).strip():
            parts.append(
                f"table content:\n{str(table_markdown).strip()}"
            )

    else:
        raise ValueError(f"Unknown record type: {record_type}")

    text = "\n".join(parts).strip()
    return text or None


def load_records(
    figures_path: Path = FIGURES_PATH,
    tables_path: Path = TABLES_PATH,
) -> list[dict]:
    figures = load_jsonl(figures_path)
    tables = load_jsonl(tables_path)

    records = []
    skipped = 0

    for record in figures:
        record_id = record.get("figure_id")
        if not record_id:
            raise ValueError("A figure record is missing figure_id.")

        searchable_text = build_searchable_text(record, "figure")
        if not searchable_text:
            print(f"Skipping figure without text: {record_id}")
            skipped += 1
            continue

        records.append({
            **record,
            "record_type": "figure",
            "record_id": str(record_id),
            "searchable_text": searchable_text,
        })

    for record in tables:
        record_id = record.get("table_id")
        if not record_id:
            raise ValueError("A table record is missing table_id.")

        searchable_text = build_searchable_text(record, "table")
        if not searchable_text:
            print(f"Skipping table without text: {record_id}")
            skipped += 1
            continue

        records.append({
            **record,
            "record_type": "table",
            "record_id": str(record_id),
            "searchable_text": searchable_text,
        })

    identities = [
        (
            record.get("document_id", ""),
            record["record_type"],
            record["record_id"],
        )
        for record in records
    ]

    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate document/type/record IDs found.")

    print(f"Figures loaded: {len(figures)}")
    print(f"Tables loaded: {len(tables)}")
    print(f"Records skipped: {skipped}")
    print(f"Searchable records: {len(records)}")

    if not records:
        raise ValueError("No searchable figure/table records found.")

    return records


def get_client() -> QdrantClient:
    url, api_key = get_qdrant_credentials()
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
        )
        print(f"Created collection: {COLLECTION_NAME}")
        return

    info = client.get_collection(COLLECTION_NAME)
    vectors = info.config.params.vectors

    if not isinstance(vectors, dict):
        raise ValueError(
            "The existing collection uses an unnamed vector, "
            "but this script expects a named vector."
        )

    if VECTOR_NAME not in vectors:
        raise ValueError(
            f"Expected vector name {VECTOR_NAME!r} not found. "
            f"Available names: {list(vectors.keys())}"
        )

    vector_config = vectors[VECTOR_NAME]

    if vector_config.size != VECTOR_SIZE:
        raise ValueError(
            f"Collection vector size is {vector_config.size}; "
            f"expected {VECTOR_SIZE}."
        )

    if vector_config.distance != models.Distance.COSINE:
        raise ValueError("Collection must use Cosine distance.")

    print(f"Using existing collection: {COLLECTION_NAME}")


def make_point_id(record: dict) -> str:
    identity = (
        f"{record.get('document_id', '')}:"
        f"{record['record_type']}:"
        f"{record['record_id']}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def index_figures_tables() -> None:
    records = load_records()
    client = get_client()
    ensure_collection(client)

    before_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print(f"Points before indexing: {before_count}")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=[
                record["searchable_text"]
                for record in batch
            ],
        )

        embeddings = response["embeddings"]

        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding count mismatch.")

        points = []

        for record, embedding in zip(batch, embeddings):
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Expected {VECTOR_SIZE}-dimensional embedding; "
                    f"received {len(embedding)}."
                )

            payload = {
                key: value
                for key, value in record.items()
                if key != "searchable_text"
            }

            points.append(
                models.PointStruct(
                    id=make_point_id(record),
                    vector={VECTOR_NAME: embedding},
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"Indexed {min(start + len(batch), len(records))}"
            f"/{len(records)} records"
        )

    after_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    print("\nIndexing finished.")
    print(f"Points before: {before_count}")
    print(f"Points after: {after_count}")
    print(f"Expected searchable records: {len(records)}")

    if after_count != len(records):
        print(
            "WARNING: The collection count does not match "
            "the number of searchable records. Check whether "
            "the collection contains unrelated points."
        )
    else:
        print("Point count matches the expected record count.")


if __name__ == "__main__":
    index_figures_tables()
