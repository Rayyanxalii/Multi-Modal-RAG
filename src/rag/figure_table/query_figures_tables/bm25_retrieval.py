
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# ==================================================
# Configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHUNKS_DIR = PROJECT_ROOT / "src" / "pdf" / "chunks"

FIGURES_PATH = CHUNKS_DIR / "attention_figures_cleaned.jsonl"
TABLES_PATH = CHUNKS_DIR / "attention_tables.jsonl"


# ==================================================
# Load source records
# ==================================================

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
                    f"Expected a JSON object in {path.name}, "
                    f"line {line_number}"
                )

            records.append(record)

    return records


def build_searchable_text(
    record: dict,
    record_type: str,
) -> str:
    parts = []

    if record_type == "figure":
        fields = (
            "parent_figure",
            "panel",
            "panel_caption",
            "shared_caption",
            "caption",
        )
    else:
        fields = ("caption", "table_markdown")

    for field in fields:
        value = record.get(field)

        if value is not None and str(value).strip():
            parts.append(f"{field}: {str(value).strip()}")

    return "\n".join(parts).strip()


# ==================================================
# Tokenization
# ==================================================

def tokenize(text: str) -> list[str]:
    """
    Preserve technical identifiers such as:
    multi-head, Q/K/V, BERT, and model dimensions.
    """
    return re.findall(
        r"[a-zA-Z0-9]+(?:[-_./][a-zA-Z0-9]+)*",
        text.lower(),
    )


# ==================================================
# Build BM25 index
# ==================================================

def build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    figures = load_jsonl(FIGURES_PATH)
    tables = load_jsonl(TABLES_PATH)

    documents = []

    for record in figures:
        record_id = record.get("figure_id")

        if not record_id:
            raise ValueError("Figure record missing figure_id.")

        text = build_searchable_text(record, "figure")

        if text:
            documents.append({
                **record,
                "record_type": "figure",
                "record_id": str(record_id),
                "searchable_text": text,
            })

    for record in tables:
        record_id = record.get("table_id")

        if not record_id:
            raise ValueError("Table record missing table_id.")

        text = build_searchable_text(record, "table")

        if text:
            documents.append({
                **record,
                "record_type": "table",
                "record_id": str(record_id),
                "searchable_text": text,
            })

    if not documents:
        raise ValueError("No searchable figure/table records found.")

    tokenized_documents = [
        tokenize(document["searchable_text"])
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    print(f"Figures loaded: {len(figures)}")
    print(f"Tables loaded: {len(tables)}")
    print(f"BM25 documents indexed: {len(documents)}")

    return bm25, documents


# ==================================================
# BM25 retrieval
# ==================================================

def bm25_retrieve(
    query: str,
    top_k: int = 5,
    record_type: str | None = None,
) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if record_type not in (None, "figure", "table"):
        raise ValueError(
            "record_type must be None, 'figure', or 'table'."
        )

    bm25, documents = build_bm25_index()

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    results = []

    for idx in ranked_indices:
        document = documents[idx]

        if record_type and document["record_type"] != record_type:
            continue

        score = float(scores[idx])

        # BM25 can assign zero to documents without matching terms.
        if score <= 0:
            continue

        results.append({
            "rank": len(results) + 1,
            "record_type": document["record_type"],
            "record_id": document["record_id"],
            "page": document.get("page"),
            "score": score,
            "searchable_text": document["searchable_text"],
            "metadata": document,
        })

        if len(results) >= top_k:
            break

    return results


# ==================================================
# Quick test
# ==================================================

if __name__ == "__main__":
    query = input("Enter your search query: ").strip()

    results = bm25_retrieve(query, top_k=5)

    if not results:
        print("No keyword matches found.")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Type: {result['record_type']}")
        print(f"ID: {result['record_id']}")
        print(f"Page: {result['page']}")
        print(f"BM25 score: {result['score']:.4f}")
        print(f"Content:\n{result['searchable_text']}")
