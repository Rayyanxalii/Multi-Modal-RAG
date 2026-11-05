import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

from rag.config import TEXT_CHUNKS_PATH as CHUNKS_PATH


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


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*",
        text.lower(),
    )


def build_field_indexes(chunks: list[dict]) -> tuple[BM25Okapi, BM25Okapi]:
    """Build BM25 indexes for both body and heading fields of chunks."""
    body_tokens = []
    heading_tokens = []

    for chunk in chunks:
        body_text = chunk.get("text") or ""
        headings = chunk.get("headings") or []
        heading_text = " ".join(headings)

        body_tokens.append(tokenize(body_text))
        heading_tokens.append(tokenize(heading_text))

    body_bm25 = BM25Okapi(body_tokens)
    heading_bm25 = BM25Okapi(heading_tokens)

    return body_bm25, heading_bm25


def search_headings(
    query: str,
    chunks: list[dict],
    heading_bm25: BM25Okapi,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve chunks using only their headings and BM25."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("Query contains no searchable tokens.")

    scores = heading_bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices:
        # Don't return chunks with no heading-term match.
        if scores[index] <= 0:
            continue

        chunk = chunks[index]

        results.append({
            "heading_score": float(scores[index]),
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "page_numbers": chunk.get("page_numbers", []),
            "headings": chunk.get("headings", []),
            "text": chunk.get("text", ""),
            "contextualized_text": chunk.get("contextualized_text", ""),
        })

        if len(results) >= top_k:
            break

    return results
