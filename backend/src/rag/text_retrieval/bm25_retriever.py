import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

from functools import lru_cache
from rag.config import TEXT_CHUNKS_PATH as CHUNKS_PATH


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


@lru_cache(maxsize=16)
def get_bm25_index_for_file(file_path_str: str):
    """Cache parsed chunks and BM25 index per document file path."""
    chunks = load_chunks(Path(file_path_str))
    body_bm25, heading_bm25 = build_field_indexes(chunks)
    return chunks, body_bm25, heading_bm25


def search_bm25(query: str, top_k: int = 5, chunks_path: Path | None = None) -> list[dict]:
    """Search chunks for a specific document using BM25."""
    p = Path(chunks_path or CHUNKS_PATH)
    if not p.exists():
        return []
    chunks, _, heading_bm25 = get_bm25_index_for_file(str(p.resolve()))
    try:
        return search_headings(query, chunks, heading_bm25, top_k=top_k)
    except Exception:
        return []
