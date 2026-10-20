"""
evaluate.py
===========
Evaluates dense (Qdrant), sparse (BM25) and hybrid (RRF) retrievers
against a manually labelled golden set.

Metrics computed per query and averaged across all queries:
  - Precision@K   — fraction of retrieved chunks that are relevant
  - Recall@K      — fraction of relevant chunks that were retrieved
  - MRR@K         — reciprocal rank of the first relevant result
  - Hit Rate@K    — 1 if at least one relevant chunk appears in top-K

Usage
-----
  python evaluate.py [--top-k 5] [--golden src/evaluation/golden_set.json]

Requirements
------------
  - Qdrant collection must already be indexed  (run index_chunks.py first)
  - Ollama must be running locally              (for embeddings)
  - .env must contain QDRANT_URL and QDRANT_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ollama
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import re

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLDEN_SET_PATH = PROJECT_ROOT / "src" / "evaluation" / "golden_set.json"
CHUNKS_PATH = (
    PROJECT_ROOT / "src" / "pdf" / "chunks" / "attention_chunks.jsonl"
)

COLLECTION_NAME = "research_papers"
EMBEDDING_MODEL = "nomic-embed-text"

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_golden_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Dense retriever (Qdrant)
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError("Set QDRANT_URL and QDRANT_API_KEY in .env")
    return QdrantClient(url=url, api_key=api_key)


def dense_retrieve(query: str, client: QdrantClient, top_k: int) -> list[str]:
    """Return the top-K chunk_ids from Qdrant dense search."""
    response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    query_vector = response["embeddings"][0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [pt.payload["chunk_id"] for pt in results.points]


# ---------------------------------------------------------------------------
# Sparse retriever (BM25, heading-aware)
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)*", text.lower())


def build_bm25_indexes(chunks: list[dict]) -> tuple[BM25Okapi, BM25Okapi]:
    body_tokens = [tokenize(c.get("text") or "") for c in chunks]
    heading_tokens = [
        tokenize(" ".join(c.get("headings") or [])) for c in chunks
    ]
    return BM25Okapi(body_tokens), BM25Okapi(heading_tokens)


def _normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def bm25_retrieve(
    query: str,
    chunks: list[dict],
    body_bm25: BM25Okapi,
    heading_bm25: BM25Okapi,
    top_k: int,
    body_weight: float = 0.3,
    heading_weight: float = 0.7,
) -> list[str]:
    """Return the top-K chunk_ids from heading-aware BM25 search."""
    tokens = tokenize(query)
    if not tokens:
        raise ValueError("Query contains no searchable tokens.")

    body_scores = _normalize(body_bm25.get_scores(tokens).tolist())
    heading_scores = _normalize(heading_bm25.get_scores(tokens).tolist())

    combined = [
        (i, body_weight * b + heading_weight * h)
        for i, (b, h) in enumerate(zip(body_scores, heading_scores))
    ]
    combined.sort(key=lambda x: x[1], reverse=True)

    return [chunks[i]["chunk_id"] for i, _ in combined[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid RRF retriever
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """
    Combine multiple ranked lists with Reciprocal Rank Fusion.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    body_bm25: BM25Okapi,
    heading_bm25: BM25Okapi,
    client: QdrantClient,
    top_k: int,
    rrf_candidate_k: int = 20,
) -> list[str]:
    """
    Retrieve using dense + BM25, fuse with RRF, return top-K chunk_ids.
    rrf_candidate_k: how many candidates each retriever fetches before fusion.
    """
    dense_ids = dense_retrieve(query, client, top_k=rrf_candidate_k)
    bm25_ids = bm25_retrieve(
        query, chunks, body_bm25, heading_bm25, top_k=rrf_candidate_k
    )
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str]) -> float:
    return float(any(r in relevant for r in retrieved))


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_retriever(
    name: str,
    retrieve_fn,          # callable(query: str) -> list[str]
    golden_set: list[dict],
    top_k: int,
) -> dict:
    """
    Run a single retriever over every question in the golden set.
    Returns a dict of average metric scores.
    """
    totals = dict(precision=0.0, recall=0.0, mrr=0.0, hit_rate=0.0)
    per_query: list[dict] = []

    for item in golden_set:
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])

        retrieved = retrieve_fn(query)[:top_k]

        p = precision_at_k(retrieved, relevant)
        r = recall_at_k(retrieved, relevant)
        m = mrr_at_k(retrieved, relevant)
        h = hit_rate_at_k(retrieved, relevant)

        totals["precision"] += p
        totals["recall"] += r
        totals["mrr"] += m
        totals["hit_rate"] += h

        per_query.append({
            "query": query,
            "retrieved": retrieved,
            "relevant": list(relevant),
            f"precision@{top_k}": round(p, 4),
            f"recall@{top_k}": round(r, 4),
            f"mrr@{top_k}": round(m, 4),
            f"hit_rate@{top_k}": round(h, 4),
        })

    n = len(golden_set)
    averages = {k: round(v / n, 4) for k, v in totals.items()}

    return {
        "retriever": name,
        "top_k": top_k,
        "num_queries": n,
        f"avg_precision@{top_k}": averages["precision"],
        f"avg_recall@{top_k}": averages["recall"],
        f"avg_mrr@{top_k}": averages["mrr"],
        f"avg_hit_rate@{top_k}": averages["hit_rate"],
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], top_k: int) -> None:
    header = (
        f"\n{'Retriever':<20} "
        f"{'P@' + str(top_k):<10} "
        f"{'R@' + str(top_k):<10} "
        f"{'MRR@' + str(top_k):<10} "
        f"{'HR@' + str(top_k):<10}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['retriever']:<20} "
            f"{r[f'avg_precision@{top_k}']:<10.4f} "
            f"{r[f'avg_recall@{top_k}']:<10.4f} "
            f"{r[f'avg_mrr@{top_k}']:<10.4f} "
            f"{r[f'avg_hit_rate@{top_k}']:<10.4f}"
        )
    print()


def save_results(results: list[dict], top_k: int) -> None:
    out_path = PROJECT_ROOT / "src" / "evaluation" / f"results_at_{top_k}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate dense, BM25 and hybrid retrievers."
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--golden", type=Path, default=GOLDEN_SET_PATH,
        help="Path to the golden set JSON file.",
    )
    args = parser.parse_args()

    top_k: int = args.top_k
    golden_path: Path = args.golden

    print(f"Loading golden set from: {golden_path}")
    golden_set = load_golden_set(golden_path)
    print(f"  {len(golden_set)} questions loaded.")

    print(f"Loading chunks from: {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"  {len(chunks)} chunks loaded.")

    print("Building BM25 indexes...")
    body_bm25, heading_bm25 = build_bm25_indexes(chunks)

    print("Connecting to Qdrant...")
    client = get_qdrant_client()
    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' does not exist. "
            "Run index_chunks.py first."
        )

    # ------------------------------------------------------------------
    # Define retrieval functions (closures capture shared state)
    # ------------------------------------------------------------------

    def dense_fn(query: str) -> list[str]:
        return dense_retrieve(query, client, top_k=top_k)

    def bm25_fn(query: str) -> list[str]:
        return bm25_retrieve(
            query, chunks, body_bm25, heading_bm25, top_k=top_k
        )

    def hybrid_fn(query: str) -> list[str]:
        return hybrid_retrieve(
            query, chunks, body_bm25, heading_bm25, client, top_k=top_k
        )

    retrievers = [
        ("Dense (Qdrant)", dense_fn),
        ("Sparse (BM25)", bm25_fn),
        ("Hybrid (RRF)", hybrid_fn),
    ]

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------

    all_results = []

    for name, fn in retrievers:
        print(f"\nEvaluating: {name} ...")
        result = evaluate_retriever(name, fn, golden_set, top_k=top_k)
        all_results.append(result)
        print(
            f"  P@{top_k}={result[f'avg_precision@{top_k}']:.4f}  "
            f"R@{top_k}={result[f'avg_recall@{top_k}']:.4f}  "
            f"MRR@{top_k}={result[f'avg_mrr@{top_k}']:.4f}  "
            f"HR@{top_k}={result[f'avg_hit_rate@{top_k}']:.4f}"
        )

    print_summary(all_results, top_k)
    save_results(all_results, top_k)


if __name__ == "__main__":
    main()
