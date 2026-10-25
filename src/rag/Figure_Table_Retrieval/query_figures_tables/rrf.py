
from rag.Figure_Table_Retrieval.query_figures_tables.dense_retrieval import (
    dense_retrieve,
)
from rag.Figure_Table_Retrieval.query_figures_tables.bm25_retrieval import (
    bm25_retrieve,
)


# ==================================================
# Configuration
# ==================================================

RRF_K = 60


# ==================================================
# Reciprocal Rank Fusion
# ==================================================

def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    top_k: int = 5,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """
    Combine ranked result lists using Reciprocal Rank Fusion.

    Each result must contain:
        - rank
        - record_type
        - record_id

    RRF score = sum(1 / (rrf_k + rank))
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    fused = {}

    for ranking in rankings:
        seen_in_ranking = set()

        for result in ranking:
            record_type = result.get("record_type")
            record_id = result.get("record_id")

            if record_type is None or record_id is None:
                raise ValueError(
                    "Every result must contain record_type and record_id."
                )

            # The same figure ID and table ID may not represent
            # the same underlying record, so include the type.
            key = (record_type, str(record_id))

            # Avoid counting duplicate results from one retriever twice.
            if key in seen_in_ranking:
                continue

            seen_in_ranking.add(key)

            if key not in fused:
                fused[key] = {
                    **result,
                    "rrf_score": 0.0,
                    "retrieval_ranks": [],
                }

            rank = result.get("rank")

            if not isinstance(rank, int) or rank < 1:
                raise ValueError(
                    f"Invalid rank for {key}: {rank!r}"
                )

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[key]["retrieval_ranks"].append(rank)

    results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank

    return results[:top_k]


# ==================================================
# Hybrid retrieval
# ==================================================

def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    record_type: str | None = None,
) -> list[dict]:
    """
    Retrieve candidates with dense + BM25, then fuse their rankings.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1.")

    if record_type not in (None, "figure", "table"):
        raise ValueError(
            "record_type must be None, 'figure', or 'table'."
        )

    dense_results = dense_retrieve(
        query=query,
        top_k=candidate_k,
        record_type=record_type,
    )

    bm25_results = bm25_retrieve(
        query=query,
        top_k=candidate_k,
        record_type=record_type,
    )

    fused_results = reciprocal_rank_fusion(
        rankings=[dense_results, bm25_results],
        top_k=top_k,
    )

    return fused_results


# ==================================================
# Quick test
# ==================================================

if __name__ == "__main__":
    query = input("Enter your search query: ").strip()

    results = hybrid_retrieve(
        query=query,
        top_k=5,
        candidate_k=10,
    )

    if not results:
        print("No results found by either retriever.")

    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Type: {result['record_type']}")
        print(f"ID: {result['record_id']}")
        print(f"Page: {result.get('page')}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Original score: {result.get('score')}")
        print(f"Content:\n{result.get('searchable_text')}")
