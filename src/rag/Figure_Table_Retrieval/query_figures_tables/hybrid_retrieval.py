from rag.Figure_Table_Retrieval.query_figures_tables.dense_retrieval import (
    dense_retrieve,
)
from rag.Figure_Table_Retrieval.query_figures_tables.bm25_retrieval import (
    bm25_retrieve,
)

from rag.Figure_Table_Retrieval.query_figures_tables.rrf import (
    reciprocal_rank_fusion,
)



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
