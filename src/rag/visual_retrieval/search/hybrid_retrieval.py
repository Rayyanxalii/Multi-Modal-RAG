"""Visual retrieval interface using ColPali page-to-asset mapping (Option A)."""

from typing import Any
from rag.visual_retrieval.search.colpali_retriever import retrieve_visual_assets


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    record_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve figures and tables using ColPali visual retrieval.

    Maintains backwards compatibility with the orchestrator interface
    while delegating directly to ColPali visual page-to-asset mapping.
    """
    return retrieve_visual_assets(
        query=query,
        top_pages=candidate_k,
        top_k=top_k,
        record_type=record_type,
    )


if __name__ == "__main__":
    query = input("Enter your search query: ").strip()
    if query:
        results = hybrid_retrieve(query=query, top_k=5)
        for item in results:
            print(
                f"Rank {item['rank']} | {item['record_type']} | "
                f"{item['record_id']} | Page: {item['page']} | "
                f"Score: {item.get('colpali_score', 0):.4f}"
            )
