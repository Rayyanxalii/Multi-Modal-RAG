
from pathlib import Path

from rag.text.bm25_retriever import load_chunks
from rag.text.colpali_model import get_colpali_retriever


# maps pages to chunks and return them
def search_colpali_chunks(
    query: str,
    top_pages: int = 5,
    top_k: int = 10,
) -> list[dict]:
    """Map ColPali page rankings to matching document chunks."""

    if top_pages < 1:
        raise ValueError("top_pages must be at least 1.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    chunks = load_chunks()
    retriever = get_colpali_retriever()

    page_results = retriever.search(
        query=query,
        top_k=min(top_pages, len(retriever.page_names)),
    )

    # Map retrieved PDF page numbers to their ColPali rankings.
    page_rankings = {}

    for result in page_results:
        # Support the page filename field returned by ColPali.
        page_name = Path(
            result.get("page_numbers", result.get("page", ""))
        ).stem

        try:
            page_number = int(page_name.removeprefix("page_"))
        except ValueError as exc:
            raise ValueError(
                f"Unexpected ColPali page filename: {page_name}"
            ) from exc

        page_rankings[page_number] = result

    # Map retrieved pages to the document chunks on those pages.
    ranked_chunks = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            continue

        raw_pages = chunk.get("page_numbers")

        if raw_pages is None:
            raw_page = chunk.get("page_number")
            raw_pages = [raw_page] if raw_page is not None else []

        # Normalize a single page number to a list.
        if not isinstance(raw_pages, (list, tuple, set)):
            raw_pages = [raw_pages]

        matching_pages = [
            page
            for page in raw_pages
            if page in page_rankings
        ]

        if not matching_pages:
            continue

        # Choose the highest-ranked matching page.
        best_page = min(
            matching_pages,
            key=lambda page: page_rankings[page]["rank"],
        )

        page_result = page_rankings[best_page]

        ranked_chunks[chunk_id] = {
            **chunk,
            "page_numbers": chunk.get("page_numbers", []),
            "colpali_page_numbers": best_page,
            "colpali_rank": page_result["rank"],
            "colpali_score": page_result["score"],
            "retrieved_by": ["colpali"],
        }

    # Return chunks in ColPali ranking order.
    return sorted(
        ranked_chunks.values(),
        key=lambda item: (
            item["colpali_rank"],
            -item["colpali_score"],
        ),
    )[:top_k]


