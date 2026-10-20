
from pathlib import Path

from rag.bm25_retriever import load_chunks
from rag.colpali_model import get_colpali_retriever


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

    # Convert page filenames into one-based page numbers.
    page_rankings = {}

    for result in page_results:
        page_name = Path(result["page"]).stem
        try:
            page_number = int(page_name.removeprefix("page_"))
        except ValueError as exc:
            raise ValueError(
                f"Unexpected ColPali page filename: {result['page']}"
            ) from exc

        page_rankings[page_number] = result

    # Map retrieved pages to their chunks.
    ranked_chunks = {}

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue

        raw_pages = chunk.get("page_numbers")

        if raw_pages is None:
            raw_page = chunk.get("page_number")
            raw_pages = [raw_page] if raw_page is not None else []

        matching_pages = [
            page for page in raw_pages
            if page in page_rankings
        ]

        if not matching_pages:
            continue

        # Use the best-ranked matching page for each chunk.
        best_page = min(
            matching_pages,
            key=lambda page: page_results.index(page_rankings[page]),
        )
        page_result = page_rankings[best_page]

        ranked_chunks[chunk_id] = {
            **chunk,
            "colpali_rank": page_result["rank"],
            "colpali_score": page_result["score"],
            "colpali_page_number": best_page,
            "retrieved_by": ["colpali"],
        }

    return sorted(
        ranked_chunks.values(),
        key=lambda item: (
            item["colpali_rank"],
            -item["colpali_score"],
        ),
    )[:top_k]



def main():
    
    results = search_colpali_chunks("What is attention mechanism?", 5, 10)
    
    for result in results:
        print('Chunk ID:', result['chunk_id'])
        


if __name__ == "__main__":
    main()