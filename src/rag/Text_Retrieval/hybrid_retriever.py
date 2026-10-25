from rag.Text_Retrieval.search_chunks import search_chunks
from rag.Text_Retrieval.bm25_retriever import (
    load_chunks,
    build_field_indexes,
    search_headings,
)
from rag.Text_Retrieval.colpali_chunk_retriever import search_colpali_chunks


def reciprocal_rank_fusion(
    dense_results: list[dict],
    heading_results: list[dict],
    colpali_results: list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """Fuse dense, heading BM25, and ColPali chunk rankings."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    fused: dict[str, dict] = {}

    ranked_lists = {
        "dense": dense_results,
        "heading_bm25": heading_results,
        "colpali": colpali_results,
    }

    for source, results in ranked_lists.items():
        seen_in_source = set()

        for rank, result in enumerate(results, start=1):
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    f"{source} result at rank {rank} "
                    "is missing chunk_id."
                )

            # Prevent duplicate contributions from one retriever.
            if chunk_id in seen_in_source:
                continue

            seen_in_source.add(chunk_id)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "chunk_id": chunk_id,
                    "rrf_score": 0.0,
                    "retrieved_by": [],
                    "source_ranks": {},
                }

            item = fused[chunk_id]

            # Reciprocal Rank Fusion score.
            item["rrf_score"] += 1.0 / (k + rank)

            item["retrieved_by"].append(source)
            item["source_ranks"][source] = rank

            # Preserve useful metadata from each retriever.
            for key, value in result.items():

                if key == "page_numbers":
                    existing_pages = item.get("page_numbers") or []

                    if not existing_pages and value:
                        item["page_numbers"] = list(value)

                elif key == "retrieved_by":
                    # Use the canonical source names above.
                    continue

                elif key in {"rrf_score", "source_ranks"}:
                    # These are calculated by RRF.
                    continue

                elif key not in item or item[key] is None:
                    item[key] = value

    # Highest RRF score first.
    fused_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return fused_results[:top_k]




def retrieve_and_fuse(
    question: str,
    top_k: int = 10,
) -> tuple[str, list[dict]]:
    """Run all retrievers and fuse their rankings."""

    chunks = load_chunks()
    _, heading_bm25 = build_field_indexes(chunks)

    dense_results = search_chunks(
        question,
        top_k=10,
    )

    heading_results = search_headings(
        query=question,
        chunks=chunks,
        heading_bm25=heading_bm25,
        top_k=10,
    )

    colpali_results = search_colpali_chunks(
        query=question,
        top_pages=5,
        top_k=10,
    )

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        heading_results=heading_results,
        colpali_results=colpali_results,
        k=60,
        top_k=top_k,
    )

    return question, fused_results


def main():
    question = input(
        "Ask a question about the paper: "
    ).strip()

    if not question:
        print("Please enter a question.")
        return

    question, fused_results = retrieve_and_fuse(
        question,
        top_k=10,
    )


    print("\nQUESTION")
    print(question)

    print("\nRRF FUSED RESULTS")

    for rank, result in enumerate(fused_results, start=1):
        print(f"\n{rank}. {result['chunk_id']}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Retrieved by: {result['retrieved_by']}")
        print(f"Source ranks: {result['source_ranks']}")
        print(f"Pages: {result.get('page_numbers', [])}")
        print(f"Headings: {result.get('headings', [])}")
        print(f"Text:\n{result.get('text', '')[:500]}")

    # Next stage:
    # Send `question` and `context` to your answer-generation LLM.
    # Require source-page citations and prohibit unsupported claims.

    return question, fused_results


if __name__ == "__main__":
    main()
