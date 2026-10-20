from rag.search_chunks import search_chunks
from rag.bm25_retriever import load_chunks, build_field_indexes, search_headings
from rag.colpali_chunk_retriever import search_colpali_chunks





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

    fused = {}

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
            item["rrf_score"] += 1.0 / (k + rank)
            item["retrieved_by"].append(source)
            item["source_ranks"][source] = rank

            # Preserve useful metadata from any retriever.
            for key, value in result.items():
                if key not in item or item[key] is None:
                    item[key] = value

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )[:top_k]






def main():
    chunks = load_chunks()
    _, heading_bm25 = build_field_indexes(chunks)

    question = input("Ask a question about the paper: ")

    # Retrieve independently from both systems.
    dense_results = search_chunks(question, top_k=10)

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

    # Fuse the two rankings.
    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        heading_results=heading_results,
        colpali_results= colpali_results,
        k=60,
        top_k=10,
    )

    print("\nDENSE RESULTS")
    for rank, result in enumerate(dense_results, start=1):
        print(
            f"{rank}. {result['chunk_id']} "
            f"| {result['headings']}"
        )

    print("\nHEADING BM25 RESULTS")
    for rank, result in enumerate(heading_results, start=1):
        print(
            f"{rank}. {result['chunk_id']} "
            f"| {result['headings']}"
        )
        
    print("\n COLPALI RESULTS")
    for rank, result in enumerate(colpali_results, start=1):
        print(
            f"{rank}. {result['chunk_id']} "
            f"| {result['headings']}"
        )


    print("\nRRF FUSED RESULTS")
    for rank, result in enumerate(fused_results, start=1):
        print(f"\n{rank}. {result['chunk_id']}")
        print(f"RRF score: {result['rrf_score']:.6f}")
        print(f"Retrieved by: {result['retrieved_by']}")
        print(f"Source ranks: {result['source_ranks']}")
        print(f"Headings: {result.get('headings', [])}")
        print(f"Pages: {result.get('page_numbers', [])}")
        print(f"Text:\n{result.get('text', '')[:500]}")


if __name__ == "__main__":
    main()
