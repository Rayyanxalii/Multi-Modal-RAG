from rag.search_chunks import search_chunks
from rag.bm25_retriever import load_chunks, build_field_indexes, search_headings


def reciprocal_rank_fusion(
    dense_results: list[dict],
    heading_results: list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:
    """
    Fuse dense retrieval and heading BM25 rankings using RRF.

    Each result must contain a unique 'chunk_id'.
    Rank positions start at 1.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    fused = {}

    ranked_lists = {    
        "dense": dense_results,
        "heading_bm25": heading_results,
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

            # Avoid counting duplicate chunks twice in one ranking.
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

            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            fused[chunk_id]["retrieved_by"].append(source)
            fused[chunk_id]["source_ranks"][source] = rank

            # Preserve useful chunk metadata if another retriever
            # returned the same chunk with fewer fields.
            for key, value in result.items():
                if key not in fused[chunk_id] or fused[chunk_id][key] is None:
                    fused[chunk_id][key] = value

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

    # Fuse the two rankings.
    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        heading_results=heading_results,
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
