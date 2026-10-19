import json
from random import sample
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
import math

from sympy import python

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "src"
    / "pdf"
    / "chunks"
    / "attention_chunks.jsonl"
)


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
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
    


# It build bm25 indexes for both the body and heading fields of the chunks. It returns a tuple containing the two BM25Okapi objects.
def build_field_indexes(chunks: list[dict]) -> tuple[BM25Okapi, BM25Okapi]:
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

    

# Normalizes a list of scores to a range between 0 and 1. If all scores are the same, it returns a list of zeros.
def normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.0] * len(scores)

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


# it returns a combined score for each chunk based on the body and heading scores, weighted by the specified weights. The results are sorted in descending order of the final score.
def score_chunks(
    query: str,
    body_bm25: BM25Okapi,
    heading_bm25: BM25Okapi,
    body_weight: float = 0,      # you can control the weight of body and heading scores by adjusting these parameters
    heading_weight: float = 1,
) -> list[dict]:
    
    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("Query contains no searchable tokens.")

    body_scores = body_bm25.get_scores(query_tokens).tolist()
    heading_scores = heading_bm25.get_scores(query_tokens).tolist()

    normalized_body = normalize_scores(body_scores)
    normalized_heading = normalize_scores(heading_scores)

    combined_scores = []

    for index, (body_score, heading_score) in enumerate(
        zip(normalized_body, normalized_heading)
    ):
        final_score = (
            body_weight * body_score
            + heading_weight * heading_score
        )

        combined_scores.append({
            "index": index,
            "body_score": body_score,
            "heading_score": heading_score,
            "final_score": final_score,
        })

    return sorted(
        combined_scores,
        key=lambda item: item["final_score"],
        reverse=True,
    )



# It searches for the top_k chunks that are most relevant to the given query, considering both the body and heading scores. It returns a list of dictionaries containing the chunk ID, scores, page numbers, headings, and text for each of the top_k results.
def search_heading_aware(
    query: str,
    chunks: list[dict],
    body_bm25: BM25Okapi,
    heading_bm25: BM25Okapi,
    top_k: int = 5,
) -> list[dict]:
    
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    ranked_scores = score_chunks(
        query=query,
        body_bm25=body_bm25,
        heading_bm25=heading_bm25,
    )

    results = []

    for item in ranked_scores[:top_k]:
        chunk = chunks[item["index"]]

        results.append({
            "chunk_id": chunk.get("chunk_id"),
            "body_score": item["body_score"],
            "heading_score": item["heading_score"],
            "final_score": item["final_score"],
            "page_numbers": chunk.get("page_numbers", []),
            "headings": chunk.get("headings", []),
            "text": chunk.get("text", ""),
        })

    return results


    
    

def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    body_bm25, heading_bm25 = build_field_indexes(chunks)
    print("Body and heading BM25 indexes built successfully.")

    query = input("\nAsk a question about the paper: ")

    results = search_heading_aware(
        query=query,
        chunks=chunks,
        body_bm25=body_bm25,
        heading_bm25=heading_bm25,
        top_k=8,
    )

    for rank, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Rank: {rank}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Body score: {result['body_score']:.4f}")
        print(f"Heading score: {result['heading_score']:.4f}")
        print(f"Final score: {result['final_score']:.4f}")
        print(f"Pages: {result['page_numbers']}")
        print(f"Headings: {result['headings']}")
        print(f"Text:\n{result['text'][:700]}")


if __name__ == "__main__":
    main()

    
    
