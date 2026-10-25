import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Existing text hybrid retrieval + text RRF.
# Change "rag.rrf" if your text RRF code is in another module.
from rag.Text_Retrieval.hybrid_retriever import retrieve_and_fuse as retrieve_text_hybrid

# Figure/table hybrid retrieval + its own RRF.
from rag.Figure_Table_Retrieval.query_figures_tables.hybrid_retrieval import (
    hybrid_retrieve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "retrieval_runs"


def run_hybrid_retrieval(
    question: str,
    text_top_k: int = 10,
    figure_table_top_k: int = 5,
    figure_table_candidate_k: int = 10,
    record_type: str | None = None,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Run text and figure/table hybrid retrieval independently.

    Returns one structured object containing both result lists.
    Optionally persists the complete retrieval output as JSON.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if text_top_k < 1:
        raise ValueError("text_top_k must be at least 1.")

    if figure_table_top_k < 1:
        raise ValueError("figure_table_top_k must be at least 1.")

    if figure_table_candidate_k < 1:
        raise ValueError("figure_table_candidate_k must be at least 1.")

    if record_type not in (None, "figure", "table"):
        raise ValueError(
            "record_type must be None, 'figure', or 'table'."
        )

    question = question.strip()

    # Branch 1: text dense + heading BM25 + ColPali + text RRF.
    _, text_results = retrieve_text_hybrid(
        question,
        top_k=text_top_k,
    )

    # Branch 2: figure/table dense + BM25 + figure/table RRF.
    figure_table_results = hybrid_retrieve(
        query=question,
        top_k=figure_table_top_k,
        candidate_k=figure_table_candidate_k,
        record_type=record_type,
    )

    # Keep the branches separate for downstream VLM/LLM processing.
    output: dict[str, Any] = {
        "question": question,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_config": {
            "text_top_k": text_top_k,
            "figure_table_top_k": figure_table_top_k,
            "figure_table_candidate_k": figure_table_candidate_k,
            "record_type": record_type,
        },
        "text_results": text_results,
        "figure_table_results": figure_table_results,
        "counts": {
            "text": len(text_results),
            "figures_and_tables": len(figure_table_results),
        },
    }

    if save_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        output_path = OUTPUT_DIR / f"retrieval_{timestamp}.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                output,
                file,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        output["saved_to"] = str(output_path)

    return output




if __name__ == "__main__":
    query = input("Enter your research question: ").strip()

    result = run_hybrid_retrieval(query)

    print("\n=== Retrieval Summary ===")
    print(f"Question: {result['question']}")
    print(f"Text results: {result['counts']['text']}")
    print(
        "Figure/table results: "
        f"{result['counts']['figures_and_tables']}"
    )

    if result.get("saved_to"):
        print(f"Saved results to: {result['saved_to']}")

    print("\n=== Text Results ===")
    for item in result["text_results"]:
        print(
            f"Rank: {item.get('rank')} | "
            f"Page(s): {item.get('page_numbers')} | "
            f"RRF score: {item.get('rrf_score')}"
        )

    print("\n=== Figure/Table Results ===")
    for item in result["figure_table_results"]:
        print(
            f"Rank: {item.get('rank')} | "
            f"Type: {item.get('record_type')} | "
            f"ID: {item.get('record_id')} | "
            f"Page: {item.get('page')} | "
            f"RRF score: {item.get('rrf_score')}"
        )
