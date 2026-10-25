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

from rag.map_retrieval_results import map_orchestrator_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "retrieval_runs"


def format_eval_run(output: dict[str, Any]) -> dict[str, Any]:
    """
    Format a lightweight run record suitable for evaluation and experiment tracking.
    Strips heavy static text and layout geometries that can always be hydrated
    on demand using chunk_id / record_id.
    """
    lean_text_results = []
    for rank, item in enumerate(output.get("text_results", []), start=1):
        clean_item = {
            "rank": rank,
            "chunk_id": item.get("chunk_id"),
            "document_id": item.get("document_id"),
            "page_numbers": item.get("page_numbers", []),
            "rrf_score": item.get("rrf_score"),
            "retrieved_by": item.get("retrieved_by", []),
            "source_ranks": item.get("source_ranks", {}),
        }
        for score_key in ("dense_score", "heading_score", "colpali_score"):
            if item.get(score_key) is not None:
                clean_item[score_key] = item[score_key]
        lean_text_results.append(clean_item)

    lean_fig_tab_results = []
    for item in output.get("figure_table_results", []):
        clean_item = {
            "rank": item.get("rank"),
            "record_type": item.get("record_type"),
            "record_id": item.get("record_id"),
            "page": item.get("page"),
            "rrf_score": item.get("rrf_score"),
            "retrieval_ranks": item.get("retrieval_ranks", []),
        }
        if item.get("score") is not None:
            clean_item["score"] = item["score"]
        lean_fig_tab_results.append(clean_item)

    return {
        "question": output.get("question"),
        "created_at": output.get("created_at"),
        "retrieval_config": output.get("retrieval_config"),
        "counts": output.get("counts"),
        "text_results": lean_text_results,
        "figure_table_results": lean_fig_tab_results,
    }


def run_hybrid_retrieval(
    question: str,
    text_top_k: int = 10,
    figure_table_top_k: int = 5,
    figure_table_candidate_k: int = 10,
    record_type: str | None = None,
    save_results: bool = True,
    lean_save: bool = True,
) -> dict[str, Any]:
    """
    Run text and figure/table hybrid retrieval independently.

    Returns one structured object containing both result lists.
    Optionally persists the retrieval output as JSON (lean by default for evaluation).
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

        data_to_save = format_eval_run(output) if lean_save else output

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                data_to_save,
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

    mapped = map_orchestrator_results(result)
    
    print("Mapped text chunks:", len(mapped["text_chunks"]))
    print("Mapped figure/table chunks:", len(mapped["figure_table_chunks"]))
