import json
import re
from pathlib import Path
from typing import Any

from rag.config import (
    FIGURES_CLEANED_JSONL as FIGURES_PATH,
    TABLES_JSONL as TABLES_PATH,
)
from rag.visual_retrieval.colpali_model import get_colpali_retriever


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines records from a file."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path.name}, line {line_number}"
                ) from exc
    return records




def extract_page_number(page_str: str) -> int:
    """Extract integer page number from page names like 'page_0003.png' or '3'."""
    stem = Path(page_str).stem
    # Match any sequence of digits in the stem
    match = re.search(r"\d+", stem)
    if match:
        return int(match.group())
    raise ValueError(f"Cannot extract page number from: {page_str}")




def retrieve_visual_assets(
    query: str,
    top_pages: int = 5,
    top_k: int = 10,
    record_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve figures and tables by mapping ColPali page rankings.

    Option A Implementation:
    1. Runs ColPali Late-Interaction (MaxSim) on rendered page embeddings.
    2. Maps top-scoring pages to the figures and tables residing on those pages.
    3. Returns sorted visual records with ColPali rank, score, and content.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if record_type not in (None, "figure", "table"):
        raise ValueError("record_type must be None, 'figure', or 'table'.")

    # 1. Run ColPali page search
    retriever = get_colpali_retriever()
    try:
        page_results = retriever.search(
            query=query.strip(),
            top_k=top_pages,
        )
    except Exception as exc:
        print(f"[Visual Retrieval] ColPali search encountered exception: {exc}")
        page_results = []

    # Map page number to ColPali ranking metadata
    page_rankings: dict[int, dict[str, Any]] = {}
    for res in page_results:
        page_num = extract_page_number(res["page"])
        page_rankings[page_num] = res

    # 2. Load figures and tables
    figures = load_jsonl(FIGURES_PATH) if FIGURES_PATH.exists() else []
    tables = load_jsonl(TABLES_PATH) if TABLES_PATH.exists() else []

    matched_assets: list[dict[str, Any]] = []

    # 3. Match figures on retrieved pages
    if record_type in (None, "figure"):
        for fig in figures:
            fig_page = fig.get("page_number")
            if fig_page is None:
                pages = fig.get("page_numbers") or []
                fig_page = pages[0] if pages else None

            if fig_page in page_rankings:
                p_info = page_rankings[fig_page]
                matched_assets.append({
                    "record_type": "figure",
                    "record_id": fig.get("figure_id", "unknown_fig"),
                    "figure_id": fig.get("figure_id", "unknown_fig"),
                    "page": fig_page,
                    "colpali_rank": p_info["rank"],
                    "colpali_score": p_info["score"],
                    "caption": fig.get("caption") or "",
                    "panel_caption": fig.get("panel_caption") or "",
                    "image_path": fig.get("image_path", ""),
                    "bounding_boxes": fig.get("bounding_boxes", []),
                    "item_type": fig.get("item_type", "PictureItem"),
                })

    # 4. Match tables on retrieved pages
    if record_type in (None, "table"):
        for tab in tables:
            tab_page = tab.get("page_number")
            if tab_page is None:
                pages = tab.get("page_numbers") or []
                tab_page = pages[0] if pages else None

            if tab_page in page_rankings:
                p_info = page_rankings[tab_page]
                matched_assets.append({
                    "record_type": "table",
                    "record_id": tab.get("table_id", "unknown_tab"),
                    "table_id": tab.get("table_id", "unknown_tab"),
                    "page": tab_page,
                    "colpali_rank": p_info["rank"],
                    "colpali_score": p_info["score"],
                    "caption": tab.get("caption") or "",
                    "table_markdown": tab.get("table_markdown") or "",
                    "image_path": tab.get("image_path", ""),
                    "bounding_boxes": tab.get("bounding_boxes", []),
                    "item_type": tab.get("item_type", "TableItem"),
                })

    # Fallback if ColPali returns 0 assets (e.g. cloud ZeroGPU rate-limit or network timeout)
    if not matched_assets:
        query_words = set(re.findall(r"\w+", query.lower()))
        if record_type in (None, "figure"):
            for fig in figures:
                fig_text = f"{fig.get('figure_id', '')} {fig.get('caption', '')} {fig.get('panel_caption', '')}".lower()
                matches = sum(1 for w in query_words if len(w) > 2 and w in fig_text)
                if matches > 0:
                    fig_page = fig.get("page_number") or (fig.get("page_numbers") or [1])[0]
                    matched_assets.append({
                        "record_type": "figure",
                        "record_id": fig.get("figure_id", "unknown_fig"),
                        "figure_id": fig.get("figure_id", "unknown_fig"),
                        "page": fig_page,
                        "colpali_rank": 99,
                        "colpali_score": float(matches),
                        "caption": fig.get("caption") or "",
                        "panel_caption": fig.get("panel_caption") or "",
                        "image_path": fig.get("image_path", ""),
                        "bounding_boxes": fig.get("bounding_boxes", []),
                        "item_type": fig.get("item_type", "PictureItem"),
                    })
        if record_type in (None, "table"):
            for tab in tables:
                tab_text = f"{tab.get('table_id', '')} {tab.get('caption', '')} {tab.get('table_markdown', '')}".lower()
                matches = sum(1 for w in query_words if len(w) > 2 and w in tab_text)
                if matches > 0:
                    tab_page = tab.get("page_number") or (tab.get("page_numbers") or [1])[0]
                    matched_assets.append({
                        "record_type": "table",
                        "record_id": tab.get("table_id", "unknown_tab"),
                        "table_id": tab.get("table_id", "unknown_tab"),
                        "page": tab_page,
                        "colpali_rank": 99,
                        "colpali_score": float(matches),
                        "caption": tab.get("caption") or "",
                        "table_markdown": tab.get("table_markdown") or "",
                        "image_path": tab.get("image_path", ""),
                        "bounding_boxes": tab.get("bounding_boxes", []),
                        "item_type": tab.get("item_type", "TableItem"),
                    })

    # 5. Sort by ColPali page rank, then highest ColPali score
    matched_assets.sort(
        key=lambda item: (
            item["colpali_rank"],
            -item["colpali_score"],
        )
    )

    # 6. Assign final overall visual ranks
    for rank, item in enumerate(matched_assets, start=1):
        item["rank"] = rank

    return matched_assets[:top_k]


def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
    record_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve figures and tables using ColPali visual retrieval.

    Maintains backwards compatibility while delegating directly to
    ColPali visual page-to-asset mapping.
    """
    return retrieve_visual_assets(
        query=query,
        top_pages=candidate_k,
        top_k=top_k,
        record_type=record_type,
    )




if __name__ == "__main__":
    test_query = input("Enter visual search query: ").strip()
    if test_query:
        results = retrieve_visual_assets(test_query, top_k=5)
        print(f"\n--- ColPali Visual Results for: '{test_query}' ---")
        for res in results:
            print(
                f"Rank {res['rank']} | Type: {res['record_type']} | "
                f"ID: {res['record_id']} | Page: {res['page']} | "
                f"ColPali Score: {res['colpali_score']:.4f}"
            )
            print(f"Caption: {res['caption'][:120]}")
            if res.get("table_markdown"):
                print("Table preview:\n" + res["table_markdown"][:200])
            print("-" * 50)
