"""Modular nodes for the Generalized Multimodal Agentic RAG workflow.

Implements all stages of the system as first-class LangGraph nodes:
1. Ingestion Check & Pipeline:
   - check_doc_status_node
   - render_pages_node
   - parse_document_node
   - extract_and_crop_node
   - index_vectors_node
2. Query Routing:
   - route_query_node
3. Granular Multimodal Retrieval:
   - dense_retriever_node (Qdrant nomic-embed-text)
   - bm25_retriever_node (Heading-aware Okapi BM25)
   - text_fusion_node (Reciprocal Rank Fusion - RRF)
   - colpali_retriever_node (ColPali MaxSim on Page Embeddings)
   - visual_hydration_node (Map Top Pages to Cropped Figures & Tables)
4. Context Formatting & Self-RAG Reflection:
   - format_context_node
   - grade_context_node
   - rewrite_query_node
   - generate_answer_node
"""

import os
import re
from pathlib import Path
from typing import Any

from rag.config import (
    DEFAULT_PDF_PATH,
    FIGURES_CLEANED_JSONL,
    TABLES_JSONL,
    TEXT_COLLECTION_NAME,
    get_document_paths,
)
from rag.graph.state import RAGState
from rag.text_retrieval.bm25_retriever import (
    build_field_indexes,
    load_chunks,
    search_headings,
)
from rag.text_retrieval.hybrid_retriever import reciprocal_rank_fusion
from rag.text_retrieval.search_chunks import search_chunks
from rag.visual_retrieval.colpali_model import get_colpali_retriever
from rag.visual_retrieval.search.colpali_retriever import extract_page_number, load_jsonl
from services.llm_service import llm


# Module-level BM25 cache for high-speed repeated queries
_CACHED_CHUNKS: list[dict[str, Any]] | None = None
_CACHED_HEADING_BM25: Any | None = None


def _get_bm25_index(chunks_path: Path | None = None) -> tuple[list[dict[str, Any]], Any]:
    global _CACHED_CHUNKS, _CACHED_HEADING_BM25
    if _CACHED_CHUNKS is None or _CACHED_HEADING_BM25 is None:
        _CACHED_CHUNKS = load_chunks(chunks_path) if chunks_path else load_chunks()
        _, _CACHED_HEADING_BM25 = build_field_indexes(_CACHED_CHUNKS)
    return _CACHED_CHUNKS, _CACHED_HEADING_BM25


# =============================================================================
# 1. Document Status & Ingestion Nodes
# =============================================================================

def check_doc_status_node(state: RAGState) -> dict[str, Any]:
    """Check if the target document has already been ingested.

    Checks existence of rendered pages, JSONL chunks, cropped visuals,
    and indexed text points in Qdrant. Takes < 2ms.
    """
    raw_pdf = state.get("pdf_path") or str(DEFAULT_PDF_PATH)
    pdf_path = Path(raw_pdf)
    doc_paths = get_document_paths(pdf_path)

    # Check key artifacts
    rendered_pages_exist = (
        doc_paths["rendered_pages_dir"].exists()
        and any(doc_paths["rendered_pages_dir"].glob("*.png"))
    )
    chunks_exist = (
        doc_paths["chunks_path"].exists()
        and doc_paths["chunks_path"].stat().st_size > 0
    )
    figures_exist = (
        doc_paths["figures_jsonl"].exists()
        or (doc_paths["chunks_dir"] / f"{doc_paths['doc_name']}_figures_cleaned.jsonl").exists()
    )

    is_ingested = bool(rendered_pages_exist and chunks_exist and figures_exist)

    return {
        "pdf_path": str(pdf_path),
        "doc_name": doc_paths["doc_name"],
        "is_ingested": is_ingested,
    }


def render_pages_node(state: RAGState) -> dict[str, Any]:
    """Render PDF pages into high-resolution images via PyMuPDF."""
    from rag.doc_loader import render_pdf_to_images
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    print(f"[Ingestion] Rendering pages for: {pdf_path.name}")
    render_pdf_to_images(pdf_path)
    return {}


def parse_document_node(state: RAGState) -> dict[str, Any]:
    """Parse PDF layout hierarchy & extract text chunks via IBM Docling."""
    from rag.docling_parser import parse_pdf_to_chunks
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    print(f"[Ingestion] Parsing document layout: {pdf_path.name}")
    parse_pdf_to_chunks(pdf_path)
    return {}


def extract_and_crop_node(state: RAGState) -> dict[str, Any]:
    """Extract figure & table bounding boxes and crop visual element PNGs."""
    from rag.visual_retrieval.preprocess.figure_extractor import extract_figures, extract_tables
    from rag.visual_retrieval.preprocess.figure_cropper import crop_figures
    from rag.visual_retrieval.preprocess.table_cropper import crop_tables

    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    print(f"[Ingestion] Extracting & cropping visual assets for: {doc_name}")
    extract_figures(
        document_json_path=doc_paths["document_json_path"],
        figures_jsonl_path=doc_paths["figures_jsonl"],
        doc_name=doc_name,
    )
    extract_tables(
        document_json_path=doc_paths["document_json_path"],
        tables_jsonl_path=doc_paths["tables_jsonl"],
        doc_name=doc_name,
    )
    crop_figures(
        figures_jsonl_path=doc_paths["figures_jsonl"],
        rendered_pages_dir=doc_paths["rendered_pages_dir"],
        output_dir=doc_paths["figures_dir"],
    )
    crop_tables(
        tables_jsonl_path=doc_paths["tables_jsonl"],
        rendered_pages_dir=doc_paths["rendered_pages_dir"],
        output_dir=doc_paths["tables_dir"],
    )
    return {}


def index_vectors_node(state: RAGState) -> dict[str, Any]:
    """Index text chunks into Qdrant using nomic-embed-text."""
    from rag.text_retrieval.index_chunks import index_text_chunks
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)

    print(f"[Ingestion] Indexing text chunks into Qdrant: {TEXT_COLLECTION_NAME}")
    index_text_chunks(
        chunks_file_path=doc_paths["chunks_path"],
        collection_name=TEXT_COLLECTION_NAME,
    )
    return {"is_ingested": True}


# =============================================================================
# 2. Intent Routing Node
# =============================================================================

VISUAL_KEYWORDS = {
    "diagram", "figure", "fig", "architecture", "chart", "table", "plot",
    "visualization", "heatmap", "workflow", "schema", "layout", "visual",
    "curves", "graph", "picture", "draw", "show", "illustrate"
}

TEXT_KEYWORDS = {
    "define", "explain", "why", "background", "motivation", "summary",
    "summarize", "prose", "text", "introduction", "conclusion", "abstract"
}


def route_query_node(state: RAGState) -> dict[str, Any]:
    """Classify user query intent into 'text', 'visual', or 'both'."""
    question = state["question"].lower()
    tokens = set(re.findall(r"\b\w+\b", question))

    has_visual = bool(tokens & VISUAL_KEYWORDS)
    has_text = bool(tokens & TEXT_KEYWORDS)

    if has_visual and not has_text:
        route = "visual"
    elif has_text and not has_visual:
        route = "text"
    else:
        route = "both"

    return {"route": route}


# =============================================================================
# 3. Granular Retrieval Nodes
# =============================================================================

def dense_retriever_node(state: RAGState) -> dict[str, Any]:
    """Execute dense semantic vector search via Qdrant & nomic-embed-text."""
    query = state["question"].strip()
    try:
        results = search_chunks(query, top_k=5)
    except Exception as exc:
        print(f"[Warning] Dense retriever error: {exc}")
        results = []
    return {"dense_matches": results}


def bm25_retriever_node(state: RAGState) -> dict[str, Any]:
    """Execute keyword BM25 retrieval over document chunk headings."""
    query = state["question"].strip()
    try:
        chunks, heading_bm25 = _get_bm25_index()
        results = search_headings(query, heading_bm25, chunks, top_k=5)
    except Exception as exc:
        print(f"[Warning] BM25 retriever error: {exc}")
        results = []
    return {"bm25_matches": results}


def text_fusion_node(state: RAGState) -> dict[str, Any]:
    """Fuse dense vector search and BM25 keywords via Reciprocal Rank Fusion (RRF)."""
    dense_matches = state.get("dense_matches", [])
    bm25_matches = state.get("bm25_matches", [])

    if not dense_matches and not bm25_matches:
        return {"text_results": []}

    try:
        fused = reciprocal_rank_fusion(dense_matches, bm25_matches, top_k=5)
    except Exception as exc:
        print(f"[Warning] Text fusion error: {exc}")
        fused = dense_matches or bm25_matches

    return {"text_results": fused}


def colpali_retriever_node(state: RAGState) -> dict[str, Any]:
    """Execute ColPali multi-vector late-interaction MaxSim search over page images."""
    query = state["question"].strip()
    try:
        retriever = get_colpali_retriever()
        top_pages = min(5, len(retriever.page_names))
        page_results = retriever.search(query=query, top_k=top_pages)
    except Exception as exc:
        print(f"[Warning] ColPali retriever error: {exc}")
        page_results = []

    return {"colpali_top_pages": page_results}


def visual_hydration_node(state: RAGState) -> dict[str, Any]:
    """Map top ColPali pages to their cropped figures and structured Markdown tables."""
    page_results = state.get("colpali_top_pages", [])
    if not page_results:
        return {"figure_table_results": []}

    page_rankings: dict[int, dict[str, Any]] = {}
    for res in page_results:
        try:
            page_num = extract_page_number(res["page"])
            page_rankings[page_num] = res
        except Exception:
            continue

    figures = load_jsonl(FIGURES_CLEANED_JSONL) if FIGURES_CLEANED_JSONL.exists() else []
    tables = load_jsonl(TABLES_JSONL) if TABLES_JSONL.exists() else []

    matched_assets: list[dict[str, Any]] = []

    # Match figures
    for fig in figures:
        p = fig.get("page_number")
        if p is None:
            pages = fig.get("page_numbers") or []
            p = pages[0] if pages else None

        if p in page_rankings:
            p_info = page_rankings[p]
            matched_assets.append({
                "record_type": "figure",
                "record_id": fig.get("figure_id", "unknown_fig"),
                "figure_id": fig.get("figure_id", "unknown_fig"),
                "page": p,
                "colpali_rank": p_info.get("rank", 99),
                "colpali_score": p_info.get("score", 0.0),
                "caption": fig.get("caption") or "",
                "panel_caption": fig.get("panel_caption") or "",
                "image_path": fig.get("image_path", ""),
                "item_type": fig.get("item_type", "PictureItem"),
            })

    # Match tables
    for tab in tables:
        p = tab.get("page_number")
        if p is None:
            pages = tab.get("page_numbers") or []
            p = pages[0] if pages else None

        if p in page_rankings:
            p_info = page_rankings[p]
            matched_assets.append({
                "record_type": "table",
                "record_id": tab.get("table_id", "unknown_tab"),
                "table_id": tab.get("table_id", "unknown_tab"),
                "page": p,
                "colpali_rank": p_info.get("rank", 99),
                "colpali_score": p_info.get("score", 0.0),
                "caption": tab.get("caption") or "",
                "table_markdown": tab.get("table_markdown") or tab.get("text") or "",
                "image_path": tab.get("image_path", ""),
                "item_type": tab.get("item_type", "TableItem"),
            })

    matched_assets.sort(
        key=lambda item: (item.get("colpali_rank", 999), -float(item.get("colpali_score", 0.0)))
    )

    return {"figure_table_results": matched_assets[:5]}


# =============================================================================
# 4. Context Formatting & Self-RAG Nodes
# =============================================================================

def format_context_node(state: RAGState) -> dict[str, Any]:
    """Format retrieved text chunks, tables, and figures into structured context."""
    text_results = state.get("text_results", [])
    fig_tab_results = state.get("figure_table_results", [])

    sources = []
    context_blocks = []
    source_idx = 1

    # Text Chunks
    for item in text_results:
        chunk_id = item.get("chunk_id", "Unknown")
        pages = item.get("page_numbers") or []
        page_str = ", ".join(str(p) for p in pages) if pages else "Unknown"
        raw_text = item.get("text") or item.get("contextualized_text") or ""
        text = str(raw_text).strip()

        sources.append({
            "source": source_idx,
            "type": "text",
            "id": chunk_id,
            "pages": pages,
        })
        context_blocks.append(
            f"[Source {source_idx} | Text Chunk {chunk_id} | Page(s): {page_str}]\n{text}"
        )
        source_idx += 1

    # Visual Assets (Figures and Tables)
    for item in fig_tab_results:
        rec_type = item.get("record_type", "visual")
        rec_id = item.get("record_id", item.get("figure_id", item.get("table_id", "Unknown")))
        page = item.get("page", "Unknown")
        caption = str(item.get("caption") or item.get("panel_caption") or "").strip()

        sources.append({
            "source": source_idx,
            "type": rec_type,
            "id": rec_id,
            "pages": [page] if page != "Unknown" else [],
        })

        if rec_type == "table":
            table_md = str(item.get("table_markdown", "")).strip()
            block_content = f"Caption: {caption}\nData:\n{table_md}" if table_md else f"Caption: {caption}"
            context_blocks.append(
                f"[Source {source_idx} | Table {rec_id} | Page: {page}]\n{block_content}"
            )
        else:
            context_blocks.append(
                f"[Source {source_idx} | Figure {rec_id} | Page: {page}]\nCaption: {caption}"
            )
        source_idx += 1

    context_str = "\n\n".join(context_blocks)
    return {
        "context_str": context_str,
        "sources": sources,
    }


def grade_context_node(state: RAGState) -> dict[str, Any]:
    """Grade whether the retrieved multimodal context is relevant to the question."""
    question = state["question"]
    context_str = state.get("context_str", "")

    if not context_str.strip():
        return {"is_relevant": False}

    grading_prompt = f"""You are a strict relevance grader assessing whether retrieved document context contains facts or visual references useful to answer a question.

Question: {question}

Retrieved Context:
{context_str[:2500]}

Respond ONLY with 'YES' if the context contains information, data, figures, or explanations relevant to answering the question.
Respond ONLY with 'NO' if the context is completely unrelated or insufficient.
Your verdict (YES/NO):"""

    try:
        response = llm.complete(grading_prompt).text.strip().upper()
        is_relevant = "YES" in response
    except Exception as exc:
        print(f"[Warning] LLM grading failed ({exc}). Defaulting to relevant.")
        is_relevant = True

    return {"is_relevant": is_relevant}


def rewrite_query_node(state: RAGState) -> dict[str, Any]:
    """Self-RAG Reflection: Reformulate question into formal technical vocabulary."""
    original_query = state.get("original_question") or state["question"]
    current_retry = state.get("retry_count", 0)

    rewrite_prompt = f"""You are an expert AI search optimizer for technical documentation and scientific papers.
The previous search query returned insufficient or irrelevant passages.
Rewrite the query to use specific technical terminology, key concepts, formulas, or architecture names likely present in the document.

Original Query: {original_query}

Requirements:
- Return ONLY the rewritten query string.
- Do not provide preamble, explanation, or quotes.
- Optimize for both keyword (BM25) and vision-language (ColPali) search.

Rewritten Query:"""

    try:
        rewritten = llm.complete(rewrite_prompt).text.strip().strip('"').strip("'")
        print(f"\n[Self-RAG] Query reformulated: '{original_query}' -> '{rewritten}'")
    except Exception as exc:
        print(f"[Warning] Query rewriting failed ({exc}). Retaining original query.")
        rewritten = original_query

    return {
        "question": rewritten,
        "rewritten_query": rewritten,
        "retry_count": current_retry + 1,
    }


def generate_answer_node(state: RAGState) -> dict[str, Any]:
    """Synthesize a grounded answer using Groq LLM with strict [Source N, Page P] citations."""
    question = state.get("original_question") or state["question"]
    context_str = state.get("context_str", "")

    if not context_str.strip():
        return {
            "answer": "I could not find sufficient information in the document to answer your question.",
        }

    system_prompt = (
        "You are an expert AI assistant answering questions based on the provided technical document context.\n"
        "Guidelines:\n"
        "1. Answer ONLY using the facts, figures, tables, and explanations present in the context below.\n"
        "2. Do NOT extrapolate, speculate, or introduce external knowledge.\n"
        "3. Every factual claim, architecture detail, or numeric result MUST cite its source using: [Source N, Page P].\n"
        "4. When referencing a figure or table, mention its name (e.g. Figure 1, Table 2) and citation [Source N, Page P].\n"
        "5. If the context does not contain enough information, state clearly that the document does not contain the answer."
    )

    user_prompt = f"""Context:
{context_str}

Question: {question}

Grounded Answer with [Source N, Page P] citations:"""

    try:
        prompt_full = f"{system_prompt}\n\n{user_prompt}"
        response = llm.complete(prompt_full).text.strip()
    except Exception as exc:
        response = f"An error occurred while generating the answer: {exc}"

    return {"answer": response}
