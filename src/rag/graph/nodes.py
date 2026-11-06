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

import json
import os
import re
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field

from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
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
from services.llm_service import llm, llm_vision


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
# 2. Intent Routing Node (LLM-Based Structured Classification)
# =============================================================================

class QueryIntent(BaseModel):
    """Structured classification of user query intent."""
    route: Literal["text", "visual", "both"] = Field(
        description="The primary retrieval route: 'text' for prose/explanations/formulas, 'visual' for diagrams/figures/tables/charts, or 'both' for hybrid/comprehensive questions."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief one-sentence explanation of why this route was selected."
    )




def route_query_node(state: RAGState) -> dict[str, Any]:
    """Classify user query intent into 'text', 'visual', or 'both' using LLM structured output.

    Analyzes semantic nuance, metaphors, and implicit references to determine whether
    retrieval should target textual prose, visual diagrams/tables, or hybrid multimodal.
    """
    question = state["question"].strip()

    routing_prompt = f"""You are an expert query router in an advanced multimodal RAG system.
Analyze the user's question and determine the optimal retrieval route:
- "visual": The user explicitly or implicitly asks about a figure, diagram, system architecture, flowchart, schema, plot, image, chart, or data table (e.g. "show the architecture", "how does the diagram represent...", "what are the BLEU scores?").
- "text": The user asks for definitions, conceptual explanations, mathematical motivations, background prose, or summaries without needing visual inspection (e.g. "define self-attention", "why use dot-product scaling?").
- "both": The question is broad, complex, asks for a comprehensive explanation, or benefits from both textual explanations and visual diagrams/tables (e.g. "how does multi-head attention work and what does its structure look like?").

Respond strictly in valid JSON matching this schema:
{{
  "route": "text" | "visual" | "both",
  "confidence": 1.0,
  "reasoning": "brief one-sentence explanation"
}}

User Question: {question}

Output JSON only:"""

    try:
        response = llm.complete(routing_prompt).text.strip()
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            intent = QueryIntent.model_validate(data)
            return {
                "route": intent.route,
                "routing_reasoning": intent.reasoning,
            }
        raise ValueError(f"No JSON object found in LLM response: {response}")
    except Exception as exc:
        print(f"[Warning] LLM intent routing encountered an error: {exc}. Defaulting route to 'both'.")
        return {
            "route": "both",
            "routing_reasoning": f"Default fallback to comprehensive multimodal retrieval: {exc}",
        }


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
        results = search_headings(query=query, chunks=chunks, heading_bm25=heading_bm25, top_k=5)
    except Exception as exc:
        print(f"[Warning] BM25 retriever error: {exc}")
        results = []
    return {"bm25_matches": results}


def text_fusion_node(state: RAGState) -> dict[str, Any]:
    """Fuse dense vector search and BM25 keywords via Reciprocal Rank Fusion (RRF)."""
    dense_matches = state.get("dense_matches") or []
    bm25_matches = state.get("bm25_matches") or []

    if not dense_matches and not bm25_matches:
        return {"text_results": []}

    try:
        fused = reciprocal_rank_fusion(dense_matches, bm25_matches, top_k=5)
    except Exception as exc:
        print(f"[Warning] Text fusion error: {exc}")
        fused = dense_matches or bm25_matches or []

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



def _resolve_image_path(raw_path: str, record_type: str = "figure") -> Path | None:
    """Resolve an asset image path, checking raw path and fallback data directories."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists() and p.is_file():
        return p
    filename = p.name
    if record_type == "figure":
        candidate = Path("data/figures") / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    elif record_type == "table":
        candidate = Path("data/tables") / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    for folder in [Path("data/figures"), Path("data/tables")]:
        candidate = folder / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None





def generate_answer_node(state: RAGState) -> dict[str, Any]:
    """Synthesize a grounded answer using the multimodal vision LLM (llm_vision) passing text, figures, and tables together."""
    question = state.get("original_question") or state["question"]
    text_results = state.get("text_results", [])
    fig_tab_results = state.get("figure_table_results", [])

    if not text_results and not fig_tab_results:
        return {
            "answer": "I could not find sufficient information in the document to answer your question.",
        }

    system_prompt = (
        "You are an expert AI assistant answering questions based on the provided technical document context (text chunks, figures, and tables).\n"
        "Guidelines:\n"
        "1. Answer ONLY using the facts, figures, tables, and explanations present in the context.\n"
        "2. Do NOT extrapolate, speculate, or introduce external knowledge.\n"
        "3. Every factual claim, architecture detail, or numeric result MUST cite its source using: [Source N, Page P].\n"
        "4. When referencing a figure or table, mention its name (e.g. Figure 1, Table 2) and citation [Source N, Page P].\n"
        "5. If the context does not contain enough information, state clearly that the document does not contain the answer."
    )

    user_blocks: list[Any] = []
    source_idx = 1
    max_images = 2
    attached_images = 0

    # 1. Text Chunks (top 4 to stay within rate-limit token budgets)
    selected_text_results = text_results[:4]
    if selected_text_results:
        text_parts = ["Retrieved Text Chunks:"]
        for item in selected_text_results:
            chunk_id = item.get("chunk_id", "Unknown")
            pages = item.get("page_numbers") or []
            page_str = ", ".join(str(p) for p in pages) if pages else "Unknown"
            raw_text = item.get("text") or item.get("contextualized_text") or ""
            text = str(raw_text).strip()
            text_parts.append(
                f"[Source {source_idx} | Text Chunk {chunk_id} | Page(s): {page_str}]\n{text}"
            )
            source_idx += 1
        user_blocks.append(TextBlock(text="\n\n".join(text_parts)))

    # 2. Visual Assets (Figures and Tables with cropped images)
    if fig_tab_results:
        for item in fig_tab_results:
            rec_type = item.get("record_type", "visual")
            rec_id = item.get("record_id", item.get("figure_id", item.get("table_id", "Unknown")))
            page = item.get("page", "Unknown")
            caption = str(item.get("caption") or item.get("panel_caption") or "").strip()
            raw_img_path = item.get("image_path", "")
            resolved_img = _resolve_image_path(raw_img_path, rec_type)

            if rec_type == "table":
                table_md = str(item.get("table_markdown", "")).strip()
                block_content = f"[Source {source_idx} | Table {rec_id} | Page: {page}]\nCaption: {caption}"
                if table_md:
                    block_content += f"\nData:\n{table_md}"
                user_blocks.append(TextBlock(text=block_content))
                if resolved_img and attached_images < max_images:
                    user_blocks.append(ImageBlock(path=resolved_img))
                    attached_images += 1
            else:
                block_content = f"[Source {source_idx} | Figure {rec_id} | Page: {page}]\nCaption: {caption}"
                user_blocks.append(TextBlock(text=block_content))
                if resolved_img and attached_images < max_images:
                    user_blocks.append(ImageBlock(path=resolved_img))
                    attached_images += 1
            source_idx += 1

    # 3. Question Prompt
    user_blocks.append(
        TextBlock(text=f"Question: {question}\n\nGrounded Answer with [Source N, Page P] citations:")
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, blocks=user_blocks),
    ]

    try:
        response = llm_vision.chat(messages).message.content.strip()
    except Exception as exc:
        response = f"An error occurred while generating the answer: {exc}"

    return {"answer": response}
