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
4. Context Formatting & CRAG Correction:
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

import sys
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _safe_str(text: Any, max_len: int = 80) -> str:
    """Safely truncate and sanitize arbitrary text for console printing."""
    if not text:
        return ""
    cleaned = str(text).replace("\n", " ").strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "..."
    return cleaned.encode("ascii", errors="replace").decode("ascii")
from rag.config import (
    DEFAULT_PDF_PATH,
    FIGURES_CLEANED_JSONL,
    FIGURES_DIR,
    TABLES_DIR,
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
        doc_paths["figures_raw_jsonl"].exists()
        or doc_paths["figures_cleaned"].exists()
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
    doc_paths = get_document_paths(pdf_path)
    print(f"[Ingestion] Rendering pages for: {pdf_path.name}")
    render_pdf_to_images(pdf_path, output_dir=doc_paths["rendered_pages_dir"])
    return {}


def parse_document_node(state: RAGState) -> dict[str, Any]:
    """Parse PDF layout hierarchy & extract text chunks via IBM Docling."""
    from rag.docling_parser import parse_pdf_to_chunks
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    print(f"[Ingestion] Parsing document layout: {pdf_path.name}")
    parse_pdf_to_chunks(pdf_path, doc_paths=doc_paths)
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
        document=doc_paths["document_json_path"],
        doc_name=doc_name,
        output_dir=doc_paths["figures_dir"],
        figures_path=doc_paths["figures_raw_jsonl"],
    )
    extract_tables(
        document=doc_paths["document_json_path"],
        doc_name=doc_name,
        tables_path=doc_paths["tables_jsonl"],
    )
    crop_figures(
        pdf_path=pdf_path,
        figures_jsonl=doc_paths["figures_raw_jsonl"],
        output_dir=doc_paths["figures_dir"],
        rendered_pages_dir=doc_paths["rendered_pages_dir"],
    )
    crop_tables(
        pdf_path=pdf_path,
        tables_jsonl=doc_paths["tables_jsonl"],
        output_dir=doc_paths["tables_dir"],
        rendered_pages_dir=doc_paths["rendered_pages_dir"],
    )
    return {}


def index_vectors_node(state: RAGState) -> dict[str, Any]:
    """Index text chunks into Qdrant using nomic-embed-text."""
    from rag.text_retrieval.index_chunks import index_text_chunks
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    print(f"[Ingestion] Indexing text chunks into Qdrant: {TEXT_COLLECTION_NAME} (doc_id={doc_name})")
    index_text_chunks(
        chunks_file_path=doc_paths["chunks_path"],
        collection_name=TEXT_COLLECTION_NAME,
        doc_id=doc_name,
    )
    return {}


def index_colpali_node(state: RAGState) -> dict[str, Any]:
    """Send rendered page images to Hugging Face ZeroGPU Space to index visual embeddings."""
    from rag.visual_retrieval.colpali_client import get_colpali_cloud_client
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    page_images = sorted(list(doc_paths["rendered_pages_dir"].glob("*.png")))
    if page_images:
        print(f"[Ingestion] Registering {len(page_images)} pages with ColPali Cloud Space for '{doc_name}'...")
        try:
            client = get_colpali_cloud_client()
            client.index_document(doc_id=doc_name, image_paths=page_images)
        except Exception as exc:
            print(f"[Warning] ColPali Cloud indexing encountered error: {exc}")

    return {"is_ingested": True}


# =============================================================================
# 2. Intent Routing Node (LLM-Based Structured Classification)
# =============================================================================

class QueryIntent(BaseModel):
    """Structured classification of user query intent."""
    route: Literal["text", "visual", "both"] = Field(
        description="The primary retrieval route: 'text' for verbal prose/definitions/background, 'visual' for pure diagrams/figures/plots, or 'both' for equations/formulas, architectural mechanics, tables, calculations, or hybrid queries."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief one-sentence explanation of why this route was selected."
    )




def contextualize_query(question: str, history: list[dict[str, Any]] | None) -> str:
    """Reformulate a follow-up query using conversation history into a standalone search query."""
    if not history:
        return question

    recent = history[-4:]
    formatted_history = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        if len(content) > 300:
            content = content[:300] + "..."
        if content:
            formatted_history.append(f"{role}: {content}")

    if not formatted_history:
        return question

    history_str = "\n".join(formatted_history)

    prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a clear, standalone search query that contains all necessary entities, subject names, and context from previous turns.
If the question is already complete, self-contained, or does not refer to previous messages, return the question unchanged. Do not answer the question; output ONLY the reformulated standalone question text.

Conversation History:
{history_str}

Follow-up Question: {question}

Standalone Question:"""

    try:
        reformulated = llm.complete(prompt).text.strip()
        reformulated = reformulated.strip('"').strip("'").strip()
        if reformulated and len(reformulated) >= 3:
            return reformulated
    except Exception as exc:
        print(f"[Warning] Contextual query reformulation fallback: {exc}")
    return question


def route_query_node(state: RAGState) -> dict[str, Any]:
    """Classify user query intent into 'text', 'visual', or 'both' using LLM structured output.

    If conversation history is present, first contextualizes follow-up questions to resolve coreferences.
    """
    original_q = state.get("original_question") or state["question"].strip()
    history = state.get("chat_history", [])

    # Resolve follow-up references using short-term conversational memory
    active_question = contextualize_query(original_q, history)

    # -------------------------------------------------------------------------
    # Routing code commented out per user instruction: always invoke 'both'
    # (ColPali figures & tables + Dense retrieval + BM25)
    # -------------------------------------------------------------------------
    # routing_prompt = f"""You are an expert query router in an advanced multimodal RAG system..."""
    # try:
    #     response = llm.complete(routing_prompt).text.strip()
    #     ...
    # -------------------------------------------------------------------------

    print("\n" + "=" * 65)
    print("[Query Intent Router] Routing bypassed: Always invoking BOTH (Dense + BM25 + ColPali)")
    if active_question != original_q:
        print(f"  └─ Contextualized from: '{original_q}' -> '{active_question}'")
    print("=" * 65)

    return {
        "question": active_question,
        "original_question": original_q,
        "route": "both",
        "routing_reasoning": "Routing bypassed: always executing parallel multimodal retrieval (Dense + BM25 + ColPali).",
    }


# =============================================================================
# 3. Granular Retrieval Nodes
# =============================================================================

def dense_retriever_node(state: RAGState) -> dict[str, Any]:
    """Execute dense semantic vector search via Qdrant & nomic-embed-text."""
    query = state["question"].strip()
    doc_id = state.get("doc_name")
    try:
        results = search_chunks(query, top_k=5, doc_id=doc_id)
        print(f"  [Dense Retriever] Found {len(results)} candidate chunks in Qdrant.")
    except Exception as exc:
        print(f"[Warning] Dense retriever error: {exc}")
        results = []
    return {"dense_matches": results}


def bm25_retriever_node(state: RAGState) -> dict[str, Any]:
    """Execute keyword BM25 retrieval over document chunk headings."""
    from rag.text_retrieval.bm25_retriever import search_bm25
    query = state["question"].strip()
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    try:
        results = search_bm25(query=query, top_k=5, chunks_path=doc_paths["chunks_path"])
        print(f"  [BM25 Retriever] Found {len(results)} lexical matches via Okapi BM25.")
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
        print(f"  [Text Fusion] Reciprocal Rank Fusion fused into top {len(fused)} text chunks.")
    except Exception as exc:
        print(f"[Warning] Text fusion error: {exc}")
        fused = dense_matches or bm25_matches or []

    return {"text_results": fused}


def colpali_retriever_node(state: RAGState) -> dict[str, Any]:
    """Execute ColPali multi-vector late-interaction MaxSim search via Hugging Face Space."""
    query = state["question"].strip()
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    try:
        retriever = get_colpali_retriever()
        page_results = retriever.search(query=query, top_k=5, doc_id=doc_name)
        print(f"  [ColPali Retriever] MaxSim late interaction retrieved {len(page_results)} top pages on A100 ZeroGPU.")
    except Exception as exc:
        print(f"[Warning] ColPali cloud search error: {exc}")
        page_results = []

    return {"colpali_top_pages": page_results}



def visual_hydration_node(state: RAGState) -> dict[str, Any]:
    """Map top ColPali pages to their cropped figures and structured Markdown tables."""
    page_results = state.get("colpali_top_pages", [])
    if not page_results:
        return {"figure_table_results": []}

    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)

    page_rankings: dict[int, dict[str, Any]] = {}
    for res in page_results:
        try:
            page_num = res.get("page_number") or extract_page_number(res["page"])
            page_rankings[page_num] = res
        except Exception:
            continue

    figures_file = doc_paths["figures_cleaned"] if doc_paths["figures_cleaned"].exists() else doc_paths["figures_raw_jsonl"]
    figures = load_jsonl(figures_file) if figures_file.exists() else []
    tables = load_jsonl(doc_paths["tables_jsonl"]) if doc_paths["tables_jsonl"].exists() else []

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
                "parent_figure": fig.get("parent_figure") or "",
                "panel": fig.get("panel") or "",
                "panel_caption": fig.get("panel_caption") or "",
                "shared_caption": fig.get("shared_caption") or "",
                "page": p,
                "colpali_rank": p_info.get("rank", 99),
                "colpali_score": p_info.get("score", 0.0),
                "caption": fig.get("caption") or "",
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
    print(f"  [Visual Hydration] Hydrated {len(matched_assets)} figures/tables from top ColPali pages.")

    return {"figure_table_results": matched_assets[:10]}


# =============================================================================
# 4. Context Formatting & CRAG Nodes
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
            "parent_figure": item.get("parent_figure"),
            "panel": item.get("panel"),
            "panel_caption": item.get("panel_caption"),
            "caption": caption,
            "table_markdown": item.get("table_markdown", ""),
            "image_path": item.get("image_path", ""),
        })

        if rec_type == "table":
            table_md = str(item.get("table_markdown", "")).strip()
            block_content = f"Caption: {caption}"
            if table_md:
                block_content += f"\nTable Markdown Grid:\n{table_md}"
            context_blocks.append(
                f"[Source {source_idx} | Table {rec_id} | Page: {page}]\n{block_content}"
            )
        else:
            fig_lines = []
            parent_fig = item.get("parent_figure")
            panel = item.get("panel")
            panel_cap = item.get("panel_caption")
            if parent_fig:
                fig_lines.append(f"Parent Figure: {parent_fig}")
            if panel:
                fig_lines.append(f"Panel: {panel}")
            if panel_cap:
                fig_lines.append(f"Panel Caption: {panel_cap}")
            if caption:
                fig_lines.append(f"Caption: {caption}")
            block_content = "\n".join(fig_lines) if fig_lines else f"Caption: {caption}"
            context_blocks.append(
                f"[Source {source_idx} | Figure {rec_id} | Page: {page}]\n{block_content}"
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

    grading_prompt = f"""You are a strict relevance grader assessing whether retrieved document context contains facts, figures, formulas, or explanations useful to answer the user's specific question.

Question: {question}

Retrieved Context:
{context_str[:3000]}

Respond ONLY with 'YES' if the context contains information, data, figures, formulas, or explanations directly useful to address the question.
Respond ONLY with 'NO' if the context is completely unrelated, irrelevant, or missing the requested mechanism/formula/table.
Your verdict (YES/NO):"""

    try:
        response = llm.complete(grading_prompt).text.strip().upper()
        is_relevant = "YES" in response
    except Exception as exc:
        print(f"[Warning] LLM grading failed ({exc}). Defaulting to relevant.")
        is_relevant = True

    return {"is_relevant": is_relevant}




def rewrite_query_node(state: RAGState) -> dict[str, Any]:
    """Corrective RAG (CRAG): Reformulate question into formal technical vocabulary."""
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
        print(f"\n[CRAG] Query reformulated: '{original_query}' -> '{rewritten}'")
    except Exception as exc:
        print(f"[Warning] Query rewriting failed ({exc}). Retaining original query.")
        rewritten = original_query

    return {
        "question": rewritten,
        "rewritten_query": rewritten,
        "retry_count": current_retry + 1,
    }



def _resolve_image_path(raw_path: str, record_type: str = "figure") -> Path | None:
    """Resolve an asset image path, checking raw path and canonical data directories."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists() and p.is_file():
        return p
    filename = p.name
    if record_type == "figure":
        candidate = FIGURES_DIR / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    elif record_type == "table":
        candidate = TABLES_DIR / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    for folder in [FIGURES_DIR, TABLES_DIR]:
        candidate = folder / filename
        if candidate.exists() and candidate.is_file():
            return candidate
    return None





def _is_unanswerable_response(text: str) -> bool:
    """Detect whether LLM response indicates that the document lacks information for the query."""
    if not text:
        return True
    lower = text.lower().strip()

    # Exact or prefix match
    if lower.startswith("i don't have information") or lower.startswith("i do not have information"):
        return True
    if lower.startswith("there is no information") or lower.startswith("there's no information"):
        return True
    if lower.startswith("the document does not contain") or lower.startswith("the provided document does not"):
        return True
    if lower.startswith("the provided text chunks") or lower.startswith("based on the provided text chunks"):
        return True
    if lower.startswith("based on the provided document context, there is no"):
        return True

    # Substring checks
    unanswerable_phrases = [
        "the provided text chunks exclusively discuss",
        "the provided text chunks do not contain",
        "not contain any information regarding",
        "does not contain any information about",
        "does not mention anything about",
        "no information about that in the document",
        "no information regarding this in the document",
    ]
    for phrase in unanswerable_phrases:
        if phrase in lower:
            return True

    return False


def generate_answer_node(state: RAGState) -> dict[str, Any]:
    """Synthesize a grounded answer using the multimodal vision LLM (llm_vision) passing text, figures, tables, and conversation history."""
    question = state.get("original_question") or state["question"]
    text_results = state.get("text_results", [])
    fig_tab_results = state.get("figure_table_results", [])
    is_relevant = state.get("is_relevant", True)
    chat_history = state.get("chat_history") or []

    if not text_results and not fig_tab_results:
        ans = "I don't have information about that."
        return {
            "answer": ans,
            "chat_history": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": ans, "sources": []},
            ],
        }

# Removed premature is_relevant blocker; model synthesizes with all available evidence

    system_prompt = (
        "You are an expert scientific and technical AI assistant answering questions based strictly on the provided technical document context.\n"
        "Guidelines:\n"
        "1. Style & Tone: Write in clear, authoritative, fluent prose with clean markdown formatting. Use **bold** for key terms and concepts. When presenting lists, ensure each item starts on a new line. Directly address the user's question using facts from the document.\n"
        "2. Grounding: Answer strictly using the retrieved text chunks, tables, and figures. Synthesize explanations accurately without speculating or hallucinating unmentioned facts.\n"
        "3. Mathematical Equations: When expressing formulas, equations, or variables, ALWAYS use standard LaTeX syntax:\n"
        "   - Use display math blocks '$$ ... $$' on their own separate lines for primary equations (e.g. $$ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V $$).\n"
        "   - Use inline math '$...$' for variables, dimensions, and short expressions (e.g. '$d_k$', '$d_v$', '$Q$', '$K$', '$V$', '$\\sqrt{d_k}$', '$O(n^2 \\cdot d)$').\n"
        "4. Citations: Ground your statements with concise source citations [Source N, Page P] placed at the end of sentences where facts are introduced. Do not interrupt formulas or insert citations between words.\n"
        "5. Visual References: When referencing figures or tables, explicitly mention them (e.g. 'as illustrated in Figure 2 [Source N, Page P]').\n"
        "6. Unanswerable Questions: If the document contains genuinely no information to answer the question, respond ONLY with: 'I don't have information about that in the document.'\n"
        "   - Do NOT add any extra explanation, summary, or commentary.\n"
        "   - Do NOT mention 'The provided text chunks...' or 'The provided context...'.\n"
        "   - Do NOT list or describe what topics, formulas, or architectures ARE in the document.\n"
        "   - Stop immediately after stating that you do not have information about it."
    )

    user_blocks: list[Any] = []
    source_idx = 1
    max_images = 2
    attached_images = 0

    # 1. Text Chunks (stay comfortably within Groq's 7,000 ITPM rate limit)
    max_chunks = 5
    selected_text_results = text_results[:max_chunks]
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
    max_images = 1  # 1 image keeps total payload safely under Groq's 7,000 ITPM limit
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
                # Truncate giant table markdown to 800 chars to prevent rate limit spikes
                if len(table_md) > 850:
                    table_md = table_md[:850] + "\n... [table truncated for token budget]"
                lines = [f"[Source {source_idx} | Table {rec_id} | Page: {page}]"]
                if caption:
                    lines.append(f"Caption: {caption}")
                if table_md:
                    lines.append(f"Table Markdown Grid:\n{table_md}")
                block_content = "\n".join(lines)
                user_blocks.append(TextBlock(text=block_content))
                # Tables with text markdown do not need redundant heavy image tokens
                if resolved_img and not table_md and attached_images < max_images:
                    user_blocks.append(ImageBlock(path=resolved_img))
                    attached_images += 1
            else:
                lines = [f"[Source {source_idx} | Figure {rec_id} | Page: {page}]"]
                parent_fig = item.get("parent_figure")
                panel = item.get("panel")
                panel_cap = item.get("panel_caption")
                if parent_fig:
                    lines.append(f"Parent Figure: {parent_fig}")
                if panel:
                    lines.append(f"Panel: {panel}")
                if panel_cap:
                    lines.append(f"Panel Caption: {panel_cap}")
                if caption:
                    lines.append(f"Caption: {caption}")
                block_content = "\n".join(lines)
                user_blocks.append(TextBlock(text=block_content))
                if resolved_img and attached_images < max_images:
                    user_blocks.append(ImageBlock(path=resolved_img))
                    attached_images += 1
            source_idx += 1

    # 3. Question Prompt
    user_blocks.append(
        TextBlock(text=f"Question: {question}\n\nPlease provide a clear, detailed, grounded explanation citing relevant sources.")
    )

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
    ]

    # Prepend up to last 4 conversation messages (2 dialogue turns) as short-term memory context
    if chat_history:
        for msg in chat_history[-4:]:
            role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
            content = str(msg.get("content", "")).strip()
            if content:
                messages.append(ChatMessage(role=role, content=content))

    messages.append(ChatMessage(role=MessageRole.USER, blocks=user_blocks))

    # -------------------------------------------------------------------------
    # Terminal Logging: Complete VLM Payload Inspection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(">>> [VLM INPUT DISPATCH] Payload sent to Vision LLM (qwen/qwen3.8-27b):")
    print(f"  * Question: '{_safe_str(question, 120)}'")

    print(f"  * Retrieved Text Chunks ({len(selected_text_results)} attached):")
    for idx, c in enumerate(selected_text_results, 1):
        cid = c.get("chunk_id", "Unknown")
        pages = c.get("page_numbers") or ["?"]
        raw_t = c.get("text") or c.get("contextualized_text") or ""
        print(f"    [{idx}] Chunk {cid} (Page {pages}): \"{_safe_str(raw_t, 85)}\"")

    v_count = len(fig_tab_results or [])
    print(f"  * Visual Assets ({v_count} candidates evaluated, max {max_images} attached as PNG ImageBlock):")
    v_img_count = 0
    for idx, v in enumerate(fig_tab_results or [], 1):
        rtype = v.get("record_type", "visual")
        rid = v.get("record_id", v.get("figure_id", v.get("table_id", "Unknown")))
        page = v.get("page", "?")
        cap = _safe_str(v.get("caption") or v.get("panel_caption") or "No caption", 75)
        raw_img = v.get("image_path", "")
        resolved = _resolve_image_path(raw_img, rtype)
        is_attached = resolved and v_img_count < max_images
        if is_attached:
            v_img_count += 1
            v_status = f"ATTACHED AS ImageBlock ({resolved.name})"
        elif resolved:
            v_status = "SKIPPED IMAGE (Limit reached; caption/table kept in text)"
        else:
            v_status = "TEXT/TABLE-ONLY (No cropped image path)"

        parent_fig = v.get("parent_figure")
        panel = v.get("panel")
        panel_cap = v.get("panel_caption")
        meta_sub = []
        if parent_fig:
            meta_sub.append(f"Parent: {parent_fig}")
        if panel:
            meta_sub.append(f"Panel: {panel}")
        meta_str = f" [{', '.join(meta_sub)}]" if meta_sub else ""

        print(f"    [{idx}] {rtype.upper()} {rid} (Page {page}){meta_str}: \"{cap}\"")
        if panel_cap and panel_cap != cap:
            print(f"        |-- Panel Caption: \"{_safe_str(panel_cap, 70)}\"")
        if v.get("table_markdown"):
            md_prev = _safe_str(v["table_markdown"], 60)
            print(f"        |-- Table Grid Markdown: \"{md_prev}\" ({len(v['table_markdown'])} chars)")
        print(f"        |-- Vision Status: [{v_status}]")

    if chat_history:
        print(f"  * Conversational Memory ({len(chat_history[-4:])} recent message(s) included):")
        for h in chat_history[-4:]:
            role = h.get("role", "unknown")
            snippet = _safe_str(h.get("content", ""), 65)
            print(f"    [{role.upper()}]: {snippet}")
    else:
        print("  * Conversational Memory: None (first turn)")

    print(f"  * System Prompt: Citation grounding [Source N, Page P] enforced.")
    print("=" * 75 + "\n")

    try:
        response = llm_vision.chat(messages).message.content.strip()
    except Exception as exc:
        err_str = str(exc)
        if "413" in err_str or "rate_limit_exceeded" in err_str or "tokens" in err_str.lower():
            print(f"\n[Self-Healing] Groq rate-limit/token ceiling encountered ({err_str[:70]}). Retrying with lean text payload...")
            # Strip heavy ImageBlocks and retry with pure text/table markdown
            lean_blocks = [b for b in user_blocks if not isinstance(b, ImageBlock)]
            lean_messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, blocks=lean_blocks),
            ]
            try:
                response = llm_vision.chat(lean_messages).message.content.strip()
                print("[Self-Healing] Recovery successful: generated grounded answer without visual tokens.")
            except Exception as retry_exc:
                response = f"An error occurred while generating the answer: {retry_exc}"
        else:
            response = f"An error occurred while generating the answer: {exc}"

    # If response indicates unanswerable topic, enforce clean single-sentence response and purge sources
    if _is_unanswerable_response(response):
        response = "I don't have information about that in the document."
        sources = []
    else:
        sources = state.get("sources", [])

    new_turn = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response, "sources": sources},
    ]

    return {
        "answer": response,
        "sources": sources,
        "chat_history": new_turn,
    }
