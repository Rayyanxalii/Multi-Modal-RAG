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
from services.llm_service import get_llm, get_llm_vision


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
    """Parse PDF layout hierarchy & extract text chunks (PyMuPDF Lite by default, Docling only if USE_DOCLING=1)."""
    from rag.fast_pdf_parser import is_low_memory_env, parse_pdf_fast
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)

    use_docling = os.getenv("USE_DOCLING", "").strip().lower() in ("1", "true", "yes")
    if not use_docling or is_low_memory_env():
        print(f"[Ingestion] Parsing '{pdf_path.name}' with high-speed PyMuPDF Lite engine (<35 MB RAM)...")
        parse_pdf_fast(pdf_path, doc_paths=doc_paths)
        return {}

    print(f"[Ingestion] Parsing document layout with Docling: {pdf_path.name}")
    try:
        from rag.docling_parser import parse_pdf_to_chunks
        parse_pdf_to_chunks(pdf_path, doc_paths=doc_paths)
    except (MemoryError, Exception) as exc:
        print(f"[Ingestion Warning] Docling unavailable or OOM ({exc}). Falling back safely to PyMuPDF Lite parser...")
        parse_pdf_fast(pdf_path, doc_paths=doc_paths)
    return {}


def extract_and_crop_node(state: RAGState) -> dict[str, Any]:
    """Extract figure & table bounding boxes and crop visual element PNGs."""
    from rag.fast_pdf_parser import is_fast_parsed_doc
    from rag.visual_retrieval.preprocess.figure_cropper import crop_figures
    from rag.visual_retrieval.preprocess.table_cropper import crop_tables

    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    print(f"[Ingestion] Extracting & cropping visual assets for: {doc_name}")
    if not is_fast_parsed_doc(doc_paths["document_json_path"]):
        from rag.visual_retrieval.preprocess.figure_extractor import extract_figures, extract_tables
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

    if doc_paths["figures_raw_jsonl"].exists():
        crop_figures(
            pdf_path=pdf_path,
            figures_jsonl=doc_paths["figures_raw_jsonl"],
            output_dir=doc_paths["figures_dir"],
            rendered_pages_dir=doc_paths["rendered_pages_dir"],
        )
    if doc_paths["tables_jsonl"].exists():
        crop_tables(
            pdf_path=pdf_path,
            tables_jsonl=doc_paths["tables_jsonl"],
            output_dir=doc_paths["tables_dir"],
            rendered_pages_dir=doc_paths["rendered_pages_dir"],
        )
    return {}


def check_qdrant_doc_exists(doc_name: str) -> int:
    """Check how many points for doc_name already exist in Qdrant omni_documents collection."""
    try:
        from qdrant_client import models
        from rag.text_retrieval.index_chunks import get_client
        client = get_client()
        if not client.collection_exists(TEXT_COLLECTION_NAME):
            return 0
        return client.count(
            collection_name=TEXT_COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=str(doc_name).strip().lower()),
                    )
                ]
            ),
            exact=True,
        ).count
    except Exception:
        return 0


def index_vectors_node(state: RAGState) -> dict[str, Any]:
    """Index text chunks into Qdrant using Google Gemini embeddings (skipping if already indexed)."""
    from rag.text_retrieval.index_chunks import index_text_chunks
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    existing_count = check_qdrant_doc_exists(doc_name)
    if existing_count > 0:
        print(f"[Ingestion] Qdrant vectors for '{doc_name}' already exist ({existing_count} chunks). Reusing index.")
        return {"qdrant_existed": True, "qdrant_points": existing_count}

    print(f"[Ingestion] Indexing text chunks into Qdrant: {TEXT_COLLECTION_NAME} (doc_id={doc_name})")
    try:
        index_text_chunks(
            chunks_file_path=doc_paths["chunks_path"],
            collection_name=TEXT_COLLECTION_NAME,
            doc_id=doc_name,
        )
    except Exception as exc:
        print(f"[Warning] Dense vector indexing skipped ({exc}). Document remains fully searchable via Okapi BM25 and ColPali visual embeddings.")
    return {"qdrant_existed": False, "qdrant_points": 0}


def index_colpali_node(state: RAGState) -> dict[str, Any]:
    """Send rendered page images to Hugging Face ZeroGPU Space (reusing if already indexed)."""
    from rag.visual_retrieval.colpali_client import get_colpali_cloud_client
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    page_images = sorted(list(doc_paths["rendered_pages_dir"].glob("*.png")))
    colpali_existed = False
    try:
        client = get_colpali_cloud_client()
        if client.document_exists(doc_name):
            print(f"[Ingestion] Visual pages for '{doc_name}' already exist on ColPali Space — reusing existing visual index.")
            colpali_existed = True
        elif page_images:
            print(f"[Ingestion] Registering {len(page_images)} pages with ColPali Cloud Space for '{doc_name}'...")
            res = client.index_document(doc_id=doc_name, image_paths=page_images)
            colpali_existed = bool(
                isinstance(res, dict) and (res.get("already_exists") or res.get("status") == "already_exists")
            )
    except Exception as exc:
        print(f"[Warning] ColPali Cloud indexing encountered error: {exc}")

    return {"is_ingested": True, "colpali_existed": colpali_existed}


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




def is_overview_query(query: str) -> bool:
    """Detect whether user query asks for a high-level summary or overview of the entire document."""
    q = query.lower().strip()
    patterns = [
        r"\bwhat is (?:this|the) (?:document|paper|report|pdf|file)\b",
        r"\bsummarize (?:this|the)?\s*(?:document|paper|report|pdf|file|core topic)?\b",
        r"\b(?:overview|summary) of (?:this|the)?\s*(?:document|paper|report|pdf)?\b",
        r"\bwhat does (?:this|the)?\s*(?:document|paper|report|pdf)?\s*(?:cover|describe|talk about|discuss)\b",
        r"\btell me about (?:this|the)?\s*(?:document|paper|report|pdf)\b",
        r"\bcore topic\b",
    ]
    return any(re.search(pat, q) for pat in patterns)


def contextualize_query(
    question: str,
    history: list[dict[str, Any]] | None,
    api_key: str | None = None,
) -> str:
    """Reformulate a follow-up query using conversation history into a standalone search query.

    Bypasses LLM rewriting if the query is already self-contained or contains no coreference pronouns.
    """
    if not history:
        return question

    clean_q = question.strip()

    # Rule 1: Check if the question contains anaphoric pronouns or relative references
    coref_pattern = r"\b(it|its|they|them|their|theirs|this|that|these|those|the former|the latter|above|previous|earlier|same|another|also|too|again)\b"
    elliptical_pattern = r"^(what about|how about|why\b|and for|explain more)\b"
    has_coref = bool(re.search(coref_pattern, clean_q, re.IGNORECASE) or re.search(elliptical_pattern, clean_q, re.IGNORECASE))

    # If the question does not contain coreference markers, it is self-contained. Keep it verbatim!
    if not has_coref:
        return clean_q

    # Rule 2: Filter out uninformative or error responses from history
    meaningful_turns = []
    for turn in history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        content = str(turn.get("content", "")).strip()
        # Skip unanswerable boilerplate or error responses
        if "I don't have information about that in the document" in content:
            continue
        if len(content) > 300:
            content = content[:300] + "..."
        if content:
            meaningful_turns.append(f"{role}: {content}")

    if not meaningful_turns:
        return clean_q

    history_str = "\n".join(meaningful_turns)

    prompt = f"""You are an expert conversational assistant. The user asked a follow-up question that contains pronouns or references to earlier messages.
Rephrase ONLY the ambiguous pronouns or references in the follow-up question into specific entities mentioned in the conversation history.

STRICT INSTRUCTIONS:
1. DO NOT invent entities, benchmark datasets, model architectures, or topics (such as ResNet, ImageNet, COCO) not explicitly present in the conversation history.
2. If the question is already clear, return it unchanged.
3. Output ONLY the standalone question text without quotes or explanations.

Conversation History:
{history_str}

Follow-up Question: {clean_q}

Standalone Question:"""

    try:
        reformulated = get_llm(api_key).complete(prompt).text.strip()
        reformulated = reformulated.strip('"').strip("'").strip()
        # Prevent repeating old questions if the user asked something new
        prev_user_questions = [
            str(t.get("content", "")).strip().lower()
            for t in history
            if t.get("role") == "user"
        ]
        if reformulated.lower() in prev_user_questions and clean_q.lower() not in prev_user_questions:
            return clean_q
        if reformulated and len(reformulated) >= 3:
            return reformulated
    except Exception as exc:
        print(f"[Warning] Contextual query reformulation fallback: {exc}")
    return clean_q


def route_query_node(state: RAGState) -> dict[str, Any]:
    """Classify user query intent into 'text', 'visual', or 'both'.

    If conversation history is present and query contains coreferences, resolves references using short-term memory.
    """
    original_q = state.get("original_question") or state["question"].strip()
    retry_count = state.get("retry_count", 0)
    history = state.get("chat_history", [])
    api_key = state.get("groq_api_key")

    # If this is a retry within the CRAG loop, do NOT re-contextualize the original question!
    if retry_count > 0:
        active_question = state["question"].strip()
    else:
        active_question = contextualize_query(original_q, history, api_key=api_key)

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
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_id = (state.get("doc_name") or doc_paths.get("doc_name") or pdf_path.stem).strip().lower()

    try:
        results = search_chunks(query, top_k=5, doc_id=doc_id)
        # Defense-in-depth: Discard any point that does not belong to this document
        filtered = [
            r for r in results
            if not r.get("document_id") or str(r.get("document_id")).strip().lower() == doc_id
        ]
        print(f"  [Dense Retriever] Found {len(filtered)} isolated candidate chunks for '{doc_id}' in Qdrant.")
    except Exception as exc:
        print(f"[Warning] Dense retriever error: {exc}")
        filtered = []
    return {"dense_matches": filtered}


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
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    active_doc = (state.get("doc_name") or doc_paths.get("doc_name") or pdf_path.stem).strip().lower()

    dense_matches = state.get("dense_matches") or []
    bm25_matches = state.get("bm25_matches") or []

    # Strict isolation: ensure no cross-document dense matches leak through
    if active_doc:
        dense_matches = [
            m for m in dense_matches
            if not m.get("document_id") or str(m.get("document_id")).strip().lower() == active_doc
        ]

    if not dense_matches and not bm25_matches:
        fused = []
    else:
        try:
            fused = reciprocal_rank_fusion(dense_matches, bm25_matches, top_k=5)
            print(f"  [Text Fusion] Reciprocal Rank Fusion fused into top {len(fused)} text chunks.")
        except Exception as exc:
            print(f"[Warning] Text fusion error: {exc}")
            fused = (dense_matches or bm25_matches or [])[:5]

    # If the user is asking for a document overview or summary, inject introductory chunks
    q_to_check = state.get("original_question") or state.get("question", "")
    if is_overview_query(q_to_check):
        pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
        doc_paths = get_document_paths(pdf_path)
        chunks_file = doc_paths.get("chunks_path")
        if chunks_file and chunks_file.exists():
            try:
                intro_chunks = []
                with chunks_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            intro_chunks.append(json.loads(line))
                        if len(intro_chunks) >= 3:
                            break
                # Prepend intro chunks if not already in fused
                existing_ids = {c.get("chunk_id") for c in fused}
                prepends = [c for c in intro_chunks if c.get("chunk_id") not in existing_ids]
                if prepends:
                    print(f"  [Text Fusion] Injected {len(prepends)} introductory document chunk(s) for overview query.")
                    fused = (prepends + fused)[:5]
            except Exception as e_intro:
                print(f"[Warning] Failed injecting intro chunks: {e_intro}")

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
    """Map top ColPali pages and targeted table/figure requests to cropped figures and structured Markdown tables."""
    page_results = state.get("colpali_top_pages", [])
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)

    query = state.get("original_question") or state.get("question", "")
    target_table_nums = [int(m) for m in re.findall(r"\btable\s*(\d+)\b", query, re.IGNORECASE)]
    target_fig_nums = [int(m) for m in re.findall(r"\b(?:fig|figure)\.?\s*(\d+)\b", query, re.IGNORECASE)]

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
    seen_ids: set[str] = set()

    # Priority 1: Explicit Table targeting (e.g. "what is table 1 shows?")
    if target_table_nums:
        for tab in tables:
            t_id = tab.get("table_id", "")
            cap = str(tab.get("caption") or "").lower()
            is_target = any(
                t_id.endswith(f"_{n:03d}") or t_id.endswith(f"_{n}") or f"table {n}" in cap or f"table_{n}" in t_id
                for n in target_table_nums
            )
            if is_target and t_id not in seen_ids:
                p = tab.get("page_number")
                if p is None:
                    pages = tab.get("page_numbers") or []
                    p = pages[0] if pages else 1
                matched_assets.append({
                    "record_type": "table",
                    "record_id": t_id,
                    "table_id": t_id,
                    "page": p,
                    "colpali_rank": 0,
                    "colpali_score": 99.0,
                    "caption": tab.get("caption") or "",
                    "table_markdown": tab.get("table_markdown") or tab.get("text") or "",
                    "image_path": tab.get("image_path", ""),
                    "item_type": tab.get("item_type", "TableItem"),
                })
                seen_ids.add(t_id)

    # Priority 2: Explicit Figure targeting (e.g. "what is the figure 5 about?")
    if target_fig_nums:
        for fig in figures:
            f_id = fig.get("figure_id", "")
            cap = str(fig.get("caption") or "").lower()
            parent = str(fig.get("parent_figure") or "").lower()
            is_target = any(
                f_id.endswith(f"_{n:03d}") or f_id.endswith(f"_{n}") or f"figure {n}" in cap or f"fig {n}" in cap or f"figure {n}" in parent or f"fig_{n}" in f_id
                for n in target_fig_nums
            )
            if is_target and f_id not in seen_ids:
                p = fig.get("page_number")
                if p is None:
                    pages = fig.get("page_numbers") or []
                    p = pages[0] if pages else 1
                matched_assets.append({
                    "record_type": "figure",
                    "record_id": f_id,
                    "figure_id": f_id,
                    "parent_figure": fig.get("parent_figure") or "",
                    "panel": fig.get("panel") or "",
                    "panel_caption": fig.get("panel_caption") or "",
                    "shared_caption": fig.get("shared_caption") or "",
                    "page": p,
                    "colpali_rank": 0,
                    "colpali_score": 99.0,
                    "caption": fig.get("caption") or "",
                    "image_path": fig.get("image_path", ""),
                    "item_type": fig.get("item_type", "PictureItem"),
                })
                seen_ids.add(f_id)

    # Priority 3: Match assets from top ColPali late-interaction pages
    for fig in figures:
        f_id = fig.get("figure_id", "unknown_fig")
        if f_id in seen_ids:
            continue
        p = fig.get("page_number")
        if p is None:
            pages = fig.get("page_numbers") or []
            p = pages[0] if pages else None

        if p in page_rankings:
            p_info = page_rankings[p]
            matched_assets.append({
                "record_type": "figure",
                "record_id": f_id,
                "figure_id": f_id,
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
            seen_ids.add(f_id)

    for tab in tables:
        t_id = tab.get("table_id", "unknown_tab")
        if t_id in seen_ids:
            continue
        p = tab.get("page_number")
        if p is None:
            pages = tab.get("page_numbers") or []
            p = pages[0] if pages else None

        if p in page_rankings:
            p_info = page_rankings[p]
            matched_assets.append({
                "record_type": "table",
                "record_id": t_id,
                "table_id": t_id,
                "page": p,
                "colpali_rank": p_info.get("rank", 99),
                "colpali_score": p_info.get("score", 0.0),
                "caption": tab.get("caption") or "",
                "table_markdown": tab.get("table_markdown") or tab.get("text") or "",
                "image_path": tab.get("image_path", ""),
                "item_type": tab.get("item_type", "TableItem"),
            })
            seen_ids.add(t_id)

    # Priority 4: Overview queries fallback
    if is_overview_query(query) and len(matched_assets) < 3:
        for tab in tables:
            t_id = tab.get("table_id", "")
            p = tab.get("page_number") or (tab.get("page_numbers") or [99])[0]
            if p in (1, 2) and t_id not in seen_ids:
                matched_assets.append({
                    "record_type": "table",
                    "record_id": t_id,
                    "table_id": t_id,
                    "page": p,
                    "colpali_rank": 5,
                    "colpali_score": 10.0,
                    "caption": tab.get("caption") or "",
                    "table_markdown": tab.get("table_markdown") or tab.get("text") or "",
                    "image_path": tab.get("image_path", ""),
                    "item_type": tab.get("item_type", "TableItem"),
                })
                seen_ids.add(t_id)

    matched_assets.sort(
        key=lambda item: (item.get("colpali_rank", 999), -float(item.get("colpali_score", 0.0)))
    )
    print(f"  [Visual Hydration] Hydrated {len(matched_assets)} figures/tables from top ColPali pages and targeted queries.")

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
            "text": text,
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
        response = get_llm(state.get("groq_api_key")).complete(grading_prompt).text.strip().upper()
        is_relevant = "YES" in response
    except Exception as exc:
        print(f"[Warning] LLM grading failed ({exc}). Defaulting to relevant.")
        is_relevant = True

    return {"is_relevant": is_relevant}




def rewrite_query_node(state: RAGState) -> dict[str, Any]:
    """Corrective RAG (CRAG): Classical Query Expansion with domain terms and technical synonyms."""
    original_query = state.get("original_question") or state["question"]
    current_retry = state.get("retry_count", 0)
    pdf_path = Path(state.get("pdf_path") or DEFAULT_PDF_PATH)
    doc_paths = get_document_paths(pdf_path)
    doc_name = doc_paths["doc_name"]

    # Rule 1: Overview questions -> Expand with executive summary, abstract, and core topic domain terms
    if is_overview_query(original_query):
        expanded = f"{original_query} executive summary introduction abstract core overview background findings {doc_name}"
        print(f"\n[CRAG Query Expansion] Overview query expanded: '{original_query}' -> '{expanded}'")
        return {
            "question": expanded,
            "rewritten_query": expanded,
            "retry_count": current_retry + 1,
        }

    # Rule 2: Specific Table questions -> Expand with data, rows, columns, and grid terminology
    table_m = re.search(r"\btable\s*(\d+)\b", original_query, re.IGNORECASE)
    if table_m:
        table_num = table_m.group(1)
        expanded = f"{original_query} Table {table_num} data grid rows columns values results breakdown {doc_name}"
        print(f"\n[CRAG Query Expansion] Table query expanded: '{original_query}' -> '{expanded}'")
        return {
            "question": expanded,
            "rewritten_query": expanded,
            "retry_count": current_retry + 1,
        }

    # Rule 3: Specific Figure questions -> Expand with visual, chart, diagram, and plot terminology
    fig_m = re.search(r"\b(?:fig|figure)\.?\s*(\d+)\b", original_query, re.IGNORECASE)
    if fig_m:
        fig_num = fig_m.group(1)
        expanded = f"{original_query} Figure {fig_num} visual chart diagram plot image illustration graph structure {doc_name}"
        print(f"\n[CRAG Query Expansion] Figure query expanded: '{original_query}' -> '{expanded}'")
        return {
            "question": expanded,
            "rewritten_query": expanded,
            "retry_count": current_retry + 1,
        }

    # Rule 4: General domain query expansion with LLM (completely document-agnostic)
    expansion_prompt = f"""You are an expert search specialist and information retrieval engineer.
The initial search for the query below in the document '{doc_name}' returned insufficient passages.
Perform CLASSICAL PRODUCTION QUERY EXPANSION: expand the query by appending domain-specific technical terms, authoritative synonyms, and related vocabulary likely found in '{doc_name}' without altering the user's intent.

Original Query: {original_query}
Document: {doc_name}

STRICT REQUIREMENTS:
1. Retain the user's original query intact at the beginning as the semantic anchor.
2. Append 4 to 8 highly relevant domain-specific technical terms, author terminology, or official domain synonyms that document authors use for this subject in '{doc_name}'.
3. Do NOT invent external benchmark datasets or machine learning architectures (such as ResNet, ImageNet, COCO) unless specifically requested by the user.
4. Output ONLY the expanded query string combining the original query with the appended domain terms.

Expanded Query:"""

    try:
        expanded = get_llm(state.get("groq_api_key")).complete(expansion_prompt).text.strip().strip('"').strip("'")
        # Ensure original query is retained at the start
        if original_query.lower() not in expanded.lower():
            expanded = f"{original_query} {expanded}"
        print(f"\n[CRAG Query Expansion] Query expanded: '{original_query}' -> '{expanded}'")
    except Exception as exc:
        print(f"[Warning] Query expansion failed ({exc}). Retaining original query.")
        expanded = original_query

    return {
        "question": expanded,
        "rewritten_query": expanded,
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


def filter_relevant_sources(response: str, sources: list[dict[str, Any]], question: str = "") -> list[dict[str, Any]]:
    """Filter candidate retrieval sources so only genuinely relevant text and visual assets are presented to the user.

    Eliminates redundant figures/tables by:
    1. Retaining visual assets specifically targeted by the user query (e.g. Table 1, Figure 5).
    2. Retaining visual assets explicitly cited in the answer (e.g. [Source N], [Page P]).
    3. Retaining visual assets explicitly mentioned by name or ID in the answer prose.
    4. Capping any visual deck to at most 2 relevant items.
    5. Omitting visual assets completely if the answer is purely textual or unanswerable.
    """
    if not sources:
        return []

    if _is_unanswerable_response(response):
        return []

    text_sources = [s for s in sources if s.get("type") == "text"]
    visual_sources = [s for s in sources if s.get("type") in ("figure", "table", "visual")]

    # 1. Parse cited source indices from LLM response (e.g. [Source 1, Page 2], [Source 3])
    cited_source_indices = set()
    for m in re.finditer(r"\[Source\s*(\d+)", response, re.IGNORECASE):
        try:
            cited_source_indices.add(int(m.group(1)))
        except ValueError:
            pass

    # 2. Check if the question explicitly requested specific figures or tables
    target_table_nums = [int(m) for m in re.findall(r"\btable\s*(\d+)\b", question, re.IGNORECASE)]
    target_fig_nums = [int(m) for m in re.findall(r"\b(?:fig|figure)\.?\s*(\d+)\b", question, re.IGNORECASE)]

    relevant_visuals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Priority A: Targeted visual assets explicitly requested in the query
    if target_table_nums or target_fig_nums:
        for v in visual_sources:
            v_id = str(v.get("id", "")).lower()
            v_cap = str(v.get("caption", "")).lower()
            v_type = (v.get("type") or "").lower()

            is_match = False
            if v_type == "table" and target_table_nums:
                is_match = any(
                    v_id.endswith(f"_{n:03d}") or v_id.endswith(f"_{n}") or f"table_{n}" in v_id or f"table {n}" in v_cap
                    for n in target_table_nums
                )
            elif v_type == "figure" and target_fig_nums:
                is_match = any(
                    v_id.endswith(f"_{n:03d}") or v_id.endswith(f"_{n}") or f"fig_{n}" in v_id or f"figure {n}" in v_cap or f"fig {n}" in v_cap
                    for n in target_fig_nums
                )

            if is_match and v.get("id") not in seen_ids:
                relevant_visuals.append(v)
                seen_ids.add(v.get("id"))

    # Priority B: Visual assets that are explicitly cited by [Source N] in the answer
    for v in visual_sources:
        if v.get("source") in cited_source_indices and v.get("id") not in seen_ids:
            relevant_visuals.append(v)
            seen_ids.add(v.get("id"))

    # Priority C: Visual assets mentioned by name/number in the answer prose (e.g. "Figure 1", "Table 2")
    resp_lower = response.lower()
    for v in visual_sources:
        v_id = str(v.get("id", "")).lower()
        v_type = (v.get("type") or "").lower()
        id_num_match = re.search(r"_(\d+)$", v_id)
        if id_num_match:
            num = int(id_num_match.group(1))
            name_pattern = rf"\b{v_type}\s*{num}\b"
            if re.search(name_pattern, resp_lower) and v.get("id") not in seen_ids:
                relevant_visuals.append(v)
                seen_ids.add(v.get("id"))

    # Priority D: If the query is an explicit visual query, and nothing was matched yet,
    # include AT MOST the single highest-scoring visual asset
    q_lower = question.lower()
    has_visual_keyword = any(k in q_lower for k in ["figure", "fig", "table", "diagram", "chart", "plot", "image", "visual", "architecture", "flowchart"])
    if not relevant_visuals and visual_sources and has_visual_keyword:
        relevant_visuals.append(visual_sources[0])
        seen_ids.add(visual_sources[0].get("id"))

    # Priority E: If the user is asking a broad overview/summary without requesting figures/tables,
    # suppress visual mounts so the response remains a clean, focused summary.
    if is_overview_query(question) and not has_visual_keyword and not (target_table_nums or target_fig_nums):
        relevant_visuals = []

    # Strictly cap relevant visual sources to at most 2 to avoid visual clutter
    relevant_visuals = relevant_visuals[:2]

    # Filter text sources: keep cited text sources, or top 3 if citations are unindexed
    relevant_text = [t for t in text_sources if t.get("source") in cited_source_indices]
    if not relevant_text:
        relevant_text = text_sources[:3]

    combined = relevant_text + relevant_visuals
    print(f"  [Sources Filter] Filtered {len(sources)} candidate sources -> {len(combined)} presented ({len(relevant_visuals)} visual).")
    return combined


def build_vlm_messages(state: RAGState) -> tuple[list[ChatMessage], list[Any], str, str, list[dict[str, Any]]]:
    """Prepare multimodal ChatMessages, user blocks, system prompt, dialogue question, and text results."""
    question = state.get("original_question") or state["question"]
    text_results = state.get("text_results", [])
    fig_tab_results = state.get("figure_table_results", [])
    chat_history = state.get("chat_history") or []
    user_dialogue_question = state.get("original_question") or question

    if not text_results and not fig_tab_results:
        return [], [], "", user_dialogue_question, []

    system_prompt = (
        "You are an expert scientific, financial, and technical AI assistant answering questions based on the provided document context.\n"
        "Guidelines:\n"
        "1. Style & Tone: Write in clear, authoritative, fluent prose with clean markdown formatting. Use **bold** for key terms, metrics, and concepts. When presenting lists, ensure each item starts on a new line. Directly address the user's question using facts from the document.\n"
        "2. Grounding & Synthesis: Answer using facts, data, findings, and numbers present in the retrieved text chunks, tables, and figures. Synthesize explanations accurately without speculating or inventing unmentioned facts.\n"
        "3. Document Summaries & Overviews: When asked to summarize the whole document, provide an overview, or explain what the document is about:\n"
        "   - Synthesize a comprehensive, well-structured executive summary based on the provided document excerpts (such as title, abstract, introduction, key findings, and results).\n"
        "   - Organize your summary with clear markdown sections or bullet points (e.g. **Core Objective / Subject**, **Key Highlights & Methodology**, **Major Results / Financial Highlights**, and **Conclusion**).\n"
        "   - Keep your summary focused and concise (around 250-350 words) to present the key findings clearly within output token budgets.\n"
        "   - NEVER refuse or state you lack information when asked for a summary or overview of the document; utilize the available document context to represent the document's central theme and main contributions.\n"
        "4. Mathematical Equations: When expressing formulas, equations, or variables, ALWAYS use standard LaTeX syntax:\n"
        "   - Use display math blocks '$$ ... $$' on their own separate lines for primary equations (e.g. $$ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V $$).\n"
        "   - Use inline math '$...$' for variables, dimensions, and short expressions (e.g. '$d_k$', '$d_v$', '$Q$', '$K$', '$V$', '$\\sqrt{d_k}$', '$O(n^2 \\cdot d)$'). Do NOT wrap financial currency figures, prices, or dollar amounts in LaTeX syntax (write '$1,773 million' or '$50' directly as standard text).\n"
        "5. Citations: Ground your statements with concise source citations [Source N, Page P] placed at the end of sentences where facts are introduced. Do not interrupt formulas or insert citations between words.\n"
        "6. Visual References: When referencing figures or tables, explicitly mention them (e.g. 'as illustrated in Figure 2 [Source N, Page P]' or 'detailed in Table 1 [Source N, Page P]').\n"
        "7. Unanswerable Questions: ONLY if the user asks about an external, unrelated topic or specific fact that has genuinely zero basis or mention in the document (e.g. an unrelated company, external person, or topic completely outside the document), respond ONLY with: 'I don't have information about that in the document.'\n"
        "   - Do NOT trigger this for document summaries, overviews, or high-level explanations of the document itself.\n"
        "   - When triggered, stop immediately without extra commentary or explanation."
    )

    user_blocks: list[Any] = []
    source_idx = 1
    max_images = 1
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
                # Truncate giant table markdown to 850 chars to prevent rate limit spikes
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
    q_to_eval = state.get("original_question") or question
    if is_overview_query(q_to_eval):
        prompt_instruction = (
            f"Question: {question}\n\n"
            "Please synthesize a clear, concise (around 250-350 words), and well-structured executive summary of the document based on the provided context, citing relevant sources."
        )
    else:
        prompt_instruction = (
            f"Question: {question}\n\n"
            "Please provide a clear, detailed, grounded explanation citing relevant sources."
        )
    user_blocks.append(TextBlock(text=prompt_instruction))

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

    # Terminal Logging: Complete VLM Payload Inspection
    print("\n" + "=" * 75)
    print(">>> [VLM INPUT DISPATCH] Payload sent to Vision LLM (qwen/qwen3.8-27b):")
    print(f"  * Question: '{_safe_str(question, 120)}'")
    print(f"  * Retrieved Text Chunks ({len(selected_text_results)} attached)")
    print(f"  * Attached Visual Images: {attached_images}")
    print("=" * 75 + "\n")

    return messages, user_blocks, system_prompt, user_dialogue_question, selected_text_results


def generate_answer_node(state: RAGState) -> dict[str, Any]:
    """Synthesize a grounded answer using the multimodal vision LLM (llm_vision) passing text, figures, tables, and conversation history."""
    messages, user_blocks, system_prompt, user_dialogue_question, selected_text_results = build_vlm_messages(state)

    if not messages:
        ans = "I don't have information about that in the document."
        return {
            "answer": ans,
            "chat_history": [
                {"role": "user", "content": user_dialogue_question},
                {"role": "assistant", "content": ans, "sources": []},
            ],
        }

    active_vision = get_llm_vision(state.get("groq_api_key"))
    try:
        response = active_vision.chat(messages).message.content.strip()
    except Exception as exc:
        err_str = str(exc)
        if "413" in err_str or "rate_limit_exceeded" in err_str or "tokens" in err_str.lower():
            print(f"\n[Self-Healing] Groq rate-limit/token ceiling encountered ({err_str[:70]}). Retrying with lean text payload...")
            lean_blocks = [b for b in user_blocks if not isinstance(b, ImageBlock)]
            lean_messages = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, blocks=lean_blocks),
            ]
            try:
                response = active_vision.chat(lean_messages).message.content.strip()
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
        raw_sources = state.get("sources", [])
        sources = filter_relevant_sources(response, raw_sources, question=user_dialogue_question)

    new_turn = [
        {"role": "user", "content": user_dialogue_question},
        {"role": "assistant", "content": response, "sources": sources},
    ]

    return {
        "answer": response,
        "sources": sources,
        "chat_history": new_turn,
    }
