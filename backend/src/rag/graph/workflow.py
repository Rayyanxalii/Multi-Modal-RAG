"""LangGraph workflow definition for the Multimodal Agentic RAG system.

Coordinates document ingestion checking, intent routing, parallel dense/BM25/ColPali
retrieval, RRF text fusion, visual asset hydration, Self-RAG relevance grading,
query reformulation, and grounded answer synthesis.
"""

from pathlib import Path
from typing import Any, Literal
from langgraph.graph import StateGraph, START, END

from rag.config import DEFAULT_PDF_PATH
from rag.graph.state import RAGState
from rag.graph.nodes import (
    check_doc_status_node,
    render_pages_node,
    parse_document_node,
    extract_and_crop_node,
    index_vectors_node,
    index_colpali_node,
    route_query_node,
    dense_retriever_node,
    bm25_retriever_node,
    text_fusion_node,
    colpali_retriever_node,
    visual_hydration_node,
    format_context_node,
    grade_context_node,
    rewrite_query_node,
    generate_answer_node,
)


def decide_ingestion(state: RAGState) -> Literal["render_pages", "route_query"]:
    """Conditional edge: check if document is already ingested or needs pipeline execution."""
    if not state.get("is_ingested", False):
        return "render_pages"
    return "route_query"


def branch_retrieval(state: RAGState) -> list[str]:
    """Conditional edge: route to Dense, BM25, and/or ColPali based on classified intent."""
    route = (state.get("route") or "both").lower()
    if route == "dense":
        return ["dense_retriever"]
    if route == "bm25":
        return ["bm25_retriever"]
    if route == "text":
        return ["dense_retriever", "bm25_retriever"]
    if route == "visual":
        return ["colpali_retriever"]
    return ["dense_retriever", "bm25_retriever", "colpali_retriever"]


def decide_after_grading(state: RAGState) -> Literal["generate_answer", "rewrite_query"]:
    """Conditional edge: decide whether to generate the answer or reformulate the query."""
    is_relevant = state.get("is_relevant", True)
    retry_count = state.get("retry_count", 0)

    # If relevant OR maximum retries reached
    if is_relevant or retry_count >= 1:
        return "generate_answer"

    # CRAG corrective loop: reformulate query and retry retrieval
    return "rewrite_query"


def build_rag_graph() -> StateGraph:
    """Build and compile the unified Multimodal Agentic RAG StateGraph."""
    workflow = StateGraph(RAGState)

    # 1. Document Ingestion Nodes
    workflow.add_node("check_doc_status", check_doc_status_node)
    workflow.add_node("render_pages", render_pages_node)
    workflow.add_node("parse_document", parse_document_node)
    workflow.add_node("extract_and_crop", extract_and_crop_node)
    workflow.add_node("index_vectors", index_vectors_node)
    workflow.add_node("index_colpali", index_colpali_node)

    # 2. Query Routing Node
    workflow.add_node("route_query", route_query_node)

    # 3. Granular Retrieval Nodes
    workflow.add_node("dense_retriever", dense_retriever_node)
    workflow.add_node("bm25_retriever", bm25_retriever_node)
    workflow.add_node("text_fusion", text_fusion_node)
    workflow.add_node("colpali_retriever", colpali_retriever_node)
    workflow.add_node("visual_hydration", visual_hydration_node)

    # 4. Context Formatting & CRAG Nodes
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("grade_context", grade_context_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # -------------------------------------------------------------------------
    # Edges & Flow
    # -------------------------------------------------------------------------

    # Entry point -> Check document status
    workflow.add_edge(START, "check_doc_status")

    # Conditional Branch 1: Ingest if new document, else fast path to route_query
    workflow.add_conditional_edges(
        "check_doc_status",
        decide_ingestion,
        {
            "render_pages": "render_pages",
            "route_query": "route_query",
        },
    )

    # Ingestion Pipeline Linear Flow
    workflow.add_edge("render_pages", "parse_document")
    workflow.add_edge("parse_document", "extract_and_crop")
    workflow.add_edge("extract_and_crop", "index_vectors")
    workflow.add_edge("index_vectors", "index_colpali")
    workflow.add_edge("index_colpali", "route_query")

    # Conditional Branch 2: Intent-based Multi-Modal Retrieval
    workflow.add_conditional_edges(
        "route_query",
        branch_retrieval,
        ["dense_retriever", "bm25_retriever", "colpali_retriever"],
    )

    # Text Retrieval Path (Dense + BM25 -> RRF Fusion -> Format Context)
    workflow.add_edge("dense_retriever", "text_fusion")
    workflow.add_edge("bm25_retriever", "text_fusion")
    workflow.add_edge("text_fusion", "format_context")

    # Visual Retrieval Path (ColPali -> Asset Hydration -> Format Context)
    workflow.add_edge("colpali_retriever", "visual_hydration")
    workflow.add_edge("visual_hydration", "format_context")

    # CRAG Corrective Loop Path
    workflow.add_edge("format_context", "grade_context")
    workflow.add_conditional_edges(
        "grade_context",
        decide_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )

    # Self-Correction Loop: Route reformulated query back to intent router
    workflow.add_edge("rewrite_query", "route_query")

    # Final Edge: Grounded generation to END
    workflow.add_edge("generate_answer", END)

    return workflow


# Compile the graph into a runnable application
app = build_rag_graph().compile()


def run_agentic_rag(
    question: str,
    pdf_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the full agentic multimodal RAG graph for a user question."""
    question = question.strip()
    if not question:
        return {
            "question": "",
            "answer": "No question provided.",
            "sources": [],
            "route": "none",
            "retry_count": 0,
        }

    target_pdf = str(pdf_path) if pdf_path else str(DEFAULT_PDF_PATH)

    initial_state: RAGState = {
        "question": question,
        "original_question": question,
        "pdf_path": target_pdf,
        "retry_count": 0,
    }

    final_state = app.invoke(initial_state)

    return {
        "question": final_state.get("original_question", question),
        "rewritten_query": final_state.get("rewritten_query"),
        "route": final_state.get("route", "both"),
        "routing_reasoning": final_state.get("routing_reasoning", ""),
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "retry_count": final_state.get("retry_count", 0),
        "is_relevant": final_state.get("is_relevant", True),
        "is_ingested": final_state.get("is_ingested", True),
    }
