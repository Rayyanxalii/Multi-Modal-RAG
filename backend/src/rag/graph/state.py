"""State definition for the Multimodal Agentic RAG workflow."""

from typing import Annotated, Any, TypedDict
import operator


class RAGState(TypedDict, total=False):
    """Represents the complete state of the Multimodal RAG Graph.

    Document-agnostic state structure supporting text passages,
    diagrams, charts, figures, structured tables, and automated ingestion.
    """
    # Conversational Short-Term Memory (Accumulated across multi-turn session)
    chat_history: Annotated[list[dict[str, Any]], operator.add]
    # Document Input & Ingestion Status
    pdf_path: str                              # Path to source document (e.g., data/raw/attention.pdf)
    doc_name: str                              # Identifier (e.g., "attention")
    is_ingested: bool                          # True if document assets & vectors exist on disk/DB

    # Query & Intent
    question: str                              # Current active query (can be rewritten)
    original_question: str                     # Original user input verbatim
    original_query: str                        # Original user input
    route: str                                 # Intent classification: "text", "visual", or "both"
    routing_reasoning: str                     # LLM explanation of the chosen route

    # Granular Intermediate Retrieval Artifacts
    dense_matches: list[dict[str, Any]]        # Raw Qdrant vector matches
    bm25_matches: list[dict[str, Any]]         # Raw BM25 keyword matches
    text_results: list[dict[str, Any]]          # Fused text chunks from text_fusion (RRF)
    colpali_top_pages: list[dict[str, Any]]     # Top-ranked pages from ColPali MaxSim
    figure_table_results: list[dict[str, Any]]   # Hydrated figures & tables mapped to top pages

    # Formatted Grounding & Self-RAG Reflection
    context_str: str                           # Formatted grounded context for generation
    sources: list[dict[str, Any]]              # Traceable source citations with pages
    retry_count: int                           # Number of query rewrite attempts (guardrail)
    is_relevant: bool                          # Document relevance grade
    rewritten_query: str                       # Reformulated query if graded irrelevant
    is_grounded: bool                          # Grounding and citation verification

    # Final Answer
    answer: str                                # LLM synthesized response
