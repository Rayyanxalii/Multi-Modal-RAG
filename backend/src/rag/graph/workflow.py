"""LangGraph workflow definition for the Multimodal Agentic RAG system.

Coordinates document ingestion checking, intent routing, parallel dense/BM25/ColPali
retrieval, RRF text fusion, visual asset hydration, Self-RAG relevance grading,
query reformulation, and grounded answer synthesis.
"""

import sqlite3
from pathlib import Path
from typing import Any, Literal, Generator
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from llama_index.core.llms import ChatMessage, MessageRole, ImageBlock

from services.llm_service import get_llm_vision
from rag.config import DEFAULT_PDF_PATH, get_document_paths
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
    build_vlm_messages,
    _is_unanswerable_response,
    filter_relevant_sources,
)


def decide_ingestion(state: RAGState) -> Literal["render_pages", "route_query"]:
    """Conditional edge: check if document is already ingested or needs pipeline execution."""
    if not state.get("is_ingested", False):
        return "render_pages"
    return "route_query"


def branch_retrieval(state: RAGState) -> list[str]:
    """Conditional edge: always invoke both Dense + BM25 and ColPali visual retrieval (routing commented out)."""
    # -------------------------------------------------------------------------
    # Intent-based routing logic (commented out per user instruction):
    # route = (state.get("route") or "both").lower()
    # if route == "dense":
    #     print(f"[Query Path: DENSE ONLY] Executing dense semantic vector search (Qdrant).")
    #     branches = ["dense_retriever"]
    # elif route == "bm25":
    #     print(f"[Query Path: BM25 ONLY] Executing lexical keyword search (BM25).")
    #     branches = ["bm25_retriever"]
    # elif route == "text":
    #     print(f"[Query Path: TEXT ONLY] Executing Dense Vector (Qdrant) + BM25 Keywords (Reciprocal Rank Fusion). ColPali is bypassed.")
    #     branches = ["dense_retriever", "bm25_retriever"]
    # elif route == "visual":
    #     print(f"[Query Path: COLPALI ONLY] Executing ColPali visual late-interaction on page embeddings. Dense/BM25 are bypassed.")
    #     branches = ["colpali_retriever"]
    # else:
    #     print(f"[Query Path: BOTH MULTIMODAL] Executing parallel Dense, BM25, AND ColPali visual late-interaction.")
    #     branches = ["dense_retriever", "bm25_retriever", "colpali_retriever"]
    # -------------------------------------------------------------------------
    print("\n" + "-" * 65)
    print("[Query Path: FIXED MULTIMODAL] Executing parallel Dense, BM25, AND ColPali (figures & tables).")
    print("-" * 65)
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


# Persistent SQLite database checkpointer for Short-Term Memory (STM)
DB_PATH = Path("data/chat_history.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_sqlite_conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
checkpointer = SqliteSaver(_sqlite_conn)
try:
    checkpointer.setup()
except Exception as exc:
    print(f"[Warning] SqliteSaver setup notice: {exc}")

# Compile the graph into a runnable application with persistent SQLite checkpointer
app = build_rag_graph().compile(checkpointer=checkpointer)


def run_agentic_rag(
    question: str,
    pdf_path: str | Path | None = None,
    thread_id: str | None = None,
    groq_api_key: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Execute the full agentic multimodal RAG graph for a user question with conversational memory."""
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
    clean_doc = Path(target_pdf).stem.lower()
    clean_user = "".join(c for c in (user_id or "") if c.isalnum() or c in ("-", "_")).strip() or "default"
    active_thread = thread_id or f"user_{clean_user}_doc_{clean_doc}"

    config = {"configurable": {"thread_id": active_thread}}

    initial_state: RAGState = {
        "question": question,
        "original_question": question,
        "original_query": question,
        "pdf_path": target_pdf,
        "doc_name": clean_doc,
        "user_id": clean_user,
        "retry_count": 0,
        "groq_api_key": groq_api_key,
    }

    final_state = app.invoke(initial_state, config=config)

    return {
        "question": final_state.get("original_question", question),
        "rewritten_query": final_state.get("rewritten_query"),
        "route": final_state.get("route", "both"),
        "routing_reasoning": final_state.get("routing_reasoning", ""),
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "context_str": final_state.get("context_str", ""),
        "retry_count": final_state.get("retry_count", 0),
        "is_relevant": final_state.get("is_relevant", True),
        "is_ingested": final_state.get("is_ingested", True),
        "thread_id": active_thread,
        "history_length": len(final_state.get("chat_history", [])),
    }


def stream_ingest_document(
    pdf_path: str | Path,
    user_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Execute the multi-stage document ingestion pipeline yielding live progress events for real-time UI bars."""
    target_pdf = str(pdf_path)
    pdf_obj = Path(target_pdf)
    clean_id = pdf_obj.stem.lower()
    clean_user = "".join(c for c in (user_id or "") if c.isalnum() or c in ("-", "_")).strip() or "default"
    doc_paths = get_document_paths(pdf_obj, user_id=clean_user)
    state: RAGState = {
        "pdf_path": target_pdf,
        "doc_name": clean_id,
        "user_id": clean_user,
        "question": "",
        "original_question": "",
        "retry_count": 0,
    }

    local_pages_exist = (
        doc_paths["rendered_pages_dir"].exists()
        and any(doc_paths["rendered_pages_dir"].glob("*.png"))
    )
    local_chunks_exist = (
        doc_paths["chunks_path"].exists()
        and doc_paths["chunks_path"].stat().st_size > 0
    )
    local_visuals_exist = (
        doc_paths["figures_raw_jsonl"].exists()
        or doc_paths["figures_cleaned"].exists()
    )

    # Stage 1: Render pages
    import gc
    if local_pages_exist:
        yield {"stage": "rendering", "progress": 15, "message": "Rendered pages already exist locally — skipping rendering..."}
    else:
        yield {"stage": "rendering", "progress": 15, "message": "Rendering high-resolution document pages..."}
        render_pages_node(state)
        gc.collect()

    # Stage 2: Layout parsing & hierarchical chunks
    if local_chunks_exist:
        yield {"stage": "parsing", "progress": 40, "message": "Layout & text chunks already exist — skipping re-parsing..."}
    else:
        yield {"stage": "parsing", "progress": 40, "message": "Parsing document layout, headings & tables..."}
        parse_document_node(state)
        gc.collect()

    # Stage 3: Figure & table extraction and cropping
    if local_visuals_exist:
        yield {"stage": "cropping", "progress": 65, "message": "Cropped figures & tables already exist — skipping cropping..."}
    else:
        yield {"stage": "cropping", "progress": 65, "message": "Extracting and cropping figures & data tables..."}
        extract_and_crop_node(state)
        gc.collect()

    # Stage 4: Dense vector indexing into Qdrant
    yield {"stage": "vectors", "progress": 82, "message": "Indexing text vectors into Qdrant..."}
    vec_res = index_vectors_node(state) or {}
    gc.collect()
    qdrant_existed = bool(vec_res.get("qdrant_existed"))
    if qdrant_existed:
        pts = vec_res.get("qdrant_points", "")
        yield {"stage": "vectors", "progress": 88, "message": f"Text vectors already exist in Qdrant ({pts} chunks) — reusing index!"}

    # Stage 5: ColPali visual late-interaction indexing
    yield {"stage": "colpali", "progress": 92, "message": "Checking visual pages on ColPali ZeroGPU Space..."}
    colpali_res = index_colpali_node(state) or {}
    colpali_existed = bool(colpali_res.get("colpali_existed"))
    if colpali_existed:
        yield {
            "stage": "colpali",
            "progress": 97,
            "message": f"Document pages already exist on ColPali Space — reused existing visual index!",
        }

    doc_filename = pdf_obj.name
    if colpali_existed and local_chunks_exist:
        complete_msg = f"'{doc_filename}' already exists on ColPali & workspace — added to chat and ready to query!"
    elif colpali_existed:
        complete_msg = f"'{doc_filename}' pages already existed on ColPali — reused visual index & added to chat!"
    else:
        complete_msg = f"Successfully ingested {doc_filename} and added to chat!"

    yield {
        "stage": "complete",
        "progress": 100,
        "message": complete_msg,
        "doc_id": clean_id,
        "pdf_path": target_pdf,
        "colpali_existed": colpali_existed,
        "qdrant_existed": qdrant_existed,
        "already_existed": bool(colpali_existed or local_chunks_exist or qdrant_existed),
    }


def ingest_document(pdf_path: str | Path) -> dict[str, Any]:
    """Execute the multi-stage document ingestion pipeline synchronously without invoking Q&A or polluting chat memory."""
    target_pdf = str(pdf_path)
    clean_id = Path(target_pdf).stem.lower()
    last_event: dict[str, Any] = {}
    for ev in stream_ingest_document(pdf_path):
        last_event = ev

    return {
        "status": "success",
        "pdf_path": target_pdf,
        "doc_id": clean_id,
        "is_ingested": True,
        "colpali_existed": bool(last_event.get("colpali_existed", False)),
        "already_existed": bool(last_event.get("already_existed", False)),
        "message": last_event.get("message", f"Successfully ingested {Path(target_pdf).name}!"),
    }


def stream_agentic_rag(
    question: str,
    pdf_path: str | Path | None = None,
    thread_id: str | None = None,
    groq_api_key: str | None = None,
    user_id: str | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Execute the agentic RAG workflow yielding real-time SSE progress events, status updates, and token deltas."""
    question = question.strip()
    if not question:
        yield {"type": "error", "message": "No question provided."}
        return

    target_pdf = str(pdf_path) if pdf_path else str(DEFAULT_PDF_PATH)
    clean_doc = Path(target_pdf).stem.lower()
    clean_user = "".join(c for c in (user_id or "") if c.isalnum() or c in ("-", "_")).strip() or "default"
    active_thread = thread_id or f"user_{clean_user}_doc_{clean_doc}"
    config = {"configurable": {"thread_id": active_thread}}

    # Retrieve existing checkpointer history for multi-turn dialogue
    try:
        checkpoint_state = app.get_state(config)
        existing_history = (
            checkpoint_state.values.get("chat_history", [])
            if checkpoint_state and checkpoint_state.values
            else []
        )
    except Exception:
        existing_history = []

    state: RAGState = {
        "question": question,
        "original_question": question,
        "original_query": question,
        "pdf_path": target_pdf,
        "doc_name": clean_doc,
        "user_id": clean_user,
        "retry_count": 0,
        "chat_history": existing_history,
        "groq_api_key": groq_api_key,
    }

    # 1. Routing & Anaphora resolution
    yield {"type": "status", "stage": "intent", "message": "Looking into document..."}
    state.update(route_query_node(state))

    # 2. Parallel Tri-Hybrid Retrieval (Dense + BM25 + ColPali)
    yield {"type": "status", "stage": "retrieval", "message": "Searching document..."}
    state.update(dense_retriever_node(state))
    state.update(bm25_retriever_node(state))
    state.update(colpali_retriever_node(state))

    # 3. Text Fusion & Visual Hydration
    yield {"type": "status", "stage": "fusion", "message": "Combining results..."}
    state.update(text_fusion_node(state))
    state.update(visual_hydration_node(state))
    state.update(format_context_node(state))

    # 4. Relevance Grading & CRAG Query Expansion
    state.update(grade_context_node(state))

    if not state.get("is_relevant", True) and state.get("retry_count", 0) < 1:
        yield {"type": "status", "stage": "crag", "message": "Refining search..."}
        state.update(rewrite_query_node(state))
        state.update(dense_retriever_node(state))
        state.update(bm25_retriever_node(state))
        state.update(colpali_retriever_node(state))
        state.update(text_fusion_node(state))
        state.update(visual_hydration_node(state))
        state.update(format_context_node(state))

    # 5. Build Vision LLM Payload
    messages, user_blocks, system_prompt, user_dialogue_question, selected_text = build_vlm_messages(state)

    if not messages:
        yield {"type": "token", "delta": "I don't have information about that in the document."}
        yield {
            "type": "metadata",
            "doc_id": clean_doc,
            "thread_id": active_thread,
            "question": user_dialogue_question,
            "sources": [],
            "route": state.get("route", "both"),
            "full_answer": "I don't have information about that in the document.",
        }
        yield {"type": "done"}
        return

    # 6. Stream Answer Generation
    yield {"type": "status", "stage": "generating", "message": "Synthesizing answer..."}
    full_answer = ""
    try:
        active_vision = get_llm_vision(groq_api_key)
        stream_res = active_vision.stream_chat(messages)
        for chunk in stream_res:
            delta = chunk.delta or ""
            if delta:
                full_answer += delta
                yield {"type": "token", "delta": delta}
    except Exception as exc:
        err_str = str(exc)
        if "413" in err_str or "rate_limit_exceeded" in err_str or "tokens" in err_str.lower():
            yield {"type": "status", "stage": "self_healing", "message": "Rate limit exceeded. Adapting token payload..."}
            top_chunk_text = ""
            if selected_text:
                raw_c = selected_text[0].get("text") or selected_text[0].get("contextualized_text") or ""
                top_chunk_text = f"Context:\n{str(raw_c)[:450]}\n\n"
            lean_user_prompt = f"{top_chunk_text}Question: {user_dialogue_question}\n\nAnswer concisely based on the context above."
            lean_messages = [
                ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful AI assistant. Answer the user question concisely using the provided context."),
                ChatMessage(role=MessageRole.USER, content=lean_user_prompt),
            ]
            try:
                active_vision = get_llm_vision(groq_api_key)
                stream_res = active_vision.stream_chat(lean_messages)
                for chunk in stream_res:
                    delta = chunk.delta or ""
                    if delta:
                        full_answer += delta
                        yield {"type": "token", "delta": delta}
            except Exception as retry_exc:
                err_msg = "The AI service is currently busy due to free-tier rate limits. Please wait a few moments and ask again."
                full_answer = err_msg
                yield {"type": "token", "delta": err_msg}
        elif any(tok in err_str.lower() for tok in ("api key", "invalid_api_key", "401", "authentication", "unauthorized")):
            yield {
                "type": "error",
                "content": "Invalid or missing Groq Vision LLM Key. Please click 'Vision LLM Key' (🔑) to update your key.",
            }
            return
        else:
            err_msg = f"An error occurred: {exc}"
            full_answer = err_msg
            yield {"type": "token", "delta": err_msg}

    # 7. Grounding filter & checkpointer update
    is_error_answer = full_answer.startswith("An error occurred:")
    if is_error_answer:
        sources = []
    elif _is_unanswerable_response(full_answer):
        full_answer = "I don't have information about that in the document."
        sources = []
    else:
        raw_sources = state.get("sources", [])
        sources = filter_relevant_sources(full_answer, raw_sources, question=user_dialogue_question)

    if not is_error_answer:
        new_turn = [
            {"role": "user", "content": user_dialogue_question},
            {"role": "assistant", "content": full_answer, "sources": sources},
        ]

        try:
            app.update_state(config, {
                "answer": full_answer,
                "sources": sources,
                "chat_history": new_turn,
            })
        except Exception as exc:
            print(f"[Warning] Failed updating checkpointer in streaming mode: {exc}")

    yield {
        "type": "metadata",
        "doc_id": clean_doc,
        "thread_id": active_thread,
        "question": user_dialogue_question,
        "rewritten_query": state.get("rewritten_query"),
        "route": state.get("route", "both"),
        "routing_reasoning": state.get("routing_reasoning", ""),
        "sources": sources,
        "full_answer": full_answer,
        "retry_count": state.get("retry_count", 0),
    }
    yield {"type": "done"}

