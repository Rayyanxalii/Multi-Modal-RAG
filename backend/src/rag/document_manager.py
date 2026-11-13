"""Comprehensive document and chat lifecycle manager.

Handles complete multi-tenant deletion across all layers:
1. Local disk assets (rendered pages, cropped figures/tables, Docling chunks, metadata)
2. Qdrant Cloud vector database points (filtered by document_id)
3. Hugging Face Space visual embeddings (embeddings/{doc_id}.pt)
4. In-memory BM25 index and LangGraph state caches
"""

import shutil
from pathlib import Path
from typing import Any
import json

from qdrant_client import models

from rag.config import (
    DATA_DIR,
    CHUNKS_DIR,
    RAW_DATA_DIR,
    RENDERED_PAGES_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    EMBEDDINGS_DIR,
    TEXT_COLLECTION_NAME,
    get_document_paths,
)
from rag.text_retrieval.index_chunks import get_client as get_qdrant_client
from rag.text_retrieval.bm25_retriever import get_bm25_index_for_file
from rag.visual_retrieval.colpali_client import get_colpali_cloud_client


def delete_document(
    doc_id: str,
    delete_raw_pdf: bool = False,
) -> dict[str, Any]:
    """Purge all data, vectors, visual assets, and cloud embeddings for a document."""
    doc_id = (doc_id or "").strip().lower()
    if not doc_id:
        return {"status": "error", "message": "Document ID cannot be empty."}

    deleted_items: list[str] = []
    errors: list[str] = []

    # -------------------------------------------------------------------------
    # 1. Delete Local Asset Folders
    # -------------------------------------------------------------------------
    # Rendered pages directory
    pages_dir = RENDERED_PAGES_DIR / doc_id
    if pages_dir.exists() and pages_dir.is_dir():
        try:
            shutil.rmtree(pages_dir)
            deleted_items.append(f"Folder: {pages_dir.name}/")
        except Exception as exc:
            errors.append(f"Failed deleting pages dir: {exc}")

    # Cropped figures directory
    fig_dir = FIGURES_DIR / doc_id
    if fig_dir.exists() and fig_dir.is_dir():
        try:
            shutil.rmtree(fig_dir)
            deleted_items.append(f"Folder: {fig_dir.name}/")
        except Exception as exc:
            errors.append(f"Failed deleting figures dir: {exc}")

    # Cropped tables directory
    tbl_dir = TABLES_DIR / doc_id
    if tbl_dir.exists() and tbl_dir.is_dir():
        try:
            shutil.rmtree(tbl_dir)
            deleted_items.append(f"Folder: {tbl_dir.name}/")
        except Exception as exc:
            errors.append(f"Failed deleting tables dir: {exc}")

    # -------------------------------------------------------------------------
    # 2. Delete Local Chunk & Metadata Files
    # -------------------------------------------------------------------------
    chunk_patterns = [
        f"{doc_id}_chunks.jsonl",
        f"{doc_id}_document.json",
        f"{doc_id}_parsed.md",
        f"{doc_id}_figures.jsonl",
        f"{doc_id}_figures_cleaned.jsonl",
        f"{doc_id}_figure_review.jsonl",
        f"{doc_id}_tables.jsonl",
    ]
    for pattern in chunk_patterns:
        target_file = CHUNKS_DIR / pattern
        if target_file.exists():
            try:
                target_file.unlink()
                deleted_items.append(f"File: chunks/{pattern}")
            except Exception as exc:
                errors.append(f"Failed deleting {pattern}: {exc}")

    # Local embeddings cache (if any)
    emb_file = EMBEDDINGS_DIR / f"{doc_id}.pt"
    if emb_file.exists():
        try:
            emb_file.unlink()
            deleted_items.append(f"File: embeddings/{doc_id}.pt")
        except Exception as exc:
            errors.append(f"Failed deleting {doc_id}.pt: {exc}")

    # Optional: Delete uploaded raw PDF
    if delete_raw_pdf and doc_id != "attention":
        raw_pdf = RAW_DATA_DIR / f"{doc_id}.pdf"
        if raw_pdf.exists():
            try:
                raw_pdf.unlink()
                deleted_items.append(f"Raw PDF: raw/{doc_id}.pdf")
            except Exception as exc:
                errors.append(f"Failed deleting raw PDF: {exc}")

    # -------------------------------------------------------------------------
    # 3. Delete Points from Qdrant Cloud
    # -------------------------------------------------------------------------
    qdrant_deleted = False
    try:
        client = get_qdrant_client()
        client.delete(
            collection_name=TEXT_COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        qdrant_deleted = True
        deleted_items.append(f"Qdrant: vectors with document_id='{doc_id}'")
    except Exception as exc:
        errors.append(f"Qdrant deletion error: {exc}")

    # -------------------------------------------------------------------------
    # 4. Delete Embeddings from Hugging Face Space
    # -------------------------------------------------------------------------
    cloud_deleted = False
    try:
        colpali_client = get_colpali_cloud_client()
        res = colpali_client.delete_document(doc_id)
        if res.get("status") in ("success", "not_found"):
            cloud_deleted = True
            deleted_items.append(f"HF Space: embeddings/{doc_id}.pt")
    except Exception as exc:
        errors.append(f"HF Space deletion notice: {exc}")

    # -------------------------------------------------------------------------
    # 5. Invalidate In-Memory Caches
    # -------------------------------------------------------------------------
    try:
        get_bm25_index_for_file.cache_clear()
    except Exception:
        pass

    return {
        "status": "success" if not errors else "partial",
        "doc_id": doc_id,
        "deleted_count": len(deleted_items),
        "deleted_items": deleted_items,
        "qdrant_deleted": qdrant_deleted,
        "cloud_deleted": cloud_deleted,
        "errors": errors,
    }


def list_all_documents() -> list[dict[str, Any]]:
    """Scan and list all documents currently ingested and available in the system."""
    docs = []
    seen = set()

    # Discover from chunks directory
    for chunk_file in sorted(CHUNKS_DIR.glob("*_chunks.jsonl")):
        stem = chunk_file.stem.replace("_chunks", "")
        if stem in seen:
            continue
        seen.add(stem)

        # Count chunks
        try:
            with chunk_file.open("r", encoding="utf-8") as f:
                chunk_count = sum(1 for line in f if line.strip())
        except Exception:
            chunk_count = 0

        # Count figures
        fig_count = 0
        fig_file = CHUNKS_DIR / f"{stem}_figures_cleaned.jsonl"
        if not fig_file.exists():
            fig_file = CHUNKS_DIR / f"{stem}_figures.jsonl"
        if fig_file.exists():
            try:
                with fig_file.open("r", encoding="utf-8") as f:
                    fig_count = sum(1 for line in f if line.strip())
            except Exception:
                pass

        # Count tables
        tbl_count = 0
        tbl_file = CHUNKS_DIR / f"{stem}_tables.jsonl"
        if tbl_file.exists():
            try:
                with tbl_file.open("r", encoding="utf-8") as f:
                    tbl_count = sum(1 for line in f if line.strip())
            except Exception:
                pass

        # Count rendered pages
        pages_dir = RENDERED_PAGES_DIR if stem == "attention" else RENDERED_PAGES_DIR / stem
        page_count = len(list(pages_dir.glob("*.png"))) if pages_dir.exists() else 0

        docs.append({
            "doc_id": stem,
            "title": stem.replace("_", " ").title(),
            "pages": page_count,
            "chunks": chunk_count,
            "figures": fig_count,
            "tables": tbl_count,
        })

    # Discover from raw PDFs directory in case chunks are in progress or omitted
    for pdf_file in sorted(RAW_DATA_DIR.glob("*.pdf")):
        stem = pdf_file.stem.lower()
        if stem.startswith(".tmp_") or stem in seen:
            continue
        seen.add(stem)

        pages_dir = RENDERED_PAGES_DIR if stem == "attention" else RENDERED_PAGES_DIR / stem
        page_count = len(list(pages_dir.glob("*.png"))) if pages_dir.exists() else 0
        if page_count == 0:
            try:
                import pymupdf
                with pymupdf.open(pdf_file) as doc_obj:
                    page_count = len(doc_obj)
            except Exception:
                page_count = 0

        docs.append({
            "doc_id": stem,
            "title": stem.replace("_", " ").title(),
            "pages": page_count,
            "chunks": 0,
            "figures": 0,
            "tables": 0,
        })

    return docs
