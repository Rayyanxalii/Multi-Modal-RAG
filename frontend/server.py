"""FastAPI Web Server for the Multimodal Agentic RAG """

import os
import shutil
from pathlib import Path
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config import DATA_DIR, RAW_DATA_DIR, DEFAULT_PDF_PATH, get_document_paths
from rag.graph import run_agentic_rag
from rag.document_manager import list_all_documents, delete_document

app = FastAPI(title="Multimodal Agentic RAG - Sketch Interface")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent
STATIC_DIR = FRONTEND_DIR / "static"

# Mount data directory so cropped figures, tables, and rendered pages can be rendered in UI
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# Mount frontend static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    question: str
    doc_id: str | None = "attention"
    thread_id: str | None = None


@app.get("/")
def serve_home():
    """Serve the primary working sheet UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(index_file)


@app.get("/api/documents")
def get_documents() -> list[dict[str, Any]]:
    """List all ingested documents available for chat."""
    return list_all_documents()


@app.get("/api/documents/{doc_id}/pdf")
def get_document_pdf(doc_id: str):
    """Serve the raw PDF file directly for inline browser rendering in side-by-side view."""
    clean_id = (doc_id or "attention").strip().lower()
    pdf_path = RAW_DATA_DIR / f"{clean_id}.pdf"
    if not pdf_path.exists():
        if clean_id == "attention":
            pdf_path = DEFAULT_PDF_PATH
        else:
            matches = list(RAW_DATA_DIR.glob(f"{clean_id}*.pdf"))
            if matches:
                pdf_path = matches[0]
            else:
                raise HTTPException(status_code=404, detail=f"Document '{doc_id}' PDF not found.")

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file does not exist on disk.")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={pdf_path.name}"},
    )


@app.get("/api/documents/{doc_id}/pages")
def get_document_pages(doc_id: str) -> dict[str, Any]:
    """List all rendered page images for side-by-side inspection."""
    clean_id = (doc_id or "attention").strip().lower()
    pdf_path = RAW_DATA_DIR / f"{clean_id}.pdf"
    if not pdf_path.exists():
        if clean_id == "attention":
            pdf_path = DEFAULT_PDF_PATH
        else:
            matches = list(RAW_DATA_DIR.glob(f"{clean_id}*.pdf"))
            pdf_path = matches[0] if matches else None

    if not pdf_path or not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    paths = get_document_paths(pdf_path)
    pages_dir: Path = paths["rendered_pages_dir"]

    # Render pages on demand if not yet rendered
    if not pages_dir.exists() or not any(pages_dir.glob("page_*.png")):
        from rag.doc_loader import render_pdf_to_images
        render_pdf_to_images(pdf_path=pdf_path, output_dir=pages_dir)

    page_files = sorted(
        pages_dir.glob("page_*.png"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or "0"),
    )

    pages = []
    for idx, p in enumerate(page_files, 1):
        rel_path = p.relative_to(DATA_DIR).as_posix()
        pages.append({
            "page_number": idx,
            "image_url": f"/data/{rel_path}",
            "filename": p.name,
        })

    return {
        "status": "success",
        "doc_id": clean_id,
        "filename": pdf_path.name,
        "total_pages": len(pages),
        "pages": pages,
        "pdf_url": f"/api/documents/{clean_id}/pdf",
    }


@app.post("/api/chat")
def submit_question(payload: ChatRequest) -> dict[str, Any]:
    """Execute question answering via the Multimodal LangGraph agent."""
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    doc_id = (payload.doc_id or "attention").strip().lower()
    pdf_path = RAW_DATA_DIR / f"{doc_id}.pdf"
    if not pdf_path.exists():
        if doc_id == "attention":
            pdf_path = DEFAULT_PDF_PATH
        else:
            # Check if any matching pdf exists in RAW_DATA_DIR
            matches = list(RAW_DATA_DIR.glob(f"{doc_id}*.pdf"))
            pdf_path = matches[0] if matches else DEFAULT_PDF_PATH

    try:
        result = run_agentic_rag(question=question, pdf_path=pdf_path)

        # Normalize image paths in sources for frontend rendering
        sources = result.get("sources", [])
        for src in sources:
            img = src.get("image_path")
            if img:
                # Normalize relative path to /data/ URL
                clean_img = str(img).replace("\\", "/").replace("data/", "")
                src["web_image_url"] = f"/data/{clean_img}"

        return {
            "status": "success",
            "doc_id": doc_id,
            "question": result.get("question", question),
            "rewritten_query": result.get("rewritten_query"),
            "route": result.get("route", "both"),
            "routing_reasoning": result.get("routing_reasoning", ""),
            "answer": result.get("answer", ""),
            "sources": sources,
            "retry_count": result.get("retry_count", 0),
        }
    except Exception as exc:
        print(f"[Error in /api/chat]: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))



@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a new PDF file and initiate automated ingestion."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Sanitize document ID from filename
    raw_stem = Path(file.filename).stem.lower()
    doc_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw_stem).strip("_")
    if not doc_id:
        doc_id = f"doc_{Path(file.filename).stem[:8]}"

    target_pdf = RAW_DATA_DIR / f"{doc_id}.pdf"
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with target_pdf.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[Upload] Saved '{file.filename}' to '{target_pdf}' (doc_id={doc_id})")

    # Run ingestion check / first pass
    try:
        # Trigger ingestion graph with a priming status check
        ingest_result = run_agentic_rag(
            question="Summarize the core topic of this document.",
            pdf_path=target_pdf,
        )
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "summary_answer": ingest_result.get("answer", ""),
        }
    except Exception as exc:
        print(f"[Error during ingestion]: {exc}")
        return {
            "status": "uploaded",
            "doc_id": doc_id,
            "filename": file.filename,
            "notice": f"Saved, background ingestion note: {exc}",
        }


@app.delete("/api/documents/{doc_id}")
def purge_document(doc_id: str) -> dict[str, Any]:
    """Purge document assets, Qdrant vectors, and cloud embeddings."""
    return delete_document(doc_id, delete_raw_pdf=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
