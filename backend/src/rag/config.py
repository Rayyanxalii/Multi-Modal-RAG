import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

# Subdirectories for data assets
RAW_DATA_DIR = DATA_DIR / "raw"
DEFAULT_PDF_PATH = RAW_DATA_DIR / "attention.pdf"
RENDERED_PAGES_DIR = DATA_DIR / "rendered_pages"
CHUNKS_DIR = DATA_DIR / "chunks"
FIGURES_DIR = DATA_DIR / "figures"
TABLES_DIR = DATA_DIR / "tables"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RUNS_DIR = DATA_DIR / "retrieval_runs"

# Canonical chunk & metadata files
TEXT_CHUNKS_PATH = CHUNKS_DIR / "attention_chunks.jsonl"
DOCUMENT_JSON_PATH = CHUNKS_DIR / "attention_document.json"
PARSED_MD_PATH = CHUNKS_DIR / "attention_parsed.md"
FIGURES_JSONL = CHUNKS_DIR / "attention_figures.jsonl"
FIGURES_CLEANED_JSONL = CHUNKS_DIR / "attention_figures_cleaned.jsonl"
FIGURE_REVIEW_JSONL = CHUNKS_DIR / "attention_figure_review.jsonl"
TABLES_JSONL = CHUNKS_DIR / "attention_tables.jsonl"
COLPALI_EMBEDDINGS_PATH = (
    EMBEDDINGS_DIR / "pages_embeddings.pt"
    if (EMBEDDINGS_DIR / "pages_embeddings.pt").exists()
    else EMBEDDINGS_DIR / "attention_colpali.pt"
)

# Vector database & retrieval constants
TEXT_COLLECTION_NAME = os.getenv("QDRANT_TEXT_COLLECTION", "omni_documents")
VISUAL_COLLECTION_NAME = "research_figures_tables"
EMBEDDING_MODEL = "gemini-embedding-001"
VECTOR_SIZE = 768
RRF_K = 60
DEFAULT_TEXT_TOP_K = 5
DEFAULT_FIGURE_TABLE_TOP_K = 10
DEFAULT_CANDIDATE_K = 10

# Environment variables
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=True)


def get_qdrant_credentials() -> tuple[str, str]:
    """Retrieve Qdrant URL and API key from environment."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise ValueError(
            f"QDRANT_URL or QDRANT_API_KEY is missing in {ENV_PATH}"
        )
    return url, api_key


def get_ollama_client():
    """Return configured Ollama Client respecting OLLAMA_HOST."""
    import ollama
    host = os.getenv("OLLAMA_HOST")
    return ollama.Client(host=host) if host else ollama.Client()


def get_document_paths(
    pdf_path: Path | str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Return canonical data paths for any given document, scoped by user_id if provided."""
    if pdf_path is None:
        p = DEFAULT_PDF_PATH
    else:
        p = Path(pdf_path)

    doc_id = p.stem.lower()
    clean_user = "".join(c for c in (user_id or "") if c.isalnum() or c in ("-", "_")).strip()

    if clean_user and clean_user != "default":
        user_root = DATA_DIR / "users" / clean_user
        raw_dir = user_root / "raw"
        chunks_dir = user_root / "chunks"
        rendered_pages_dir = user_root / "rendered_pages" / doc_id
        figures_dir = user_root / "figures" / doc_id
        tables_dir = user_root / "tables" / doc_id
        embeddings_path = EMBEDDINGS_DIR / clean_user / f"{doc_id}.pt"
    else:
        clean_user = "default"
        raw_dir = RAW_DATA_DIR
        chunks_dir = CHUNKS_DIR
        if doc_id == "attention":
            rendered_pages_dir = RENDERED_PAGES_DIR
            figures_dir = FIGURES_DIR
            tables_dir = TABLES_DIR
        else:
            rendered_pages_dir = RENDERED_PAGES_DIR / doc_id
            figures_dir = FIGURES_DIR / doc_id
            tables_dir = TABLES_DIR / doc_id
        embeddings_path = EMBEDDINGS_DIR / f"{doc_id}.pt"

    # Cleaned figures preference
    cleaned_fig = chunks_dir / f"{doc_id}_figures_cleaned.jsonl"
    raw_fig = chunks_dir / f"{doc_id}_figures.jsonl"
    figures_jsonl = cleaned_fig if cleaned_fig.exists() else raw_fig

    return {
        "user_id": clean_user,
        "pdf_path": p,
        "raw_dir": raw_dir,
        "doc_id": doc_id,
        "doc_name": doc_id,
        "chunks_dir": chunks_dir,
        "rendered_pages_dir": rendered_pages_dir,
        "figures_dir": figures_dir,
        "tables_dir": tables_dir,
        "chunks_path": chunks_dir / f"{doc_id}_chunks.jsonl",
        "document_json": chunks_dir / f"{doc_id}_document.json",
        "document_json_path": chunks_dir / f"{doc_id}_document.json",
        "parsed_md": chunks_dir / f"{doc_id}_parsed.md",
        "figures_jsonl": figures_jsonl,
        "figures_raw_jsonl": raw_fig,
        "figures_cleaned": cleaned_fig,
        "tables_jsonl": chunks_dir / f"{doc_id}_tables.jsonl",
        "colpali_embeddings_path": embeddings_path,
    }
