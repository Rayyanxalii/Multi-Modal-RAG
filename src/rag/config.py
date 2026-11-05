import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
COLPALI_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "attention_colpali.pt"

# Vector database & retrieval constants
TEXT_COLLECTION_NAME = "research_papers"
VISUAL_COLLECTION_NAME = "research_figures_tables"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768
RRF_K = 60
DEFAULT_TEXT_TOP_K = 10
DEFAULT_FIGURE_TABLE_TOP_K = 5
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
