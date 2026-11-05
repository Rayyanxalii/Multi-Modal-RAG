"""Unified Ingestion Pipeline for Multimodal RAG.

This module coordinates document preprocessing and indexing end-to-end:
1. Render PDF pages to images (PyMuPDF) -> data/rendered_pages/
2. Parse layout and generate contextual text chunks (Docling) -> data/chunks/
3. Extract figures and tables metadata (Docling) -> data/figures/, data/tables/
4. Crop figure and table images from rendered pages -> data/figures/, data/tables/
5. Clean captions and normalize visual assets -> data/figures/figures_cleaned.jsonl
6. Index text chunks into Qdrant -> collection: text_chunks
7. Index figures and tables into Qdrant -> collection: figures_tables

Usage:
    python -m rag.ingest --pdf data/raw/attention.pdf
    python -m rag.ingest --skip-rendering
    python -m rag.ingest --only-indexing
"""

import argparse
import sys
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from rag.config import (
    DEFAULT_PDF_PATH,
    RENDERED_PAGES_DIR,
    TEXT_CHUNKS_PATH,
    DOCUMENT_JSON_PATH,
    FIGURES_JSONL,
    FIGURES_DIR,
    TABLES_JSONL,
    TABLES_DIR,
    FIGURES_CLEANED_JSONL,
    FIGURE_REVIEW_JSONL,
)
from rag.doc_loader import render_pdf_images
from rag.docling_parser import create_chunks
from rag.visual_retrieval.preprocess.figure_extractor import (
    extract_figures,
    extract_tables,
)
from rag.visual_retrieval.preprocess.figure_cropper import crop_figures
from rag.visual_retrieval.preprocess.table_cropper import crop_tables
from rag.visual_retrieval.preprocess.caption_processor import process_captions


def run_pipeline(
    pdf_path: Path = DEFAULT_PDF_PATH,
    skip_rendering: bool = False,
    skip_parsing: bool = False,
    skip_cropping: bool = False,
    skip_indexing: bool = False,
) -> None:
    """Run the multimodal document ingestion pipeline."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

    print("=" * 60)
    print(f"Starting Multimodal Ingestion Pipeline for: {pdf_path.name}")
    print("=" * 60)

    # Stage 1: Render PDF pages
    if not skip_rendering:
        print("\n[Stage 1/5] Rendering PDF pages as images...")
        render_pdf_images(pdf_path, output_dir=RENDERED_PAGES_DIR)
    else:
        print("\n[Stage 1/5] Skipping PDF page rendering.")

    # Stage 2: Parse layout and create chunks with Docling
    if not skip_parsing:
        print("\n[Stage 2/5] Parsing document layout and creating text chunks...")
        create_chunks(pdf_path, chunks_path=TEXT_CHUNKS_PATH)
    else:
        print("\n[Stage 2/5] Skipping Docling parsing.")

    # Stage 3: Extract figures and tables metadata
    if not skip_cropping:
        print("\n[Stage 3/5] Extracting figure and table metadata...")
        if not DOCUMENT_JSON_PATH.exists():
            print(
                f"Warning: {DOCUMENT_JSON_PATH} not found. "
                "Ensure Stage 2 has run previously."
            )
        else:
            doc = DoclingDocument.load_from_json(filename=DOCUMENT_JSON_PATH)
            extract_figures(doc, figures_path=FIGURES_JSONL)
            extract_tables(doc, tables_path=TABLES_JSONL)

        print("\n[Stage 4/5] Cropping figures, tables, and processing captions...")
        crop_figures(
            pdf_path=pdf_path,
            figures_jsonl=FIGURES_JSONL,
            output_dir=FIGURES_DIR,
            rendered_pages_dir=RENDERED_PAGES_DIR,
        )
        crop_tables(
            pdf_path=pdf_path,
            tables_jsonl=TABLES_JSONL,
            output_dir=TABLES_DIR,
            rendered_pages_dir=RENDERED_PAGES_DIR,
        )
        process_captions(
            input_path=FIGURES_JSONL,
            cleaned_path=FIGURES_CLEANED_JSONL,
            report_path=FIGURE_REVIEW_JSONL,
        )
    else:
        print("\n[Stage 3/5 & 4/5] Skipping visual extraction and cropping.")

    # Stage 5: Vector Indexing in Qdrant
    if not skip_indexing:
        print("\n[Stage 5/5] Indexing text chunks and visual assets in Qdrant...")
        try:
            from rag.text_retrieval.index_chunks import index_chunks
            from rag.visual_retrieval.search.index import index_figures_tables

            print("\n--- Indexing Text Chunks ---")
            index_chunks()

            print("\n--- Indexing Figures & Tables ---")
            index_figures_tables()
        except Exception as exc:
            print(f"Indexing encountered an error or Qdrant is unreachable: {exc}")
    else:
        print("\n[Stage 5/5] Skipping Qdrant vector indexing.")

    print("\n" + "=" * 60)
    print("Ingestion pipeline completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a PDF document into the multimodal RAG system."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF_PATH,
        help="Path to the PDF file to ingest (default: data/raw/attention.pdf).",
    )
    parser.add_argument(
        "--skip-rendering",
        action="store_true",
        help="Skip rendering PDF pages into images.",
    )
    parser.add_argument(
        "--skip-parsing",
        action="store_true",
        help="Skip Docling document parsing.",
    )
    parser.add_argument(
        "--skip-cropping",
        action="store_true",
        help="Skip cropping visual figures and tables.",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip Qdrant vector indexing.",
    )
    parser.add_argument(
        "--only-indexing",
        action="store_true",
        help="Run only the Qdrant vector indexing stage.",
    )

    args = parser.parse_args()

    if args.only_indexing:
        run_pipeline(
            pdf_path=args.pdf,
            skip_rendering=True,
            skip_parsing=True,
            skip_cropping=True,
            skip_indexing=False,
        )
    else:
        run_pipeline(
            pdf_path=args.pdf,
            skip_rendering=args.skip_rendering,
            skip_parsing=args.skip_parsing,
            skip_cropping=args.skip_cropping,
            skip_indexing=args.skip_indexing,
        )


if __name__ == "__main__":
    main()
