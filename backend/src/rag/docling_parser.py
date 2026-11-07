import json
from pathlib import Path
from typing import Any

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from rag.config import (
    DEFAULT_PDF_PATH,
    CHUNKS_DIR,
    TEXT_CHUNKS_PATH,
    PARSED_MD_PATH,
    DOCUMENT_JSON_PATH,
)


def get_chunk_metadata(chunk: Any) -> dict:
    """
    Extract useful metadata from a Docling chunk.
    No additional LLM call is used.
    """
    meta = chunk.meta

    # Extract page numbers and bounding boxes from source provenance.
    page_numbers = set()
    bounding_boxes = []

    for item in (meta.doc_items or []):
        for provenance in (item.prov or []):
            page_no = getattr(provenance, "page_no", None)

            if page_no is not None:
                page_numbers.add(page_no)

            bbox = getattr(provenance, "bbox", None)

            if bbox is not None:
                bounding_boxes.append({
                    "page_number": page_no,
                    "left": getattr(bbox, "l", None),
                    "top": getattr(bbox, "t", None),
                    "right": getattr(bbox, "r", None),
                    "bottom": getattr(bbox, "b", None),
                })

    page_numbers = sorted(page_numbers)

    # Identify the types of source document items.
    item_types = sorted({
        type(item).__name__
        for item in (meta.doc_items or [])
    })

    # Extract headings and captions assigned by Docling.
    headings = list(meta.headings or [])
    captions = list(meta.captions or [])

    # Document origin information.
    origin = getattr(meta, "origin", None)

    return {
        "page_number": page_numbers[0] if page_numbers else None,
        "page_numbers": page_numbers,
        "headings": headings,
        "captions": captions,
        "item_types": item_types,
        "bounding_boxes": bounding_boxes,
        "source_file": getattr(origin, "filename", None),
        "source_mimetype": getattr(origin, "mimetype", None),
    }


def parse_pdf_to_chunks(
    pdf_path: Path = DEFAULT_PDF_PATH,
    doc_paths: dict | None = None,
) -> list[dict]:
    """Parse document layout via IBM Docling and chunk using HybridChunker."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    if doc_paths is None:
        from rag.config import get_document_paths
        doc_paths = get_document_paths(pdf_path)

    output_dir = doc_paths["chunks_dir"]
    document_json_path = doc_paths["document_json_path"]
    markdown_path = doc_paths["parsed_md"]
    chunks_path = doc_paths["chunks_path"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert the PDF into a structured document.
    print(f"Parsing PDF: {pdf_path.name}")
    try:
        from docling.document_converter import PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False  # Skip redundant local RapidOCR on digital PDFs
        pipeline_options.do_table_structure = True
        converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
    except Exception:
        converter = DocumentConverter()

    result = converter.convert(str(pdf_path))
    document = result.document

    document.save_as_json(document_json_path)
    print(f"Structured document saved to: {document_json_path}")

    # 2. Save Markdown for inspection.
    markdown = document.export_to_markdown()
    markdown_path.write_text(markdown, encoding="utf-8")
    print(f"Parsed Markdown saved to: {markdown_path}")

    # 3. Chunk the document using its structure.
    chunker = HybridChunker()
    chunks = []

    for index, chunk in enumerate(chunker.chunk(document)):
        contextualized_text = chunker.contextualize(chunk)
        metadata = get_chunk_metadata(chunk)

        chunk_data = {
            "chunk_id": f"{pdf_path.stem}_{index:04d}",
            "document_id": pdf_path.stem.lower(),
            "file_name": pdf_path.name,
            "chunk_index": index,
            **metadata,
            "text": chunk.text,
            "contextualized_text": contextualized_text,
        }
        chunks.append(chunk_data)

    # 4. Save as JSON Lines.
    with chunks_path.open("w", encoding="utf-8") as file:
        for chunk_data in chunks:
            file.write(
                json.dumps(chunk_data, ensure_ascii=False) + "\n"
            )

    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Chunks saved to: {chunks_path}")

    return chunks


def create_chunks(
    pdf_path: Path = DEFAULT_PDF_PATH,
    output_dir: Path = CHUNKS_DIR,
    document_json_path: Path = DOCUMENT_JSON_PATH,
    markdown_path: Path = PARSED_MD_PATH,
    chunks_path: Path = TEXT_CHUNKS_PATH,
) -> list[dict]:
    doc_paths = {
        "chunks_dir": output_dir,
        "document_json_path": document_json_path,
        "parsed_md": markdown_path,
        "chunks_path": chunks_path,
    }
    return parse_pdf_to_chunks(pdf_path, doc_paths=doc_paths)


def main():
    chunks = create_chunks()

    for chunk in chunks[:5]:
        print("\n" + "=" * 70)
        print("Chunk ID:", chunk["chunk_id"])
        print("Pages:", chunk["page_numbers"])
        print("Headings:", chunk["headings"])
        print("Item types:", chunk["item_types"])
        print("Captions:", chunk["captions"])
        print("Text preview:", chunk["text"][:300])


if __name__ == "__main__":
    main()
