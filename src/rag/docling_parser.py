
import json
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = (
    PROJECT_ROOT
    / "src"
    / "pdf"
    / "attention.pdf"
)

OUTPUT_DIR = PROJECT_ROOT/ "src" / "pdf"
CHUNKS_PATH = OUTPUT_DIR / 'chunks'/ "attention_chunks.jsonl"
MARKDOWN_PATH = OUTPUT_DIR / 'chunks'/ "attention_parsed.md"


def create_chunks(pdf_path: Path) -> list[dict]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Convert the original PDF into a structured document
    print(f"Parsing PDF: {pdf_path.name}")

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    # print(f"This is the DOCUMENT_CONVERTER result: {result.document}")
    document = result.document
    
    

    # 2. Save structured Markdown for inspection
    markdown = document.export_to_markdown()
    MARKDOWN_PATH.write_text(markdown, encoding="utf-8")
    print(f"Parsed Markdown saved to: {MARKDOWN_PATH}")




    # 3. Chunk the document using its structure
    chunker = HybridChunker()

    chunks = []

    for index, chunk in enumerate(chunker.chunk(document)):
        # Add heading context to the text used for retrieval
        contextualized_text = chunker.contextualize(chunk)

        headings = chunk.meta.headings or []

        origin = getattr(chunk.meta, "origin", None)
        page_number = getattr(origin, "page_no", None)

        chunk_data = {
            "chunk_id": f"{pdf_path.stem}_{index:04d}",
            "document_id": pdf_path.stem,
            "file_name": pdf_path.name,
            "chunk_index": index,
            "page_number": page_number,
            "headings": headings,
            "text": chunk.text,
            "contextualized_text": contextualized_text,
        }

        chunks.append(chunk_data)




    # 4. Save chunks as JSON Lines
    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Chunks saved to: {CHUNKS_PATH}")

    return chunks





def main():
    chunks = create_chunks(PDF_PATH)

    # Inspect a few chunks before building indexes
    for chunk in chunks[:11]:
        print("\n" + "=" * 70)
        print("Chunk ID:", chunk["chunk_id"])
        print("Document ID:", chunk["document_id"])
        print("File Name:", chunk["file_name"])
        print("Chunk Index:", chunk["chunk_index"])
        print("Page:", chunk["page_number"])
        print("Headings:", chunk["headings"])
        print("Text:", chunk["text"])
        print('Contextualized_text:', chunk["contextualized_text"])


if __name__ == "__main__":
    main()
