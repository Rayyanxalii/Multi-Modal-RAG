from pathlib import Path
import pymupdf
from llama_index.core import Document

# Paths
SRC_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = SRC_DIR / "pdf" / "attention.pdf"
IMAGE_DIR = SRC_DIR / "pdf" / "rendered_pages"


def ingest_pdf(pdf_path: Path) -> list[Document]:
    
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    documents = []

    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            page_number = index + 1

            # Extract text from the current page
            page_text = page.get_text("text").strip()

            # Render the current page as an image for ColPali
            image_path = IMAGE_DIR / f"page_{page_number:04d}.png"

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False,
            )
            pixmap.save(str(image_path))

            # Create one LlamaIndex Document per page
            document = Document(
                text=page_text,
                metadata={
                    "document_id": pdf_path.stem,
                    "file_name": pdf_path.name,
                    "page_number": page_number,
                    "image_path": str(image_path),
                },
            )

            documents.append(document)

            print(
                f"Processed page {page_number}/{len(pdf)} "
                f"| Text characters: {len(page_text)}"
            )

    print(f"\nTotal pages processed: {len(documents)}")
    print(f"Rendered images saved in: {IMAGE_DIR.resolve()}")

    return documents


def document_loader():
    documents = ingest_pdf(PDF_PATH)

    if documents:
        print('Returning documents')
        return documents
    
    else:
        print('No documents found')
        return 



