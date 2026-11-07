from pathlib import Path
import pymupdf

from rag.config import DEFAULT_PDF_PATH, RENDERED_PAGES_DIR


def render_pdf_to_images(
    pdf_path: Path = DEFAULT_PDF_PATH,
    output_dir: Path | None = None,
) -> list[Path]:
    """Render all PDF pages as PNG images at 2x resolution."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    if output_dir is None:
        from rag.config import get_document_paths
        output_dir = get_document_paths(pdf_path)["rendered_pages_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            page_number = index + 1
            image_path = output_dir / f"page_{page_number}.png"

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False,
            )
            pixmap.save(str(image_path))
            image_paths.append(image_path)

    return image_paths


render_pdf_images = render_pdf_to_images


def main():
    image_paths = render_pdf_images()
    print(f"Rendered {len(image_paths)} pages.")
    print(f"Images saved to: {RENDERED_PAGES_DIR}")


if __name__ == "__main__":
    main()
