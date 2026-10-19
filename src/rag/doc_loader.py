from pathlib import Path

import pymupdf

# Paths
SRC_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = SRC_DIR / "pdf" / "attention.pdf"
IMAGE_DIR = SRC_DIR / "pdf" / "rendered_pages"


def render_pdf_images(pdf_path: Path) -> list[Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = []

    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            page_number = index + 1

            image_path = IMAGE_DIR / f"page_{page_number:04d}.png"

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False,
            )
            pixmap.save(str(image_path))
            image_paths.append(image_path)

    return image_paths



def main():
    image_paths = render_pdf_images(PDF_PATH)

    print(f"Rendered {len(image_paths)} pages.")
    print(f"Images saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()
