"""Render PDF pages to images."""

import gc
import os
from pathlib import Path
import pymupdf

from rag.config import DEFAULT_PDF_PATH, RENDERED_PAGES_DIR
from rag.memory_utils import (
    configure_mupdf_low_memory,
    log_memory,
    release_mupdf_caches,
)


def render_pdf_to_images(
    pdf_path: Path = DEFAULT_PDF_PATH,
    output_dir: Path | None = None,
) -> list[Path]:
    """Render all PDF pages as PNG images (memory-safe page-by-page cleanup)."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}"
        )

    if output_dir is None:
        from rag.config import get_document_paths
        output_dir = get_document_paths(pdf_path)["rendered_pages_dir"]

    from rag.fast_pdf_parser import is_low_memory_env
    low_mem = is_low_memory_env()
    base_scale = 1.5 if low_mem else 2.0
    max_dimension = 1200 if low_mem else 2400

    if low_mem:
        configure_mupdf_low_memory()

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    log_memory("render:start")

    with pymupdf.open(pdf_path) as pdf:
        total_pages = len(pdf)
        for index, page in enumerate(pdf):
            page_number = index + 1
            image_path = output_dir / f"page_{page_number}.png"

            scale = base_scale
            max_dim = max(page.rect.width, page.rect.height)
            if max_dim > 0 and (max_dim * scale) > max_dimension:
                scale = max_dimension / max_dim

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                alpha=False,
            )
            # Save atomically: a crash mid-write must not leave a truncated PNG
            # that later runs mistake for a completed page.
            tmp_path = output_dir / f".page_{page_number}.{os.getpid()}.tmp.png"
            try:
                try:
                    pixmap.save(str(tmp_path))
                finally:
                    del pixmap
                os.replace(tmp_path, image_path)
            finally:
                tmp_path.unlink(missing_ok=True)
            image_paths.append(image_path)

            # Release each page's pixmap buffers before rendering the next one.
            if low_mem or page_number % 5 == 0:
                release_mupdf_caches()
            else:
                gc.collect()

    log_memory(f"render:end ({total_pages} pages)")
    return image_paths


render_pdf_images = render_pdf_to_images


def main():
    """Render the default PDF and report the saved image paths."""
    image_paths = render_pdf_images()
    print(f"Rendered {len(image_paths)} pages.")
    print(f"Images saved to: {RENDERED_PAGES_DIR}")


if __name__ == "__main__":
    main()
