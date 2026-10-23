
import json
from pathlib import Path

import fitz
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = PROJECT_ROOT / "src" / "pdf" / "attention.pdf"
RENDERED_PAGES_DIR = PROJECT_ROOT / "src" / "pdf" / "rendered_pages"
FIGURES_JSONL = (
    PROJECT_ROOT / "src" / "pdf" / "chunks" / "attention_figures.jsonl"
)
FIGURES_DIR = PROJECT_ROOT / "src" / "pdf" / "figures" / "cropped"


def crop_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not FIGURES_JSONL.exists():
        raise FileNotFoundError(f"Metadata not found: {FIGURES_JSONL}")

    # Read existing figure metadata.
    with FIGURES_JSONL.open("r", encoding="utf-8") as file:
        figures = [json.loads(line) for line in file if line.strip()]

    # Read page dimensions only; no Docling conversion is performed.
    pdf = fitz.open(PDF_PATH)

    updated_figures = []

    try:
        for figure in figures:
            figure_id = figure["figure_id"]
            page_number = figure.get("page_number")
            boxes = figure.get("bounding_boxes") or []

            if not page_number or not boxes:
                print(f"Skipping {figure_id}: missing page or bbox.")
                updated_figures.append(figure)
                continue

            page_index = page_number - 1

            if not 0 <= page_index < len(pdf):
                print(f"Skipping {figure_id}: invalid page number.")
                updated_figures.append(figure)
                continue

            page_image_path = (
                RENDERED_PAGES_DIR / f"page_{page_number:04d}.png"
            )

            if not page_image_path.exists():
                print(f"Skipping {figure_id}: {page_image_path.name} missing.")
                updated_figures.append(figure)
                continue

            # This version handles the first provenance box.
            bbox = boxes[0]

            left = float(bbox["left"])
            top = float(bbox["top"])
            right = float(bbox["right"])
            bottom = float(bbox["bottom"])

            page = pdf[page_index]

            if page.rotation != 0:
                print(
                    f"Skipping {figure_id}: rotated PDF page "
                    "needs rotation-aware coordinate mapping."
                )
                updated_figures.append(figure)
                continue

            # Convert bottom-left PDF coordinates to top-left coordinates.
            if bbox.get("coordinate_origin", "").endswith("BOTTOMLEFT"):
                x0_pt = left
                y0_pt = page.rect.height - top
                x1_pt = right
                y1_pt = page.rect.height - bottom
            else:
                x0_pt, y0_pt, x1_pt, y1_pt = (
                    left, top, right, bottom
                )

            # Map PDF points to the actual rendered image dimensions.
            with Image.open(page_image_path) as page_image:
                page_image = page_image.convert("RGB")

                scale_x = page_image.width / page.rect.width
                scale_y = page_image.height / page.rect.height

                x0 = round(x0_pt * scale_x)
                y0 = round(y0_pt * scale_y)
                x1 = round(x1_pt * scale_x)
                y1 = round(y1_pt * scale_y)

                # Clamp coordinates to image boundaries.
                x0 = max(0, min(x0, page_image.width))
                x1 = max(0, min(x1, page_image.width))
                y0 = max(0, min(y0, page_image.height))
                y1 = max(0, min(y1, page_image.height))

                if x1 <= x0 or y1 <= y0:
                    print(f"Skipping {figure_id}: invalid crop coordinates.")
                    updated_figures.append(figure)
                    continue

                cropped = page_image.crop((x0, y0, x1, y1))

                output_path = FIGURES_DIR / f"{figure_id}.png"
                cropped.save(output_path)

                figure["image_path"] = str(
                    output_path.relative_to(PROJECT_ROOT)
                )
                figure["image_width"], figure["image_height"] = cropped.size

                print(
                    f"Cropped {figure_id} | page {page_number} | "
                    f"size {cropped.size} | saved {output_path.name}"
                )

            updated_figures.append(figure)

    finally:
        pdf.close()

    # Update metadata with the paths and dimensions of the crops.
    temporary_path = FIGURES_JSONL.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        for figure in updated_figures:
            file.write(json.dumps(figure, ensure_ascii=False) + "\n")

    temporary_path.replace(FIGURES_JSONL)

    print(f"\nFinished. Crops saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    crop_figures()
