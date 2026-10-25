
import json
from pathlib import Path

import fitz
from PIL import Image


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]

PDF_PATH = PROJECT_ROOT / "src" / "pdf" / "attention.pdf"

RENDERED_PAGES_DIR = (
    PROJECT_ROOT / "src" / "pdf" / "rendered_pages"
)

TABLES_JSONL = (
    PROJECT_ROOT / "src" / "pdf" / "chunks" / "attention_tables.jsonl"
)

TABLES_DIR = PROJECT_ROOT / "src" / "pdf" / "tables" / "cropped"


def crop_tables():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if not TABLES_JSONL.exists():
        raise FileNotFoundError(
            f"Table metadata not found: {TABLES_JSONL}"
        )

    with TABLES_JSONL.open("r", encoding="utf-8") as file:
        tables = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    pdf = fitz.open(PDF_PATH)
    updated_tables = []

    try:
        for table in tables:
            table_id = table["table_id"]
            page_numbers = table.get("page_numbers") or []
            boxes = table.get("bounding_boxes") or []

            if not page_numbers or not boxes:
                print(f"Skipping {table_id}: missing page or bbox.")
                updated_tables.append(table)
                continue

            page_number = page_numbers[0]
            page_index = page_number - 1

            if not 0 <= page_index < len(pdf):
                print(f"Skipping {table_id}: invalid page number.")
                updated_tables.append(table)
                continue

            page_image_path = (
                RENDERED_PAGES_DIR
                / f"page_{page_number:04d}.png"
            )

            if not page_image_path.exists():
                print(
                    f"Skipping {table_id}: "
                    f"{page_image_path.name} not found."
                )
                updated_tables.append(table)
                continue

            bbox = boxes[0]

            left = float(bbox["left"])
            top = float(bbox["top"])
            right = float(bbox["right"])
            bottom = float(bbox["bottom"])

            page = pdf[page_index]

            if page.rotation != 0:
                print(
                    f"Skipping {table_id}: rotated page "
                    "needs rotation-aware coordinate mapping."
                )
                updated_tables.append(table)
                continue

            # Convert Docling coordinates to top-left PDF coordinates.
            origin = bbox.get("coordinate_origin", "")

            if origin.endswith("BOTTOMLEFT"):
                x0_pt = left
                y0_pt = page.rect.height - top
                x1_pt = right
                y1_pt = page.rect.height - bottom
            else:
                x0_pt, y0_pt, x1_pt, y1_pt = (
                    left, top, right, bottom
                )

            with Image.open(page_image_path) as source:
                page_image = source.convert("RGB")

                # Use actual image dimensions instead of assuming DPI.
                scale_x = page_image.width / page.rect.width
                scale_y = page_image.height / page.rect.height

                x0 = round(x0_pt * scale_x)
                y0 = round(y0_pt * scale_y)
                x1 = round(x1_pt * scale_x)
                y1 = round(y1_pt * scale_y)

                # Clamp to page-image boundaries.
                x0 = max(0, min(x0, page_image.width))
                x1 = max(0, min(x1, page_image.width))
                y0 = max(0, min(y0, page_image.height))
                y1 = max(0, min(y1, page_image.height))

                if x1 <= x0 or y1 <= y0:
                    print(f"Skipping {table_id}: invalid crop bounds.")
                    updated_tables.append(table)
                    continue

                cropped = page_image.crop((x0, y0, x1, y1))

                output_path = TABLES_DIR / f"{table_id}.png"
                cropped.save(output_path)

                table["image_path"] = str(
                    output_path.relative_to(PROJECT_ROOT)
                )
                table["image_width"] = cropped.width
                table["image_height"] = cropped.height

                print(
                    f"Cropped {table_id} | page {page_number} | "
                    f"size {cropped.size}"
                )

            updated_tables.append(table)

    finally:
        pdf.close()

    # Save updated metadata without risking a partial overwrite.
    temp_path = TABLES_JSONL.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        for table in updated_tables:
            file.write(
                json.dumps(table, ensure_ascii=False) + "\n"
            )

    temp_path.replace(TABLES_JSONL)

    print(f"\nTable crops saved to: {TABLES_DIR}")
    print(f"Updated metadata: {TABLES_JSONL}")


if __name__ == "__main__":
    crop_tables()
