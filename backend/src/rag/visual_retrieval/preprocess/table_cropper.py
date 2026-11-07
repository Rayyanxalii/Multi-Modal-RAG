import json
from pathlib import Path

import fitz
from PIL import Image

from rag.config import (
    PROJECT_ROOT,
    DEFAULT_PDF_PATH as PDF_PATH,
    RENDERED_PAGES_DIR,
    TABLES_JSONL,
    TABLES_DIR,
)


def crop_tables(
    pdf_path: Path = PDF_PATH,
    tables_jsonl: Path = TABLES_JSONL,
    output_dir: Path = TABLES_DIR,
    rendered_pages_dir: Path = RENDERED_PAGES_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tables_jsonl.exists():
        raise FileNotFoundError(
            f"Table metadata not found: {tables_jsonl}"
        )

    with tables_jsonl.open("r", encoding="utf-8") as file:
        tables = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    pdf = fitz.open(pdf_path)
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

            candidate_paths = [
                rendered_pages_dir / f"page_{page_number:04d}.png",
                rendered_pages_dir / f"page_{page_number}.png",
            ]
            page_image_path = next((p for p in candidate_paths if p.exists()), None)

            if not page_image_path:
                print(f"Skipping {table_id}: page {page_number} image not found.")
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

            with Image.open(page_image_path) as page_image:
                page_image = page_image.convert("RGB")

                scale_x = page_image.width / page.rect.width
                scale_y = page_image.height / page.rect.height

                x0 = round(x0_pt * scale_x)
                y0 = round(y0_pt * scale_y)
                x1 = round(x1_pt * scale_x)
                y1 = round(y1_pt * scale_y)

                x0 = max(0, min(x0, page_image.width))
                x1 = max(0, min(x1, page_image.width))
                y0 = max(0, min(y0, page_image.height))
                y1 = max(0, min(y1, page_image.height))

                if x1 <= x0 or y1 <= y0:
                    print(f"Skipping {table_id}: invalid crop bounds.")
                    updated_tables.append(table)
                    continue

                cropped = page_image.crop((x0, y0, x1, y1))

                output_path = output_dir / f"{table_id}.png"
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

    temp_path = tables_jsonl.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        for table in updated_tables:
            file.write(
                json.dumps(table, ensure_ascii=False) + "\n"
            )

    temp_path.replace(tables_jsonl)

    print(f"\nTable crops saved to: {output_dir}")
    print(f"Updated metadata: {tables_jsonl}")


if __name__ == "__main__":
    crop_tables()
