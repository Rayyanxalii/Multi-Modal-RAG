import json
from pathlib import Path

from docling_core.types.doc import (
    DoclingDocument,
    PictureItem,
    TableItem,
)

from rag.config import (
    PROJECT_ROOT,
    DOCUMENT_JSON_PATH,
    RENDERED_PAGES_DIR,
    FIGURES_DIR,
    FIGURES_JSONL as FIGURES_PATH,
    TABLES_JSONL as TABLES_PATH,
)


def get_item_metadata(item) -> dict:
    """Extract provenance metadata from a Docling document item."""

    page_numbers = set()
    bounding_boxes = []

    for provenance in (getattr(item, "prov", None) or []):
        page_number = getattr(provenance, "page_no", None)

        if page_number is not None:
            page_numbers.add(page_number)

        bbox = getattr(provenance, "bbox", None)

        if bbox is not None:
            bounding_boxes.append({
                "page_number": page_number,
                "left": getattr(bbox, "l", None),
                "top": getattr(bbox, "t", None),
                "right": getattr(bbox, "r", None),
                "bottom": getattr(bbox, "b", None),
                "coordinate_origin": str(
                    getattr(bbox, "coord_origin", None)
                ),
            })

    page_numbers = sorted(page_numbers)

    return {
        "page_number": page_numbers[0] if page_numbers else None,
        "page_numbers": page_numbers,
        "bounding_boxes": bounding_boxes,
    }


def get_caption(item, document) -> str | None:
    """Return the caption if Docling provides one."""

    try:
        caption = item.caption_text(doc=document)
        return caption or None
    except (AttributeError, TypeError, ValueError):
        return None


def extract_figures(
    document,
    output_dir: Path = FIGURES_DIR,
    figures_path: Path = FIGURES_PATH,
) -> list[dict]:
    """Extract picture elements from the saved Docling document."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    for item, _level in document.iterate_items():
        if not isinstance(item, PictureItem):
            continue

        metadata = get_item_metadata(item)
        figure_index = len(figures) + 1
        figure_id = f"attention_fig_{figure_index:03d}"

        image_path = None
        image_width = None
        image_height = None

        try:
            image = item.get_image(doc=document)

            if image is not None:
                output_path = output_dir / f"{figure_id}.png"
                image.save(output_path)

                image_path = str(
                    output_path.relative_to(PROJECT_ROOT)
                )
                image_width, image_height = image.size

        except Exception as exc:
            print(f"Could not export {figure_id}: {exc}")

        figure_data = {
            "figure_id": figure_id,
            "document_id": "attention",
            "file_name": "attention.pdf",
            **metadata,
            "caption": get_caption(item, document),
            "image_path": image_path,
            "image_width": image_width,
            "image_height": image_height,
            "item_type": type(item).__name__,
        }

        figures.append(figure_data)

    with figures_path.open("w", encoding="utf-8") as file:
        for figure in figures:
            file.write(
                json.dumps(figure, ensure_ascii=False) + "\n"
            )

    print(f"\nPicture elements found: {len(figures)}")
    print(f"Figure metadata saved to: {figures_path}")
    print(f"Figure images directory: {output_dir}")

    return figures


def extract_tables(
    document,
    tables_path: Path = TABLES_PATH,
) -> list[dict]:
    """Extract table elements as structured Markdown and metadata."""

    tables_path.parent.mkdir(parents=True, exist_ok=True)
    tables = []

    for item, _level in document.iterate_items():
        if not isinstance(item, TableItem):
            continue

        metadata = get_item_metadata(item)
        table_index = len(tables) + 1
        table_id = f"attention_table_{table_index:03d}"

        try:
            table_markdown = item.export_to_markdown(doc=document)
        except Exception as exc:
            print(f"Could not export {table_id} as Markdown: {exc}")
            table_markdown = ""

        table_data = {
            "table_id": table_id,
            "document_id": "attention",
            "file_name": "attention.pdf",
            **metadata,
            "caption": get_caption(item, document),
            "table_markdown": table_markdown,
            "item_type": type(item).__name__,
        }

        tables.append(table_data)

    with tables_path.open("w", encoding="utf-8") as file:
        for table in tables:
            file.write(
                json.dumps(table, ensure_ascii=False) + "\n"
            )

    print(f"\nTable elements found: {len(tables)}")
    print(f"Table metadata saved to: {tables_path}")

    return tables


def inspect_rendered_pages(rendered_pages_dir: Path = RENDERED_PAGES_DIR):
    """Check whether the existing rendered page images are available."""

    if not rendered_pages_dir.exists():
        print(f"Rendered pages directory not found: {rendered_pages_dir}")
        return

    images = sorted(rendered_pages_dir.glob("page_*.png"))

    print(f"\nExisting rendered page images: {len(images)}")

    if images:
        print("First image:", images[0].name)
        print("Last image:", images[-1].name)


def main():
    if not DOCUMENT_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Saved Docling JSON not found: {DOCUMENT_JSON_PATH}\n"
            "Run docling_parser.py first to create it."
        )

    print(f"Loading saved document: {DOCUMENT_JSON_PATH}")

    document = DoclingDocument.load_from_json(
        filename=DOCUMENT_JSON_PATH
    )

    figures = extract_figures(document)
    tables = extract_tables(document)

    inspect_rendered_pages()

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Figures: {len(figures)}")
    print(f"Tables: {len(tables)}")


if __name__ == "__main__":
    main()
